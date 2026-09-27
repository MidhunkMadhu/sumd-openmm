"""
omm_utils.py

OpenMM building blocks used by the MD setup: pressure coupling, CHARMM
force switching of Lennard-Jones interactions, making molecules whole in
the periodic box, and reading CHARMM topology, coordinate and parameter
files.

Force switching follows Steinbach and Brooks, J. Comput. Chem. 15, 667
(1994): for r_on < r < r_off the Lennard-Jones force goes smoothly to
zero at r_off,

    E(r) = A k12 (r^-6 - r_off^-6)^2 - B k6 (r^-3 - r_off^-3)^2,
    k12 = r_off^6 / (r_off^6 - r_on^6),   k6 = r_off^3 / (r_off^3 - r_on^3),

and below r_on the unmodified potential is shifted to join it,

    E(r) = A (r^-12 - (r_on r_off)^-6) - B (r^-6 - (r_on r_off)^-3),

with A = 4 eps sigma^12 and B = 4 eps sigma^6 (Lorentz-Berthelot), or
A = a^2 and B = b from NBFIX tables written as (a/r^6)^2 - b/r^6.
"""

from collections import deque

import numpy as np

__all__ = ["barostat", "vfswitch", "rewrap", "read_top", "read_crd", "read_params", "gen_box"]

_SWITCH = ("select(step(r - r_on), A*k12*(1/r^6 - 1/r_off^6)^2 - B*k6*(1/r^3 - 1/r_off^3)^2,"
           " A*(1/r^12 - 1/(r_on*r_off)^6) - B*(1/r^6 - 1/(r_on*r_off)^3));"
           " k12 = r_off^6/(r_off^6 - r_on^6); k6 = r_off^3/(r_off^3 - r_on^3);"
           " r_on = %r; r_off = %r")


def barostat(system, inputs):
    """Add the Monte Carlo barostat named by inputs.p_type (isotropic or membrane)."""
    import openmm
    from openmm import unit

    pressure = inputs.p_ref * unit.bar
    temperature = inputs.temp * unit.kelvin
    if inputs.p_type == "isotropic":
        force = openmm.MonteCarloBarostat(pressure, temperature, inputs.p_freq)
    elif inputs.p_type == "membrane":
        tension = inputs.p_tens * 10.0 * unit.bar * unit.nanometer      # dyn/cm -> bar nm
        force = openmm.MonteCarloMembraneBarostat(pressure, tension, temperature,
                                                  inputs.p_XYMode, inputs.p_ZMode, inputs.p_freq)
    else:
        raise ValueError("p_type must be isotropic or membrane, not %s" % inputs.p_type)
    system.addForce(force)
    return system


def vfswitch(system, psf=None, inputs=None, r_on=None, r_off=None):
    """
    Switch Lennard-Jones forces to zero between r_on and r_off (nm), taken
    from inputs.r_on/inputs.r_off unless given. Plain Lennard-Jones moves
    from the NonbondedForce to a switched CustomNonbondedForce; an NBFIX
    table force is switched in place; 1-4 Lennard-Jones terms move to a
    switched CustomBondForce. Electrostatics are unchanged.
    """
    import openmm

    r_on = inputs.r_on if r_on is None else r_on
    r_off = inputs.r_off if r_off is None else r_off
    nonbonded = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    tables = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)
              and f.getNumTabulatedFunctions() == 2]

    if tables:
        table = tables[0]
        table.setEnergyFunction(_SWITCH % (r_on, r_off) +
                                "; A = acoef(type1, type2)^2; B = bcoef(type1, type2)")
        table.setUseLongRangeCorrection(False)
    else:
        switched = openmm.CustomNonbondedForce(
            _SWITCH % (r_on, r_off) + "; A = 4*eps*sigma^12; B = 4*eps*sigma^6;"
            " sigma = half_sigma1 + half_sigma2; eps = root_eps1*root_eps2")
        switched.addPerParticleParameter("half_sigma")
        switched.addPerParticleParameter("root_eps")
        switched.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        switched.setCutoffDistance(nonbonded.getCutoffDistance())
        for i in range(nonbonded.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            switched.addParticle([0.5 * sigma._value, np.sqrt(epsilon._value)])
            nonbonded.setParticleParameters(i, charge, sigma, 0.0)
        for i in range(nonbonded.getNumExceptions()):
            a, b = nonbonded.getExceptionParameters(i)[:2]
            switched.addExclusion(a, b)
        switched.setForceGroup(nonbonded.getForceGroup())
        system.addForce(switched)

    pairs = openmm.CustomBondForce(_SWITCH % (r_on, r_off) +
                                   "; A = 4*eps*sigma^12; B = 4*eps*sigma^6")
    pairs.addPerBondParameter("sigma")
    pairs.addPerBondParameter("eps")
    for i in range(nonbonded.getNumExceptions()):
        a, b, charge, sigma, epsilon = nonbonded.getExceptionParameters(i)
        if epsilon._value != 0:
            pairs.addBond(a, b, [sigma._value, epsilon._value])
            nonbonded.setExceptionParameters(i, a, b, charge, sigma, 0.0)
    pairs.setForceGroup(nonbonded.getForceGroup())
    system.addForce(pairs)
    return system


def rewrap(simulation):
    """
    Make every molecule whole: walk each molecule's bonds and place every
    atom at the periodic image nearest to the atom it is bonded to. Works
    for molecules of any size; atoms without bonds are left in place.
    """
    from openmm import unit

    state = simulation.context.getState(getPositions=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).copy()
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    inverse = np.linalg.inv(box)
    neighbours = [[] for _ in range(len(x))]
    for bond in simulation.topology.bonds():
        neighbours[bond[0].index].append(bond[1].index)
        neighbours[bond[1].index].append(bond[0].index)
    placed = np.zeros(len(x), bool)
    for start in range(len(x)):
        if placed[start]:
            continue
        placed[start] = True
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j in neighbours[i]:
                if not placed[j]:
                    d = x[j] - x[i]
                    x[j] = x[i] + d - np.round(d @ inverse) @ box
                    placed[j] = True
                    queue.append(j)
    simulation.context.setPositions(x * unit.nanometer)
    return simulation


def read_top(filename, fftype="CHARMM"):
    from openmm import app
    return app.CharmmPsfFile(filename) if fftype == "CHARMM" else app.AmberPrmtopFile(filename)


def read_crd(filename, fftype="CHARMM"):
    from openmm import app
    return app.CharmmCrdFile(filename) if fftype == "CHARMM" else app.AmberInpcrdFile(filename)


def read_params(filename):
    """CharmmParameterSet from a list of parameter files, one per line ('!' starts a comment)."""
    from openmm import app
    files = [line.split("!", 1)[0].strip() for line in open(filename)]
    return app.CharmmParameterSet(*[f for f in files if f])


def gen_box(psf, crd):
    """Rectangular box spanning the coordinates, set on a CharmmPsfFile."""
    from openmm import unit
    xyz = np.asarray(crd.positions.value_in_unit(unit.nanometer))
    size = xyz.max(axis=0) - xyz.min(axis=0)
    psf.setBox(*(float(v) * unit.nanometer for v in size))
    return psf
