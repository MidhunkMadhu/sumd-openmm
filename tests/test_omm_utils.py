"""
OpenMM setup helpers: the CHARMM force switch (omm_utils.vfswitch) and the
input handling of build_simulation (yes/no keys, dispersion correction,
barostats) on a small Amber water box.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit

from sumd_openmm.omm_setup import build_simulation, load_production_helpers, yes_no
from sumd_openmm.omm_utils import vfswitch

R_ON, R_OFF = 1.0, 1.2
SIGMA, EPS = 0.35, 0.5


def _switched(r):
    """Steinbach-Brooks force-switched Lennard-Jones energy."""
    A, B = 4 * EPS * SIGMA**12, 4 * EPS * SIGMA**6
    if r >= R_OFF:
        return 0.0
    if r >= R_ON:
        k12 = R_OFF**6 / (R_OFF**6 - R_ON**6)
        k6 = R_OFF**3 / (R_OFF**3 - R_ON**3)
        return A * k12 * (r**-6 - R_OFF**-6)**2 - B * k6 * (r**-3 - R_OFF**-3)**2
    return A * (r**-12 - (R_ON * R_OFF)**-6) - B * (r**-6 - (R_ON * R_OFF)**-3)


def _pair_system(exception):
    """Two LJ particles in a 10 nm box; `exception` makes them a 1-4 pair."""
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*(openmm.Vec3(*row) for row in 10 * np.eye(3)))
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(R_OFF)
    for _ in range(2):
        system.addParticle(1.0)
        nb.addParticle(0.0, SIGMA, EPS)
    if exception:
        nb.addException(0, 1, 0.0, SIGMA, EPS)
    system.addForce(nb)
    return vfswitch(system, r_on=R_ON, r_off=R_OFF)


def _energy(system, r):
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([(0, 0, 0), (r, 0, 0)])
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


@pytest.mark.parametrize("exception", [False, True], ids=["nonbonded", "1-4 pair"])
def test_force_switch_matches_analytic_and_is_zero_past_cutoff(exception):
    system = _pair_system(exception)
    for r in (0.4, 0.9, 1.05, 1.19, 1.25, 1.5, 3.0):
        assert _energy(system, r) == pytest.approx(_switched(r), abs=1e-12), r


def _in_subprocess(code, cwd=None):
    """
    Run `code` in a fresh interpreter: an OpenMM C++ abort (as a long-range
    correction that does not converge used to cause) would kill pytest.
    """
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.path.dirname(__file__)]))
    run = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    return run.stdout


def test_force_switch_with_long_range_correction():
    out = _in_subprocess("""
        import openmm
        from test_omm_utils import _pair_system, _energy
        system = _pair_system(False)
        for force in system.getForces():
            if isinstance(force, openmm.CustomNonbondedForce):
                force.setUseLongRangeCorrection(True)
        print(_energy(system, 3.0))
        """)
    assert float(out) == pytest.approx(0.0, abs=1e-12)


def test_force_switch_keeps_non_periodic_cutoff():
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    system.addParticle(1.0)
    nb.addParticle(0.0, SIGMA, EPS)
    system.addForce(nb)
    vfswitch(system, r_on=R_ON, r_off=R_OFF)
    switched = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)]
    assert switched[0].getNonbondedMethod() == openmm.CustomNonbondedForce.CutoffNonPeriodic


def test_yes_no():
    assert yes_no("Yes", "k") and yes_no("TRUE", "k") and yes_no(" on ", "k")
    assert not yes_no("No", "k") and not yes_no("0", "k")
    with pytest.raises(ValueError, match="k must be yes or no"):
        yes_no("maybe", "k")


@pytest.fixture(scope="module")
def water_box(tmp_path_factory):
    """Amber prmtop/inpcrd of a small TIP3P box, written with ParmEd."""
    parmed = pytest.importorskip("parmed")
    ff = app.ForceField("amber14/tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, boxSize=openmm.Vec3(2.5, 2.5, 2.5) * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer, rigidWater=False)
    structure = parmed.openmm.load_topology(modeller.topology, system, modeller.positions)
    path = tmp_path_factory.mktemp("water")
    structure.save(str(path / "water.parm7"))
    structure.save(str(path / "water.rst7"))
    return path


def _build(water_box, monkeypatch, **keys):
    monkeypatch.chdir(water_box)
    cfg = dict(force_field="AMBER", topology_file="water.parm7", coordinate_file="water.rst7",
               dt="0.002", temp="300", genvel="yes", platform="CPU", r_off="1.0", r_on="0.8")
    cfg.update(keys)
    notes = []
    omm = build_simulation(load_production_helpers(), cfg, 1234, log=notes.append)
    return omm, notes


def _forces(omm, cls):
    return [f for f in omm.system.getForces() if isinstance(f, cls)]


def test_lj_lrc(water_box, monkeypatch):
    def correction(**keys):
        omm, _ = _build(water_box, monkeypatch, pcouple="no", vdw="Cutoff", **keys)
        return _forces(omm, openmm.NonbondedForce)[0].getUseDispersionCorrection()
    assert correction()                     # Amber default kept when the key is left out
    assert not correction(lj_lrc="no")
    assert correction(lj_lrc="Yes")


def test_force_switch_with_lj_lrc_builds(water_box):
    _in_subprocess("""
        import pytest
        from test_omm_utils import _build
        omm, _ = _build(%r, pytest.MonkeyPatch(), pcouple="no", vdw="Force-switch",
                        lj_lrc="yes")
        omm.simulation.step(10)
        """ % str(water_box))


@pytest.mark.parametrize("p_type, cls", [("isotropic", "MonteCarloBarostat"),
                                         ("anisotropic", "MonteCarloAnisotropicBarostat"),
                                         ("membrane", "MonteCarloMembraneBarostat")])
def test_barostat(water_box, monkeypatch, p_type, cls):
    omm, notes = _build(water_box, monkeypatch, pcouple="Yes", p_type=p_type)
    barostats = _forces(omm, getattr(openmm, cls))
    assert len(barostats) == 1 and omm.has_barostat
    assert barostats[0].getRandomNumberSeed() == 1234
    assert not any("p_type not set" in n for n in notes)


def test_default_barostat_is_reported(water_box, monkeypatch):
    omm, notes = _build(water_box, monkeypatch)
    assert _forces(omm, openmm.MonteCarloMembraneBarostat)
    assert any("p_type not set" in n for n in notes)


def test_no_barostat_and_bad_yes_no(water_box, monkeypatch):
    omm, _ = _build(water_box, monkeypatch, pcouple="NO")
    assert not omm.has_barostat
    assert not [f for f in omm.system.getForces() if "Barostat" in type(f).__name__]
    with pytest.raises(ValueError, match="pcouple must be yes or no"):
        _build(water_box, monkeypatch, pcouple="maybe")
