"""
equilibration.py

Run a CHARMM-GUI equilibration protocol (charmmgui.Protocol) in OpenMM
before SuMD production.

Each stage builds its own System from the topology, with that stage's
restraints and barostat, and continues from the previous stage's State
(positions, velocities, box). Restraint references are the initial
structure, as CHARMM-GUI's -ref / -r files are. Per stage the folder gets

    <stage>.log   OpenMM StateDataReporter output
    <stage>.dcd   trajectory at the stage's output frequency
    <stage>.xml   final State, written last; its presence marks the stage done

and equilibration.log records every stage's settings and outcome. A rerun
skips completed stages, so a job that hits its wall-time limit resumes.
The last State is copied to equilibrated.xml (and equilibrated.rst7).
"""

import datetime
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np

from .omm_setup import choose_platform, available_platforms, force_switch, platform_properties, precision

TEST_STEPS = 50


def _restraint_forces(stage, reference):
    import openmm

    forces = []
    for group in stage.positions:
        force = openmm.CustomExternalForce(
            "kx*periodicdistance(x, 0, 0, x0, 0, 0)^2 + ky*periodicdistance(0, y, 0, 0, y0, 0)^2"
            " + kz*periodicdistance(0, 0, z, 0, 0, z0)^2")
        for name in ("kx", "ky", "kz", "x0", "y0", "z0"):
            force.addPerParticleParameter(name)
        for atom, k in zip(group.atoms, group.k):
            force.addParticle(int(atom), [*map(float, k), *map(float, reference[atom])])
        force.setName("restraint: %s" % group.label)
        forces.append(force)
    for group in stage.torsions:
        force = openmm.CustomTorsionForce(
            "select(step(l1 - d), klo*((l1 - lo)^2 + 2*(l1 - lo)*(d - l1)),"
            " select(step(lo - d), klo*(d - lo)^2,"
            " select(step(d - u4), khi*((u4 - hi)^2 + 2*(u4 - hi)*(d - u4)),"
            " select(step(d - hi), khi*(d - hi)^2, 0))));"
            " d = theta - c - 2*pi*floor((theta - c + pi)/(2*pi)); pi = 3.141592653589793")
        for name in ("c", "l1", "lo", "hi", "u4", "klo", "khi"):
            force.addPerTorsionParameter(name)
        for quad, params in zip(group.atoms, group.params):
            force.addTorsion(*map(int, quad), list(map(float, params)))
        force.setName("restraint: %s" % group.label)
        forces.append(force)
    return forces


class Equilibration:

    def __init__(self, protocol, outdir, platform="auto", gpu_precision="single",
                 device_index=None, seed=None, log=print, test=False):
        self.p = protocol
        self.out = outdir
        self.platform_request = platform
        self.precision = gpu_precision
        self.device_index = device_index
        self.seed = seed
        self.log = log
        self.test = test
        os.makedirs(outdir, exist_ok=True)
        self.journal = open(os.path.join(outdir, "equilibration.log"), "a", buffering=1)
        self._load()

    def note(self, message):
        line = "[%s] %s" % (datetime.datetime.now().isoformat(timespec="seconds"), message)
        self.journal.write(line + "\n")
        self.log(message)

    def _load(self):
        from openmm import app
        from openmm.unit import nanometer

        if self.p.format == "AMBER":
            self.top = app.AmberPrmtopFile(self.p.topology)
            crd = app.AmberInpcrdFile(self.p.coordinates)
            self.box = crd.getBoxVectors()
            self.initial = crd.getPositions(asNumpy=True).value_in_unit(nanometer)
        else:
            gro = app.GromacsGroFile(self.p.coordinates)
            self.box = gro.getPeriodicBoxVectors()
            self.top = app.GromacsTopFile(self.p.topology, periodicBoxVectors=self.box,
                                          includeDir=self.p.gmx_include)
            self.initial = gro.getPositions(asNumpy=True).value_in_unit(nanometer)
        self.topology = self.top.topology
        self.topology.setPeriodicBoxVectors(self.box)

    def system(self, stage):
        import openmm
        from openmm import app, unit

        system = self.top.createSystem(nonbondedMethod=app.PME,
                                       nonbondedCutoff=self.p.cutoff_nm * unit.nanometer,
                                       constraints=app.HBonds, rigidWater=True)
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(self.p.dispersion_correction)
            if isinstance(force, openmm.CustomNonbondedForce):
                force.setUseLongRangeCorrection(self.p.dispersion_correction)
        if self.p.switch_nm:
            from .omm_setup import load_production_helpers
            system = force_switch(load_production_helpers(), system,
                                  SimpleNamespace(r_on=self.p.switch_nm, r_off=self.p.cutoff_nm))
        for force in _restraint_forces(stage, self.initial):
            system.addForce(force)
        if stage.barostat and not stage.minimize:
            pressure, temperature = stage.pressure_bar * unit.bar, stage.temperature * unit.kelvin
            if stage.barostat == "membrane":
                barostat = openmm.MonteCarloMembraneBarostat(
                    pressure, stage.surface_tension * unit.bar * unit.nanometer, temperature,
                    openmm.MonteCarloMembraneBarostat.XYIsotropic,
                    openmm.MonteCarloMembraneBarostat.ZFree, 100)
            elif stage.barostat == "anisotropic":
                barostat = openmm.MonteCarloAnisotropicBarostat(
                    openmm.Vec3(1, 1, 1) * stage.pressure_bar * unit.bar, temperature, True, True, True, 100)
            else:
                barostat = openmm.MonteCarloBarostat(pressure, temperature, 100)
            system.addForce(barostat)
        return system

    def simulation(self, stage):
        import openmm
        from openmm import app, unit

        integrator = openmm.LangevinMiddleIntegrator(stage.temperature * unit.kelvin,
                                                     stage.friction / unit.picosecond,
                                                     stage.dt_ps * unit.picoseconds)
        if self.seed is not None:
            integrator.setRandomNumberSeed(int(self.seed))
        name = choose_platform(self.platform_request, available_platforms(), self.note)
        platform = openmm.Platform.getPlatformByName(name)
        sim = app.Simulation(self.topology, self.system(stage), integrator, platform,
                             platform_properties(platform, self.precision, self.device_index, self.note))
        return sim, name

    def run(self):
        """Run every stage not yet done. Returns the path of equilibrated.xml."""
        import openmm
        from openmm import app, unit

        final = os.path.join(self.out, "equilibrated.xml")
        stages = self.p.stages
        self.note("equilibration: CHARMM-GUI %s protocol from %s, %d stages%s"
                  % (self.p.format, self.p.folder, len(stages),
                     " (test: %d steps each)" % TEST_STEPS if self.test else ""))
        for message in self.p.notes:
            self.note("NOTE: " + message)
        with open(os.path.join(self.out, "protocol.json"), "w") as fh:
            json.dump(dict(format=self.p.format, folder=self.p.folder, cutoff_nm=self.p.cutoff_nm,
                           switch_nm=self.p.switch_nm, stages=[s.describe() for s in stages],
                           test=self.test), fh, indent=2)
        previous = None
        for number, stage in enumerate(stages, 1):
            xml = os.path.join(self.out, stage.name + ".xml")
            if os.path.exists(xml):
                self.note("stage %d/%d %s already done; skipping" % (number, len(stages), stage.name))
                previous = xml
                continue
            steps = min(stage.nsteps, TEST_STEPS) if self.test else stage.nsteps
            self.note("stage %d/%d started: %s" % (number, len(stages), stage.describe()))
            sim, platform = self.simulation(stage)
            if previous is None:
                sim.context.setPositions(self.initial)
                sim.context.setPeriodicBoxVectors(*self.box)
            else:
                with open(previous) as fh:
                    sim.context.setState(openmm.XmlSerializer.deserialize(fh.read()))
            start = sim.context.getState(getEnergy=True).getPotentialEnergy()
            begin = datetime.datetime.now()
            if stage.minimize:
                sim.minimizeEnergy(tolerance=stage.tolerance * unit.kilojoule_per_mole / unit.nanometer,
                                   maxIterations=steps)
            else:
                if stage.generate_velocities:
                    sim.context.setVelocitiesToTemperature(
                        stage.velocity_temperature * unit.kelvin,
                        *([int(self.seed)] if self.seed is not None else []))
                every = max(1, min(stage.report_steps, steps))
                sim.reporters.append(app.StateDataReporter(
                    os.path.join(self.out, stage.name + ".log"), every, step=True, time=True,
                    potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True,
                    density=True, speed=True, elapsedTime=True))
                if stage.dcd_steps and not self.test:
                    sim.reporters.append(app.DCDReporter(os.path.join(self.out, stage.name + ".dcd"),
                                                         stage.dcd_steps, enforcePeriodicBox=False))
                sim.step(steps)
            state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                                         getParameters=True, enforcePeriodicBox=False)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if stage.minimize and energy > start.value_in_unit(unit.kilojoule_per_mole) and not self.test:
                self.note("WARNING: energy rose during minimization, usually because the input "
                          "coordinates do not satisfy the constraints; check the structure")
            if not np.isfinite(energy):
                raise RuntimeError("stage %s: potential energy is %s; the system blew up"
                                   % (stage.name, energy))
            box = np.diag(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom))
            temperature = ""
            if not stage.minimize:
                dof = 3 * sim.system.getNumParticles() - sim.system.getNumConstraints() - 3
                ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                temperature = ", T %.1f K" % (2 * ke / (dof * 0.0083144626))
            tmp = xml + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(openmm.XmlSerializer.serialize(state))
            os.replace(tmp, xml)
            self.note("stage %d/%d finished on %s in %s: Epot %.1f -> %.1f kJ/mol%s, box %.2f %.2f %.2f A"
                      % (number, len(stages), platform,
                         str(datetime.datetime.now() - begin).split(".")[0],
                         start.value_in_unit(unit.kilojoule_per_mole), energy, temperature, *box))
            previous = xml
            del sim
        shutil.copyfile(previous, final)
        self.note("equilibration finished: %s" % final)
        return final

    def close(self):
        self.journal.close()


def production_settings(protocol, equilibrated_xml, cfg):
    """
    MD_openmm keys for production from the equilibrated State: the same
    topology, cutoff and switching as the protocol, and step7_production's
    thermostat and barostat unless the input file sets them.
    """
    cfg = dict(cfg)
    if "coordinate_file" in cfg:
        raise ValueError("equilibration = charmm-gui provides coordinate_file; remove it from the input")
    implied = dict(force_field=protocol.format, topology_file=protocol.topology,
                   coulomb="PME", r_off=str(protocol.cutoff_nm), cons="HBonds",
                   lj_lrc="yes" if protocol.dispersion_correction else "no",
                   vdw="Force-switch" if protocol.switch_nm else "No-switch")
    if protocol.switch_nm:
        implied["r_on"] = str(protocol.switch_nm)
    if protocol.gmx_include:
        implied["gmx_include"] = protocol.gmx_include
    implied.update({k: str(v) for k, v in protocol.production.items()})
    for key, value in implied.items():
        cfg.setdefault(key, value)
    cfg["coordinate_file"] = equilibrated_xml
    cfg["genvel"] = "no"
    return cfg
