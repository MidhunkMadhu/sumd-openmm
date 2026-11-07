"""
engine.py

The only code that touches an OpenMM Context: CV evaluation on a frame,
State snapshot/restore, velocity reseeding, running one window, and the
"packet" form of a State used to move states between MPI ranks.

A packet is plain numpy (positions, velocities, box in nm; time in ps;
context parameters; step; Epot), so broadcasting it costs a few MB and
restoring it costs milliseconds - much cheaper than XML round-trips.
"""

import os
import time

import numpy as np

from .energylog import COLUMNS
from .backend import read_state, write_state, is_portable_state


class Snap:
    __slots__ = ("state", "step", "epot")

    def __init__(self, state, step, epot):
        self.state, self.step, self.epot = state, step, epot


class Engine:
    name = "openmm"

    def __init__(self, omm, log):
        from openmm import unit
        self.u = unit
        self.omm = omm
        for name in ("dt_ps", "temperature_K", "platform_name", "crdfile"):
            setattr(self, name, getattr(omm, name, None))
        self.sim = omm.simulation
        self.ctx = omm.simulation.context
        self.log = log
        self.periodic = omm.system.usesPeriodicBoundaryConditions()
        self.terms = getattr(omm, "energy_groups", {})
        self.log_constants = None
        self.barostat = next((force for force in omm.system.getForces()
                              if hasattr(force, "computeCurrentPressure")), None)

    def molecules(self):
        return [np.array(m, dtype=int) for m in self.ctx.getMolecules()]

    # ---------------------------------------------- states

    def snapshot(self):
        st = self.ctx.getState(getPositions=True, getVelocities=True, getEnergy=True,
                               getParameters=True, enforcePeriodicBox=False)
        epot = st.getPotentialEnergy().value_in_unit(self.u.kilojoule_per_mole)
        return Snap(st, int(self.sim.currentStep), float(epot))

    def time_ps(self, snap):
        return float(snap.state.getTime().value_in_unit(self.u.picosecond))

    def frame(self, wrap=True):
        st = self.ctx.getState(getPositions=True, enforcePeriodicBox=wrap and self.periodic)
        box = st.getPeriodicBoxVectors(asNumpy=True) if self.periodic else None
        return st, st.getPositions(asNumpy=True), box

    def _energy_check(self, epot, check):
        if not check:
            return None

        e = self.ctx.getState(getEnergy=True).getPotentialEnergy()
        dE = float(e.value_in_unit(self.u.kilojoule_per_mole) - epot)
        tol = max(1.0, 1e-5 * abs(epot))

        if abs(dE) > tol:
            self.log("WARNING: restored state energy differs from saved by %.4g kJ/mol "
                     "(tolerance %.3g). Check barostat/box restoration." % (dE, tol))

        return dE

    def restore(self, snap, check=True):
        """
        Restore positions, velocities, box vectors, time and parameters from
        a saved State. Returns the potential-energy deviation from the saved
        value (kJ/mol), or None if not checked.
        """
        self.ctx.setState(snap.state)
        self.sim.currentStep = snap.step
        return self._energy_check(snap.epot, check)

    def packet(self, snap):
        u, st = self.u, snap.state
        return dict(
            pos=np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=np.float64),
            vel=np.asarray(st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond),
                           dtype=np.float64),
            box=np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer), dtype=np.float64),
            time=float(st.getTime().value_in_unit(u.picosecond)),
            params=dict(st.getParameters()),
            step=int(snap.step), epot=float(snap.epot),
        )

    def restore_packet(self, pk, check=True):
        """Same as restore(), from a packet. Box first, then coordinates."""
        from openmm import Vec3

        self.ctx.setPeriodicBoxVectors(*[Vec3(*row) for row in pk["box"]])
        self.ctx.setPositions(pk["pos"])
        self.ctx.setVelocities(pk["vel"])
        self.ctx.setTime(pk["time"])
        for k, v in pk["params"].items():
            self.ctx.setParameter(k, v)
        self.sim.currentStep = pk["step"]

        return self._energy_check(pk["epot"], check)

    def reseed(self, seed):
        """New Maxwell-Boltzmann velocities at the run temperature."""
        self.ctx.setVelocitiesToTemperature(self.omm.temperature_K * self.u.kelvin, int(seed))

    def save_state(self, snap, path):
        from openmm import XmlSerializer
        write_state(path, self.packet(snap), self.name,
                    XmlSerializer.serialize(snap.state).encode(), "openmm-state-xml")

    def load_state(self, path, step, epot):
        from openmm import XmlSerializer
        if is_portable_state(path):
            packet = read_state(path)
            if packet["engine"] != self.name:
                raise ValueError("cannot resume another MD engine's state; use an interchange restart")
            return Snap(XmlSerializer.deserialize(packet["native"].decode()),
                        packet["step"], packet["epot"])
        with open(path) as f:
            return Snap(XmlSerializer.deserialize(f.read()), step, epot)

    # Compatibility for callers of the original OpenMM engine.
    save_xml = save_state
    load_xml = load_state

    # ---------------------------------------------- dynamics

    def current_cvs(self, evaluator):
        _, pos, box = self.frame(wrap=True)
        return evaluator(pos.value_in_unit(self.u.angstrom),
                         None if box is None else box.value_in_unit(self.u.angstrom))

    def _log_constants(self):
        """Degrees of freedom and total mass (Da) of the System."""
        import openmm

        system = self.omm.system
        masses = [system.getParticleMass(i).value_in_unit(self.u.dalton)
                  for i in range(system.getNumParticles())]
        dof = 3 * sum(m > 0 for m in masses)
        for i in range(system.getNumConstraints()):
            a, b, _ = system.getConstraintParameters(i)
            dof -= masses[a] > 0 or masses[b] > 0
        if any(isinstance(f, openmm.CMMotionRemover) for f in system.getForces()):
            dof -= 3
        return dof, sum(masses)

    def energy_row(self, metrics, wall=0.0):
        """One line of a window's energy log, for the Context as it stands."""
        u = self.u
        kj = u.kilojoule_per_mole

        if self.log_constants is None:
            self.log_constants = self._log_constants()
        dof, mass = self.log_constants

        st = self.ctx.getState(getEnergy=True)
        epot = st.getPotentialEnergy().value_in_unit(kj)
        ekin = st.getKineticEnergy().value_in_unit(kj)
        integrator = self.omm.simulation.integrator
        if hasattr(integrator, "computeSystemTemperature"):
            temperature = integrator.computeSystemTemperature().value_in_unit(u.kelvin)
        else:
            temperature = 2 * ekin / (dof * u.MOLAR_GAS_CONSTANT_R.value_in_unit(kj / u.kelvin))

        pressure = ""
        if self.barostat is not None:
            bar = self.barostat.computeCurrentPressure(self.ctx)
            components = bar.value_in_unit(u.bar) if u.is_quantity(bar) else bar
            # Membrane and anisotropic barostats return Px, Py, Pz.
            # The common scalar column is their hydrostatic mean.
            pressure = "%.3f" % float(np.mean(components))

        volume = density = ""
        if self.periodic:
            nm3 = st.getPeriodicBoxVolume().value_in_unit(u.nanometer ** 3)
            volume, density = "%.5f" % nm3, "%.5f" % (mass * 1.66053906660e-3 / nm3)

        fields = ["%d" % self.sim.currentStep,
                  "%.4f" % st.getTime().value_in_unit(u.picosecond),
                  "%.4f" % epot, "%.4f" % ekin, "%.4f" % (epot + ekin), "%.3f" % temperature,
                  pressure, volume, density, "%.3f" % wall]
        from .energylog import ENERGY_TERMS
        fields += [("%.4f" % self.ctx.getState(getEnergy=True, groups={self.terms[name]})
                    .getPotentialEnergy().value_in_unit(kj)) if name in self.terms else ""
                   for name in ENERGY_TERMS]
        fields += ["%.4f" % value for value in metrics]
        return ",".join(fields) + "\n"

    def run_window(self, n_samples, steps_per_sample, evaluator, dcd_path, stride,
                   log_path=None, log_stride=0):
        """
        Run one window. Samples CVs every `steps_per_sample` steps (the first
        sample is at t = delta, the last at t = tau, as in an Amber trajectory
        without frame 0) and writes every `stride`-th sample to its own DCD.
        Frames are wrapped per molecule, as in the production DCDReporter.
        With log_path, every `log_stride`-th sample adds a row to that energy CSV.
        """
        from .dcdtools import DCDWriter

        u = self.u
        t0 = self.ctx.getState().getTime().value_in_unit(u.picosecond)
        t, values = [], []
        started = time.time()

        with open(dcd_path, "wb") as fh, open(log_path or os.devnull, "w") as log:
            if log_path:
                from .energylog import ENERGY_TERMS
                log.write(",".join(COLUMNS + list(ENERGY_TERMS) + list(evaluator.names)) + "\n")
            dcd = DCDWriter(fh, self.omm.system.getNumParticles(), self.omm.dt_ps,
                            self.sim.currentStep + steps_per_sample * stride,
                            steps_per_sample * stride, self.periodic)

            for k in range(1, n_samples + 1):
                self.sim.step(steps_per_sample)
                st, pos, box = self.frame(wrap=True)

                row = evaluator(pos.value_in_unit(u.angstrom),
                                None if box is None else box.value_in_unit(u.angstrom))
                t.append(st.getTime().value_in_unit(u.picosecond) - t0)
                values.append(row)

                if k % stride == 0:
                    dcd.write(pos.value_in_unit(u.nanometer),
                              None if box is None else box.value_in_unit(u.nanometer))
                if log_path and k % log_stride == 0:
                    log.write(self.energy_row(row, time.time() - started))

        return np.asarray(t), np.asarray(values)
