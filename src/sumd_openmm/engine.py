"""
engine.py

The only code that touches an OpenMM Context: CV evaluation on a frame,
State snapshot/restore, velocity reseeding, running one window, and the
"packet" form of a State used to move states between MPI ranks.

A packet is plain numpy (positions, velocities, box in nm; time in ps;
context parameters; step; Epot), so broadcasting it costs a few MB and
restoring it costs milliseconds - much cheaper than XML round-trips.
"""

import numpy as np

from . import cv as CV


class CVEvaluator:

    def __init__(self, scfg, sel, align_groups=None):
        self.s = scfg
        self.sel = sel
        self.align_groups = align_groups

    def cv1(self, X, box):
        s, sel = self.s, self.sel

        if s.cv1_type == "rmsd":
            return CV.ligand_rmsd_receptor_frame(
                X, sel.align, sel.cv1, sel.ref_align_xyz, sel.ref_cv1_xyz,
                box=box, w_align=sel.w_align, w_lig=sel.w_cv1, align_groups=self.align_groups)

        if s.cv1_type == "distance":
            return CV.com_distance(X, sel.cv1_site, sel.cv1, box=box, w_a=sel.w_site, w_b=sel.w_cv1)

        return CV.min_distance(X, sel.cv1_site, sel.cv1, box=box)

    def cv2(self, X, box):
        if self.sel.cv2_a is None:
            return None
        return CV.min_distance(X, self.sel.cv2_a, self.sel.cv2_b, box=box)

    def __call__(self, X, box):
        return self.cv1(X, box), self.cv2(X, box)


class Snap:
    __slots__ = ("state", "step", "epot")

    def __init__(self, state, step, epot):
        self.state, self.step, self.epot = state, step, epot


class Engine:

    def __init__(self, omm, log):
        from openmm import unit
        self.u = unit
        self.omm = omm
        self.sim = omm.simulation
        self.ctx = omm.simulation.context
        self.log = log

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
        st = self.ctx.getState(getPositions=True, enforcePeriodicBox=wrap)
        return st, st.getPositions(asNumpy=True), st.getPeriodicBoxVectors(asNumpy=True)

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

    def save_xml(self, snap, path):
        from openmm import XmlSerializer
        with open(path, "w") as f:
            f.write(XmlSerializer.serialize(snap.state))

    def load_xml(self, path, step, epot):
        from openmm import XmlSerializer
        with open(path) as f:
            return Snap(XmlSerializer.deserialize(f.read()), step, epot)

    # ---------------------------------------------- dynamics

    def current_cvs(self, evaluator):
        _, pos, box = self.frame(wrap=True)
        return evaluator(pos.value_in_unit(self.u.angstrom), box.value_in_unit(self.u.angstrom))

    def run_window(self, n_samples, steps_per_sample, evaluator, dcd_path, stride):
        """
        Run one window. Samples CVs every `steps_per_sample` steps (the first
        sample is at t = delta, the last at t = tau, as in an Amber trajectory
        without frame 0) and writes every `stride`-th sample to its own DCD.
        Frames are wrapped per molecule, as in the production DCDReporter.
        """
        from openmm.app import DCDFile

        u = self.u
        t0 = self.ctx.getState().getTime().value_in_unit(u.picosecond)
        t, c1, c2 = [], [], []

        with open(dcd_path, "wb") as fh:
            dcd = DCDFile(fh, self.sim.topology, self.omm.dt_ps * u.picoseconds,
                          firstStep=self.sim.currentStep + steps_per_sample * stride,
                          interval=steps_per_sample * stride)

            for k in range(1, n_samples + 1):
                self.sim.step(steps_per_sample)
                st, pos, box = self.frame(wrap=True)

                a, b = evaluator(pos.value_in_unit(u.angstrom), box.value_in_unit(u.angstrom))
                t.append(st.getTime().value_in_unit(u.picosecond) - t0)
                c1.append(a)
                c2.append(b)

                if k % stride == 0:
                    dcd.writeModel(pos, periodicBoxVectors=box)

        return np.array(t), np.array(c1), (None if c2[0] is None else np.array(c2))
