"""
A stand-in for engine.Engine: the "system" is a random walk in (CV1, CV2),
so the whole SuMD loop and both executors can be exercised without OpenMM.

Each window's fake DCD holds its CV1 series as JSON, so a test can check that
every committed node's file really is the window that produced that node.
"""

import json
import os

import numpy as np

from sumd_openmm.config import SumdConfig
from sumd_openmm.engine import Snap

BASE = dict(cv1_traj_selection=":LIG", cv1_ref_selection=":LIG", align_traj_selection=":1-10@CA",
            align_ref_selection=":1-10@CA", reference_structure="ref.pdb",
            cv2_selection_a=":3", cv2_selection_b=":7",
            window_ps="1", cv_sample_ps="0.1", colvarcut="-1", random_seed="3")


def make_config(**over):
    return SumdConfig.from_cfg({**BASE, **{k: str(v) for k, v in over.items()}})


class FakeEngine:

    def __init__(self, seed, drift=0.0, noise=0.2):
        self.rs = np.random.default_rng(seed)
        self.drift, self.noise = drift, noise
        self.x, self.g, self.t, self.step = 20.0, 4.5, 0.0, 0

    def _epot(self):
        return -1000.0 - self.x

    def snapshot(self):
        return Snap(dict(x=self.x, g=self.g, t=self.t), self.step, self._epot())

    def time_ps(self, snap):
        return snap.state["t"]

    def restore(self, snap, check=True):
        self.x, self.g, self.t = snap.state["x"], snap.state["g"], snap.state["t"]
        self.step = snap.step
        return (self._epot() - snap.epot) if check else None

    def packet(self, snap):
        return dict(state=dict(snap.state), step=snap.step, epot=snap.epot)

    def restore_packet(self, pk, check=True):
        return self.restore(Snap(pk["state"], pk["step"], pk["epot"]), check)

    def reseed(self, seed):
        self.rs = np.random.default_rng(seed)

    def save_xml(self, snap, path):
        with open(path, "w") as f:
            json.dump(dict(state=snap.state, step=snap.step, epot=snap.epot), f)

    def load_xml(self, path, step, epot):
        with open(path) as f:
            d = json.load(f)
        return Snap(d["state"], d["step"], d["epot"])

    def current_cvs(self, evaluator):
        return self.x, self.g

    def run_window(self, n_samples, steps_per_sample, evaluator, dcd_path, stride):
        t, c1, c2 = [], [], []
        t0 = self.t
        for _ in range(n_samples):
            self.x = max(0.0, self.x + self.drift + self.noise * self.rs.normal())
            self.g = max(2.5, self.g + 0.3 * self.rs.normal())
            self.t += 0.1
            self.step += steps_per_sample
            t.append(self.t - t0)
            c1.append(self.x)
            c2.append(self.g)
        with open(dcd_path, "w") as f:
            json.dump(c1, f)
        return np.array(t), np.array(c1), np.array(c2)


def check_run_invariants(outdir, runner, walkers):
    """Assertions shared by the serial, in-process MPI and mpirun tests."""
    nodes = runner.nodes

    for nid, n in nodes.items():
        if n["parent"] is not None:
            assert n["parent"] < nid, "parent must be an earlier node"
        assert os.path.exists(os.path.join(outdir, n["state_xml"])), "node state written"
        if n["dcd"]:
            with open(os.path.join(outdir, n["dcd"])) as f:
                series = json.load(f)
            assert abs(series[-1] - n["cv1"]) < 1e-3, "node %d DCD is not its own window" % nid
            with open(os.path.join(outdir, n["state_xml"])) as f:
                assert abs(json.load(f)["state"]["x"] - n["cv1"]) < 1e-3, "node %d state mismatch" % nid

    assert os.listdir(os.path.join(outdir, "windows", "tmp")) == [], "undecided windows left behind"

    rows = [json.loads(l) for l in open(os.path.join(outdir, "windows.jsonl"))]
    cycles = sorted({r["cycle"] for r in rows})
    assert cycles == list(range(1, len(cycles) + 1)), "cycles must be consecutive"
    assert len(rows) == len(cycles) * walkers, "one row per walker per cycle"
    committed = {(r["cycle"], r["walker"]) for r in rows if r["accepted"]} | {
        (r["committed"]["cycle"], r["committed"]["walker"]) for r in rows if r["committed"]}
    assert committed == {(n["cycle"], n["walker"]) for n in nodes.values() if n["dcd"]}
    return rows
