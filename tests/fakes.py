"""A deterministic coordinate-free engine for testing the window loop."""

import json
import os

import numpy as np

from sumd_openmm.config import SumdConfig
from sumd_openmm.engine import Snap

BASE = dict(metric_1_type="distance", metric_1_a="indices:0",
            metric_1_b="indices:1", metric_1_role="supervise",
            metric_1_direction="decrease", metric_1_target="-1",
            window_ps="1", cv_sample_ps="0.1", random_seed="3",
            check_stride="0")   # its "DCDs" are JSON; test_check.py covers check/


def make_config(**over):
    return SumdConfig.from_cfg({**BASE, **{k: str(v) for k, v in over.items()}})


class FakeEngine:
    def __init__(self, seed, drift=0.0, noise=0.2, n=1):
        self.rs = np.random.default_rng(seed)
        self.drift, self.noise, self.n = drift, noise, n
        self.values = np.arange(n, dtype=float) + 20
        self.t, self.step = 0.0, 0

    def snapshot(self):
        return Snap(dict(values=self.values.tolist(), t=self.t), self.step, -1000.0)

    def time_ps(self, snap):
        return snap.state["t"]

    def restore(self, snap, check=True):
        self.values = np.asarray(snap.state["values"])
        self.t, self.step = snap.state["t"], snap.step
        return 0.0 if check else None

    def packet(self, snap):
        return dict(state=snap.state, step=snap.step, epot=snap.epot)

    def restore_packet(self, packet, check=True):
        return self.restore(Snap(packet["state"], packet["step"], packet["epot"]), check)

    def reseed(self, seed):
        self.rs = np.random.default_rng(seed)

    def save_xml(self, snap, path):
        with open(path, "w") as fh:
            json.dump(dict(state=snap.state, step=snap.step, epot=snap.epot), fh)

    def load_xml(self, path, step, epot):
        with open(path) as fh:
            data = json.load(fh)
        return Snap(data["state"], data["step"], data["epot"])

    def current_cvs(self, evaluator):
        return self.values.copy()

    def run_window(self, n_samples, steps_per_sample, evaluator, dcd_path, stride):
        times, rows = [], []
        for i in range(n_samples):
            self.values += self.drift + self.noise * self.rs.normal(size=self.n)
            self.t += 0.1
            self.step += steps_per_sample
            times.append((i + 1) * 0.1)
            rows.append(self.values.copy())
        with open(dcd_path, "w") as fh:
            json.dump(np.asarray(rows).tolist(), fh)
        return np.asarray(times), np.asarray(rows)


def check_run_invariants(outdir, runner, walkers):
    for nid, node in runner.nodes.items():
        if node["parent"] is not None:
            assert node["parent"] < nid
        with open(os.path.join(outdir, node["state_xml"])) as fh:
            state = json.load(fh)["state"]
        assert np.allclose(state["values"], node["values"])
        if node["dcd"]:
            with open(os.path.join(outdir, node["dcd"])) as fh:
                assert np.allclose(json.load(fh)[-1], node["values"])
    assert not os.listdir(os.path.join(outdir, "windows", "tmp"))
    with open(os.path.join(outdir, "windows.jsonl")) as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == walkers * len({row["cycle"] for row in rows})
    return rows
