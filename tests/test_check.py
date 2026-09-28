"""check/: the thinned path trajectory and latest State, after every AcceptedStep."""

import json
import os

import numpy as np
import pytest

from sumd_openmm import cli
from sumd_openmm.executors import LocalExecutor

from test_parallel import run_fake


def json_concat(paths, out_path, stride=1):
    """concat_dcds for the fake engine, whose window "DCDs" are JSON rows."""
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += json.load(fh)
    with open(out_path, "w") as fh:
        json.dump(rows[::stride], fh)
    return len(rows[::stride])


@pytest.mark.parametrize("cycles, every, written", [
    (12, 3, [3, 6, 9, 12]),     # the last one is already on schedule
    (7, 5, [5, 7]),             # the end of the run adds the last AcceptedStep
    (4, 10, [4]),
    (5, 1, [1, 2, 3, 4, 5]),
])
def test_check_every_nth_accepted_step_and_at_the_end(tmp_path, monkeypatch, cycles, every, written):
    calls = []

    def spy(paths, out_path, stride=1):
        calls.append(len(paths))
        return json_concat(paths, out_path, stride)

    monkeypatch.setattr(cli.dcdtools, "concat_dcds", spy)
    # no retries: every cycle adds an AcceptedStep, so AcceptedStep k has k windows
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, max_cycles=cycles, drift=-0.1,
                                 check_stride=3, check_every=every, max_retries_per_parent=0)
    assert list(runner.nodes)[-1] == cycles
    assert calls == written

    check = os.path.join(out, "check")
    assert sorted(os.listdir(check)) == ["latest.json", "latest_accepted_step.xml",
                                         "sumd_traj_stride3.dcd"]
    with open(os.path.join(check, "latest.json")) as fh:
        latest = json.load(fh)
    last = cycles
    assert latest["accepted_step"] == last and latest["path"] == runner.path_to(last)

    with open(os.path.join(check, "latest_accepted_step.xml")) as fh, \
            open(os.path.join(out, runner.nodes[last]["state_xml"])) as ref:
        assert fh.read() == ref.read()

    rows = []
    for nid in runner.path_to(last):
        if runner.nodes[nid]["dcd"]:
            with open(os.path.join(out, runner.nodes[nid]["dcd"])) as fh:
                rows += json.load(fh)
    with open(os.path.join(check, "sumd_traj_stride3.dcd")) as fh:
        assert np.allclose(json.load(fh), rows[::3])
    assert latest["frames"] == len(rows[::3])


def test_check_failure_does_not_stop_the_run(tmp_path, monkeypatch):
    def broken(paths, out_path, stride=1):
        raise OSError("disk full")

    monkeypatch.setattr(cli.dcdtools, "concat_dcds", broken)
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, max_cycles=3, drift=-0.3,
                                 check_stride=10, max_retries_per_parent=0)
    assert len(runner.nodes) > 1
    with open(os.path.join(out, "progress.log")) as fh:
        assert "check/ not updated" in fh.read()
    assert not os.path.exists(os.path.join(out, "check", "sumd_traj_stride10.dcd"))


def test_check_off(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, max_cycles=3, drift=-0.3,
                                 check_stride=0)
    assert not os.path.exists(os.path.join(out, "check"))
