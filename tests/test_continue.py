"""--continue: carry a run on after max_cycles, a crash or a time limit."""

import csv
import json
import os

import numpy as np
import pytest

from test_xml_run import _stop_at_cycle, _system

pytest.importorskip("openmm")


def _setup(tmp_path, monkeypatch, extra=""):
    _system(tmp_path)
    inp = tmp_path / "run.inp"
    inp.write_text(inp.read_text().replace("metric_1_target = 10", "metric_1_target = 1")
                   + "max_retries_per_parent = 1\ncheck_every = 2\n" + extra)
    monkeypatch.chdir(tmp_path)
    return inp


def _main(*argv):
    from sumd_openmm import cli
    cli.main(["run.inp", *argv])


def _audit(out, cycles):
    """One consistent run of `cycles` cycles, whatever the stops in between."""
    from sumd_openmm import dcdtools
    from sumd_openmm.inspect_run import inspect
    with open(out / "windows.csv") as fh:
        windows = list(csv.DictReader(fh))
    assert [int(w["cycle"]) for w in windows] == list(range(1, cycles + 1))
    with open(out / "windows.jsonl") as fh:
        assert [json.loads(line)["cycle"] for line in fh] == list(range(1, cycles + 1))
    with open(out / "accepted_steps.csv") as fh:
        steps = list(csv.DictReader(fh))
    assert [int(s["accepted_step"]) for s in steps] == list(range(len(steps)))
    with open(out / "cv_samples.csv") as fh:
        assert {int(r["cycle"]) for r in csv.DictReader(fh)} == set(range(1, cycles + 1))
    with open(out / "resume.json") as fh:
        state = json.load(fh)
    assert state["last_cycle"] == cycles and len(state["nodes"]) == len(steps)
    kept = sorted(os.listdir(out / "windows" / "accepted"))
    assert kept == sorted(os.path.basename(s["trajectory"]) for s in steps[1:])
    assert not os.listdir(out / "windows" / "tmp")
    report = inspect(str(out))
    assert report["final_traj_identical_to_path_windows"]
    assert report["max_dcd_vs_state_mismatch_A"] < 1e-3
    with open(out / "run_summary.json") as fh:
        summary = json.load(fh)
    path = summary["final_path"]
    frames = sum(len(dcdtools.read_frames(str(out / steps[i]["trajectory"]))) for i in path[1:])
    assert summary["final_traj"]["frames"] == frames
    return summary, steps


def test_extend_after_max_cycles(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _main()
    out = tmp_path / "output"
    first, _ = _audit(out, 2)
    _main("--extend", "3")
    summary, steps = _audit(out, 5)
    assert summary["stop_reason"] == "max_cycles"
    assert [c["after_cycle"] for c in summary["continuations"]] == [2]
    # the first new window starts where the first run ended
    new = [s for s in steps if int(s["cycle"]) > 2]
    if new:
        assert int(new[0]["parent_accepted_step"]) <= first["final_accepted_step"]
    with open(out / "progress.log") as fh:
        log = fh.read()
    assert "continuing after cycle 2" in log and log.count("finished: max_cycles") == 2
    with pytest.raises(SystemExit) as exc:          # nothing left: max_cycles already reached
        _main("--continue")
    assert exc.value.code == 2


def test_continue_after_crash_redoes_the_interrupted_cycle(tmp_path, monkeypatch):
    inp = _setup(tmp_path, monkeypatch)
    inp.write_text(inp.read_text().replace("max_cycles = 2", "max_cycles = 5"))

    def fail():
        raise RuntimeError("node lost")

    # the error comes after cycle 3 wrote its rows and files, before its checkpoint
    _stop_at_cycle(monkeypatch, 3, fail)
    with pytest.raises(RuntimeError):
        _main()
    out = tmp_path / "output"
    with open(out / "resume.json") as fh:
        assert json.load(fh)["last_cycle"] == 2
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    _main("--continue")
    summary, _ = _audit(out, 5)
    assert not summary["partial"]
    assert summary["continuations"][0]["previous_stop_reason"] is None


def test_continue_refuses_changed_settings(tmp_path, monkeypatch):
    inp = _setup(tmp_path, monkeypatch)
    _main()
    text = inp.read_text()
    for old, new in [("window_ps = 0.006", "window_ps = 0.008"),
                     ("metric_1_b = indices:1", "metric_1_b = indices:0"),
                     ("dt = 0.002", "dt = 0.001")]:
        inp.write_text(text.replace(old, new))
        with pytest.raises(SystemExit):
            _main("--extend", "1")
    inp.write_text(text)
    _main("--extend", "1")                                   # unchanged: fine
    _audit(tmp_path / "output", 3)


def test_converged_run_needs_a_new_target(tmp_path, monkeypatch, capsys):
    inp = _setup(tmp_path, monkeypatch)
    inp.write_text(inp.read_text().replace("metric_1_target = 1", "metric_1_target = 10"))
    _main()                                                   # starts at 5 A: converges at once
    with open(tmp_path / "output" / "run_summary.json") as fh:
        assert json.load(fh)["stop_reason"] == "converged"
    with pytest.raises(SystemExit):
        _main("--extend", "2")
    assert "change a supervised metric's target" in capsys.readouterr().err
    inp.write_text(inp.read_text().replace("metric_1_target = 10", "metric_1_target = 1"))
    _main("--extend", "2")
    with open(tmp_path / "output" / "resume.json") as fh:
        cycles = json.load(fh)["last_cycle"]
    assert cycles > 1
    _audit(tmp_path / "output", cycles)


def test_continue_without_resume_file(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _main("--continue")
    assert "resume.json not found" in capsys.readouterr().err


def test_stratified_pool_is_restored(tmp_path):
    """Fake engine: the pool, retries and random stream come back as saved."""
    from fakes import FakeEngine, check_run_invariants, make_config
    from sumd_openmm.cli import RunLog, SumdRun, load_resume
    from sumd_openmm.executors import LocalExecutor, RankWorker

    over = dict(seeding="stratified", band_width=0.2, metric_2_type="distance",
                metric_2_a="indices:2", metric_2_b="indices:3", metric_2_role="stratify",
                metric_2_bins="quantiles:3", max_retries_per_parent=2)
    out = str(tmp_path / "run")
    for d in ("accepted_steps", "windows/accepted", "windows/tmp", "windows/rejected"):
        os.makedirs(os.path.join(out, d))

    def segment(cycles, state=None):
        scfg = make_config(walkers=1, max_cycles=cycles, **over)
        worker = RankWorker(FakeEngine(seed=cycles, drift=0.02, n=2), None, out,
                            scfg.samples_per_window, 5, 1, True)
        log = RunLog(out, [m.name for m in scfg.metrics], append=state is not None)
        rng = np.random.default_rng(3)
        if state:
            rng.bit_generator.state = state["rng"]
        runner = SumdRun(scfg, LocalExecutor(worker), out, rng, log, worker.eng.current_cvs(None))
        runner.saved_config = dict(sumd_config=scfg.as_dict(), md_config={})
        if state:
            runner.load_state(state)
            assert runner.pool.entries.keys() == {e["node_id"] for e in state["pool"]["entries"]}
            assert runner.retries == {int(k): v for k, v in state["retries"].items()}
        runner.run()
        log.close()
        return runner

    first = segment(6)
    state = load_resume(out)
    assert state["pool"]["entries"] and state["pool"]["samples"][0]
    second = segment(10, state)
    assert list(second.nodes)[:len(first.nodes)] == list(first.nodes)
    assert all(second.nodes[i]["metrics"] == first.nodes[i]["metrics"] for i in first.nodes)
    assert len(second.pool.samples[0]) > len(first.pool.samples[0])
    rows = check_run_invariants(out, second, walkers=1)
    assert [r["cycle"] for r in rows] == list(range(1, 11))
