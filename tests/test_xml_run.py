"""Installed-command smoke test of the serialized-System setup path."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest


def _system(tmp_path):
    openmm = pytest.importorskip("openmm")
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.5, 100.0)
    system.addForce(bonds)
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0),
                                        openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    (tmp_path / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (tmp_path / "system.pdb").write_text(
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  C1  MOL A   1       5.000   5.000   5.000  1.00  0.00           C\n"
        "ATOM      2  C2  MOL A   1      10.000   5.000   5.000  1.00  0.00           C\nEND\n"
    )
    (tmp_path / "run.inp").write_text(
        "force_field = OPENMM_XML\n"
        "system_xml = system.xml\n"
        "topology_file = system.pdb\n"
        "platform = CPU\n"
        "output_dir = output\n"
        "dt = 0.002\nwindow_ps = 0.006\ncv_sample_ps = 0.002\n"
        "max_cycles = 2\nrandom_seed = 42\n"
        "metric_1_type = distance\nmetric_1_a = indices:0\n"
        "metric_1_b = indices:1\nmetric_1_role = supervise\n"
        "metric_1_direction = decrease\nmetric_1_target = 10\n"
    )


def test_xml_run_and_inspection(tmp_path):
    from sumd_openmm.inspect_run import inspect
    _system(tmp_path)
    cmd = [sys.executable, "-m", "sumd_openmm", "run.inp"]
    dry = subprocess.run(cmd + ["--dry-run"], cwd=tmp_path, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    with open(tmp_path / "output" / "run_summary.json") as fh:
        assert json.load(fh)["metrics"][0]["initial_value"] == pytest.approx(5)
    # the output of a dry run does not block the run that follows it
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    # a second run renames the first one's output_dir and starts
    again = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 0, again.stderr
    assert (tmp_path / "output_old00001" / "run_summary.json").exists()
    assert "renamed to output_old00001" in (tmp_path / "output" / "progress.log").read_text()
    report = inspect(str(tmp_path / "output"))
    assert report["windows_simulated"] > 0
    assert report["max_dcd_vs_state_mismatch_A"] is not None
    assert report["max_dcd_vs_state_mismatch_A"] < 1e-4
    assert report["final_traj_identical_to_path_windows"]


def test_check_folder_matches_final_trajectory(tmp_path):
    from sumd_openmm import dcdtools
    _system(tmp_path)
    inp = tmp_path / "run.inp"
    inp.write_text(inp.read_text().replace("max_cycles = 2", "max_cycles = 4")
                   .replace("metric_1_target = 10", "metric_1_target = 1") + "check_stride = 2\n")
    run = subprocess.run([sys.executable, "-m", "sumd_openmm", "run.inp"], cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    out = tmp_path / "output"
    with open(out / "run_summary.json") as fh:
        summary = json.load(fh)
    if not summary["final_traj"]["frames"]:
        pytest.skip("no window was accepted")
    with open(out / "check" / "latest.json") as fh:
        latest = json.load(fh)
    assert latest["accepted_step"] == summary["final_accepted_step"]
    full = dcdtools.read_frames(str(out / "sumd_traj.dcd"))
    thin = dcdtools.read_frames(str(out / "check" / "sumd_traj_stride2.dcd"))
    assert len(thin) == latest["frames"] and np.array_equal(thin, full[::2])
    assert ((out / "check" / "latest_accepted_step.xml").read_text() ==
            (out / "final_state.xml").read_text())


def test_torsion_with_stratified_and_monitored_metrics(tmp_path):
    from sumd_openmm.inspect_run import inspect
    openmm = pytest.importorskip("openmm")
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12)
    force = openmm.HarmonicBondForce()
    for a, b in ((0, 1), (1, 2), (2, 3)):
        force.addBond(a, b, 0.1, 100)
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0),
                                        openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    (tmp_path / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    coords = ((5, 6, 5), (5, 5, 5), (6, 5, 5), (6, 5, 6))
    lines = ["CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1"]
    for i, (x, y, z) in enumerate(coords, 1):
        lines.append("ATOM   %4d  C%d  MOL A   1    %8.3f%8.3f%8.3f  1.00  0.00           C" %
                     (i, i, x, y, z))
    (tmp_path / "system.pdb").write_text("\n".join(lines + ["END", ""]))
    (tmp_path / "run.inp").write_text(
        "force_field = OPENMM_XML\nsystem_xml = system.xml\n"
        "topology_file = system.pdb\nplatform = CPU\noutput_dir = output\n"
        "dt = 0.002\nwindow_ps = 0.006\ncv_sample_ps = 0.002\n"
        "max_cycles = 1\nmax_retries_per_parent = 0\n"
        "seeding = stratified\n"
        "metric_1_type = dihedral\nmetric_1_a = indices:0\nmetric_1_b = indices:1\n"
        "metric_1_c = indices:2\nmetric_1_d = indices:3\n"
        "metric_1_role = supervise\nmetric_1_direction = toward\n"
        "metric_1_target = 180\nmetric_1_tolerance = 20\n"
        "metric_2_type = distance\nmetric_2_a = indices:0\nmetric_2_b = indices:3\n"
        "metric_2_role = stratify\nmetric_2_bins = 2,4\n"
        "metric_3_type = angle\nmetric_3_a = indices:0\nmetric_3_b = indices:1\n"
        "metric_3_c = indices:2\nmetric_3_role = monitor\n"
    )
    cmd = [sys.executable, "-m", "sumd_openmm", "run.inp"]
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = inspect(str(tmp_path / "output"))
    assert set(report["retained_metrics"]) == {"metric_1", "metric_2", "metric_3"}
    assert report["max_dcd_vs_state_mismatch_A"] < 1e-4


def _stop_at_cycle(monkeypatch, n, action):
    from sumd_openmm import cli
    report = cli.SumdRun.cycle_report

    def patched(self, cycle, *args):
        line = report(self, cycle, *args)
        if cycle == n:
            action()
        return line

    monkeypatch.setattr(cli.SumdRun, "cycle_report", patched)


def _partial_run(tmp_path, monkeypatch, n, action, raises):
    from sumd_openmm import cli, dcdtools
    _system(tmp_path)
    inp = tmp_path / "run.inp"
    inp.write_text(inp.read_text().replace("max_cycles = 2", "max_cycles = 6")
                   .replace("metric_1_target = 10", "metric_1_target = 1")
                   + "max_retries_per_parent = 0\n")
    _stop_at_cycle(monkeypatch, n, action)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(raises):
        cli.main(["run.inp"])
    out = tmp_path / "output"
    with open(out / "run_summary.json") as fh:
        summary = json.load(fh)
    assert summary["partial"] and summary["final_accepted_step"] == n
    path = summary["final_path"]
    assert path == list(range(n + 1))
    frames = sum(len(dcdtools.read_frames(str(out / "windows" / "accepted" / ("c%05d_w0.dcd" % c))))
                 for c in range(1, n + 1))
    assert summary["final_traj"]["frames"] == frames == len(dcdtools.read_frames(str(out / "sumd_traj.dcd")))
    assert ((out / "final_state.xml").read_text() ==
            (out / "accepted_steps" / ("accepted_step_%06d.xml" % n)).read_text())
    assert (out / "final_state.rst7").exists()
    with open(out / "check" / "latest.json") as fh:
        assert json.load(fh)["accepted_step"] == n
    return summary


def test_error_writes_partial_result(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("boom")

    summary = _partial_run(tmp_path, monkeypatch, 3, fail, RuntimeError)
    assert summary["stop_reason"] == "error: RuntimeError: boom"


def test_sigterm_writes_partial_result(tmp_path, monkeypatch):
    import signal
    from sumd_openmm.cli import Terminated
    before = signal.getsignal(signal.SIGTERM)
    summary = _partial_run(tmp_path, monkeypatch, 2,
                           lambda: os.kill(os.getpid(), signal.SIGTERM), Terminated)
    assert summary["stop_reason"].startswith("terminated")
    assert signal.getsignal(signal.SIGTERM) is before
