"""Installed-command smoke test of the serialized-System setup path."""

import json
import os
import subprocess
import sys

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
    again = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 2 and "output_dir exists" in again.stderr
    assert "Traceback" not in again.stderr
    report = inspect(str(tmp_path / "output"))
    assert report["windows_simulated"] > 0
    assert report["max_dcd_vs_state_mismatch_A"] is not None
    assert report["max_dcd_vs_state_mismatch_A"] < 1e-4
    assert report["final_traj_identical_to_path_windows"]


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
