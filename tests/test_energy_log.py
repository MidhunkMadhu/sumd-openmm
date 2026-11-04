"""Energy logs: per-window files, the joined path log, and the real engine's columns."""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from sumd_openmm import energylog
from sumd_openmm.config import ConfigError
from sumd_openmm.executors import LocalExecutor

from fakes import make_config
from test_parallel import run_fake
from test_xml_run import _system


def rows_of(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write(path, header, *rows):
    path.write_text("\n".join([header, *rows]) + "\n")
    return str(path)


def test_join_keeps_order_adds_labels_and_skips_missing_logs(tmp_path):
    a = write(tmp_path / "a.log", "Step,Time_ps,Epot_kJmol,m", "10,0.02,-5.0,1.0")
    b = write(tmp_path / "b.log", "Step,Time_ps,Epot_kJmol,m", "20,0.04,-6.0,2.0", "30,0.06,-7.0,3.0")
    out = str(tmp_path / "joined.log")
    n = energylog.join_logs([(a, 1, 1, 0), (str(tmp_path / "none.log"), 2, 2, 0), (b, 3, 3, 2)], out)
    assert n == 3
    with open(out) as fh:
        assert fh.readline().strip() == "Step,Time_ps,Cycle,AcceptedStep,Walker,Epot_kJmol,m"
    rows = rows_of(out)
    assert [r["Step"] for r in rows] == ["10", "20", "30"]
    assert [(r["Cycle"], r["AcceptedStep"], r["Walker"]) for r in rows] == [
        ("1", "1", "0"), ("3", "3", "2"), ("3", "3", "2")]


def test_join_with_other_columns_leaves_them_empty(tmp_path):
    a = write(tmp_path / "a.log", "Step,Time_ps,m", "10,0.02,1.0")
    b = write(tmp_path / "b.log", "Step,Time_ps,E_bond,m", "20,0.04,3.0,2.0")
    out = str(tmp_path / "joined.log")
    energylog.join_logs([(a, 1, 1, 0), (b, 2, 2, 0)], out)
    rows = rows_of(out)
    assert rows[0]["E_bond"] == "" and rows[1]["E_bond"] == "3.0" and rows[1]["m"] == "2.0"


def test_join_without_logs_writes_nothing(tmp_path):
    out = tmp_path / "joined.log"
    assert energylog.join_logs([(str(tmp_path / "none.log"), 1, 1, 0)], str(out)) == 0
    assert not out.exists()


def test_summary_skips_labels_and_empty_columns(tmp_path):
    path = write(tmp_path / "s.log", "Step,Time_ps,Cycle,AcceptedStep,Walker,Epot_kJmol,Volume_nm3",
                 "1,0.1,1,1,0,-2.0,", "2,0.2,1,1,0,-4.0,")
    assert energylog.summarise(path) == {"Epot_kJmol": dict(mean=-3.0, std=1.0)}


def test_summary_of_one_row(tmp_path):
    path = write(tmp_path / "s.log", "Step,Time_ps,Epot_kJmol", "1,0.1,-2.0")
    assert energylog.summarise(path) == {"Epot_kJmol": dict(mean=-2.0, std=0.0)}


def test_log_stride_follows_dcd_stride_unless_set():
    assert make_config(dcd_stride=5).energy_log_stride == 5
    assert make_config(dcd_stride=5, log_stride=2).energy_log_stride == 2
    assert make_config(log_stride=0).energy_log_stride == 0


@pytest.mark.parametrize("value", [3, -1])
def test_log_stride_is_checked(value):
    with pytest.raises(ConfigError, match="log_stride"):
        make_config(log_stride=value)        # 10 samples per window


def test_logs_follow_their_windows(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, max_cycles=4, drift=-0.1)
    kept = sorted(f for f in os.listdir(os.path.join(out, "windows", "accepted")))
    assert kept and {os.path.splitext(f)[0] for f in kept if f.endswith(".log")} == \
        {os.path.splitext(f)[0] for f in kept if f.endswith(".dcd")}
    assert not os.listdir(os.path.join(out, "windows", "tmp"))
    assert not os.listdir(os.path.join(out, "windows", "rejected"))


def test_rejected_logs_are_kept_with_rejected_dcds(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, max_cycles=4, drift=-0.1,
                                 keep_rejected_dcd="yes")
    names = os.listdir(os.path.join(out, "windows", "rejected"))
    assert names and sum(f.endswith(".log") for f in names) == sum(f.endswith(".dcd") for f in names)


def test_log_off_writes_no_logs(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, max_cycles=3, drift=-0.3, log_stride=0)
    kept = os.listdir(os.path.join(out, "windows", "accepted"))
    assert kept and not any(f.endswith(".log") for f in kept)
    assert energylog.join_logs(runner.path_logs(max(runner.nodes)), str(tmp_path / "j.log")) == 0


def test_joined_path_leaves_out_abandoned_branches(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, max_cycles=30, drift=0.05, noise=0.3,
                                 on_retry_exhaustion="step_back", max_retries_per_parent=1,
                                 walker_acceptance="sumd")
    final = runner.final
    joined = str(tmp_path / "joined.log")
    energylog.join_logs(runner.path_logs(final), joined)
    rows = rows_of(joined)
    path = [i for i in runner.path_to(final) if runner.nodes[i]["dcd"]]
    assert len(path) < len(runner.nodes) - 1          # some AcceptedSteps were abandoned
    assert {int(r["AcceptedStep"]) for r in rows} == set(path)
    assert len(rows) == len(path) * scfg.samples_per_window


def _run(tmp_path, extra="", setup=True):
    if setup:
        _system(tmp_path)
    inp = tmp_path / "run.inp"
    inp.write_text(inp.read_text().replace("metric_1_target = 10", "metric_1_target = 1") + extra)
    run = subprocess.run([sys.executable, "-m", "sumd_openmm", "run.inp"], cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    out = tmp_path / "output"
    with open(out / "run_summary.json") as fh:
        return out, json.load(fh)


def test_real_run_log_matches_trajectory_and_saved_states(tmp_path):
    from sumd_openmm import dcdtools
    out, summary = _run(tmp_path, "max_cycles = 3\n")
    if not summary["final_traj"]["windows"]:
        pytest.skip("no window was accepted")
    rows = rows_of(out / "sumd_energy.log")
    assert list(rows[0])[:5] == ["Step", "Time_ps", "Cycle", "AcceptedStep", "Walker"]
    assert len(rows) == dcdtools.read_header(str(out / "sumd_traj.dcd"))["nframes"]
    steps = [int(r["Step"]) for r in rows]
    assert steps == sorted(set(steps))
    with open(out / "accepted_steps.csv") as fh:
        saved = {r["accepted_step"]: r for r in csv.DictReader(fh)}
    last = {}
    for row in rows:
        last[row["AcceptedStep"]] = row
    for step, row in last.items():
        assert int(row["Step"]) == int(saved[step]["md_step"])
        assert float(row["Epot_kJmol"]) == pytest.approx(float(saved[step]["epot_kJmol"]), abs=2e-3)
    assert rows[0]["Volume_nm3"] == "" and rows[0]["Density_gmL"] == ""     # no periodic force
    assert float(rows[0]["Temp_K"]) > 0
    assert summary["energy_log"]["rows"] == len(rows)
    assert "Epot_kJmol" in summary["energy_log"]["columns"]


def test_real_run_energy_terms_add_up_to_epot(tmp_path):
    out, summary = _run(tmp_path, "max_cycles = 3\nenergy_terms = yes\n")
    if not summary["final_traj"]["windows"]:
        pytest.skip("no window was accepted")
    for row in rows_of(out / "sumd_energy.log"):
        assert float(row["E_bond"]) == pytest.approx(float(row["Epot_kJmol"]), abs=1e-3)


def _forces():
    openmm = pytest.importorskip("openmm")
    from sumd_openmm.omm_utils import vfswitch
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 1.9, 100.0)
    torsions = openmm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 3, 0.0, 5.0)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(nonbonded.PME)
    nonbonded.setCutoffDistance(1.2)
    for charge in (0.4, -0.4, 0.3, -0.3):
        nonbonded.addParticle(charge, 0.3, 0.5)
    nonbonded.addException(0, 1, 0.0, 0.3, 0.0)
    nonbonded.addException(0, 3, 0.05, 0.3, 0.2)
    for force in (bonds, angles, torsions, nonbonded):
        system.addForce(force)
    system.addForce(openmm.CMMotionRemover())
    positions = [openmm.Vec3(*p) for p in
                 ((0.5, 0.5, 0.5), (0.65, 0.5, 0.5), (0.7, 0.64, 0.5), (0.85, 0.66, 0.6))]
    return openmm, system, positions, vfswitch


@pytest.mark.parametrize("switched", [False, True])
def test_force_groups_split_the_potential_energy_completely(switched):
    from types import SimpleNamespace
    from sumd_openmm.omm_utils import assign_energy_groups
    openmm, system, positions, vfswitch = _forces()
    if switched:
        vfswitch(system, r_on=1.0, r_off=1.2)
    groups = assign_energy_groups(system)
    expected = (["E_bond", "E_angle", "E_dihedral", "E_vdw", "E_elec_direct", "E_elec_recip"] if switched
                else ["E_bond", "E_angle", "E_dihedral", "E_nonbonded", "E_elec_recip"])
    assert list(groups) == expected
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    energy = lambda **kw: context.getState(getEnergy=True, **kw).getPotentialEnergy()._value
    parts = {term: energy(groups={g}) for term, g in groups.items()}
    assert sum(parts.values()) == pytest.approx(energy(), rel=1e-9)
    assert all(np.isfinite(list(parts.values())))
    assert parts["E_bond"] != 0 and parts["E_elec_recip"] != 0


def test_row_agrees_with_state_data_reporter():
    import io
    from types import SimpleNamespace
    from openmm import app, unit
    from sumd_openmm.engine import Engine
    openmm, system, positions, _ = _forces()
    system.addConstraint(2, 3, 0.15)
    topology = app.Topology()
    residue = topology.addResidue("MOL", topology.addChain())
    for name in "ABCD":
        topology.addAtom(name, app.Element.getBySymbol("C"), residue)
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds)
    sim = app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 7)
    report = io.StringIO()
    sim.reporters.append(app.StateDataReporter(report, 5, step=True, time=True, potentialEnergy=True,
                                               kineticEnergy=True, totalEnergy=True, temperature=True,
                                               volume=True, density=True))
    sim.step(5)
    fields = Engine(SimpleNamespace(simulation=sim, system=system), print).energy_row([1.5]).strip().split(",")
    expected = report.getvalue().strip().splitlines()[-1].split(",")
    assert len(fields) == len(expected) + 1 and fields[-1] == "1.5000"
    assert fields[0] == expected[0]
    assert [float(x) for x in fields[1:-1]] == pytest.approx([float(x) for x in expected[1:]], abs=1e-3)


def test_logging_does_not_change_the_dynamics(tmp_path):
    from sumd_openmm import dcdtools
    frames = {}
    for name, extra in (("off", "log_stride = 0\n"), ("on", "energy_terms = yes\n")):
        folder = tmp_path / name
        folder.mkdir()
        _system(folder)
        inp = folder / "run.inp"
        inp.write_text(inp.read_text().replace("platform = CPU", "platform = Reference"))
        out, summary = _run(folder, "max_cycles = 3\n" + extra, setup=False)
        assert summary["final_traj"]["windows"]
        frames[name] = dcdtools.read_frames(str(out / "sumd_traj.dcd"))
    assert np.array_equal(frames["off"], frames["on"])


def test_continue_a_run_whose_early_windows_have_no_log(tmp_path):
    out, summary = _run(tmp_path, "max_cycles = 2\n")
    assert summary["final_traj"]["windows"] == 2
    for name in os.listdir(out / "windows" / "accepted"):       # a run from before energy logs
        if name.endswith(".log"):
            os.remove(out / "windows" / "accepted" / name)
    with open(out / "resume.json") as fh:
        state = json.load(fh)
    for key in ("log_stride", "energy_terms"):
        del state["sumd_config"][key]
    with open(out / "resume.json", "w") as fh:
        json.dump(state, fh)
    inp = tmp_path / "run.inp"
    inp.write_text(inp.read_text().replace("max_cycles = 2\n", "max_cycles = 4\n", 1)
                   .replace("max_cycles = 2\n", "") + "energy_terms = yes\n")
    run = subprocess.run([sys.executable, "-m", "sumd_openmm", "run.inp", "--continue"], cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    rows = rows_of(out / "sumd_energy.log")
    assert rows and {int(r["Cycle"]) for r in rows} == {3, 4}
    assert all(r["E_bond"] != "" for r in rows)
