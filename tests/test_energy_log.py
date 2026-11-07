"""Energy logs: per-window files, the joined path CSV, the mdout, and the real engine's columns."""

import csv
import json
import os
import re
import shutil
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
    windows = {os.path.splitext(f)[0] for f in kept if f.endswith(".dcd")}
    assert windows and windows == {os.path.splitext(f)[0] for f in kept if f.endswith(".log")} \
        == {os.path.splitext(f)[0] for f in kept if f.endswith(".csv")}
    assert not os.listdir(os.path.join(out, "windows", "tmp"))
    assert not os.listdir(os.path.join(out, "windows", "rejected"))


def test_rejected_logs_are_kept_with_rejected_dcds(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, max_cycles=4, drift=-0.1,
                                 keep_rejected_dcd="yes")
    names = os.listdir(os.path.join(out, "windows", "rejected"))
    dcds = sum(f.endswith(".dcd") for f in names)
    assert dcds and sum(f.endswith(".log") for f in names) == sum(f.endswith(".csv") for f in names) == dcds


def test_log_off_writes_no_logs(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, max_cycles=3, drift=-0.3, log_stride=0)
    kept = os.listdir(os.path.join(out, "windows", "accepted"))
    assert kept and not any(f.endswith((".log", ".csv")) for f in kept)
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
    rows = rows_of(out / "sumd_energy.csv")
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
    assert rows[0]["Pressure_bar"] == "" and float(rows[-1]["Wall_s"]) > 0
    assert float(rows[0]["Temp_K"]) > 0
    assert summary["energy_log"]["rows"] == len(rows)
    assert "Epot_kJmol" in summary["energy_log"]["columns"]
    assert "Wall_s" not in summary["energy_log"]["columns"]
    assert summary["energy_log"]["data"] == "sumd_energy.csv"
    report = (out / "sumd_energy.log").read_text()
    assert len(re.findall(r"^ NSTEP =", report, re.M)) == len(rows) + 2      # averages and fluctuations
    assert "Here is the input file:" in report and "metric_1_target = 1" in report
    assert "ns/day" in report


def test_real_run_energy_terms_add_up_to_epot(tmp_path):
    out, summary = _run(tmp_path, "max_cycles = 3\nenergy_terms = yes\n")
    if not summary["final_traj"]["windows"]:
        pytest.skip("no window was accepted")
    for row in rows_of(out / "sumd_energy.csv"):
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
    assert fields[-1] == "1.5000" and fields[0] == expected[0]
    assert fields[6] == "" and fields[9] == "0.000"            # no barostat; wall time not given
    logged = fields[1:6] + fields[7:9]
    assert [float(x) for x in logged] == pytest.approx([float(x) for x in expected[1:]], abs=1e-3)


def _barostat_run(logging, kind="isotropic"):
    from types import SimpleNamespace
    from openmm import app, unit
    from sumd_openmm.engine import Engine
    openmm, system, positions, _ = _forces()
    if kind == "membrane":
        cls = openmm.MonteCarloMembraneBarostat
        barostat = cls(1 * unit.bar, 0 * unit.bar * unit.nanometer, 300 * unit.kelvin,
                       cls.XYIsotropic, cls.ZFree, 2)
    elif kind == "anisotropic":
        barostat = openmm.MonteCarloAnisotropicBarostat(
            openmm.Vec3(1, 1, 1) * unit.bar, 300 * unit.kelvin, True, True, True, 2)
    else:
        barostat = openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 2)
    barostat.setRandomNumberSeed(5)
    system.addForce(barostat)
    topology = app.Topology()
    residue = topology.addResidue("MOL", topology.addChain())
    for name in "ABCD":
        topology.addAtom(name, app.Element.getBySymbol("C"), residue)
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds)
    integrator.setRandomNumberSeed(9)
    sim = app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 7)
    engine = Engine(SimpleNamespace(simulation=sim, system=system), print)
    rows = []
    for _ in range(10):
        sim.step(5)
        if logging:
            rows.append(engine.energy_row([]))
    state = sim.context.getState(getPositions=True)
    return rows, state.getPositions(asNumpy=True)._value, state.getPeriodicBoxVolume()._value


@pytest.mark.parametrize("kind", ["isotropic", "membrane", "anisotropic"])
def test_pressure_is_logged_without_changing_the_dynamics(kind):
    rows, positions, volume = _barostat_run(True, kind)
    _, positions_off, volume_off = _barostat_run(False, kind)
    assert np.array_equal(positions, positions_off) and volume == volume_off
    assert all(np.isfinite(float(row.split(",")[6])) for row in rows)
    volumes = [float(row.split(",")[7]) for row in rows]
    assert len(set(volumes)) > 1 and volumes[-1] == pytest.approx(volume, abs=1e-4)


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
        if name.endswith((".log", ".csv")):
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
    rows = rows_of(out / "sumd_energy.csv")
    assert rows and {int(r["Cycle"]) for r in rows} == {3, 4}
    assert all(r["E_bond"] != "" for r in rows)


def test_continue_reads_windows_of_version_0_9_0(tmp_path):
    out, summary = _run(tmp_path, "max_cycles = 2\n")
    accepted = out / "windows" / "accepted"
    names = sorted(n for n in os.listdir(accepted) if n.endswith(".csv"))
    assert names
    for name in names:                                  # 0.9.0 wrote the CSV as the .log
        os.replace(accepted / name, accepted / name.replace(".csv", ".log"))
    inp = tmp_path / "run.inp"
    inp.write_text(inp.read_text().replace("max_cycles = 2\n", "max_cycles = 4\n", 1)
                   .replace("max_cycles = 2\n", ""))
    run = subprocess.run([sys.executable, "-m", "sumd_openmm", "run.inp", "--continue"], cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert {int(r["Cycle"]) for r in rows_of(out / "sumd_energy.csv")} >= {1, 2, 3}


AMBER_RECORD = """\
 NSTEP =      100   TIME(PS) =       0.200  TEMP(K) =   361.48  PRESS =   169.0
 Etot   =     -2131.5876  EKtot   =       842.5945  EPtot      =     -2974.1820
                                                    VOLUME     =     20387.7248
                                                    Density    =         0.5723
 ------------------------------------------------------------------------------
"""


def test_a_record_is_laid_out_as_amber_prints_it():
    values = {"NSTEP": 100, "TIME(PS)": 0.2, "TEMP(K)": 361.48, "PRESS": 169.0, "Etot": -2131.5876,
              "EKtot": 842.5945, "EPtot": -2974.182, "VOLUME": 20387.7248, "Density": 0.5723}
    assert "\n".join(energylog._block(values, [])[:5]) + "\n" == AMBER_RECORD


def amber_csv(path, terms, rows):
    """An energy CSV of `rows` (step, wall time), with the energy `terms` in kJ/mol and a metric."""
    names = list(terms)
    text = [",".join(energylog.COLUMNS + names + ["dist_lig"])]
    total = sum(terms.values())
    for step, wall in rows:
        text.append(",".join(map(str, [step, step * 0.002, total, 418.4, total + 418.4, 300.0, 1.5, 20.0,
                                       0.9, wall, *terms.values(), 12.5])))
    path.write_text("\n".join(text) + "\n")
    return str(path)


def fields_of(report):
    """The NAME = value pairs of the first record of an mdout."""
    record = report[report.index(" NSTEP ="):report.index(" " + "-" * 78)]
    return {name.strip(): float(value) for name, value in re.findall(r"([\w() -]+?) *= *(-?[\d.]+)", record)}


@pytest.mark.parametrize("terms, shown", [
    (dict(E_bond=41.84, E_angle=8.368, E_dihedral=4.184, E_vdw=83.68, E_elec_direct=-418.4, E_elec_recip=41.84),
     dict(BOND=10, ANGLE=2, DIHED=1, VDWAALS=20, EELEC=-90)),
    (dict(E_bond=41.84, E_nonbonded=-418.4, E_elec_recip=41.84, E_other=4.184),
     dict(BOND=10, NONBND=-100, RECIP=10, OTHER=1)),
    (dict(E_elec_direct=-418.4, E_nonbonded_recip=41.84), dict(EELEC=-100, RECIP=10)),
])
def test_mdout_terms_are_in_kcal_and_add_up_to_eptot(tmp_path, terms, shown):
    out = tmp_path / "w.log"
    energylog.render(amber_csv(tmp_path / "w.csv", terms, [(50, 1.0)]), str(out))
    fields = fields_of(out.read_text())
    assert {name: fields[name] for name in shown} == pytest.approx(shown)
    assert sum(shown.values()) == pytest.approx(fields["EPtot"], abs=1e-3)
    assert not {"EKCMT", "VIRIAL", "1-4 NB", "EHBOND"} & set(fields)


def test_mdout_units_sections_and_statistics(tmp_path):
    data = amber_csv(tmp_path / "w.csv", dict(E_bond=41.84), [(50, 1.0), (100, 2.0)])
    rows = rows_of(data)
    rows[1]["Temp_K"] = "310.0"
    with open(data, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "w.log"
    meta = dict(version="9.9", input="metric_1 = x\n", platform="CPU", dt_ps=0.002, temperature_K=300.0, ntpr=50)
    assert energylog.render(data, str(out), meta, dict(Cycle=3, Walker=1)) == 2
    report = out.read_text()

    titles = re.findall(r"^   (\d)\.  ", report, re.M)
    assert titles == list("12345")
    assert "imin    =       0" in report and "nstlim  =     100, ntpr    =      50" in report
    assert report.count("| Cycle 3  Walker 1") == 1
    assert "|  dist_lig" not in report and re.search(r"^\| dist_lig +12\.5000$", report, re.M)

    fields = fields_of(report)
    assert fields["EKtot"] == pytest.approx(100.0) and fields["VOLUME"] == pytest.approx(20000.0)
    assert fields["Density"] == pytest.approx(0.9) and fields["PRESS"] == pytest.approx(1.5)
    averages = report[report.index("A V E R A G E S   O V E R      2 R E C O R D S"):]
    assert "TEMP(K) =   305.00" in averages
    assert "TEMP(K) =     5.00" in averages[averages.index("R M S  F L U C T U A T I O N S"):]
    assert "Elapsed(s) =       2.00" in report and "ns/day =       8.64" in report


def test_mdout_without_box_or_barostat_omits_them(tmp_path):
    data = tmp_path / "w.csv"
    data.write_text("Step,Time_ps,Epot_kJmol,Ekin_kJmol,Etot_kJmol,Temp_K,Pressure_bar,Volume_nm3,Density_gmL,Wall_s\n"
                    "10,0.02,-4.184,4.184,0.0,300.0,,,,\n")
    out = tmp_path / "w.log"
    energylog.render(str(data), str(out))
    report = out.read_text()
    assert "VOLUME" not in report and "Density" not in report and "PRESS =     0.0" in report
    assert "Elapsed" not in report


def test_joined_mdout_marks_each_window_and_adds_their_wall_time(tmp_path):
    a = amber_csv(tmp_path / "a.csv", {}, [(50, 1.0), (100, 2.0)])
    b = amber_csv(tmp_path / "b.csv", {}, [(150, 3.0)])
    joined = str(tmp_path / "j.csv")
    energylog.join_logs([(a, 1, 1, 0), (b, 4, 2, 1)], joined)
    out = tmp_path / "j.log"
    energylog.render(joined, str(out))
    report = out.read_text()
    assert report.index("| Cycle 1  AcceptedStep 1  Walker 0") < report.index("| Cycle 4  AcceptedStep 2  Walker 1")
    assert report.count("| Cycle") == 2 and "Elapsed(s) =       5.00" in report


@pytest.mark.skipif(not shutil.which("process_mdout.perl"), reason="AmberTools is not on PATH")
def test_amber_process_mdout_reads_the_log(tmp_path):
    data = amber_csv(tmp_path / "w.csv", dict(E_bond=41.84), [(50, 1.0), (100, 2.0)])
    energylog.render(data, str(tmp_path / "w.log"), dict(ntpr=50, dt_ps=0.002))
    subprocess.run(["process_mdout.perl", "w.log"], cwd=tmp_path, capture_output=True, check=True)
    temperature = np.loadtxt(tmp_path / "summary.TEMP")
    assert temperature[:, 1] == pytest.approx([300.0, 300.0])
    assert np.loadtxt(tmp_path / "summary.PRES")[:, 1] == pytest.approx([1.5, 1.5])
    assert np.loadtxt(tmp_path / "summary.EKTOT")[:, 1] == pytest.approx([100.0, 100.0])
