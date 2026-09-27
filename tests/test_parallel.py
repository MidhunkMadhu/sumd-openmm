"""
Parallel layout rules, and the full SuMD loop through both executors with a
fake engine (no OpenMM). The mpirun test runs 3 real MPI ranks.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sumd_openmm.config import ConfigError
from sumd_openmm.executors import LocalExecutor, RankWorker
from sumd_openmm.parallel import (LayoutError, assign_device, check_unique_devices,
                      parse_devices, walker_ranks)
from fakes import FakeEngine, check_run_invariants, make_config

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- device assignment

def test_parse_devices():
    assert parse_devices("auto") is None
    assert parse_devices("0, 1 3") == [0, 1, 3]


def test_multi_node():
    assert assign_device("multi_node", "auto", 0, 1, {}) is None
    assert assign_device("multi_node", "2", 0, 1, {}) == "2"
    with pytest.raises(LayoutError, match="one MPI rank per node"):
        assign_device("multi_node", "auto", 1, 4, {}, host="gpu07")


def test_multi_gpu_auto_uses_local_rank():
    assert [assign_device("multi_gpu", "auto", k, 4, {}) for k in range(4)] == ["0", "1", "2", "3"]


def test_multi_gpu_explicit_list():
    assert [assign_device("multi_gpu", "4,5,6,7", k, 4, {}) for k in range(4)] == ["4", "5", "6", "7"]
    with pytest.raises(LayoutError, match="lists only 2"):
        assign_device("multi_gpu", "0,1", 2, 3, {})


def test_multi_gpu_scheduler_bound_gpu():
    # srun --gpus-per-task=1: each rank sees exactly one GPU, always ordinal 0
    assert assign_device("multi_gpu", "auto", 3, 4, {"CUDA_VISIBLE_DEVICES": "3"}) == "0"


def test_multi_gpu_too_few_visible():
    with pytest.raises(LayoutError, match="expose 2 GPUs"):
        assign_device("multi_gpu", "auto", 2, 4, {"CUDA_VISIBLE_DEVICES": "0,1"})


def test_amd_scheduler_bound_gpu():
    # Slurm --gpu-bind on AMD nodes (e.g. Dardel MI250X) sets ROCR_VISIBLE_DEVICES
    assert assign_device("multi_gpu", "auto", 5, 8, {"ROCR_VISIBLE_DEVICES": "5"}) == "0"
    assert assign_device("multi_gpu", "auto", 1, 2, {"HIP_VISIBLE_DEVICES": "2,3"}) == "1"
    with pytest.raises(LayoutError, match="expose 2 GPUs"):
        assign_device("multi_gpu", "auto", 2, 3, {"ROCR_VISIBLE_DEVICES": "0,1,2",
                                                   "HIP_VISIBLE_DEVICES": "0,1"})


def test_duplicate_device_warning():
    layout = [dict(rank=0, host="a", device="0"), dict(rank=1, host="a", device="0"),
              dict(rank=2, host="b", device="0")]
    w = check_unique_devices(layout)
    assert len(w) == 1 and "ranks 0 and 1" in w[0]


def test_walker_ranks():
    assert walker_ranks(4, 4) == [0, 1, 2, 3]
    assert walker_ranks(4, 4, first_rank=2) == [2, 0, 1, 3]     # parent holder gets walker 0
    assert walker_ranks(5, 3) == [0, 1, 2, 0, 1]


# ---------------------------------------------------------------- config

def test_walkers_auto():
    assert make_config(walkers="auto").walkers == 1
    c = make_config(walkers="auto", parallel="mpi")
    assert c.walkers == 0 and c.resolve_walkers(4) == [] and c.walkers == 4
    c = make_config(walkers="2", parallel="mpi")
    assert "idle" in c.resolve_walkers(4)[0]
    c = make_config(walkers="5", parallel="mpi")
    assert c.resolve_walkers(4) == []


def test_bad_gpu_devices():
    with pytest.raises(ConfigError, match="gpu_devices"):
        make_config(gpu_devices="0,x")


def test_auto_walkers_single_rank_smscore_refused():
    c = make_config(walkers="auto", parallel="mpi", walker_score="smscore")
    with pytest.raises(ConfigError, match="score modes"):
        c.resolve_walkers(1)


# ---------------------------------------------------------------- full loop, fake engine

def run_fake(tmp_path, executor_factory, n_walkers, drift=0.02, noise=0.2, **over):
    from sumd_openmm.cli import RunLog, SumdRun

    out = str(tmp_path / "run")
    for d in ("states", "windows/accepted", "windows/tmp", "windows/rejected"):
        os.makedirs(os.path.join(out, d))

    scfg = make_config(walkers=n_walkers, **over)
    worker = RankWorker(FakeEngine(seed=11, drift=drift, noise=noise, n=len(scfg.metrics)), None, out,
                        scfg.samples_per_window, 5, 1, True)
    ex = executor_factory(worker)
    log = RunLog(out, [m.name for m in scfg.metrics])
    runner = SumdRun(scfg, ex, out, np.random.default_rng(3), log,
                     worker.eng.current_cvs(None))
    runner.final, why = runner.run()
    ex.shutdown()
    log.close()
    return out, runner, scfg


@pytest.mark.parametrize("over", [
    dict(max_cycles=15, max_retries_per_parent=1),                                   # accept_best path
    dict(max_cycles=15, max_retries_per_parent=1, on_retry_exhaustion="step_back"),
    dict(max_cycles=15, seeding="stratified", band_width=0.2,
         metric_2_type="distance", metric_2_a="indices:2", metric_2_b="indices:3",
         metric_2_role="stratify", metric_2_bins="20", max_retries_per_parent=2),
])
def test_serial_loop_invariants(tmp_path, over):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, **over)
    rows = check_run_invariants(out, runner, walkers=1)
    assert len(rows) == 15, "ran all cycles"
    assert any(r["verdict"] == "reject" for r in rows), "test must exercise rejection"


def test_serial_multiwalker(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, walker_score="smscore", max_cycles=6)
    rows = check_run_invariants(out, runner, walkers=3)
    assert len(rows) == 18
    assert len(runner.nodes) == 7                             # smscore always extends


def test_chain_ends_at_latest_state(tmp_path):
    """mwSuMD keeps extending even when the metric worsens; the trajectory must follow."""
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, drift=0.5, walker_score="smscore",
                                 max_cycles=4)
    assert runner.final == max(runner.nodes) == 4
    assert runner.path_to(runner.final) == [0, 1, 2, 3, 4]
    assert runner.best_node() == 0                            # metric only grew


def test_mwsumd_slope_always_extends_with_selected_velocities(tmp_path):
    """Deganutti et al. 2025: one walker per batch is always productive, no new velocities."""
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, drift=0.5, max_cycles=5)
    rows = check_run_invariants(out, runner, walkers=3)
    assert not scfg.reassign_velocities
    assert len(runner.nodes) == 6
    assert {r["start_mode"] for r in rows} == {"continue", "restore+keep"}
    assert all(r["seed_used"] is None for r in rows)


def test_mwsumd_dmscore_stops_on_one_metric(tmp_path):
    """Paper-style DMscore run: the second metric's target is always met."""
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, drift=-0.3, max_cycles=30,
                                 supervision="combined", walker_score="dmscore",
                                 metric_1_target="15",
                                 metric_2_type="distance", metric_2_a="indices:2",
                                 metric_2_b="indices:3", metric_2_role="supervise",
                                 metric_2_direction="decrease", metric_2_target="1000")
    rows = check_run_invariants(out, runner, walkers=3)
    assert [r["verdict"] for r in rows if r["best_in_batch"]][-1] == "converged"
    assert runner.nodes[runner.final]["metrics"]["metric_1"] < 15
    assert len(rows) < 90


def test_walker_acceptance_sumd_rejects_batches(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, drift=0.5, max_cycles=5,
                                 walker_acceptance="sumd", retry_velocities="reassign")
    rows = check_run_invariants(out, runner, walkers=3)
    assert any(r["verdict"] == "reject" for r in rows)
    assert len(runner.nodes) < 6


def test_velocity_policy():
    assert make_config().reassign_velocities                           # SuMD
    assert not make_config(walkers=3).reassign_velocities              # mwSuMD
    assert make_config(walkers=3, retry_velocities="reassign").reassign_velocities
    assert not make_config(retry_velocities="keep").reassign_velocities
    with pytest.raises(ConfigError, match="walker_score = slope"):
        make_config(walkers=3, walker_score="smscore", walker_acceptance="sumd")


@pytest.mark.parametrize("over", [
    dict(metric_1_direction="increase", metric_1_target="100"),
    dict(supervision="combined", metric_2_type="distance",
         metric_2_a="indices:2", metric_2_b="indices:3",
         metric_2_role="supervise", metric_2_direction="increase", metric_2_target="100",
         metric_1_sigma="1", metric_2_sigma="1"),
    dict(supervision="multistep", stages="1,2", metric_2_type="distance",
         metric_2_a="indices:2", metric_2_b="indices:3",
         metric_2_role="supervise", metric_2_direction="increase", metric_2_target="100"),
    dict(seeding="stratified", metric_2_type="distance",
         metric_2_a="indices:2", metric_2_b="indices:3",
         metric_2_role="stratify", metric_2_bins="20",
         metric_3_type="distance", metric_3_a="indices:4", metric_3_b="indices:5",
         metric_3_role="stratify", metric_3_bins="22"),
])
def test_general_modes_keep_state_and_log_all_metrics(tmp_path, over):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 1, max_cycles=4, **over)
    rows = check_run_invariants(out, runner, 1)
    assert len(rows) == 4
    assert set(rows[0]["metric_series"]) == {metric.name for metric in scfg.metrics}


def test_multistep_changes_stage_after_convergence(tmp_path):
    out, runner, _ = run_fake(
        tmp_path, LocalExecutor, 1, drift=-1, noise=0, max_cycles=3,
        supervision="multistep", stages="1,2", metric_1_target="19",
        metric_2_type="distance", metric_2_a="indices:2",
        metric_2_b="indices:3", metric_2_role="supervise",
        metric_2_direction="increase", metric_2_target="100")
    rows = check_run_invariants(out, runner, 1)
    assert runner.stage == 1
    assert rows[0]["stage"] == 1 and rows[1]["stage"] == 2


def test_mpi_executor_in_process(tmp_path):
    MPI = pytest.importorskip("mpi4py.MPI")
    from sumd_openmm.executors import MPIExecutor

    out, runner, scfg = run_fake(tmp_path, lambda w: MPIExecutor(MPI.COMM_WORLD, w), 2,
                                 max_cycles=10, max_retries_per_parent=1,
                                 walker_acceptance="sumd")
    rows = check_run_invariants(out, runner, walkers=2)
    assert len(rows) == 20


def test_mpirun_three_ranks(tmp_path):
    pytest.importorskip("mpi4py")
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("mpirun not available")

    cmd = [mpirun, "-n", "3"]
    if "Open MPI" in subprocess.run([mpirun, "--version"], capture_output=True, text=True).stdout:
        cmd.append("--oversubscribe")
    cmd += [sys.executable, "-B", os.path.join(HERE, "mpi_fake_driver.py"), str(tmp_path / "mpirun")]

    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "MPI FAKE RUN OK" in p.stdout


def test_progress_line_reports_walker_and_metrics(tmp_path):
    out, runner, scfg = run_fake(tmp_path, LocalExecutor, 3, walker_score="smscore", max_cycles=2)
    lines = [l for l in open(os.path.join(out, "progress.log")) if "] cycle " in l]
    assert len(lines) == 2
    first = lines[0]
    assert "cycle 1/2 from node 0: walker w" in first and "of 3 accepted -> node 1" in first
    assert "metric_1 " in first and " A (" in first
    assert "by walker" not in first
