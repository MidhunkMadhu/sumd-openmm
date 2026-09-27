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
    with pytest.raises(LayoutError, match="exposes 2 GPUs"):
        assign_device("multi_gpu", "auto", 2, 4, {"CUDA_VISIBLE_DEVICES": "0,1"})


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
    c = make_config(walkers="auto")
    assert c.walkers == 1                                     # serial: classic SuMD
    c = make_config(walkers="auto", parallel="mpi")
    assert c.walkers == 0 and c.resolve_walkers(4) == [] and c.walkers == 4
    c = make_config(walkers="2", parallel="mpi")
    assert "idle" in c.resolve_walkers(4)[0]
    c = make_config(walkers="5", parallel="mpi")
    assert "not a multiple" in c.resolve_walkers(4)[0]


def test_bad_gpu_devices():
    with pytest.raises(ConfigError, match="gpu_devices"):
        make_config(gpu_devices="0,x")


def test_auto_walkers_single_rank_smscore_refused():
    c = make_config(walkers="auto", parallel="mpi", walker_score="smscore")
    with pytest.raises(ConfigError, match="walkers = 1"):
        c.resolve_walkers(1)


# ---------------------------------------------------------------- full loop, fake engine

def run_fake(tmp_path, executor_factory, n_walkers, drift=0.02, **over):
    from sumd_openmm.cli import RunLog, SumdRun

    out = str(tmp_path / "run")
    for d in ("states", "windows/accepted", "windows/tmp", "windows/rejected"):
        os.makedirs(os.path.join(out, d))

    scfg = make_config(walkers=n_walkers, **over)
    worker = RankWorker(FakeEngine(seed=11, drift=drift), None, out,
                        scfg.samples_per_window, 5, 1, True)
    ex = executor_factory(worker)
    log = RunLog(out)
    runner = SumdRun(scfg, ex, out, np.random.default_rng(3), log)
    final, why = runner.run()
    ex.shutdown()
    log.close()
    return out, runner, scfg


@pytest.mark.parametrize("over", [
    dict(max_cycles=15, max_retries_per_parent=1),                                   # accept_best path
    dict(max_cycles=15, max_retries_per_parent=1, on_retry_exhaustion="step_back"),
    dict(max_cycles=15, supervision="stratified", cv1_band_width=0.2, max_retries_per_parent=2, drift=0.0),
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


def test_mpi_executor_in_process(tmp_path):
    MPI = pytest.importorskip("mpi4py.MPI")
    from sumd_openmm.executors import MPIExecutor

    out, runner, scfg = run_fake(tmp_path, lambda w: MPIExecutor(MPI.COMM_WORLD, w), 2,
                                 max_cycles=10, max_retries_per_parent=1)
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
