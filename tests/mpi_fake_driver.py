"""
Launched by test_parallel.py under `mpirun -n 3`: the full SuMD loop on 3
real MPI ranks with fake engines (no OpenMM). Rank 0 checks the invariants
and prints MPI FAKE RUN OK.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "src")]

import numpy as np
from mpi4py import MPI

from sumd_openmm.executors import MPIExecutor, RankWorker, worker_loop
from fakes import FakeEngine, check_run_invariants, make_config


def main(out):
    comm = MPI.COMM_WORLD
    scfg = make_config(parallel="mpi", walkers="auto", max_cycles=12, max_retries_per_parent=4,
                       seeding="stratified", band_width=0.2,
                       metric_2_type="distance", metric_2_a="indices:2",
                       metric_2_b="indices:3", metric_2_role="stratify", metric_2_bins="20")
    scfg.resolve_walkers(comm.Get_size())

    if comm.rank == 0:
        for d in ("states", "windows/accepted", "windows/tmp", "windows/rejected"):
            os.makedirs(os.path.join(out, d), exist_ok=True)
    comm.barrier()

    # every rank has a different noise stream, like a different GPU/integrator seed
    worker = RankWorker(FakeEngine(seed=100 + comm.rank, drift=0.02, n=2), None, out,
                        scfg.samples_per_window, 5, 1, True, rank=comm.rank)

    if comm.rank != 0:
        worker_loop(comm, worker)
        return

    from sumd_openmm.cli import RunLog, SumdRun
    ex = MPIExecutor(comm, worker)
    log = RunLog(out, [m.name for m in scfg.metrics])
    runner = SumdRun(scfg, ex, out, np.random.default_rng(3), log,
                     worker.eng.current_cvs(None))
    try:
        runner.run()
    finally:
        ex.shutdown()
        log.close()

    rows = check_run_invariants(out, runner, walkers=comm.Get_size())
    assert len(rows) == 12 * comm.Get_size(), "ran all cycles"
    ranks = {r["rank"] for r in rows}
    assert ranks == set(range(comm.Get_size())), "every rank must have run walkers: %s" % ranks
    assert any(r["verdict"] == "reject" for r in rows)
    print("MPI FAKE RUN OK: %d nodes, walkers ran on ranks %s" % (len(runner.nodes), sorted(ranks)))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        MPI.COMM_WORLD.Abort(1)
