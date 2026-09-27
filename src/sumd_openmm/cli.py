#!/usr/bin/env python3
"""
cli.py - the sumd-openmm command

Supervised Molecular Dynamics (SuMD) on OpenMM, built on the MD_openmm
production setup path (bundled in sumd_openmm.md_openmm).

    sumd-openmm sumd.inp                 run (serial: one GPU)
    sumd-openmm sumd.inp --dry-run       resolve selections, build the system,
                                         print initial CVs, stop
    mpirun -n 4 sumd-openmm sumd.inp     run with parallel = mpi: 4 walkers at
                                         once, one per GPU
    (python -m sumd_openmm ... works the same)

SuMD methodology: Sabbadin & Moro, J Chem Inf Model 54:372 (2014).
Amber reference implementation (behaviour reproduced by supervision = single):
supervisedmdamber, M. K. Madhu. Multiple-walker batches and SMscore/DMscore:
Deganutti et al., eLife 96513 (2025).

Windows are ordinary unbiased dynamics. Only the SELECTION of windows is
biased, so SuMD segments carry no valid thermodynamics or kinetics; they are a
pathway generator and a source of seeds for unbiased runs.

Execution
---------
Every rank builds its own Simulation through MD_openmm from the same .inp.
Rank 0 runs the SuMD logic (this file's SumdRun) and also runs walkers; the
executor (executors.py) decides where each walker runs:
  * an accepted window continues in place (velocities kept, as Amber irest=1);
  * every other start restores the parent State (positions, velocities, box,
    time, parameters) and, with retry_velocities = reassign, draws new
    velocities with a fresh, logged seed;
  * each window writes its own DCD, kept on acceptance, deleted on rejection;
  * every accepted node's State is written as OpenMM XML, which
    openmm_production.py accepts directly as coordinate_file for seeding.
"""

import argparse
import datetime
import json
import os
import platform as pyplatform
import shutil
import socket
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

from . import __version__
from . import cv as CV
from . import dcdtools
from . import selection
from .config import ConfigError, SumdConfig, read_key_value_file
from .engine import CVEvaluator, Engine
from .executors import LocalExecutor, MPIExecutor, RankWorker, worker_loop
from .omm_setup import build_simulation, load_production_helpers
from .parallel import LayoutError, assign_device, check_unique_devices
from .seeding import StratifiedPool
from .supervision import (DMSCORE_WARNING, JOINT_WARNING, joint_series,
                         pick_best_walker, sumd_decision)


# ============================================================
# Logging
# ============================================================

class RunLog:

    def __init__(self, outdir):
        self.progress = open(os.path.join(outdir, "progress.log"), "a", buffering=1)
        self.windows = open(os.path.join(outdir, "windows.jsonl"), "a", buffering=1)
        self.nodes_path = os.path.join(outdir, "nodes.csv")

        with open(self.nodes_path, "w") as f:
            f.write("node_id,parent_id,cycle,walker,cv1_final,cv2_final,cv1_band,cv2_stratum,"
                    "time_ps,step,epot_kJmol,reason,dcd,state_xml\n")

    def __call__(self, msg):
        line = "[%s] %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        self.progress.write(line + "\n")

    def window(self, row):
        self.windows.write(json.dumps(row, default=_json_default) + "\n")

    def node(self, n):
        cols = ["id", "parent", "cycle", "walker", "cv1", "cv2", "band", "stratum",
                "time_ps", "step", "epot", "reason", "dcd", "state_xml"]
        vals = ["" if n.get(c) is None else str(n[c]) for c in cols]
        vals[11] = '"%s"' % vals[11].replace('"', "'")

        with open(self.nodes_path, "a") as f:
            f.write(",".join(vals) + "\n")

    def close(self):
        self.progress.close()
        self.windows.close()


_FATAL_DIR = None      # output_dir once known, for FATAL_rank_N.txt


def report_fatal(rank, msg, echo=True):
    """
    Make a fatal error visible before MPI_Abort. Open MPI kills all processes
    on abort and can drop output it has not forwarded yet, so the message goes
    to stderr and (if output_dir exists) a file, then we pause.
    """
    text = "[rank %d] FATAL: %s\n" % (rank, str(msg).rstrip())
    if echo:
        try:
            sys.__stderr__.write(text)
            sys.__stderr__.flush()
        except Exception:
            pass
    if _FATAL_DIR and os.path.isdir(_FATAL_DIR):
        try:
            with open(os.path.join(_FATAL_DIR, "FATAL_rank_%03d.txt" % rank), "w") as f:
                f.write(text)
        except OSError:
            pass
    time.sleep(2.0)


def rank_logger(rank):
    def log(msg):
        print("[%s] [rank %d] %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank, msg),
              flush=True)
    return log


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _num(x, nd=4):
    """Round for the log; NaN/inf become null, never a fake number."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def steps_per_sample(scfg, dt_ps):
    sps = scfg.cv_sample_ps / dt_ps
    if abs(sps - round(sps)) > 1e-6:
        raise ConfigError("cv_sample_ps (%g) must be a multiple of dt (%g ps)" % (scfg.cv_sample_ps, dt_ps))
    return int(round(sps))


# ============================================================
# The SuMD loop (rank 0 only)
# ============================================================

class SumdRun:

    def __init__(self, scfg, executor, outdir, rng, log):
        self.s = scfg
        self.x = executor
        self.out = outdir
        self.rng = rng
        self.log = log

        self.nodes = OrderedDict()
        self.retries = {}
        self.seen_series = {}
        self.joint_mu = scfg.joint_mu
        self.joint_sigma = scfg.joint_sigma

        self.pool = None
        if scfg.supervision == "stratified":
            self.pool = StratifiedPool(scfg.cv1_band_width, scfg.gate_open_cutoff, scfg.gate_strata,
                                       scfg.pool_per_cell, scfg.frontier_bands,
                                       scfg.max_retries_per_parent, rng)

    # ---------------------------------------------- helpers

    def new_seed(self):
        return int(self.rng.integers(1, 2 ** 31 - 1))

    def band_stratum(self, cv1, cv2):
        if self.pool is not None:
            return self.pool.band(cv1), (None if cv2 is None else self.pool.stratum(cv2))
        band = int(np.floor(cv1 / self.s.cv1_band_width))
        stratum = None if cv2 is None else int(cv2 >= self.s.gate_open_cutoff)
        return band, stratum

    def xml_rel(self, nid):
        return os.path.join("states", "node_%06d.xml" % nid)

    def parent_ref(self, nid):
        n = self.nodes[nid]
        return dict(id=nid, xml=os.path.join(self.out, n["state_xml"]), step=n["step"], epot=n["epot"])

    def record_node(self, parent, cycle, walker, info, dcd, reason):
        nid = len(self.nodes)
        band, stratum = self.band_stratum(info["cv1"], info["cv2"])

        n = dict(id=nid, parent=parent, cycle=cycle, walker=walker, cv1=_num(info["cv1"]),
                 cv2=_num(info["cv2"]), band=band, stratum=stratum, time_ps=_num(info["time_ps"], 3),
                 step=info["step"], epot=_num(info["epot"], 3), reason=reason, dcd=dcd,
                 state_xml=self.xml_rel(nid))
        self.nodes[nid] = n
        self.log.node(n)

        if self.pool is not None:
            for ev in self.pool.add(nid, info["cv1"], info["cv2"]):
                self.log("pool: node %d evicted from its cell (pool_per_cell = %d)" % (ev, self.s.pool_per_cell))

        return nid

    def keep_dcd(self, tmp, cycle, walker):
        rel = os.path.join("windows", "accepted", "c%05d_w%d.dcd" % (cycle, walker))
        shutil.move(tmp, os.path.join(self.out, rel))
        return rel

    def drop_dcd(self, tmp):
        if tmp is None or not os.path.exists(tmp):
            return
        if self.s.keep_rejected_dcd:
            shutil.move(tmp, os.path.join(self.out, "windows", "rejected", os.path.basename(tmp)))
        else:
            os.remove(tmp)

    def supervised_series(self, cv1, cv2):
        """The series the slope rule sees: CV1, or the joint score s(t)."""
        if self.s.supervision != "joint":
            return cv1

        if self.joint_mu is None or self.joint_sigma is None:
            mu = [float(np.mean(cv1)), float(np.mean(cv2))]
            sd = [float(np.std(cv1, ddof=1)), float(np.std(cv2, ddof=1))]
            if min(sd) <= 0:
                raise ConfigError("joint: reference window has zero variance in a CV; "
                                  "set joint_mu / joint_sigma explicitly")
            self.joint_mu = self.joint_mu or mu
            self.joint_sigma = self.joint_sigma or sd
            self.log("joint: mu = %s, sigma = %s taken from the first window" % (self.joint_mu, self.joint_sigma))

        return joint_series(cv1, cv2, self.joint_mu, self.joint_sigma, self.s.joint_weights, self.s.cv2_direction)

    def analyse(self, r, parent):
        """Checks and trend for one walker result (on rank 0)."""
        s = self.s

        if not np.all(np.isfinite(r["cv1"])):
            raise RuntimeError("cycle %d walker %d (rank %d): non-finite CV1 %s. Check selections and imaging."
                               % (r["cycle"], r["w"], r["rank"], r["cv1"]))

        if s.retry_velocities == "keep":
            sig = tuple(np.round(r["cv1"], 6))
            if sig in self.seen_series.setdefault(parent, set()):
                self.log("WARNING: cycle %d walker %d reproduced an earlier attempt from node %d "
                         "exactly. retry_velocities = keep is not diversifying on this platform; "
                         "use reassign." % (r["cycle"], r["w"], parent))
            self.seen_series[parent].add(sig)

        r["trend"] = CV.window_trend(r["t"], self.supervised_series(r["cv1"], r["cv2"]), s.slope_points)

        if self.pool is not None and r["cv2"] is not None:
            self.pool.observe_cv2(r["cv2"])

    # ---------------------------------------------- main loop

    def run(self):
        s = self.s
        log = self.log

        root = self.x.root(os.path.join(self.out, self.xml_rel(0)))
        self.record_node(None, 0, None, root, None, "initial state")
        log("initial state: CV1 = %.3f A, CV2 = %s"
            % (root["cv1"], "null" if root["cv2"] is None else "%.3f A" % root["cv2"]))

        parent = 0
        held = None          # least-bad rejected attempt from `parent` (accept_best)
        stop_reason = "max_cycles"
        final = None

        for cycle in range(1, s.max_cycles + 1):

            if os.path.exists(os.path.join(self.out, "STOP")):
                stop_reason = "STOP file"
                break

            cell = visit = None
            if self.pool is not None:
                entry, cell, visit = self.pool.choose()
                if entry is None:
                    stop_reason = "pool exhausted: every state retired after max_retries_per_parent"
                    break
                if entry.node_id != parent and held is not None:
                    self.drop_dcd(held["dcd_tmp"])
                    held = None
                parent = entry.node_id

            batch_parent = parent
            seeds = [self.new_seed() if s.retry_velocities == "reassign" else None
                     for _ in range(s.walkers)]

            wall0 = time.time()
            batch = self.x.run_batch(cycle, self.parent_ref(parent), seeds)
            batch_wall = time.time() - wall0

            for r in batch:
                self.analyse(r, parent)

            # -------------------------------------------- decide
            if s.walkers > 1:
                best, scores = pick_best_walker(s.walker_score, [r["trend"] for r in batch],
                                                [r["cv1"] for r in batch],
                                                [r["cv2"] for r in batch], s.cv2_direction)
            else:
                best, scores = 0, [batch[0]["trend"].slope]

            b = batch[best]

            if s.walker_score == "slope":
                decision, reason = sumd_decision(b["trend"].slope, b["cv1_final"], s.colvarcut,
                                                 b["trend"].se, s.require_significant_slope)
            else:
                decision = "converged" if b["cv1_final"] < s.colvarcut else "accept"
                reason = "best of %d walkers by %s = %.4g" % (s.walkers, s.walker_score, scores[best])

            verdict = decision          # the rule's verdict on this cycle's best window
            accepted = b if decision != "reject" else None
            action = None

            if decision == "reject":
                if held is None or b["cv1_final"] < held["cv1_final"]:
                    if held is not None:
                        self.drop_dcd(held["dcd_tmp"])
                    held = b

                if self.pool is not None:
                    n_rej = self.pool.entries[parent].retries + 1
                    if self.pool.record_rejection(parent):
                        if self.pool.live_entries():
                            action = "parent %d retired from pool after %d rejections" % (parent, n_rej)
                            self.drop_dcd(held["dcd_tmp"])
                            held = None
                        else:
                            # retiring the last live state would end the run
                            accepted = held
                            held = None
                            decision = "accept"
                            action = ("parent %d retired after %d rejections and no other live state: "
                                      "accepting least-bad attempt (cycle %d walker %d, CV1 %.3f)"
                                      % (parent, n_rej, accepted["cycle"], accepted["w"], accepted["cv1_final"]))
                else:
                    self.retries[parent] = self.retries.get(parent, 0) + 1

                    if self.retries[parent] > s.max_retries_per_parent:
                        gp = self.nodes[parent]["parent"]

                        if s.on_retry_exhaustion == "step_back" and gp is not None:
                            action = "retry limit at node %d: stepping back to node %d" % (parent, gp)
                            self.drop_dcd(held["dcd_tmp"])
                            held = None
                            parent = gp
                        else:
                            accepted = held
                            held = None
                            decision = "accept"
                            action = ("retry limit at node %d: accepting least-bad attempt "
                                      "(cycle %d walker %d, CV1 %.3f)"
                                      % (parent, accepted["cycle"], accepted["w"], accepted["cv1_final"]))
                            if gp is None and s.on_retry_exhaustion == "step_back":
                                action += " (step_back impossible at the root)"

            # -------------------------------------------- commit
            child = None
            commit_key = None
            if accepted is not None:
                dcd = self.keep_dcd(accepted["dcd_tmp"], accepted["cycle"], accepted["w"])
                info = dict(cv1=accepted["cv1_final"], cv2=accepted["cv2_final"],
                            time_ps=accepted["time_ps"], step=accepted["step"], epot=accepted["epot"])
                child = self.record_node(batch_parent, accepted["cycle"], accepted["w"], info, dcd,
                                         action or reason)
                commit_key = accepted["key"]

                if self.pool is None:
                    parent = child
                if held is not None:
                    self.drop_dcd(held["dcd_tmp"])
                    held = None

            self.x.resolve(commit_key, child,
                           None if child is None else os.path.join(self.out, self.xml_rel(child)),
                           [] if held is None else [held["key"]])

            for r in batch:
                if r is not accepted and r is not held:
                    self.drop_dcd(r["dcd_tmp"])

            # -------------------------------------------- log
            for r in batch:
                band, stratum = self.band_stratum(r["cv1_final"], r["cv2_final"])
                is_best = r is b
                self.log.window(dict(
                    cycle=cycle, walker=r["w"], rank=r["rank"], host=r["host"], device=r["device"],
                    parent_state_id=batch_parent,
                    child_state_id=child if r is accepted else None,
                    accepted=r is accepted, best_in_batch=is_best,
                    verdict=verdict if is_best else "not best",
                    reason=reason if is_best else None,
                    action=action if is_best else None,
                    committed=(None if accepted is None or accepted in batch else
                               dict(cycle=accepted["cycle"], walker=accepted["w"], child_state_id=child)),
                    seed_used=r["seed"], start_mode=r["start_mode"], restore_dE_kJmol=_num(r["restore_dE"]),
                    retries_so_far=(self.pool.entries[batch_parent].retries if self.pool is not None
                                    else self.retries.get(batch_parent, 0)),
                    wallclock_s=_num(r["wall"], 2), batch_wallclock_s=_num(batch_wall, 2),
                    t_ps=[_num(v, 3) for v in r["t"]],
                    cv1_series=[_num(v) for v in r["cv1"]],
                    cv2_series=None if r["cv2"] is None else [_num(v) for v in r["cv2"]],
                    supervised=("joint" if s.supervision == "joint" else "cv1"),
                    slope_b=_num(r["trend"].slope, 6), se_b=_num(r["trend"].se, 6),
                    r_squared=_num(r["trend"].r2), slope_units=r["trend"].units,
                    walker_score=s.walker_score, score=_num(scores[r["w"]], 6),
                    cv1_final=_num(r["cv1_final"]), cv2_final=_num(r["cv2_final"]),
                    cv1_band=band, cv2_stratum=stratum,
                    cell_chosen=None if cell is None else list(cell), cell_visit_count=visit,
                ))

            t = b["trend"]
            if child is None:
                outcome = "no node"
            elif accepted is b:
                outcome = "node %d" % child
            else:
                outcome = "node %d <- cycle %d walker %d" % (child, accepted["cycle"], accepted["w"])
            walkers_txt = "" if s.walkers == 1 else " | best w%d/%d (rank %d) %.1fs" % (
                b["w"], s.walkers, b["rank"], batch_wall)
            log("cycle %4d | parent %4d | %-9s | CV1 %.3f  CV2 %s | slope %+.4g (SE %.2g) %s | %s%s%s"
                % (cycle, batch_parent, verdict.upper(), b["cv1_final"],
                   "null " if b["cv2_final"] is None else "%.3f" % b["cv2_final"],
                   t.slope, t.se if np.isfinite(t.se) else float("nan"), t.units,
                   outcome, walkers_txt, (" | " + action) if action else ""))

            if decision == "converged":
                final = child
                stop_reason = "converged: " + reason
                break

        if held is not None:
            self.drop_dcd(held["dcd_tmp"])

        if final is None:
            final = min(self.nodes, key=lambda i: self.nodes[i]["cv1"])
            log("not converged (%s); final path ends at best node %d (CV1 %.3f)"
                % (stop_reason, final, self.nodes[final]["cv1"]))

        return final, stop_reason

    def path_to(self, node_id):
        path = []
        while node_id is not None:
            path.append(node_id)
            node_id = self.nodes[node_id]["parent"]
        return path[::-1]


# ============================================================
# Setup
# ============================================================

def software_versions():
    v = dict(python=sys.version.split()[0], numpy=np.__version__, host=pyplatform.node())
    try:
        import openmm
        v["openmm"] = openmm.version.version
        v["openmm_git"] = openmm.version.git_revision
    except Exception:
        v["openmm"] = None
    try:
        import parmed
        v["parmed"] = parmed.__version__
    except Exception:
        v["parmed"] = None
    try:
        import mpi4py
        from mpi4py import MPI
        v["mpi4py"] = mpi4py.__version__
        v["mpi"] = MPI.Get_library_version().strip().splitlines()[0]
    except Exception:
        pass
    return v


def platform_details(omm):
    d = dict(platform=omm.platform_name, requested=omm.platform_request)
    try:
        plat = omm.simulation.context.getPlatform()
        for name in plat.getPropertyNames():
            d[name] = plat.getPropertyValue(omm.simulation.context, name)
    except Exception:
        pass
    return d


def md_openmm_provenance():
    """'commit <hash> (<date>)' of the bundled MD_openmm helpers."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md_openmm", "PROVENANCE.txt")
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("commit:"):
                    return "MD_openmm commit " + line.split(":", 1)[1].strip()[:40]
    except OSError:
        pass
    return "MD_openmm (provenance file missing)"


def mpi_layout(comm, scfg):
    """This rank's host, node-local rank and GPU; the full table on rank 0."""
    from mpi4py import MPI

    host = MPI.Get_processor_name()
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank, local_size = local.Get_rank(), local.Get_size()
    local.Free()

    err, device = None, None
    try:
        device = assign_device(scfg.mpi_mode, scfg.gpu_devices, local_rank, local_size, os.environ, host)
    except LayoutError as exc:
        err = str(exc)

    me = dict(rank=comm.rank, host=host, local_rank=local_rank, local_size=local_size,
              device=device, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), error=err)
    table = comm.allgather(me)

    errors = sorted({r["error"] for r in table if r["error"]})
    if errors:
        raise LayoutError("; ".join(errors))

    return me, table


def run_setup(args, scfg, comm):
    rank = 0 if comm is None else comm.rank
    size = 1 if comm is None else comm.Get_size()
    is_root = rank == 0

    prod = load_production_helpers()
    md_dir = md_openmm_provenance()
    cfg = prod.read_key_value_file(args.input_file)

    # ---------------------------------------------- output dir (shared by all ranks)
    global _FATAL_DIR
    outdir = os.path.abspath(scfg.output_dir)
    if is_root:
        if os.path.isdir(outdir) and os.listdir(outdir) and not (args.overwrite or args.dry_run):
            sys.exit("output_dir %s exists and is not empty; use --overwrite or a new output_dir" % outdir)
        if os.path.isdir(outdir) and args.overwrite:
            shutil.rmtree(outdir)
        for d in ("states", "windows/accepted", "windows/tmp", "windows/rejected", "ranks"):
            os.makedirs(os.path.join(outdir, d), exist_ok=True)
    _FATAL_DIR = outdir

    if comm is not None:
        token = "%s-%d" % (socket.gethostname(), time.time_ns()) if is_root else None
        if is_root:
            with open(os.path.join(outdir, "ranks", ".shared_fs_probe"), "w") as f:
                f.write(token)
        token = comm.bcast(token, root=0)

        seen = None
        try:
            with open(os.path.join(outdir, "ranks", ".shared_fs_probe")) as f:
                seen = f.read()
        except OSError:
            pass
        ok = comm.allgather(seen == token)
        if not all(ok):
            bad = [r for r, good in enumerate(ok) if not good]
            raise RuntimeError("output_dir %s is not visible from rank(s) %s. It must be on a "
                               "filesystem shared by all nodes (not node-local /tmp or scratch)."
                               % (outdir, bad))

        if not is_root:        # keep rank 0's terminal readable
            fh = open(os.path.join(outdir, "ranks", "rank_%03d.log" % rank), "a", buffering=1)
            sys.stdout = sys.stderr = fh

    log = RunLog(outdir) if is_root else rank_logger(rank)
    if is_root:
        log("sumd_openmm: %s" % os.path.abspath(args.input_file))
        log("sumd_openmm %s; MD_openmm helpers bundled from %s" % (__version__, md_dir))
        if scfg.md_openmm_dir is not None:
            log("NOTE: md_openmm_dir is ignored; the MD_openmm helpers are bundled in the package "
                "(re-sync with tools/vendor_md_openmm.py)")
        for w in ([JOINT_WARNING] if scfg.supervision == "joint" else []) + \
                 ([DMSCORE_WARNING] if scfg.walker_score == "dmscore" else []):
            log("WARNING: " + w)

    # ---------------------------------------------- seeds (drawn on rank 0)
    rng = seedseq = None
    if is_root:
        seedseq = np.random.SeedSequence(scfg.random_seed)
        rng = np.random.default_rng(seedseq)
        integrator_seeds = [int(rng.integers(1, 2 ** 31 - 1)) for _ in range(size)]
    else:
        integrator_seeds = None
    if comm is not None:
        integrator_seeds = comm.bcast(integrator_seeds, root=0)
    my_seed = integrator_seeds[rank]

    # ---------------------------------------------- layout
    me, table = (dict(rank=0, host=socket.gethostname(), device=None), None)
    if comm is not None:
        me, table = mpi_layout(comm, scfg)
        if is_root:
            log("MPI: %d ranks, mpi_mode = %s, gpu_devices = %s" % (size, scfg.mpi_mode, scfg.gpu_devices))
            for r in table:
                log("  rank %3d  host %-20s local %d/%d  device %-5s CUDA_VISIBLE_DEVICES=%s"
                    % (r["rank"], r["host"], r["local_rank"], r["local_size"],
                       "auto" if r["device"] is None else r["device"], r["cuda_visible"]))
            for w in check_unique_devices(table):
                log("WARNING: " + w)

    for w in scfg.resolve_walkers(size):
        if is_root:
            log("WARNING: " + w)

    if is_root:
        log("random_seed = %s (entropy %d); integrator seeds %s"
            % (scfg.random_seed, seedseq.entropy, integrator_seeds))

    # ---------------------------------------------- system (every rank)
    topfile = prod.get_str(cfg, "topology_file", required=True)
    try:
        sel = selection.resolve(scfg, topfile)
    except selection.SelectionError as exc:
        sys.exit("Selection error: %s" % exc)
    if is_root:
        log("selections: %s" % json.dumps(sel.report))

    omm = build_simulation(prod, cfg, my_seed, log=log, device_index=me["device"])
    engine = Engine(omm, log)

    if comm is not None and omm.platform_name not in ("CUDA", "OpenCL"):
        log("NOTE: parallel = mpi on the %s platform: fine for testing, but ranks on one node "
            "compete for the same cores" % omm.platform_name)

    align_groups = None
    if sel.align is not None:
        mol_of = {}
        for k, m in enumerate(engine.molecules()):
            for a in m:
                mol_of[int(a)] = k
        ids = np.array([mol_of[int(a)] for a in sel.align])
        align_groups = [np.where(ids == k)[0] for k in dict.fromkeys(ids)]
        if len(align_groups) > 1 and is_root:
            log("alignment selection spans %d molecules; they are imaged together" % len(align_groups))

    evaluator = CVEvaluator(scfg, sel, align_groups)
    sps = steps_per_sample(scfg, omm.dt_ps)

    worker = RankWorker(engine, evaluator, outdir, scfg.samples_per_window, sps, scfg.dcd_stride,
                        scfg.restore_check, rank=rank, host=me["host"], device=me["device"])

    return dict(prod=prod, cfg=cfg, md_dir=md_dir, outdir=outdir, log=log, rng=rng, seedseq=seedseq,
                integrator_seeds=integrator_seeds, table=table, sel=sel, omm=omm, engine=engine,
                evaluator=evaluator, worker=worker, size=size)


def run(args, scfg, comm):
    env = run_setup(args, scfg, comm)
    is_root = comm is None or comm.rank == 0
    log, outdir, omm = env["log"], env["outdir"], env["omm"]

    if not is_root:
        if not args.dry_run:
            worker_loop(comm, env["worker"])
        return

    summary = dict(
        input_file=os.path.abspath(args.input_file), sumd_openmm_version=__version__,
        md_openmm_helpers=env["md_dir"],
        started=datetime.datetime.now().isoformat(timespec="seconds"),
        sumd_config=scfg.as_dict(), md_config=env["cfg"],
        seeds=dict(random_seed=scfg.random_seed, entropy=str(env["seedseq"].entropy),
                   integrator=env["integrator_seeds"]),
        parallel=dict(mode=scfg.parallel, mpi_mode=scfg.mpi_mode if comm else None,
                      ranks=env["size"], walkers=scfg.walkers, layout=env["table"]),
        selections=env["sel"].report,
        resolved_indices=dict(cv1=env["sel"].cv1, cv1_site=env["sel"].cv1_site, align=env["sel"].align,
                              cv2_a=env["sel"].cv2_a, cv2_b=env["sel"].cv2_b),
        engine=dict(dt_ps=omm.dt_ps, temperature_K=omm.temperature_K, barostat=omm.has_barostat,
                    coordinate_file=omm.crdfile, coordinate_kind=omm.start_info["kind"],
                    genvel=omm.genvel, rewrap=omm.rewrap_coordinates, **platform_details(omm)),
        software=software_versions(),
        warnings=([JOINT_WARNING] if scfg.supervision == "joint" else [])
        + ([DMSCORE_WARNING] if scfg.walker_score == "dmscore" else []),
    )

    def write_summary():
        with open(os.path.join(outdir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

    if args.dry_run:
        c1, c2 = env["engine"].current_cvs(env["evaluator"])
        summary["dry_run"] = dict(initial_cv1=c1, initial_cv2=c2)
        write_summary()
        log("dry run: initial CV1 = %.4f A, CV2 = %s, walkers per cycle = %d. Nothing simulated."
            % (c1, "null" if c2 is None else "%.4f A" % c2, scfg.walkers))
        log.close()
        return

    write_summary()
    executor = MPIExecutor(comm, env["worker"]) if comm is not None else LocalExecutor(env["worker"])
    runner = SumdRun(scfg, executor, outdir, env["rng"], log)

    wall0 = time.time()
    try:
        final, stop_reason = runner.run()
    finally:
        executor.shutdown()
        summary["wallclock_s"] = round(time.time() - wall0, 1)
        summary["n_nodes"] = len(runner.nodes)
        write_summary()

    # ---------------------------------------------- final products (rank 0)
    path = runner.path_to(final)
    dcds = [runner.nodes[i]["dcd"] for i in path if runner.nodes[i]["dcd"]]

    with open(os.path.join(outdir, "final_path.txt"), "w") as f:
        f.write("# node_id parent_id cycle cv1_final cv2_final dcd\n")
        for i in path:
            n = runner.nodes[i]
            f.write("%d %s %d %s %s %s\n" % (i, n["parent"], n["cycle"], n["cv1"], n["cv2"], n["dcd"]))

    nframes = 0
    if dcds:
        nframes = dcdtools.concat_dcds([os.path.join(outdir, d) for d in dcds],
                                       os.path.join(outdir, "sumd_traj.dcd"))

    fn = runner.nodes[final]
    final_xml = os.path.join(outdir, "final_state.xml")
    shutil.copyfile(os.path.join(outdir, fn["state_xml"]), final_xml)
    engine = env["engine"]
    engine.restore(engine.load_xml(final_xml, fn["step"], fn["epot"]), check=False)
    rst7 = env["prod"].write_amber_restart(omm.simulation, os.path.join(outdir, "final_state.rst7"),
                                           netcdf=False, enforce_pbc=True)

    summary.update(
        finished=datetime.datetime.now().isoformat(timespec="seconds"),
        stop_reason=stop_reason, final_node=final, final_path=path,
        final_cv1=fn["cv1"], final_cv2=fn["cv2"],
        final_traj=dict(file="sumd_traj.dcd" if dcds else None, windows=len(dcds), frames=nframes),
        final_state_xml=final_xml, final_state_rst7=rst7,
        n_cycles_run=max((n["cycle"] for n in runner.nodes.values()), default=0),
    )
    write_summary()

    log("stop: %s" % stop_reason)
    log("final path: %d windows, %d frames -> sumd_traj.dcd; final node %d (CV1 %s)"
        % (len(dcds), nframes, final, fn["cv1"]))
    log.close()


def main():
    ap = argparse.ArgumentParser(prog="sumd-openmm", description="SuMD on OpenMM via MD_openmm")
    ap.add_argument("input_file", help="MD_openmm-style .inp with the SuMD keys added")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve selections, build the system (and the MPI layout), print initial CVs, stop")
    ap.add_argument("--overwrite", action="store_true", help="allow a non-empty output_dir")
    ap.add_argument("--version", action="version", version="%(prog)s " + __version__)
    args = ap.parse_args()

    try:
        scfg = SumdConfig.from_cfg(read_key_value_file(args.input_file))
    except ConfigError as exc:
        sys.exit("Config error: %s" % exc)

    comm = None
    if scfg.parallel == "mpi":
        try:
            from mpi4py import MPI
        except ImportError:
            sys.exit("parallel = mpi needs mpi4py in this environment "
                     "(conda install -c conda-forge mpi4py, built against the cluster MPI).")
        comm = MPI.COMM_WORLD

    try:
        run(args, scfg, comm)
    except BaseException as exc:
        # One failing rank must take the whole job down, not leave the rest
        # blocked in a collective until the wall-time limit.
        clean_exit = isinstance(exc, SystemExit) and exc.code in (None, 0)
        if comm is not None and comm.Get_size() > 1 and not clean_exit:
            if isinstance(exc, SystemExit):
                msg = exc.code
            elif isinstance(exc, (LayoutError, ConfigError)):
                msg = "%s: %s" % (type(exc).__name__, exc)
            else:
                msg = traceback.format_exc()
            # a layout error is identical on every rank: let rank 0 say it once
            report_fatal(comm.rank, msg, echo=comm.rank == 0 or not isinstance(exc, LayoutError))
            comm.Abort(1)
        raise


if __name__ == "__main__":
    main()
