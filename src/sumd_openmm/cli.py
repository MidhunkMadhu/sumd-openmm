"""Command line driver for supervised OpenMM windows."""

import argparse
import csv
import datetime
import json
import os
import shutil
import socket
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

from . import __version__, cv, dcdtools, selection
from .config import ConfigError, SumdConfig, read_key_value_file
from .engine import Engine
from .executors import LocalExecutor, MPIExecutor, RankWorker, worker_loop
from .metrics import supervised_series
from .omm_setup import build_simulation, load_production_helpers
from .parallel import LayoutError, assign_device
from .seeding import StratifiedPool
from .supervision import SELECTION_BIAS, combined_series, pick_best_walker, sumd_decision


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    raise TypeError(type(value))


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _write_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json)


class RunLog:
    def __init__(self, outdir, names):
        self.progress = open(os.path.join(outdir, "progress.log"), "w", buffering=1)
        self.windows = open(os.path.join(outdir, "windows.jsonl"), "w", buffering=1)
        self.nodes = open(os.path.join(outdir, "nodes.csv"), "w", newline="", buffering=1)
        self.fields = (["node_id", "parent_id", "cycle", "walker"] + names +
                       ["band", "cell", "time_ps", "step", "epot_kJmol",
                        "reason", "dcd", "state_xml"])
        self.writer = csv.DictWriter(self.nodes, fieldnames=self.fields)
        self.writer.writeheader()

    def __call__(self, message):
        line = "[%s] %s" % (datetime.datetime.now().isoformat(timespec="seconds"), message)
        print(line, flush=True)
        self.progress.write(line + "\n")

    def window(self, row):
        self.windows.write(json.dumps(row, default=_json, allow_nan=False) + "\n")

    def node(self, row):
        data = {name: row["metrics"].get(name) for name in row["metrics"]}
        data.update(node_id=row["id"], parent_id=row["parent"], cycle=row["cycle"],
                    walker=row["walker"], band=row.get("band"), cell=row.get("cell"),
                    time_ps=row["time_ps"], step=row["step"], epot_kJmol=row["epot"],
                    reason=row["reason"], dcd=row["dcd"], state_xml=row["state_xml"])
        self.writer.writerow(data)

    def close(self):
        self.progress.close()
        self.windows.close()
        self.nodes.close()


class SumdRun:
    def __init__(self, config, executor, outdir, rng, log, initial):
        self.s, self.x, self.out = config, executor, outdir
        self.rng, self.log = rng, log
        self.names = [m.name for m in config.metrics]
        self.supervised = [m for m in config.metrics if m.role == "supervise"]
        self.stratified = [m for m in config.metrics if m.role == "stratify"]
        self.positions = [m.number - 1 for m in self.supervised]
        self.strata_positions = [m.number - 1 for m in self.stratified]
        self.initial = np.asarray(initial, float)
        self.stage = 0
        self.mu = np.array([m.mu if m.mu is not None else np.nan for m in self.supervised])
        self.sigma = np.array([m.sigma if m.sigma is not None else np.nan for m in self.supervised])
        self.nodes = OrderedDict()
        self.retries = {}
        self.seen_series = {}
        self.pool = (StratifiedPool(config.band_width, [m.bins for m in self.stratified],
                     config.pool_per_cell, config.frontier_bands,
                     config.max_retries_per_parent, rng) if config.seeding == "stratified" else None)

    def seed(self):
        return int(self.rng.integers(1, 2 ** 31 - 1))

    def target(self, spec):
        return spec.target if spec.target_delta is None else self.initial[spec.number - 1] + spec.target_delta

    def quantity(self, raw, spec, parent=None):
        values, target = supervised_series(raw, spec, parent, self.initial[spec.number - 1])
        return values, target

    def inside(self, value, spec):
        series, target = self.quantity([value], spec)
        return bool(series[-1] > target if spec.direction == "increase" else series[-1] < target)

    def progress(self, values, memory=None):
        spec = self.supervised[0]
        raw = values[spec.number - 1]
        if memory and spec.name in memory:
            raw = memory[spec.name]
        x, _ = self.quantity([raw], spec)
        return -float(x[-1]) if spec.direction == "increase" else float(x[-1])

    def xml_rel(self, nid):
        return os.path.join("states", "node_%06d.xml" % nid)

    def parent_ref(self, nid):
        node = self.nodes[nid]
        return dict(id=nid, xml=os.path.join(self.out, node["state_xml"]),
                    step=node["step"], epot=node["epot"])

    def record_node(self, parent, cycle, walker, info, dcd, reason, memory=None):
        nid = len(self.nodes)
        values = np.asarray(info["metrics"], float)
        progress = self.progress(values, memory)
        strata = values[self.strata_positions]
        band = self.pool.band(progress) if self.pool else None
        cell = None
        row = dict(id=nid, parent=parent, cycle=cycle, walker=walker,
                   metrics=dict(zip(self.names, map(float, values))), values=values,
                   memory=memory or {}, progress=progress, band=band, cell=cell,
                   time_ps=_finite(info["time_ps"]), step=info["step"],
                   epot=_finite(info["epot"]), reason=reason,
                   dcd=dcd, state_xml=self.xml_rel(nid))
        if self.pool:
            for evicted in self.pool.add(nid, progress, strata):
                self.log("pool evicted node %d" % evicted)
            row["cell"] = self.pool.cell_of(self.pool.entries[nid])
        self.nodes[nid] = row
        self.log.node(row)
        return nid

    def keep_dcd(self, path, cycle, walker):
        relative = os.path.join("windows", "accepted", "c%05d_w%d.dcd" % (cycle, walker))
        shutil.move(path, os.path.join(self.out, relative))
        return relative

    def drop_dcd(self, path):
        if not path or not os.path.exists(path):
            return
        if self.s.keep_rejected_dcd:
            shutil.move(path, os.path.join(self.out, "windows", "rejected", os.path.basename(path)))
        else:
            os.remove(path)

    def analyse(self, result, parent):
        raw = np.asarray(result["metrics"], float)
        if raw.shape[1] != len(self.s.metrics) or not np.all(np.isfinite(raw)):
            raise RuntimeError("cycle %d walker %d: invalid metric samples" %
                               (result["cycle"], result["w"]))
        if self.s.retry_velocities == "keep":
            signature = tuple(np.round(raw.ravel(), 6))
            seen = self.seen_series.setdefault(parent, set())
            if signature in seen:
                self.log("WARNING: identical retry from node %d; try retry_velocities = reassign" % parent)
            seen.add(signature)
        columns, final_memory = [], {}
        for spec in self.supervised:
            previous = self.nodes[parent]["memory"].get(spec.name)
            values, _ = self.quantity(raw[:, spec.number - 1], spec, previous)
            columns.append(values)
            if spec.type == "dihedral" and spec.direction != "toward":
                final_memory[spec.name] = float(values[-1])
        result["supervised"] = np.column_stack(columns)
        result["memory"] = final_memory
        if self.pool:
            self.pool.observe(raw[:, self.strata_positions])

    def _scored(self, batch):
        if self.s.supervision == "combined":
            reference = batch[0]["supervised"]
            missing_mu = np.isnan(self.mu)
            missing_sd = np.isnan(self.sigma)
            self.mu[missing_mu] = np.mean(reference, axis=0)[missing_mu]
            self.sigma[missing_sd] = np.std(reference, axis=0, ddof=1)[missing_sd]
            if np.any(self.sigma <= 0):
                raise ConfigError("combined score has zero variance; supply metric_N_sigma")
        for result in batch:
            if self.s.supervision == "combined":
                series = combined_series(result["supervised"], self.supervised, self.mu, self.sigma)
                direction = "decrease"
            else:
                idx = (self.s.stages[self.stage] - 1 if self.s.supervision == "multistep" else
                       self.supervised[0].number - 1)
                spec_index = next(i for i, m in enumerate(self.supervised) if m.number - 1 == idx)
                series = result["supervised"][:, spec_index]
                direction = ("decrease" if self.supervised[spec_index].direction == "toward"
                             else self.supervised[spec_index].direction)
            result["score_series"] = series
            result["direction"] = direction
            result["trend"] = cv.window_trend(result["t"], series, self.s.slope_points)
            result["progress_slope"] = (result["trend"].slope if direction == "increase"
                                        else -result["trend"].slope)

    def _decision(self, result):
        inside_all = all(
            (result["supervised"][-1, i] > self.target(spec) if spec.direction == "increase"
             else result["supervised"][-1, i] < (spec.tolerance if spec.direction == "toward"
                                                   else self.target(spec)))
            for i, spec in enumerate(self.supervised))
        if self.s.supervision == "combined":
            verdict, reason = sumd_decision(result["trend"].slope, result["score_series"][-1],
                -float("inf"), result["trend"].se, self.s.require_significant_slope)
            if verdict == "converged":
                verdict = "accept"
            if inside_all and result["progress_slope"] >= 0:
                verdict = "converged"
        else:
            index = (next(i for i, spec in enumerate(self.supervised)
                     if spec.number == self.s.stages[self.stage])
                     if self.s.supervision == "multistep" else 0)
            spec = self.supervised[index]
            target = spec.tolerance if spec.direction == "toward" else self.target(spec)
            verdict, reason = sumd_decision(result["trend"].slope,
                result["supervised"][-1, index], target,
                result["trend"].se, self.s.require_significant_slope,
                "decrease" if spec.direction == "toward" else spec.direction)
        if self.s.walker_score != "slope":
            if self.s.supervision == "multistep":
                current = self.s.stages[self.stage]
                index = next(i for i, spec in enumerate(self.supervised) if spec.number == current)
                reached = (result["supervised"][-1, index] >
                           self.target(self.supervised[index])
                           if self.supervised[index].direction == "increase" else
                           result["supervised"][-1, index] <
                           (self.supervised[index].tolerance
                            if self.supervised[index].direction == "toward"
                            else self.target(self.supervised[index])))
            else:
                reached = inside_all
            verdict = "converged" if reached else "accept"
            reason = "best walker selected by %s" % self.s.walker_score
        return verdict, reason

    def run(self):
        root = self.x.root(os.path.join(self.out, self.xml_rel(0)))
        root_memory = {m.name: float(root["metrics"][m.number - 1])
                       for m in self.supervised if m.type == "dihedral"
                       and m.direction != "toward"}
        self.record_node(None, 0, None, root, None, "initial state", root_memory)
        parent, held, final, stop_reason = 0, None, None, "max_cycles"
        for cycle in range(1, self.s.max_cycles + 1):
            if os.path.exists(os.path.join(self.out, "STOP")):
                stop_reason = "STOP file"
                break
            cell = visits = None
            if self.pool:
                entry, cell, visits = self.pool.choose()
                if entry is None:
                    stop_reason = "pool exhausted"
                    break
                if entry.node_id != parent and held:
                    self.drop_dcd(held["dcd_tmp"])
                    held = None
                parent = entry.node_id
            batch_parent = parent
            seeds = [self.seed() if self.s.retry_velocities == "reassign" else None
                     for _ in range(self.s.walkers)]
            start = time.time()
            batch = self.x.run_batch(cycle, self.parent_ref(parent), seeds)
            batch_wall = time.time() - start
            for result in batch:
                self.analyse(result, parent)
            self._scored(batch)
            if self.s.walkers > 1:
                trend_scores = [type("Score", (), {"slope": -r["progress_slope"]}) for r in batch]
                best, scores = pick_best_walker(self.s.walker_score, trend_scores,
                    [r["supervised"] for r in batch], self.supervised)
            else:
                best, scores = 0, [batch[0]["progress_slope"]]
            chosen = batch[best]
            verdict, reason = self._decision(chosen)
            decision, accepted, action = verdict, chosen if verdict != "reject" else None, None
            if verdict == "reject":
                quality = -chosen["score_series"][-1] if chosen["direction"] == "increase" else chosen["score_series"][-1]
                prior = None if held is None else (-held["score_series"][-1] if held["direction"] == "increase" else held["score_series"][-1])
                if held is None or quality < prior:
                    if held:
                        self.drop_dcd(held["dcd_tmp"])
                    held = chosen
                if self.pool:
                    if self.pool.record_rejection(parent):
                        if self.pool.live_entries():
                            action = "parent retired after retry limit"
                            self.drop_dcd(held["dcd_tmp"])
                            held = None
                        else:
                            accepted, held = held, None
                            decision, action = "accept", "pool exhausted; accepting least-bad attempt"
                else:
                    self.retries[parent] = self.retries.get(parent, 0) + 1
                    if self.retries[parent] > self.s.max_retries_per_parent:
                        previous = self.nodes[parent]["parent"]
                        if self.s.on_retry_exhaustion == "step_back" and previous is not None:
                            parent = previous
                            action = "step back after retry limit"
                            self.drop_dcd(held["dcd_tmp"])
                            held = None
                        else:
                            accepted, held = held, None
                            decision, action = "accept", "retry limit; accepting least-bad attempt"
            child = commit_key = None
            if accepted is not None:
                path = self.keep_dcd(accepted["dcd_tmp"], accepted["cycle"], accepted["w"])
                info = dict(metrics=accepted["metrics_final"], time_ps=accepted["time_ps"],
                            step=accepted["step"], epot=accepted["epot"])
                child = self.record_node(batch_parent, accepted["cycle"], accepted["w"],
                                         info, path, action or reason, accepted["memory"])
                commit_key = accepted["key"]
                if not self.pool:
                    parent = child
                if held:
                    self.drop_dcd(held["dcd_tmp"])
                    held = None
            self.x.resolve(commit_key, child,
                os.path.join(self.out, self.xml_rel(child)) if child is not None else None,
                [held["key"]] if held else [])
            for result in batch:
                if result is not accepted and result is not held:
                    self.drop_dcd(result["dcd_tmp"])
                raw = np.asarray(result["metrics"])
                series = {name: [_finite(x) for x in raw[:, i]]
                          for i, name in enumerate(self.names)}
                unwrapped = {m.name: [_finite(x) for x in result["supervised"][:, i]]
                             for i, m in enumerate(self.supervised)
                             if m.type == "dihedral" and m.direction != "toward"}
                self.log.window(dict(cycle=cycle, walker=result["w"], rank=result["rank"],
                    host=result["host"], device=result["device"], parent_state_id=batch_parent,
                    child_state_id=child if result is accepted else None, accepted=result is accepted,
                    best_in_batch=result is chosen, verdict=verdict if result is chosen else "not best",
                    reason=reason if result is chosen else None, action=action if result is chosen else None,
                    seed_used=result["seed"], start_mode=result["start_mode"],
                    restore_dE_kJmol=_finite(result["restore_dE"]),
                    retries_so_far=self.retries.get(batch_parent, 0),
                    wallclock_s=result["wall"], batch_wallclock_s=batch_wall,
                    t_ps=list(map(_finite, result["t"])), metric_series=series,
                    metric_final={name: _finite(value) for name, value in
                                  zip(self.names, result["metrics_final"])},
                    unwrapped_series=unwrapped, stage=self.stage + 1,
                    slope_b=_finite(result["trend"].slope),
                    se_b=_finite(result["trend"].se), r_squared=_finite(result["trend"].r2),
                    walker_score=self.s.walker_score, score=_finite(scores[result["w"]]),
                    cell_chosen=list(cell) if cell is not None else None,
                    cell_visit_count=visits))
            self.log("cycle %d parent %d: %s; node %s" % (cycle, batch_parent, verdict, child))
            if verdict == "converged" and decision == "converged":
                if self.s.supervision == "multistep" and self.stage < len(self.s.stages) - 1:
                    self.stage += 1
                    self.log("stage changed to metric %d" % self.s.stages[self.stage])
                else:
                    final, stop_reason = child, "converged"
                    break
        if held:
            self.drop_dcd(held["dcd_tmp"])
        if final is None:
            final = min(self.nodes, key=lambda i: self.nodes[i]["progress"])
        return final, stop_reason

    def path_to(self, node_id):
        path = []
        while node_id is not None:
            path.append(node_id)
            node_id = self.nodes[node_id]["parent"]
        return path[::-1]


def run(args, scfg, comm=None):
    rank, size = (comm.rank, comm.Get_size()) if comm is not None else (0, 1)
    root = rank == 0
    outdir = os.path.abspath(scfg.output_dir)
    if root:
        if os.path.isdir(outdir) and os.listdir(outdir) and not (args.overwrite or args.dry_run):
            raise ConfigError("output_dir exists; choose another or use --overwrite")
        if args.overwrite and os.path.isdir(outdir):
            shutil.rmtree(outdir)
        for folder in ("states", "windows/accepted", "windows/tmp", "windows/rejected", "ranks"):
            os.makedirs(os.path.join(outdir, folder), exist_ok=True)
    if comm is not None:
        comm.barrier()
        probe = os.path.join(outdir, "ranks", ".shared_fs_probe")
        token = str(time.time_ns()) if root else None
        if root:
            with open(probe, "w") as fh:
                fh.write(token)
        token = comm.bcast(token, root=0)
        try:
            with open(probe) as fh:
                visible = fh.read() == token
        except OSError:
            visible = False
        if not all(comm.allgather(visible)):
            raise RuntimeError("output_dir must be shared across MPI ranks")
        if not root:
            handle = open(os.path.join(outdir, "ranks", "rank_%03d.log" % rank), "w", buffering=1)
            sys.stdout = sys.stderr = handle
    log = RunLog(outdir, [m.name for m in scfg.metrics]) if root else print
    rng = np.random.default_rng(scfg.random_seed) if root else None
    seeds = [int(rng.integers(1, 2**31-1)) for _ in range(size)] if root else None
    if comm is not None:
        seeds = comm.bcast(seeds, root=0)
    my_device, layout = None, None
    if comm is not None:
        from mpi4py import MPI
        local = comm.Split_type(MPI.COMM_TYPE_SHARED)
        my_device = assign_device(scfg.mpi_mode, scfg.gpu_devices, local.rank,
                                  local.size, os.environ, MPI.Get_processor_name())
        local.Free()
        layout = comm.allgather(dict(rank=rank, host=socket.gethostname(), device=my_device))
    scfg.resolve_walkers(size)
    prod = load_production_helpers()
    cfg = prod.read_key_value_file(args.input_file)
    topfile = prod.get_str(cfg, "topology_file", required=True)
    base = os.path.dirname(os.path.abspath(args.input_file))
    evaluator, report = selection.resolve(scfg, topfile, base)
    omm = build_simulation(prod, cfg, seeds[rank], log=log, device_index=my_device)
    engine = Engine(omm, log)
    initial = engine.current_cvs(evaluator)
    if root:
        for row, value in zip(report, initial):
            row["initial_value"] = float(value)
            log("%s: %s" % (row["name"], json.dumps(row)))
            if row["role"] == "supervise":
                spec = scfg.metrics[row["number"] - 1]
                values, target = supervised_series([value], spec, None, value)
                if (values[-1] > target if spec.direction == "increase" else values[-1] < target):
                    log("WARNING: %s already satisfies its target" % spec.name)
        if scfg.supervision != "single" or scfg.walker_score == "dmscore":
            log("WARNING: " + SELECTION_BIAS)
    sps = scfg.cv_sample_ps / omm.dt_ps
    if abs(sps - round(sps)) > 1e-6:
        raise ConfigError("cv_sample_ps must be a multiple of dt")
    worker = RankWorker(engine, evaluator, outdir, scfg.samples_per_window, int(round(sps)),
                        scfg.dcd_stride, scfg.restore_check, rank=rank,
                        host=socket.gethostname(), device=my_device)
    if not root:
        if not args.dry_run:
            worker_loop(comm, worker)
        return
    summary = dict(input_file=os.path.abspath(args.input_file), sumd_openmm_version=__version__,
                   sumd_config=scfg.as_dict(), md_config=cfg, metrics=report,
                   parallel=dict(mode=scfg.parallel, ranks=size, walkers=scfg.walkers, layout=layout),
                   engine=dict(dt_ps=omm.dt_ps, temperature_K=omm.temperature_K,
                               platform=omm.platform_name, coordinate_file=omm.crdfile),
                   seeds=dict(random_seed=scfg.random_seed, integrator=seeds),
                   warnings=[SELECTION_BIAS] if scfg.supervision != "single" or scfg.walker_score == "dmscore" else [])
    summary_path = os.path.join(outdir, "run_summary.json")
    if args.dry_run:
        summary["dry_run"] = dict(initial_metrics=dict(zip(evaluator.names, map(float, initial))))
        _write_json(summary_path, summary)
        log("dry run: no dynamics simulated")
        log.close()
        return
    _write_json(summary_path, summary)
    executor = MPIExecutor(comm, worker) if comm is not None else LocalExecutor(worker)
    runner = SumdRun(scfg, executor, outdir, rng, log, initial)
    try:
        final, stop_reason = runner.run()
    finally:
        executor.shutdown()
    path = runner.path_to(final)
    dcds = [runner.nodes[i]["dcd"] for i in path if runner.nodes[i]["dcd"]]
    with open(os.path.join(outdir, "final_path.txt"), "w") as fh:
        fh.write("# node_id parent_id cycle metrics dcd\n")
        for i in path:
            node = runner.nodes[i]
            fh.write("%d %s %d %s %s\n" % (i, node["parent"], node["cycle"],
                     json.dumps(node["metrics"]), node["dcd"]))
    frames = (dcdtools.concat_dcds([os.path.join(outdir, p) for p in dcds],
              os.path.join(outdir, "sumd_traj.dcd")) if dcds else 0)
    last = runner.nodes[final]
    final_xml = os.path.join(outdir, "final_state.xml")
    shutil.copyfile(os.path.join(outdir, last["state_xml"]), final_xml)
    engine.restore(engine.load_xml(final_xml, last["step"], last["epot"]), check=False)
    rst7 = prod.write_amber_restart(omm.simulation, os.path.join(outdir, "final_state.rst7"),
                                     netcdf=False, enforce_pbc=True)
    summary.update(stop_reason=stop_reason, final_node=final, final_path=path,
                   final_metrics=last["metrics"], n_nodes=len(runner.nodes),
                   final_traj=dict(file="sumd_traj.dcd" if dcds else None,
                                   windows=len(dcds), frames=frames),
                   final_state_xml=final_xml, final_state_rst7=rst7)
    _write_json(summary_path, summary)
    log("finished: %s; final node %d; %d frames" % (stop_reason, final, frames))
    log.close()


def main():
    ap = argparse.ArgumentParser(prog="sumd-openmm", description="Supervised molecular dynamics with OpenMM")
    ap.add_argument("input_file", help="key = value input file")
    ap.add_argument("--dry-run", action="store_true", help="resolve metrics and report initial values")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output directory")
    ap.add_argument("--version", action="version", version="%(prog)s " + __version__)
    args = ap.parse_args()
    try:
        config = SumdConfig.from_cfg(read_key_value_file(args.input_file))
        comm = None
        if config.parallel == "mpi":
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        run(args, config, comm)
    except BaseException as exc:
        if "comm" in locals() and comm is not None and comm.size > 1:
            with open(os.path.join(os.path.abspath(config.output_dir),
                     "FATAL_rank_%03d.txt" % comm.rank), "w") as fh:
                fh.write(traceback.format_exc())
            comm.Abort(1)
        raise


if __name__ == "__main__":
    main()
