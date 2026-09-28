"""Command line driver for supervised OpenMM windows."""

import argparse
import csv
import datetime
import json
import os
import shutil
import signal
import socket
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

from . import __version__, cv, dcdtools, selection
from .charmmgui import CharmmGuiError
from .config import _MD_KEYS, ConfigError, SumdConfig, read_key_value_file
from .engine import Engine
from .executors import LocalExecutor, MPIExecutor, RankWorker, worker_loop
from .metrics import supervised_series
from .omm_setup import PlatformError, build_simulation, load_production_helpers
from .parallel import LayoutError, assign_device
from .seeding import PoolEntry, StratifiedPool
from .supervision import SELECTION_BIAS, combined_series, pick_best_walker, sumd_decision


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    raise TypeError(type(value))


def _round(value, digits=4):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _write_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json)


def _table(outdir, name, fields, append=False):
    handle = open(os.path.join(outdir, name), "a" if append else "w", newline="", buffering=1)
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not append:
        writer.writeheader()
    return handle, writer


class RunLog:
    """progress.log, windows.jsonl and the three CSV tables of a run."""

    def __init__(self, outdir, names, samples=True, append=False):
        """append: continue the files of an earlier run (--continue)."""
        self.names = names
        mode = "a" if append else "w"
        self.progress = open(os.path.join(outdir, "progress.log"), mode, buffering=1)
        self.windows = open(os.path.join(outdir, "windows.jsonl"), mode, buffering=1)
        self.steps, self.step_rows = _table(outdir, "accepted_steps.csv",
            ["accepted_step", "parent_accepted_step", "cycle", "walker"] + names +
            ["time_ps", "md_step", "epot_kJmol", "reason", "trajectory", "restart_file", "band", "cell"],
            append)
        self.window_table, self.window_rows = _table(outdir, "windows.csv",
            ["cycle", "walker", "start_accepted_step", "outcome", "new_accepted_step", "retries_used"] + names +
            ["slope", "score", "wallclock_s"], append)
        self.samples = self.sample_rows = None
        if samples:
            self.samples, self.sample_rows = _table(outdir, "cv_samples.csv",
                                                    ["cycle", "walker", "t_ps"] + names, append)

    def __call__(self, message):
        line = "[%s] %s" % (datetime.datetime.now().isoformat(timespec="seconds"), message)
        print(line, flush=True)
        self.progress.write(line + "\n")

    def row(self, text, stamp=True):
        """A table row, stamped with the time only (the date is in the lines above)."""
        prefix = datetime.datetime.now().strftime("%H:%M:%S") if stamp else " " * 8
        line = "%s  %s" % (prefix, text)
        print(line, flush=True)
        self.progress.write(line + "\n")

    def window(self, row):
        self.windows.write(json.dumps(row, default=_json, allow_nan=False) + "\n")
        data = {name: _round(row["metric_final"][name]) for name in self.names}
        data.update(cycle=row["cycle"], walker=row["walker"], start_accepted_step=row["start_accepted_step"],
                    outcome=row["outcome"], new_accepted_step=row["new_accepted_step"],
                    retries_used=row["retries_so_far"], slope=_round(row["slope_b"], 5),
                    score=_round(row["score"], 5), wallclock_s=_round(row["wallclock_s"], 1))
        self.window_rows.writerow(data)
        if self.sample_rows is not None:
            for k, t in enumerate(row["t_ps"]):
                sample = {name: _round(row["metric_series"][name][k]) for name in self.names}
                sample.update(cycle=row["cycle"], walker=row["walker"], t_ps=_round(t, 3))
                self.sample_rows.writerow(sample)

    def accepted_step(self, row):
        data = {name: _round(row["metrics"].get(name)) for name in row["metrics"]}
        data.update(accepted_step=row["id"], parent_accepted_step=row["parent"], cycle=row["cycle"],
                    walker=row["walker"], band=row.get("band"), cell=row.get("cell"),
                    time_ps=row["time_ps"], md_step=row["step"], epot_kJmol=_round(row["epot"], 3),
                    reason=row["reason"], trajectory=row["dcd"], restart_file=row["state_xml"])
        self.step_rows.writerow(data)

    def close(self):
        for handle in (self.progress, self.windows, self.steps, self.window_table, self.samples):
            if handle is not None:
                handle.close()


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
        self.checked = None
        self.parent, self.last_cycle = 0, 0
        self.saved_config = {}
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
        return os.path.join("accepted_steps", "accepted_step_%06d.xml" % nid)

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
                self.log("pool evicted AcceptedStep %d" % evicted)
            row["cell"] = self.pool.cell_of(self.pool.entries[nid])
        self.nodes[nid] = row
        self.log.accepted_step(row)
        return nid

    def keep_dcd(self, path, cycle, walker):
        relative = os.path.join("windows", "accepted", "c%05d_w%d.dcd" % (cycle, walker))
        shutil.move(path, os.path.join(self.out, relative))
        return relative

    def write_check(self, nid):
        """
        check/: the path from the start to AcceptedStep `nid`, every
        check_stride-th frame, and its State, replaced every check_every
        AcceptedSteps and at the end, so a running job can be looked at. A
        failure here does not stop the run.
        """
        self.checked = nid
        folder = os.path.join(self.out, "check")
        node = self.nodes[nid]
        dcds = [os.path.join(self.out, self.nodes[i]["dcd"]) for i in self.path_to(nid)
                if self.nodes[i]["dcd"]]
        traj = os.path.join(folder, "sumd_traj_stride%d.dcd" % self.s.check_stride)
        state = os.path.join(folder, "latest_accepted_step.xml")
        try:
            os.makedirs(folder, exist_ok=True)
            frames = dcdtools.concat_dcds(dcds, traj + ".tmp", self.s.check_stride)
            shutil.copyfile(os.path.join(self.out, node["state_xml"]), state + ".tmp")
            os.replace(traj + ".tmp", traj)
            os.replace(state + ".tmp", state)
            _write_json(os.path.join(folder, "latest.json"),
                        dict(accepted_step=nid, cycle=node["cycle"], path=self.path_to(nid),
                             windows=len(dcds), frames=frames, check_stride=self.s.check_stride,
                             check_every=self.s.check_every,
                             metrics=node["metrics"], trajectory=os.path.basename(traj),
                             state_xml=os.path.basename(state),
                             written=datetime.datetime.now().isoformat(timespec="seconds")))
        except (OSError, ValueError) as exc:
            self.log("WARNING: check/ not updated for AcceptedStep %d: %s" % (nid, exc))

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
        if not self.s.reassign_velocities:
            signature = tuple(np.round(raw.ravel(), 6))
            seen = self.seen_series.setdefault(parent, set())
            if signature in seen:
                self.log("WARNING: identical window from AcceptedStep %d; try retry_velocities = reassign" % parent)
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
        if self.s.extends_best_walker or self.s.walker_score != "slope":
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
        self.table_layout()
        for line in self.table_header():
            self.log(line)
        if not self.nodes:
            root = self.x.root(os.path.join(self.out, self.xml_rel(0)))
            root_memory = {m.name: float(root["metrics"][m.number - 1])
                           for m in self.supervised if m.type == "dihedral"
                           and m.direction != "toward"}
            self.record_node(None, 0, None, root, None, "starting structure", root_memory)
        parent, held, final, stop_reason = self.parent, None, None, "max_cycles"
        self.expected_start = parent
        for cycle in range(self.last_cycle + 1, self.s.max_cycles + 1):
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
            seeds = [self.seed() if self.s.reassign_velocities else None
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
                            action = "back to AcceptedStep %d" % previous
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
            if child is not None and self.s.check_stride and child % self.s.check_every == 0:
                self.write_check(child)
            for result in batch:
                if result is not accepted and result is not held:
                    self.drop_dcd(result["dcd_tmp"])
                raw = np.asarray(result["metrics"])
                series = {name: [_finite(x) for x in raw[:, i]]
                          for i, name in enumerate(self.names)}
                unwrapped = {m.name: [_finite(x) for x in result["supervised"][:, i]]
                             for i, m in enumerate(self.supervised)
                             if m.type == "dihedral" and m.direction != "toward"}
                outcome = ("converged" if result is accepted and decision == "converged" else
                           "kept" if result is accepted else
                           "rejected" if result is chosen and verdict == "reject" else "not kept")
                self.log.window(dict(cycle=cycle, walker=result["w"], rank=result["rank"], outcome=outcome,
                    host=result["host"], device=result["device"], start_accepted_step=batch_parent,
                    new_accepted_step=child if result is accepted else None, accepted=result is accepted,
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
            getattr(self.log, "row", self.log)(
                self.cycle_report(cycle, batch_parent, batch, chosen, accepted, verdict, action, child))
            done = False
            if verdict == "converged" and decision == "converged":
                if self.s.supervision == "multistep" and self.stage < len(self.s.stages) - 1:
                    self.stage += 1
                    self.log("stage changed to metric %d" % self.s.stages[self.stage])
                else:
                    final, stop_reason, done = child, "converged", True
            self.parent, self.last_cycle = parent, cycle
            self.save_state()
            if done:
                break
        if held:
            self.drop_dcd(held["dcd_tmp"])
        if final is None:
            # A chain ends at its latest state, as its trajectory does; a pool
            # has no single chain, so it ends at its most advanced state.
            final = min(self.nodes, key=lambda i: self.nodes[i]["progress"]) if self.pool else parent
        if self.s.check_stride and final != self.checked and self.nodes[final]["dcd"]:
            self.write_check(final)
        self.save_state(stop_reason, final)
        return final, stop_reason

    def save_state(self, stop_reason=None, final=None):
        """resume.json: all a later --continue needs, as of the end of the last cycle."""
        def plain(array):
            return [float(x) if np.isfinite(x) else None for x in array]
        nodes = [dict(row, values=plain(row["values"]),
                      cell=list(row["cell"]) if row["cell"] is not None else None)
                 for row in self.nodes.values()]
        pool = None
        if self.pool:
            pool = dict(entries=[dict(vars(e), strata=list(e.strata)) for e in self.pool.entries.values()],
                        samples=self.pool.samples, order=self.pool._order)
        state = dict(format=1, sumd_openmm_version=__version__, last_cycle=self.last_cycle,
                     parent=self.parent, stop_reason=stop_reason, final=final, stage=self.stage,
                     initial=plain(self.initial), mu=plain(self.mu), sigma=plain(self.sigma),
                     retries={str(k): v for k, v in self.retries.items()},
                     rng=self.rng.bit_generator.state, checked=self.checked,
                     nodes=nodes, pool=pool, **self.saved_config)
        path = os.path.join(self.out, "resume.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(state, fh, default=_json)
        os.replace(path + ".tmp", path)

    def load_state(self, state):
        """
        Continue from resume.json. The random generator is restored by the
        caller. A rejected window held as a fallback is not carried over.
        """
        def array(values):
            return np.array([np.nan if x is None else x for x in values], float)
        self.last_cycle, self.parent, self.stage = state["last_cycle"], state["parent"], state["stage"]
        self.initial, self.mu, self.sigma = array(state["initial"]), array(state["mu"]), array(state["sigma"])
        self.retries = {int(k): v for k, v in state["retries"].items()}
        self.checked = state["checked"]
        self.nodes = OrderedDict()
        for row in state["nodes"]:
            row = dict(row, values=array(row["values"]),
                       cell=tuple(row["cell"]) if row["cell"] is not None else None)
            self.nodes[row["id"]] = row
        if self.pool:
            for data in state["pool"]["entries"]:
                entry = PoolEntry(**dict(data, strata=tuple(data["strata"])))
                entry.exhausted = entry.retries > self.pool.max_retries   # max_retries may have changed
                self.pool.entries[entry.node_id] = entry
            self.pool.samples = state["pool"]["samples"]
            self.pool._order = state["pool"]["order"]

    def last_saved(self):
        """
        Where a run that stops early ends: its newest AcceptedStep whose State
        is on disk (a chain), or its most advanced one (a pool). None if none.
        """
        saved = [i for i, n in self.nodes.items() if os.path.exists(os.path.join(self.out, n["state_xml"]))]
        if not saved:
            return None
        return min(saved, key=lambda i: self.nodes[i]["progress"]) if self.pool else saved[-1]

    def best_node(self):
        return min(self.nodes, key=lambda i: self.nodes[i]["progress"])

    def _unit(self, spec):
        return "" if spec.type == "contacts" else " deg" if spec.type in ("angle", "dihedral") else " A"

    def table_layout(self):
        """Fields of the per-cycle line; only those that carry information."""
        self.show_walker = self.s.walkers > 1
        self.show_result = not self.s.extends_best_walker
        self.show_step = self.show_result          # otherwise every cycle makes AcceptedStep = cycle
        self.cycle_width = len(str(self.s.max_cycles))
        self.step_width = len(str(self.s.max_cycles))
        self.window_width = len(str(self.s.walkers - 1))
        self.expected_start = 0

    def table_header(self):
        """One line per metric: which CVn of the per-cycle lines it is."""
        units = {"contacts": "", "angle": " (deg)", "dihedral": " (deg)"}
        return ["CV%d = %s%s" % (i, m.name, units.get(m.type, " (A)"))
                for i, m in enumerate(self.s.metrics, 1)]

    def cycle_report(self, cycle, parent, batch, chosen, accepted, verdict, action, child):
        """One table row: cycle, AcceptedStep made, walker, result, every metric and its change."""
        shown = accepted if accepted is not None else chosen
        notes = []
        if parent != self.expected_start:
            notes.append("from AcceptedStep %d" % parent)
        used = self.retries.get(parent, 0)
        if accepted is None:
            result = "rejected"
            if not self.pool:
                result = ("retry limit" if used > self.s.max_retries_per_parent else
                          "rejected %d/%d" % (used, self.s.max_retries_per_parent))
            if action:
                notes.append(action)
        elif accepted is not chosen:
            result = "best retry"
            notes.append("%s: kept the best rejected window (cycle %d)"
                         % (action.split(";")[0], accepted["cycle"]))
        else:
            result = "converged" if verdict == "converged" else "accepted"
        if verdict == "converged" and not self.show_result:
            notes.append("converged")
        self.expected_start = child if child is not None else parent
        before = self.nodes[parent]["metrics"]
        cols = ["Cycle=%-*d" % (self.cycle_width, cycle)]
        if self.show_step:
            cols.append("AcceptedStep=%-*s" % (self.step_width, child if child is not None else "-"))
        if self.show_walker:
            cols.append("Window=%-*d" % (self.window_width, shown["w"]))
        if self.show_result:
            cols.append("Result=%-12s" % result)
        for i, (m, v) in enumerate(zip(self.s.metrics, shown["metrics_final"]), 1):
            value = "%d" % v if m.type == "contacts" else "%.2f" % v
            cols.append("CV%d=%s%s (%+.2f)" % (i, value, self._unit(m), v - before[m.name]))
        return "  ".join(cols) + ("   " + "; ".join(notes) if notes else "")

    def path_to(self, node_id):
        path = []
        while node_id is not None:
            path.append(node_id)
            node_id = self.nodes[node_id]["parent"]
        return path[::-1]


def _metric_summary(row, spec):
    """One readable line per metric; run_summary.json keeps the full record."""
    def residues(names):
        return names[0] if len(names) == 1 else "%s..%s (%d residues)" % (names[0], names[-1], len(names))
    goal = ""
    if spec.role == "supervise" and spec.direction == "toward":
        goal = ", toward %g within %g" % (spec.target, spec.tolerance)
    elif spec.role == "supervise":
        goal = ", %s to %s" % (spec.direction, "%g" % spec.target if spec.target is not None
                               else "start%+g" % spec.target_delta)
    parts = ["%s = %s -> %d atom%s in %s" % (key, sel["expression"], sel["count"],
                                             "" if sel["count"] == 1 else "s", residues(sel["residues"]))
             for key, sel in row["selections"].items()]
    if "reference" in row:
        parts.append("reference %s" % row["reference"]["file"])
    return "metric %s (%s, %s%s): %s; initial %.2f" % (
        spec.name, spec.type, spec.role, goal, "; ".join(parts), row["initial_value"])


def _only_dry_run(outdir):
    """True when output_dir holds nothing but the output of --dry-run."""
    try:
        with open(os.path.join(outdir, "run_summary.json")) as fh:
            summary = json.load(fh)
    except (OSError, ValueError):
        return False
    if "dry_run" not in summary or "stop_reason" in summary:
        return False
    return not any(files for _, _, files in os.walk(os.path.join(outdir, "accepted_steps"))) and \
        not any(files for _, _, files in os.walk(os.path.join(outdir, "windows")))


def _rank_from_launcher():
    for var in ("SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    return 0


def _mpi_world():
    try:
        from mpi4py import MPI
    except ImportError:
        raise ConfigError("parallel = mpi needs mpi4py, which this Python (%s) does not have. "
                          "Install it against the cluster's MPI (docs/PARALLEL.md), or set "
                          "parallel = serial." % sys.executable) from None
    return MPI.COMM_WORLD


def equilibrate(args, scfg, cfg, comm, log, device, root):
    """
    Run (rank 0) or skip the CHARMM-GUI equilibration; every rank returns the
    production settings that start from its final State.
    """
    from . import charmmgui
    from .equilibration import Equilibration, production_settings
    from .omm_setup import precision

    protocol = charmmgui.read(scfg.charmm_gui_dir, scfg.charmm_gui_format)
    folder = os.path.abspath(scfg.equilibration_dir)
    final = os.path.join(folder, "equilibrated.xml")
    info = dict(format=protocol.format, source=protocol.folder, dir=folder,
                stages=[stage.describe() for stage in protocol.stages])
    previous = os.path.join(folder, "protocol.json")
    if os.path.exists(previous) and not args.test:
        with open(previous) as fh:
            done = json.load(fh)
        if (done.get("format"), done.get("folder")) != (protocol.format, protocol.folder):
            raise ConfigError("%s holds an equilibration of %s (%s); use another equilibration_dir"
                              % (folder, done.get("folder"), done.get("format")))
    if root:
        if args.test and not args.dry_run:
            shutil.rmtree(folder, ignore_errors=True)
        if args.dry_run:
            log("stage: equilibration (dry run) - CHARMM-GUI %s protocol from %s"
                % (protocol.format, protocol.folder))
            for stage in protocol.stages:
                log("  " + stage.describe())
        elif os.path.exists(final) and not args.test:
            log("stage: equilibration already finished; using %s" % final)
        else:
            log("stage: equilibration - CHARMM-GUI %s protocol, %d stages, in %s"
                % (protocol.format, len(protocol.stages), folder))
            job = Equilibration(protocol, folder, cfg.get("platform", "auto"), precision(cfg),
                                device, scfg.random_seed, log, test=args.test)
            try:
                job.run()
            finally:
                job.close()
    if comm is not None:
        comm.barrier()
    if args.dry_run:
        # the initial structure stands in for the State equilibration will produce
        cfg = production_settings(protocol, protocol.coordinates, cfg)
        cfg["genvel"] = "yes"
    else:
        cfg = production_settings(protocol, final, cfg)
    info["equilibrated_xml"] = cfg["coordinate_file"]
    return cfg, info


# Settings --continue may change: they only affect cycles still to run.
CONTINUE_MAY_CHANGE = {"max_cycles", "max_retries_per_parent", "on_retry_exhaustion",
                       "require_significant_slope", "check_stride", "check_every",
                       "keep_rejected_dcd", "restore_check", "random_seed", "parallel",
                       "mpi_mode", "gpu_devices", "walkers", "output_dir"}
TARGET_KEYS = ("target", "target_delta", "tolerance")
MD_MAY_CHANGE = {"platform", "cuda_precision", "precision"}


def load_resume(outdir):
    path = os.path.join(outdir, "resume.json")
    if not os.path.exists(path):
        raise ConfigError("--continue: %s not found; runs made before sumd-openmm saved it "
                          "cannot be continued" % path)
    with open(path) as fh:
        state = json.load(fh)
    if state.get("format") != 1:
        raise ConfigError("--continue: %s has an unknown format" % path)
    return state


def check_continue(state, scfg):
    """
    Refuse changes that alter what the saved AcceptedSteps mean. Returns
    the supervised metrics whose target changed.
    """
    old, new = state["sumd_config"], scfg.as_dict()
    changed = sorted(k for k in new if k not in CONTINUE_MAY_CHANGE | {"metrics"}
                     and new[k] != old.get(k))
    strip = lambda m: {k: v for k, v in m.items() if k not in TARGET_KEYS}
    if len(old["metrics"]) != len(new["metrics"]):
        changed.append("metric definitions")
    else:
        changed += ["metric_%d" % m["number"] for m, o in zip(new["metrics"], old["metrics"])
                    if strip(m) != strip(o)]
    if changed:
        raise ConfigError("--continue: these settings differ from the run being continued: %s. "
                          "Only %s and metric targets/tolerances may change; start a new "
                          "run for the rest" % (", ".join(changed), ", ".join(sorted(CONTINUE_MAY_CHANGE))))
    retargeted = [m["name"] for m, o in zip(new["metrics"], old["metrics"])
                  if m["role"] == "supervise" and any(m[k] != o[k] for k in TARGET_KEYS)]
    if state["stop_reason"] == "converged" and not retargeted:
        raise ConfigError("--continue: the run converged at AcceptedStep %s; change a supervised "
                          "metric's target to continue it" % state["final"])
    if scfg.max_cycles <= state["last_cycle"]:
        raise ConfigError("--continue: %d cycles are done and max_cycles = %d; raise max_cycles "
                          "or use --extend N" % (state["last_cycle"], scfg.max_cycles))
    return retargeted


def check_continue_md(state, cfg):
    old = state["md_config"]
    changed = sorted(k for k in set(old) | set(cfg)
                     if k in _MD_KEYS and k not in MD_MAY_CHANGE and old.get(k) != cfg.get(k))
    if changed:
        raise ConfigError("--continue: these MD settings differ from the run being continued: %s"
                          % ", ".join(changed))


def _keep_lines(path, keep):
    """Rewrite a table, keeping its first line and the lines keep(line) accepts."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        lines = fh.readlines()
    head = lines[:1] if path.endswith(".csv") else []
    body = lines[len(head):]
    with open(path + ".tmp", "w") as fh:
        fh.writelines(head + [line for line in body if line.strip() and keep(line)])
    os.replace(path + ".tmp", path)


def prepare_continue(outdir, state):
    """
    Drop what a cycle interrupted after the last checkpoint left behind:
    table rows, trajectories and States past it, and the STOP file.
    """
    last, n_nodes = state["last_cycle"], len(state["nodes"])
    first = lambda line: int(line.split(",", 1)[0])
    _keep_lines(os.path.join(outdir, "accepted_steps.csv"), lambda line: first(line) < n_nodes)
    for name in ("windows.csv", "cv_samples.csv"):
        _keep_lines(os.path.join(outdir, name), lambda line: first(line) <= last)
    _keep_lines(os.path.join(outdir, "windows.jsonl"), lambda line: json.loads(line)["cycle"] <= last)
    for folder in ("windows/accepted", "windows/rejected", "windows/tmp"):
        folder = os.path.join(outdir, folder)
        for name in os.listdir(folder):
            if folder.endswith("tmp") or int(name[1:6]) > last:
                os.remove(os.path.join(folder, name))
    for name in os.listdir(os.path.join(outdir, "accepted_steps")):
        if int(name.split("_")[-1].split(".")[0]) >= n_nodes:
            os.remove(os.path.join(outdir, "accepted_steps", name))
    stop = os.path.join(outdir, "STOP")
    if os.path.exists(stop):
        os.remove(stop)


def run(args, scfg, comm=None):
    rank, size = (comm.rank, comm.Get_size()) if comm is not None else (0, 1)
    root = rank == 0
    outdir = os.path.abspath(scfg.output_dir)
    resume = retargeted = None
    if root and args.resume:
        resume = load_resume(outdir)
        if args.extend:
            scfg.max_cycles = resume["last_cycle"] + args.extend
        retargeted = check_continue(resume, scfg)
        prepare_continue(outdir, resume)
    elif root:
        if os.path.isdir(outdir) and os.listdir(outdir) and not (args.overwrite or args.dry_run):
            if not _only_dry_run(outdir):
                raise ConfigError("output_dir exists; choose another or use --overwrite")
            shutil.rmtree(outdir)
        if args.overwrite and os.path.isdir(outdir):
            shutil.rmtree(outdir)
        for folder in ("accepted_steps", "windows/accepted", "windows/tmp", "windows/rejected", "ranks"):
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
    log = (RunLog(outdir, [m.name for m in scfg.metrics], scfg.write_cv_samples, append=bool(resume))
           if root else print)
    if resume:
        log("continuing after cycle %d (AcceptedStep %d, %s), up to cycle %d"
            % (resume["last_cycle"], resume["parent"], resume["stop_reason"] or "interrupted",
               scfg.max_cycles))
        if retargeted:
            log("new target for %s" % ", ".join(retargeted))
    rng = np.random.default_rng(scfg.random_seed) if root else None
    if resume:
        # integrator seeds too come from the saved stream, so no noise is repeated
        rng.bit_generator.state = resume["rng"]
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
    if resume and scfg.walkers != resume["sumd_config"]["walkers"]:
        raise ConfigError("--continue: the run has %d walker(s), this launch %d"
                          % (resume["sumd_config"]["walkers"], scfg.walkers))
    prod = load_production_helpers()
    cfg = prod.read_key_value_file(args.input_file)
    equilibrated = None
    if scfg.equilibration == "charmm-gui":
        cfg, equilibrated = equilibrate(args, scfg, cfg, comm, log, my_device, root)
    if resume:
        check_continue_md(resume, cfg)
    topfile = prod.get_str(cfg, "topology_file", required=True)
    base = os.path.dirname(os.path.abspath(args.input_file))
    evaluator, report = selection.resolve(scfg, topfile, base)
    omm = build_simulation(prod, cfg, seeds[rank], log=log, device_index=my_device)
    if root and equilibrated and equilibrated["format"] == "AMBER" and not args.dry_run:
        equilibrated["rst7"] = prod.write_amber_restart(
            omm.simulation, os.path.join(equilibrated["dir"], "equilibrated.rst7"),
            netcdf=False, enforce_pbc=True)
    engine = Engine(omm, log)
    initial = engine.current_cvs(evaluator)
    if resume:
        # targets given as a change are measured from the original start
        initial = np.array([np.nan if x is None else x for x in resume["initial"]], float)
    if root:
        for row, value in zip(report, initial):
            row["initial_value"] = float(value)
            log(_metric_summary(row, scfg.metrics[row["number"] - 1]))
            if row["role"] == "supervise" and not resume:
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
                   equilibration=equilibrated,
                   engine=dict(dt_ps=omm.dt_ps, temperature_K=omm.temperature_K,
                               platform=omm.platform_name, coordinate_file=omm.crdfile),
                   seeds=dict(random_seed=scfg.random_seed, integrator=seeds),
                   warnings=[SELECTION_BIAS] if scfg.supervision != "single" or scfg.walker_score == "dmscore" else [])
    summary_path = os.path.join(outdir, "run_summary.json")
    if resume:
        with open(summary_path) as fh:
            before = json.load(fh)
        summary["continuations"] = before.get("continuations", []) + [dict(
            at=datetime.datetime.now().isoformat(timespec="seconds"), after_cycle=resume["last_cycle"],
            previous_stop_reason=resume["stop_reason"], max_cycles=scfg.max_cycles,
            sumd_openmm_version=__version__, integrator_seeds=seeds)]
    if args.dry_run:
        summary["dry_run"] = dict(initial_metrics=dict(zip(evaluator.names, map(float, initial))))
        _write_json(summary_path, summary)
        log("dry run: no dynamics simulated")
        log.close()
        return
    _write_json(summary_path, summary)
    executor = MPIExecutor(comm, worker) if comm is not None else LocalExecutor(worker)
    runner = SumdRun(scfg, executor, outdir, rng, log, initial)
    runner.saved_config = dict(sumd_config=scfg.as_dict(), md_config=cfg)
    if resume:
        runner.load_state(resume)
    log("stage: SuMD production, %d walker(s), %s %d cycles of %g ps"
        % (scfg.walkers, "cycles %d to" % (runner.last_cycle + 1) if resume else "up to",
           scfg.max_cycles, scfg.window_ps))
    previous = _catch_sigterm(root)
    try:
        final, stop_reason = runner.run()
    except BaseException as exc:
        # write what was accepted so far, then let the error end the run as before
        final = runner.last_saved()
        if final is not None:
            try:
                _finish(outdir, runner, final, _early_stop_reason(exc), engine, omm, prod,
                        summary, summary_path, log, partial=True)
            except Exception as salvage:
                log("WARNING: could not write the partial result: %s" % salvage)
        log.close()
        raise
    finally:
        executor.shutdown()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    _finish(outdir, runner, final, stop_reason, engine, omm, prod, summary, summary_path, log)
    log.close()


class Terminated(BaseException):
    """SIGTERM, such as a batch time limit, raised so the partial run is written."""


def _raise_terminated(signum, frame):
    raise Terminated("signal %d" % signum)


def _catch_sigterm(root):
    """
    Rank 0 turns SIGTERM into Terminated. Other MPI ranks ignore it: exiting
    would abort the job before rank 0 has written the partial result.
    Returns the previous handler, or None outside the main thread.
    """
    import threading
    if threading.current_thread() is not threading.main_thread():
        return None
    return signal.signal(signal.SIGTERM, _raise_terminated if root else signal.SIG_IGN)


def _early_stop_reason(exc):
    if isinstance(exc, Terminated):
        return "terminated (%s)" % exc
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    return "error: %s: %s" % (type(exc).__name__, exc)


def _finish(outdir, runner, final, stop_reason, engine, omm, prod, summary, summary_path, log,
            partial=False):
    """sumd_traj.dcd, final_state.xml/.rst7, final_path.txt and run_summary.json for `final`."""
    path = runner.path_to(final)
    dcds = [runner.nodes[i]["dcd"] for i in path if runner.nodes[i]["dcd"]]
    with open(os.path.join(outdir, "final_path.txt"), "w") as fh:
        fh.write("# accepted_step parent_accepted_step cycle metrics trajectory\n")
        for i in path:
            node = runner.nodes[i]
            fh.write("%d %s %d %s %s\n" % (i, node["parent"], node["cycle"],
                     json.dumps(node["metrics"]), node["dcd"]))
    frames = (dcdtools.concat_dcds([os.path.join(outdir, p) for p in dcds],
              os.path.join(outdir, "sumd_traj.dcd")) if dcds else 0)
    last = runner.nodes[final]
    final_xml = os.path.join(outdir, "final_state.xml")
    shutil.copyfile(os.path.join(outdir, last["state_xml"]), final_xml)
    if partial and runner.s.check_stride and final != runner.checked and last["dcd"]:
        runner.write_check(final)
    rst7 = None
    try:
        engine.restore(engine.load_xml(final_xml, last["step"], last["epot"]), check=False)
        rst7 = prod.write_amber_restart(omm.simulation, os.path.join(outdir, "final_state.rst7"),
                                         netcdf=False, enforce_pbc=True)
    except Exception as exc:
        if not partial:
            raise
        # after a GPU error the context may be unusable; the XML is already written
        log("WARNING: final_state.rst7 not written: %s" % exc)
    summary.update(stop_reason=stop_reason, final_accepted_step=final, final_path=path,
                   final_metrics=last["metrics"], n_accepted_steps=len(runner.nodes),
                   best_accepted_step=runner.best_node(),
                   best_metrics=runner.nodes[runner.best_node()]["metrics"],
                   final_traj=dict(file="sumd_traj.dcd" if dcds else None,
                                   windows=len(dcds), frames=frames),
                   final_state_xml=final_xml, final_state_rst7=rst7, partial=partial)
    _write_json(summary_path, summary)
    log("%s: %s; final AcceptedStep %d; %d frames"
        % ("stopped early" if partial else "finished", stop_reason, final, frames))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sumd-openmm", description="Supervised molecular dynamics with OpenMM")
    ap.add_argument("input_file", nargs="?", help="key = value input file")
    ap.add_argument("--dry-run", action="store_true", help="resolve metrics and report initial values")
    ap.add_argument("--test", action="store_true",
                    help="without an input file: check OpenMM platforms and run a built-in system; "
                         "with one: run two short cycles in <output_dir>_test and audit them")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output directory")
    ap.add_argument("--continue", dest="resume", action="store_true",
                    help="continue the run in output_dir after its last completed cycle "
                         "(raise max_cycles, or use --extend)")
    ap.add_argument("--extend", type=int, metavar="N",
                    help="with --continue implied: run N cycles more than are done")
    ap.add_argument("--version", action="version", version="%(prog)s " + __version__)
    args = ap.parse_args(argv)
    if args.extend is not None:
        if args.extend < 1:
            ap.error("--extend needs a positive number of cycles")
        args.resume = True
    if args.resume and (args.overwrite or args.test or args.dry_run):
        ap.error("--continue cannot be combined with --overwrite, --test or --dry-run")
    if args.input_file is None:
        if not args.test:
            ap.error("an input file is required unless --test is given")
        from . import selftest
        sys.exit(selftest.run())
    try:
        config = SumdConfig.from_cfg(read_key_value_file(args.input_file))
        if args.test:
            from . import selftest
            selftest.shorten(config)
            args.overwrite = True
        comm = None
        if config.parallel == "mpi":
            comm = _mpi_world()
        run(args, config, comm)
        if args.test and not args.dry_run and (comm is None or comm.rank == 0):
            ok = selftest.audit(os.path.abspath(config.output_dir))
            print("test run %s: %s" % ("passed" if ok else "FAILED", os.path.abspath(config.output_dir)))
            if not ok:
                sys.exit(1)
    except (ConfigError, PlatformError, selection.SelectionError, CharmmGuiError) as exc:
        if "comm" in locals() and comm is not None and comm.size > 1:
            if comm.rank == 0:
                print("sumd-openmm: error: %s" % exc, file=sys.stderr, flush=True)
            comm.Abort(2)
        if _rank_from_launcher() == 0:
            print("sumd-openmm: error: %s" % exc, file=sys.stderr)
        sys.exit(2)
    except BaseException as exc:
        if "comm" in locals() and comm is not None and comm.size > 1:
            with open(os.path.join(os.path.abspath(config.output_dir),
                     "FATAL_rank_%03d.txt" % comm.rank), "w") as fh:
                fh.write(traceback.format_exc())
            comm.Abort(1)
        raise


if __name__ == "__main__":
    main()
