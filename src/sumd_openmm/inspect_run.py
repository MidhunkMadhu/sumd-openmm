#!/usr/bin/env python3
"""
inspect_run.py

Audit trajectories, states and sampling from a finished SuMD run.

    sumd-inspect sumd_run [sumd_run2 ...] [--json out.json]
    (or python -m sumd_openmm.inspect_run ...)

Per run it reports:
  * cycles, windows simulated, acceptance rate, retry-exhaustion events;
  * restore check: max |dEpot| after state restores;
  * trajectory integrity: sumd_traj.dcd is frame-for-frame identical to the
    accepted windows on the final path, in order, and every committed
    window's last DCD frame equals its node's saved State (so no DCD was
    mixed up with a rejected attempt);
  * the distribution and range of every named metric in retained states.

Quantities that cannot be computed are reported as null, never 0.
"""

import argparse
import csv
import json
import os

import numpy as np

from . import dcdtools


def load(run):
    with open(os.path.join(run, "run_summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(run, "nodes.csv")) as f:
        nodes = list(csv.DictReader(f))
    with open(os.path.join(run, "windows.jsonl")) as f:
        windows = [json.loads(l) for l in f if l.strip()]
    return summary, nodes, windows


def fnum(x):
    return None if x in ("", None) else float(x)


def inspect(run):
    summary, nodes, windows = load(run)
    out = dict(run=run, stop_reason=summary.get("stop_reason"),
               supervision=summary["sumd_config"]["supervision"],
               walkers=summary["sumd_config"]["walkers"])

    cycles = sorted({w["cycle"] for w in windows})
    best = [w for w in windows if w["best_in_batch"]]
    out["cycles"] = len(cycles)
    out["windows_simulated"] = len(windows)
    out["accepted_windows"] = sum(1 for w in windows if w["accepted"])
    out["acceptance_rate_per_cycle"] = (
        None if not best else round(sum(1 for w in best if w["verdict"] != "reject") / len(best), 3))
    out["retry_exhaustion_events"] = sum(1 for w in best if w.get("action"))

    ranks = {}
    for w in windows:
        ranks[w.get("rank", 0)] = ranks.get(w.get("rank", 0), 0) + 1
    out["windows_per_rank"] = dict(sorted(ranks.items()))
    out["start_modes"] = {m: sum(1 for w in windows if w["start_mode"] == m)
                          for m in sorted({w["start_mode"] for w in windows})}

    dE = [abs(w["restore_dE_kJmol"]) for w in windows if w.get("restore_dE_kJmol") is not None]
    out["restores_checked"] = len(dE)
    out["max_abs_restore_dE_kJmol"] = max(dE) if dE else None

    # ---------------------------------------------- trajectory integrity
    by_id = {int(n["node_id"]): n for n in nodes}
    path = summary.get("final_path") or []
    dcds = [by_id[i]["dcd"] for i in path if by_id[i]["dcd"]]
    committed = {(int(n["cycle"]), n["walker"]) for n in nodes if n["dcd"]}
    out["windows_committed"] = len(committed)
    out["windows_discarded"] = len(windows) - len(committed)

    traj = os.path.join(run, "sumd_traj.dcd")
    if dcds and os.path.exists(traj):
        expected = sum(dcdtools.read_header(os.path.join(run, d))["nframes"] for d in dcds)
        out["final_traj_frames"] = dcdtools.read_header(traj)["nframes"]
        out["final_traj_frames_expected"] = expected
        out["final_traj_identical_to_path_windows"] = frame_identity(run, traj, dcds)
    else:
        out["final_traj_frames"] = None
        out["final_traj_identical_to_path_windows"] = None

    out["max_dcd_vs_state_mismatch_A"] = dcd_state_consistency(run, nodes)

    out["retained_metrics"] = {}
    for metric in summary.get("metrics", []):
        name = metric["name"]
        values = np.array([fnum(n[name]) for n in nodes if n.get(name, "") != ""], float)
        out["retained_metrics"][name] = (
            dict(min=float(values.min()), max=float(values.max()),
                 sd=float(values.std(ddof=1)) if len(values) > 1 else None)
            if len(values) else None)
    cells = [n["cell"] for n in nodes if n.get("cell")]
    out["retained_states_by_cell"] = {cell: cells.count(cell) for cell in sorted(set(cells))}
    return out


def _state_xml_positions(path):
    """Positions (A) and box (A, rows a,b,c) from an OpenMM State XML."""
    import xml.etree.ElementTree as ET
    pos, box = [], []
    for _, el in ET.iterparse(path):
        if el.tag == "Position":
            pos.append((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
        elif el.tag in ("A", "B", "C"):
            box.append((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
        el.clear()
    return 10.0 * np.array(pos), 10.0 * np.array(box[:3])


def dcd_state_consistency(run, nodes):
    """
    For every committed node, the last frame of its window DCD must equal the
    node's saved State modulo lattice vectors (the DCD is wrapped, the State
    is not). Returns the worst per-atom deviation in A; float32 DCD storage
    limits this to ~1e-5 A per 100 A of coordinate.
    """
    from . import cv as CV

    worst = None
    for n in nodes:
        if not n["dcd"]:
            continue
        last = dcdtools.read_frames(os.path.join(run, n["dcd"]))[-1].astype(float)
        X, box = _state_xml_positions(os.path.join(run, n["state_xml"]))
        d = CV.minimum_image(last - X, box)
        dev = float(np.sqrt((d ** 2).sum(axis=1)).max())
        worst = dev if worst is None else max(worst, dev)
    return None if worst is None else float("%.3g" % worst)


def frame_identity(run, traj, dcds):
    """sumd_traj.dcd must be exactly the path windows' frames, in order."""
    full = dcdtools.read_frames(traj)
    parts = np.concatenate([dcdtools.read_frames(os.path.join(run, d)) for d in dcds])
    return bool(full.shape == parts.shape and np.array_equal(full, parts))


def main():
    ap = argparse.ArgumentParser(prog="sumd-inspect")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()

    res = [inspect(r) for r in args.runs]
    for r in res:
        print(json.dumps(r, indent=2))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
