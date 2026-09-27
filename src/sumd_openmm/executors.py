"""
executors.py

Where walkers run. The SuMD logic (SumdRun in sumd_openmm.py) talks to an
executor through four calls and never sees an OpenMM object:

    root(xml_path)                     -> info on the initial state (node 0)
    run_batch(cycle, parent, seeds)    -> one result dict per walker
    resolve(commit_key, node_id, xml, keep_keys)
    shutdown()

LocalExecutor runs every walker on this process's GPU, one after another.
MPIExecutor runs them on all ranks at once (rank 0 also runs walkers).

State ownership
---------------
Each rank has a RankWorker that keeps, in memory:
  * pending  - the end State of every walker it ran that is still undecided
               (the current batch, plus a held least-bad attempt);
  * nodes    - an LRU cache of committed node States it produced or loaded;
  * ctx      - which node its Context holds untouched, if any, so that
               walker 0 can continue in place (velocities kept, like Amber
               irest=1) instead of restoring.
A committed node is written to states/node_N.xml by the rank that ran it.
When a rank needs a parent it does not hold, the parent is broadcast as a
numpy packet from a rank that does; if nobody holds it, rank 0 reads the
node's XML from disk first. output_dir must be on a filesystem shared by
all ranks.
"""

import os
import socket
import time
from collections import OrderedDict

import numpy as np

from .parallel import walker_ranks


class RankWorker:

    def __init__(self, engine, evaluator, outdir, samples, steps_per_sample, stride,
                 restore_check, rank=0, host=None, device=None, cache_size=8):
        self.eng = engine
        self.ev = evaluator
        self.out = outdir
        self.samples = samples
        self.sps = steps_per_sample
        self.stride = stride
        self.check = restore_check
        self.rank = rank
        self.host = host or socket.gethostname()
        self.device = device
        self.cache_size = cache_size

        self.pending = {}
        self.nodes = OrderedDict()
        self.ctx = None

    # ---------------------------------------------- node cache

    def _cache(self, nid, snap):
        self.nodes[nid] = snap
        self.nodes.move_to_end(nid)
        while len(self.nodes) > self.cache_size:
            self.nodes.popitem(last=False)

    def has_node(self, nid):
        return nid in self.nodes

    def load_node(self, nid, xml_path, step, epot):
        self._cache(nid, self.eng.load_xml(xml_path, step, epot))

    def packet_of(self, nid):
        return self.eng.packet(self.nodes[nid])

    def ctx_node(self):
        return self.ctx[1] if self.ctx is not None and self.ctx[0] == "node" else None

    # ---------------------------------------------- work

    def root(self, xml_path):
        snap = self.eng.snapshot()
        c1, c2 = self.eng.current_cvs(self.ev)
        self.eng.save_xml(snap, xml_path)
        self._cache(0, snap)
        self.ctx = ("node", 0)
        return dict(cv1=float(c1), cv2=(None if c2 is None else float(c2)),
                    time_ps=self.eng.time_ps(snap), step=snap.step, epot=snap.epot)

    def _restore_parent(self, parent_id, packet):
        """Restore the parent into the Context; cache it for later retries."""
        if parent_id in self.nodes:
            self.nodes.move_to_end(parent_id)
            return self.eng.restore(self.nodes[parent_id], self.check)

        if packet is None:
            raise RuntimeError("rank %d: parent node %d neither cached nor sent" % (self.rank, parent_id))

        dE = self.eng.restore_packet(packet, self.check)
        self._cache(parent_id, self.eng.snapshot())
        return dE

    def run(self, cycle, specs, parent_id, packet):
        """Run this rank's walkers (specs in walker order). Returns result dicts."""
        out = []

        for i, sp in enumerate(specs):
            w, key = sp["w"], (cycle, sp["w"])

            if i == 0 and self.ctx == ("node", parent_id):
                start_mode, seed, dE = "continue", None, None
            else:
                dE = self._restore_parent(parent_id, packet)
                seed = sp["seed"]
                if seed is not None:
                    self.eng.reseed(seed)
                start_mode = "restore+reassign" if seed is not None else "restore+keep"

            tmp = os.path.join(self.out, "windows", "tmp", "c%05d_w%d.dcd" % (cycle, w))
            wall0 = time.time()
            t, c1, c2 = self.eng.run_window(self.samples, self.sps, self.ev, tmp, self.stride)
            wall = time.time() - wall0

            snap = self.eng.snapshot()
            self.pending[key] = snap
            self.ctx = ("key", key)

            out.append(dict(
                key=key, w=w, cycle=cycle, rank=self.rank, host=self.host, device=self.device,
                t=t, cv1=c1, cv2=c2, seed=seed, start_mode=start_mode, restore_dE=dE,
                dcd_tmp=tmp, wall=wall, t_start_wall=wall0,
                time_ps=self.eng.time_ps(snap),
                step=snap.step, epot=snap.epot,
                cv1_final=float(c1[-1]), cv2_final=(None if c2 is None else float(c2[-1])),
            ))

        return out

    def resolve(self, commit_key, node_id, xml_path, keep):
        """Write the committed walker's State as node `node_id`; drop the rest."""
        if commit_key is not None and commit_key in self.pending:
            snap = self.pending[commit_key]
            self.eng.save_xml(snap, xml_path)
            self._cache(node_id, snap)
            if self.ctx == ("key", commit_key):
                self.ctx = ("node", node_id)

        keep = set(keep)
        for k in list(self.pending):
            if k not in keep:
                del self.pending[k]

        if self.ctx is not None and self.ctx[0] == "key" and self.ctx[1] not in keep:
            self.ctx = None          # Context holds a discarded attempt


# ============================================================
# Serial
# ============================================================

class LocalExecutor:
    size = 1

    def __init__(self, worker):
        self.w = worker

    def root(self, xml_path):
        return self.w.root(xml_path)

    def run_batch(self, cycle, parent, seeds):
        """parent: dict(id, xml, step, epot) of the parent node."""
        if not self.w.has_node(parent["id"]):
            self.w.load_node(parent["id"], parent["xml"], parent["step"], parent["epot"])
        specs = [dict(w=w, seed=s) for w, s in enumerate(seeds)]
        return self.w.run(cycle, specs, parent["id"], None)

    def resolve(self, commit_key, node_id, xml_path, keep):
        self.w.resolve(commit_key, node_id, xml_path, keep)

    def shutdown(self):
        pass


# ============================================================
# MPI
# ============================================================

def _batch_collective(comm, worker, msg):
    """Executed by every rank for one batch. Returns all results on rank 0."""
    parent = msg["parent"]
    pid = parent["id"]

    has = comm.allgather(worker.has_node(pid))
    if not any(has):
        if comm.rank == 0:
            worker.load_node(pid, parent["xml"], parent["step"], parent["epot"])
        has[0] = True

    packet = None
    if not all(has):
        src = has.index(True)
        packet = comm.bcast(worker.packet_of(pid) if comm.rank == src else None, root=src)

    mine = [sp for sp in msg["specs"] if sp["rank"] == comm.rank]
    results = worker.run(msg["cycle"], mine, pid, packet)

    gathered = comm.gather(results, root=0)
    if comm.rank != 0:
        return None
    return sorted((r for part in gathered for r in part), key=lambda r: r["w"])


def _resolve_collective(comm, worker, msg):
    worker.resolve(msg["commit_key"], msg["node_id"], msg["xml"], msg["keep"])
    # gather returns only after every rank has finished writing its XML
    return comm.gather(worker.ctx_node(), root=0)


def worker_loop(comm, worker):
    """Ranks > 0: obey rank 0 until told to stop."""
    while True:
        msg = comm.bcast(None, root=0)

        if msg["op"] == "batch":
            _batch_collective(comm, worker, msg)
        elif msg["op"] == "resolve":
            _resolve_collective(comm, worker, msg)
        elif msg["op"] == "stop":
            return
        else:
            raise RuntimeError("unknown op %r" % msg["op"])


class MPIExecutor:

    def __init__(self, comm, worker):
        self.comm = comm
        self.w = worker
        self.size = comm.Get_size()
        self.ctx_nodes = [None] * self.size
        self.stopped = False

    def root(self, xml_path):
        info = self.w.root(xml_path)
        self.ctx_nodes[0] = 0
        return info

    def run_batch(self, cycle, parent, seeds):
        pid = parent["id"]
        first = self.ctx_nodes.index(pid) if pid in self.ctx_nodes else 0
        ranks = walker_ranks(len(seeds), self.size, first)
        specs = [dict(w=w, seed=s, rank=r) for w, (s, r) in enumerate(zip(seeds, ranks))]

        msg = dict(op="batch", cycle=cycle, parent=parent, specs=specs)
        self.comm.bcast(msg, root=0)
        return _batch_collective(self.comm, self.w, msg)

    def resolve(self, commit_key, node_id, xml_path, keep):
        msg = dict(op="resolve", commit_key=commit_key, node_id=node_id, xml=xml_path, keep=list(keep))
        self.comm.bcast(msg, root=0)
        self.ctx_nodes = _resolve_collective(self.comm, self.w, msg)

    def shutdown(self):
        if not self.stopped:
            self.comm.bcast(dict(op="stop"), root=0)
            self.stopped = True
