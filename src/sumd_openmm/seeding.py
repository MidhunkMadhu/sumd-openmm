"""
seeding.py

Stratified seed pool for supervision = stratified. Pure python + numpy.

Every accepted window's final state becomes a pool entry tagged with its
final CV1 (ligand progress) and CV2 (gate aperture). Entries fall into cells

    c = (CV1 band, CV2 stratum)
    band    = floor(CV1 / cv1_band_width)
    stratum = 0 (closed) / 1 (open) at gate_open_cutoff, or quantile bin

Each cycle the parent is drawn from the least-visited cell on the frontier,

    F  = { c : band(c) <= best_band + frontier_bands }
    c* = argmin_{c in F} n_c        ties broken at random

where best_band is the lowest CV1 band held by any live entry and n_c the
number of times a parent was drawn from c. Gate-closed states near the
frontier are therefore revisited on purpose instead of being out-competed by
open-gate states, which is what keeps the gate waiting time in the seeds.

Visit counts are kept on entries, including evicted and retired ones, and n_c
is summed over whatever entries currently map to c. That stays consistent
when quantile edges move as more CV2 data arrive.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PoolEntry:
    node_id: int
    cv1: float
    cv2: float
    order: int
    visits: int = 0
    retries: int = 0
    evicted: bool = False
    exhausted: bool = False

    @property
    def live(self):
        return not (self.evicted or self.exhausted)


class StratifiedPool:

    def __init__(self, band_width, gate_cutoff, n_strata=2, per_cell=20,
                 frontier_bands=1, max_retries=8, rng=None):
        if band_width <= 0:
            raise ValueError("cv1_band_width must be > 0")
        if n_strata < 2:
            raise ValueError("gate_strata must be >= 2")

        self.band_width = float(band_width)
        self.gate_cutoff = float(gate_cutoff)
        self.n_strata = int(n_strata)
        self.per_cell = int(per_cell)
        self.frontier_bands = int(frontier_bands)
        self.max_retries = int(max_retries)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.entries = {}
        self._order = 0
        self._cv2_samples = []

    # -------------------------------------------------- cells

    def observe_cv2(self, samples):
        """Feed CV2 samples; used only for quantile strata (gate_strata > 2)."""
        self._cv2_samples.extend(float(x) for x in samples)

    def band(self, cv1):
        return int(math.floor(cv1 / self.band_width))

    def stratum_edges(self):
        if self.n_strata == 2:
            return [self.gate_cutoff]

        if len(self._cv2_samples) < self.n_strata:
            return [self.gate_cutoff]

        qs = np.linspace(0, 1, self.n_strata + 1)[1:-1]
        return list(np.quantile(self._cv2_samples, qs))

    def stratum(self, cv2, edges=None):
        edges = self.stratum_edges() if edges is None else edges
        return int(np.searchsorted(edges, cv2, side="right"))

    def cell_of(self, entry, edges=None):
        return (self.band(entry.cv1), self.stratum(entry.cv2, edges))

    def cell_visits(self, edges=None):
        edges = self.stratum_edges() if edges is None else edges
        counts = {}

        for e in self.entries.values():
            c = self.cell_of(e, edges)
            counts[c] = counts.get(c, 0) + e.visits

        return counts

    # -------------------------------------------------- mutation

    def add(self, node_id, cv1, cv2):
        """Insert a state. Returns node ids evicted by the per-cell cap."""
        e = PoolEntry(int(node_id), float(cv1), float(cv2), self._order)
        self._order += 1
        self.entries[e.node_id] = e

        edges = self.stratum_edges()
        cell = self.cell_of(e, edges)
        same = sorted(
            (x for x in self.entries.values() if x.live and self.cell_of(x, edges) == cell),
            key=lambda x: x.order,
        )

        evicted = []
        while len(same) > self.per_cell:
            old = same.pop(0)
            old.evicted = True
            evicted.append(old.node_id)

        return evicted

    def record_rejection(self, node_id):
        """Count a rejected window from this parent; retire it past the limit."""
        e = self.entries[node_id]
        e.retries += 1

        if e.retries > self.max_retries:
            e.exhausted = True

        return e.exhausted

    # -------------------------------------------------- selection

    def live_entries(self):
        return [e for e in self.entries.values() if e.live]

    def frontier(self):
        live = self.live_entries()

        if not live:
            return []

        best = min(self.band(e.cv1) for e in live)
        return [e for e in live if self.band(e.cv1) <= best + self.frontier_bands]

    def choose(self):
        """
        Draw the next parent. Returns (entry, cell, n_c before this visit),
        or (None, None, None) when no live entry is left.
        """
        edges = self.stratum_edges()
        front = self.frontier()

        if not front:
            return None, None, None

        visits = self.cell_visits(edges)
        cells = sorted({self.cell_of(e, edges) for e in front})
        n_min = min(visits.get(c, 0) for c in cells)
        tied = [c for c in cells if visits.get(c, 0) == n_min]
        cell = tied[int(self.rng.integers(len(tied)))]

        members = [e for e in front if self.cell_of(e, edges) == cell]
        m_min = min(e.visits for e in members)
        members = [e for e in members if e.visits == m_min]
        entry = members[int(self.rng.integers(len(members)))]

        entry.visits += 1

        return entry, cell, n_min
