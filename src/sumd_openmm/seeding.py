"""Frontier sampling across progress bands and stratification bins."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PoolEntry:
    node_id: int
    progress: float
    strata: tuple
    order: int
    visits: int = 0
    retries: int = 0
    evicted: bool = False
    exhausted: bool = False

    @property
    def live(self):
        return not self.evicted and not self.exhausted


class StratifiedPool:
    def __init__(self, band_width, bins, per_cell=20, frontier_bands=1,
                 max_retries=8, rng=None):
        if band_width <= 0 or per_cell < 1:
            raise ValueError("band_width and per_cell must be positive")
        self.band_width, self.bins = float(band_width), list(bins)
        self.per_cell, self.frontier_bands = per_cell, frontier_bands
        self.max_retries = max_retries
        self.rng = rng if rng is not None else np.random.default_rng()
        self.entries = {}
        self.samples = [[] for _ in bins]
        self._order = 0

    def band(self, progress):
        return int(math.floor(progress / self.band_width))

    def observe(self, values):
        array = np.asarray(values, float)
        for k in range(len(self.bins)):
            self.samples[k].extend(array[:, k].tolist())

    def _edges(self, i):
        spec = str(self.bins[i])
        if spec.startswith("quantiles:"):
            n = int(spec.split(":", 1)[1])
            if n < 2:
                raise ValueError("quantile bin count must be >= 2")
            values = self.samples[i]
            return np.quantile(values, np.arange(1, n) / n) if len(values) >= n else []
        return [float(x) for x in spec.replace(",", " ").split()]

    def cell_of(self, entry):
        return (self.band(entry.progress), *[
            int(np.searchsorted(self._edges(i), value, side="right"))
            for i, value in enumerate(entry.strata)])

    def add(self, node_id, progress, strata):
        entry = PoolEntry(node_id, float(progress), tuple(strata), self._order)
        self._order += 1
        self.entries[node_id] = entry
        same = sorted((x for x in self.entries.values()
                       if x.live and self.cell_of(x) == self.cell_of(entry)),
                      key=lambda x: x.order)
        removed = []
        while len(same) > self.per_cell:
            old = same.pop(0)
            old.evicted = True
            removed.append(old.node_id)
        return removed

    def record_rejection(self, node_id):
        entry = self.entries[node_id]
        entry.retries += 1
        entry.exhausted = entry.retries > self.max_retries
        return entry.exhausted

    def live_entries(self):
        return [x for x in self.entries.values() if x.live]

    def choose(self):
        live = self.live_entries()
        if not live:
            return None, None, None
        best = min(self.band(x.progress) for x in live)
        frontier = [x for x in live if self.band(x.progress) <= best + self.frontier_bands]
        totals = {}
        for entry in self.entries.values():
            cell = self.cell_of(entry)
            totals[cell] = totals.get(cell, 0) + entry.visits
        least = min(totals[self.cell_of(x)] for x in frontier)
        cells = sorted({self.cell_of(x) for x in frontier if totals[self.cell_of(x)] == least})
        cell = cells[int(self.rng.integers(len(cells)))]
        members = [x for x in frontier if self.cell_of(x) == cell]
        count = min(x.visits for x in members)
        tied = [x for x in members if x.visits == count]
        entry = tied[int(self.rng.integers(len(tied)))]
        entry.visits += 1
        return entry, cell, least
