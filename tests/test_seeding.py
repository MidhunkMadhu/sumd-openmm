"""Stratified pool: cells, frontier, least-visited choice, retirement, caps."""

import numpy as np

from sumd_openmm.seeding import StratifiedPool


def pool(**kw):
    args = dict(band_width=1.0, gate_cutoff=4.7, n_strata=2, per_cell=20,
                frontier_bands=1, max_retries=2, rng=np.random.default_rng(0))
    args.update(kw)
    return StratifiedPool(**args)


def test_cells():
    p = pool()
    p.add(0, 7.3, 3.9)
    p.add(1, 7.9, 5.2)
    e0, e1 = p.entries[0], p.entries[1]
    assert p.cell_of(e0) == (7, 0)      # band 7, gate closed
    assert p.cell_of(e1) == (7, 1)      # band 7, gate open
    assert p.stratum(4.7) == 1          # boundary counts as open (>=)


def test_frontier_excludes_far_bands():
    p = pool()
    p.add(0, 12.0, 4.0)
    p.add(1, 5.2, 4.0)
    p.add(2, 6.4, 6.0)
    assert {e.node_id for e in p.frontier()} == {1, 2}


def test_gate_closed_cell_gets_revisited():
    """
    The point of stratified mode: an open-gate state at the frontier must not
    monopolise the parents. With one closed and one open state in the same
    band, draws alternate between the cells.
    """
    p = pool()
    p.add(0, 5.5, 3.9)      # closed
    p.add(1, 5.6, 6.1)      # open
    picks = [p.choose()[1][1] for _ in range(10)]
    assert picks.count(0) == 5 and picks.count(1) == 5


def test_least_visited_cell_wins():
    p = pool()
    p.add(0, 5.5, 3.9)
    for _ in range(3):
        p.choose()           # only one cell: visits accumulate there
    p.add(1, 5.6, 6.1)
    entry, cell, n = p.choose()
    assert entry.node_id == 1 and n == 0


def test_retirement_and_exhaustion():
    p = pool(max_retries=2)
    p.add(0, 5.0, 4.0)
    assert not p.record_rejection(0)
    assert not p.record_rejection(0)
    assert p.record_rejection(0)          # third rejection > max_retries
    assert p.choose() == (None, None, None)


def test_retired_frontier_widens():
    p = pool(max_retries=0)
    p.add(0, 5.0, 4.0)
    p.add(1, 9.0, 4.0)
    p.record_rejection(0)
    assert {e.node_id for e in p.frontier()} == {1}


def test_per_cell_cap_evicts_oldest():
    p = pool(per_cell=2)
    assert p.add(0, 5.1, 4.0) == []
    assert p.add(1, 5.2, 4.0) == []
    assert p.add(2, 5.3, 4.0) == [0]
    assert not p.entries[0].live


def test_evicted_visits_still_count():
    p = pool(per_cell=1)
    p.add(0, 5.1, 4.0)
    p.choose()
    p.choose()
    p.add(1, 5.2, 4.0)                    # evicts 0, whose 2 visits stay on the cell
    assert p.cell_visits()[(5, 0)] == 2


def test_quantile_strata():
    p = pool(n_strata=4)
    p.observe_cv2(np.linspace(3.0, 7.0, 101))
    edges = p.stratum_edges()
    assert np.allclose(edges, [4.0, 5.0, 6.0])
    assert [p.stratum(x) for x in (3.5, 4.5, 5.5, 6.5)] == [0, 1, 2, 3]
