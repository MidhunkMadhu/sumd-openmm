"""Multiple stratification axes, direction bands and eviction."""

import numpy as np

from sumd_openmm.seeding import StratifiedPool


def test_two_strata_and_frontier():
    pool = StratifiedPool(1, ["2", "10"], per_cell=1, rng=np.random.default_rng(3))
    pool.add(0, 7.1, [1, 9])
    pool.add(1, 7.2, [3, 11])
    pool.add(2, 9.2, [0, 0])
    assert pool.cell_of(pool.entries[0]) == (7, 0, 0)
    assert pool.cell_of(pool.entries[1]) == (7, 1, 1)
    assert 2 not in {pool.choose()[0].node_id for _ in range(8)}
    pool.add(3, 7.3, [1, 9])
    assert pool.entries[0].evicted
    assert pool.entries[0].visits > 0


def test_quantile_bins_and_retries():
    pool = StratifiedPool(1, ["quantiles:3"], max_retries=0)
    pool.observe(np.arange(9)[:, None])
    pool.add(0, -2, [4])
    assert pool.cell_of(pool.entries[0])[1] == 1
    assert pool.record_rejection(0)
    assert pool.choose() == (None, None, None)
