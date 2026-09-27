"""Acceptance rules and scores over arbitrary directions and metric counts."""

import numpy as np
import pytest

from sumd_openmm.config import MetricSpec
from sumd_openmm.metrics import supervised_series, unwrap
from sumd_openmm.supervision import combined_series, dmscores, pick_best_walker, smscore, sumd_decision


@pytest.mark.parametrize("direction,sign", [("decrease", -1), ("increase", 1)])
def test_decision_table(direction, sign):
    for inside in (False, True):
        for progress in (-1, 0, 1):
            target = 5
            final = target + (sign if inside else -sign)
            verdict, _ = sumd_decision(sign * progress, final, target, direction=direction)
            expected = ("converged" if inside and progress >= 0 else
                        "accept" if inside or progress >= 0 else "reject")
            assert verdict == expected


def test_significance_only_changes_progress_acceptance():
    assert sumd_decision(-1, 8, 5, se=1, require_significant=True)[0] == "reject"
    assert sumd_decision(-1, 4, 5, se=1, require_significant=True)[0] == "converged"


def test_torsion_unwrap_continues_from_parent_and_toward_is_circular():
    np.testing.assert_allclose(unwrap([179, -179, -175], 178), [179, 181, 185])
    np.testing.assert_allclose(unwrap([-179, -175], 179), [181, 185])
    spec = MetricSpec(1, "rotation", "dihedral", "supervise", "toward", target=180, tolerance=10)
    values, target = supervised_series([-175, 179], spec)
    np.testing.assert_allclose(values, [5, 1])
    assert target == 10
    spec.direction, spec.target, spec.target_delta = "increase", None, 180
    values, target = supervised_series([-179, -175], spec, parent=179, initial=179)
    np.testing.assert_allclose(values, [181, 185])
    assert target == 359


def test_combined_and_multimetric_scores():
    a = MetricSpec(1, "a", "distance", "supervise", "decrease")
    b = MetricSpec(2, "b", "distance", "supervise", "increase")
    x = np.array([[4, 1], [3, 2], [2, 3]], float)
    assert np.all(np.diff(combined_series(x, [a, b], [0, 0], [1, 1])) < 0)
    scores = dmscores([x, x + [1, -1]], [a, b])
    assert scores[0] > scores[1]
    assert smscore([4, 2]) == pytest.approx(np.sqrt(6))
    trends = [type("Trend", (), {"slope": -1}), type("Trend", (), {"slope": 1})]
    assert pick_best_walker("slope", trends, [x, x], [a, b])[0] == 0


def test_dmscore_is_equation_2():
    """DMscore = ((X'last / mean X' - 1) + (X''last / mean X'' - 1)) * 100, sign -1 if decreasing."""
    up = MetricSpec(1, "up", "distance", "supervise", "increase", 10.0)
    down = MetricSpec(2, "down", "rmsd", "supervise", "decrease", 1.0)
    walkers = [np.array([[4.0, 6.0], [6.0, 4.0]]), np.array([[2.0, 8.0], [4.0, 6.0]])]
    mean = np.concatenate(walkers).mean(axis=0)                  # over every walker in the batch
    expected = [100 * ((w[-1, 0] / mean[0] - 1) - (w[-1, 1] / mean[1] - 1)) for w in walkers]
    assert dmscores(walkers, [up, down]) == pytest.approx(expected)
    assert pick_best_walker("dmscore", None, walkers, [up, down])[0] == 0


def test_smscore_is_equation_1():
    x = np.array([[5.0], [4.0], [3.0]])
    assert smscore(x[:, 0]) == pytest.approx(np.sqrt(3.0 * 4.0))
    spec = MetricSpec(1, "d", "distance", "supervise", "decrease", 2.0)
    best, scores = pick_best_walker("smscore", None, [x, x + 1], [spec])
    assert best == 0 and scores[0] < scores[1]
