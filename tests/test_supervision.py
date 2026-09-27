"""Decision rules, walker scores, config validation."""

import numpy as np
import pytest

from sumd_openmm.config import ConfigError, SumdConfig
from sumd_openmm.cv import Trend
from sumd_openmm.supervision import (dmscores, joint_series, pick_best_walker, smscore,
                         sumd_decision)


# ---------------------------------------------------------------- Amber rule

@pytest.mark.parametrize("slope,cv,expected", [
    (-0.1, 1.5, "converged"),      # inside cut, decreasing
    (0.0, 1.5, "converged"),       # slope == 0 counts as <= 0
    (+0.1, 1.5, "accept"),         # inside cut, rising: not punished
    (-0.1, 5.0, "accept"),         # outside, progressing
    (0.0, 5.0, "accept"),
    (+0.1, 5.0, "reject"),         # outside, receding
    (+0.1, 2.0, "reject"),         # cv == cut is NOT inside (strict <, as reference)
])
def test_reference_rule_table(slope, cv, expected):
    assert sumd_decision(slope, cv, 2.0)[0] == expected


def test_reference_rule_is_exactly_supervisedmdamber():
    """Brute-force against the literal expressions in supervisedmdamber.py."""
    rng = np.random.default_rng(0)
    cut = 2.0
    for slope, cv in zip(rng.normal(size=2000) * 0.1, rng.uniform(0, 6, 2000)):
        if cv < cut and slope <= 0:
            ref = "converged"
        elif slope <= 0 or (slope > 0 and cv < cut):
            ref = "accept"
        else:
            ref = "reject"
        assert sumd_decision(slope, cv, cut)[0] == ref


def test_require_significant_slope():
    # noisy small negative slope: accepted by default, rejected when strict
    assert sumd_decision(-0.01, 5.0, 2.0, se=0.02)[0] == "accept"
    assert sumd_decision(-0.01, 5.0, 2.0, se=0.02, require_significant=True)[0] == "reject"
    assert sumd_decision(-0.10, 5.0, 2.0, se=0.02, require_significant=True)[0] == "accept"
    # the inside-the-cutoff clause and convergence are unchanged
    assert sumd_decision(+0.10, 1.0, 2.0, se=0.02, require_significant=True)[0] == "accept"
    assert sumd_decision(-0.01, 1.0, 2.0, se=0.02, require_significant=True)[0] == "converged"


# ---------------------------------------------------------------- joint

def test_joint_series_direction():
    # ligand approaching (cv1 down) and gate opening (cv2 up) are both progress
    cv1 = np.array([10.0, 9.0, 8.0])
    cv2 = np.array([4.0, 5.0, 6.0])
    s = joint_series(cv1, cv2, mu=[9.0, 5.0], sigma=[1.0, 1.0], weights=[1.0, 1.0])
    assert np.all(np.diff(s) < 0)
    s_dec = joint_series(cv1, cv2, [9.0, 5.0], [1.0, 1.0], [1.0, 1.0], cv2_direction="decrease")
    assert np.allclose(np.diff(s_dec), 0.0)


# ---------------------------------------------------------------- mwSuMD scores

def test_smscore_formula():
    x = np.array([4.0, 3.0, 2.0, 1.0])
    assert np.isclose(smscore(x), np.sqrt(1.0 * 2.5))
    with pytest.raises(ValueError):
        smscore(np.array([1.0, -1.0]))


def test_dmscore_formula_and_neutral_metric():
    c1 = [np.array([10.0, 8.0]), np.array([10.0, 10.0])]
    c2 = [np.array([5.0, 5.0]), np.array([5.0, 5.0])]
    m1 = np.mean([10, 8, 10, 10])
    s = dmscores(c1, c2)
    assert np.isclose(s[0], 100 * (-(8.0 / m1 - 1)))
    assert s[0] > s[1]      # walker 0 got closer; gate identical so contributes 0


def test_pick_best_walker():
    trends = [Trend(0.1, 0, 0, 0, 5, "A/ps"), Trend(-0.2, 0, 0, 0, 5, "A/ps")]
    cv1 = [np.array([5.0, 4.0]), np.array([5.0, 4.5])]
    assert pick_best_walker("slope", trends, cv1)[0] == 1
    assert pick_best_walker("smscore", trends, cv1)[0] == 0


# ---------------------------------------------------------------- config

BASE = dict(cv1_traj_selection=":314&!@H=", cv1_ref_selection=":314&!@H=",
            align_traj_selection=":1-313@CA", align_ref_selection=":1-313@CA",
            reference_structure="ref.pdb")


def test_defaults_are_single_and_reference_like():
    c = SumdConfig.from_cfg(dict(BASE))
    assert c.supervision == "single"
    assert c.walkers == 1 and c.walker_score == "slope"
    assert not c.require_significant_slope and not c.cv2_in_acceptance


@pytest.mark.parametrize("extra,msg", [
    (dict(method="GaMD"), "GaMD"),
    (dict(supervision="stratified"), "cv2_selection"),
    (dict(walker_score="smscore"), "walkers = 1"),
    (dict(walkers="4", walker_score="dmscore", supervision="stratified",
          cv2_selection_a=":136", cv2_selection_b=":179"), "contradictory"),
    (dict(window_ps="20", cv_sample_ps="3"), "integer"),
    (dict(cv2_selection_a=":136"), "both"),
])
def test_config_refusals(extra, msg):
    with pytest.raises(ConfigError, match=msg):
        SumdConfig.from_cfg({**BASE, **extra})


def test_distance_cv_needs_site():
    with pytest.raises(ConfigError, match="cv1_site_selection"):
        SumdConfig.from_cfg(dict(cv1_type="distance", cv1_traj_selection=":314"))
