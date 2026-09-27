"""
supervision.py

Accept/reject rules and walker scores. Pure python + numpy, no OpenMM.

Two independent axes are configured in the .inp:

  supervision  = single | stratified | joint
      which quantity the slope rule is applied to, and how parents are seeded
  walker_score = slope | smscore | dmscore
      how the best of a batch of walkers is picked (walkers > 1 only)

single / stratified accept on CV1 alone. joint and dmscore let the gate
coordinate (CV2) into the acceptance decision and carry JOINT_WARNING.
"""

import numpy as np


JOINT_WARNING = (
    "joint mode applies an AND-style filter that discards windows in which the "
    "ligand waits outside a closed gate. Since the gate is open only 0.8-3 % of "
    "the time in the apo receptor, kon estimates derived from joint-mode seeds "
    "are expected to be biased fast. Use for comparison only."
)

DMSCORE_WARNING = (
    "walker_score = dmscore ranks walkers on the gate coordinate as well as the "
    "ligand coordinate, so it preferentially retains walkers that found an open "
    "gate. It mitigates stalling, not selection bias: " + JOINT_WARNING
)


# ============================================================
# Single-window rule (Amber reference, supervisedmdamber.py)
# ============================================================

def sumd_decision(slope, cv_final, colvarcut, se=None, require_significant=False):
    """
    Returns (decision, reason), decision in {"converged", "accept", "reject"}.

        converged  iff  cv < cut  and  b <= 0
        accept     iff  b <= 0  or  (b > 0 and cv < cut)
        reject     otherwise

    Identical to the reference, where the stop test is evaluated on the same
    window's values before the restart decision.

    require_significant_slope = yes tightens only the "progress" clause:
    b must satisfy b < 0 and |b| > 2 SE(b). The inside-the-cutoff clause and
    the convergence test are unchanged.
    """
    inside = cv_final < colvarcut

    if inside and slope <= 0:
        return "converged", "cv %.3f < cut %.3f and slope %.4g <= 0" % (cv_final, colvarcut, slope)

    if require_significant:
        if se is None or not np.isfinite(se):
            progress = False
        else:
            progress = slope < 0 and abs(slope) > 2.0 * se
        progress_txt = "slope %.4g < 0 and |slope| > 2 SE (%.4g)" % (slope, 2.0 * (se or np.nan))
    else:
        progress = slope <= 0
        progress_txt = "slope %.4g <= 0" % slope

    if progress:
        return "accept", progress_txt

    if inside:
        return "accept", "slope %.4g > 0 but cv %.3f < cut %.3f" % (slope, cv_final, colvarcut)

    return "reject", "slope %.4g not progress and cv %.3f >= cut %.3f" % (slope, cv_final, colvarcut)


# ============================================================
# Joint score
# ============================================================

def joint_series(cv1, cv2, mu, sigma, weights, cv2_direction="increase"):
    """
    s(t) = w1 z1 + w2 z2,  z_k = (x_k - mu_k) / sigma_k

    Progress is a DECREASE of s. CV1 (ligand) progresses by decreasing; the
    gate aperture progresses by increasing, so z2 is negated when
    cv2_direction = increase.
    """
    z1 = (np.asarray(cv1, dtype=float) - mu[0]) / sigma[0]
    z2 = (np.asarray(cv2, dtype=float) - mu[1]) / sigma[1]

    if cv2_direction == "increase":
        z2 = -z2

    return weights[0] * z1 + weights[1] * z2


# ============================================================
# mwSuMD scores (Deganutti et al., eLife 2025, 96513)
# ============================================================

def smscore(series):
    """
    Single-metric score  sqrt(X_last * X_mean).

    Geometric mean of the endpoint and the window mean: weights where the
    walker ended while rewarding consistent progress. For a decreasing CV
    (ligand RMSD / distance) LOWER is better. Requires X >= 0.
    """
    x = np.asarray(series, dtype=float)

    if (x < 0).any():
        raise ValueError("smscore requires a non-negative metric")

    return float(np.sqrt(x[-1] * x.mean()))


def dmscores(batch_cv1, batch_cv2, cv2_direction="increase"):
    """
    Dual-metric score for every walker of one batch, HIGHER is better:

        DM_w = [ s1 (X'_last,w / <X'>_batch - 1) + s2 (X''_last,w / <X''>_batch - 1) ] * 100

    <X>_batch is the mean over all samples of all walkers in the batch.
    s = -1 for a metric that should decrease (CV1), +1 for one that should
    increase (the gate, when cv2_direction = increase). A metric sitting at
    the batch average contributes zero, so a stalled metric does not block
    the other.
    """
    m1 = float(np.mean(np.concatenate([np.asarray(s, float) for s in batch_cv1])))
    m2 = float(np.mean(np.concatenate([np.asarray(s, float) for s in batch_cv2])))

    if m1 == 0 or m2 == 0:
        raise ValueError("dmscore undefined: batch mean of a metric is zero")

    s2 = 1.0 if cv2_direction == "increase" else -1.0

    return [
        100.0 * (-(c1[-1] / m1 - 1.0) + s2 * (c2[-1] / m2 - 1.0))
        for c1, c2 in zip(batch_cv1, batch_cv2)
    ]


def pick_best_walker(walker_score, trends, cv1_series, cv2_series=None, cv2_direction="increase"):
    """
    Index of the best walker in a batch and the per-walker scores.

      slope    lowest slope of the supervised series (ties: lower final CV1)
      smscore  lowest SMscore of CV1
      dmscore  highest DMscore of (CV1, CV2)
    """
    n = len(cv1_series)

    if walker_score == "slope":
        scores = [t.slope for t in trends]
        best = min(range(n), key=lambda i: (scores[i], cv1_series[i][-1]))

    elif walker_score == "smscore":
        scores = [smscore(s) for s in cv1_series]
        best = min(range(n), key=lambda i: scores[i])

    elif walker_score == "dmscore":
        scores = dmscores(cv1_series, cv2_series, cv2_direction)
        best = max(range(n), key=lambda i: scores[i])

    else:
        raise ValueError("unknown walker_score: %s" % walker_score)

    return best, scores
