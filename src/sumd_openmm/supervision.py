"""Direction-aware acceptance and multiple-walker scores."""

import numpy as np


SELECTION_BIAS = (
    "Accepting windows on several metrics preferentially retains pathways where "
    "those metrics progress together and may discard waiting periods. SuMD segments "
    "do not provide valid kinetics or thermodynamics; estimate rates from unbiased "
    "simulations seeded from saved states."
)


def sumd_decision(slope, final, target, se=None, require_significant=False,
                  direction="decrease"):
    """Return (converged|accept|reject, explanation)."""
    if direction not in ("increase", "decrease"):
        raise ValueError("direction must be increase or decrease")
    progress = slope if direction == "increase" else -slope
    inside = final > target if direction == "increase" else final < target
    if inside and progress >= 0:
        return "converged", "inside target with nonnegative progress"
    significant = (not require_significant or
                   (se is not None and np.isfinite(se) and progress > 2 * se))
    if progress >= 0 and significant:
        return "accept", "progress toward target"
    if inside:
        return "accept", "inside target"
    return "reject", "no progress toward target"


def combined_series(columns, specs, mu, sigma):
    """Signed standardized score; lower is better."""
    X = np.asarray(columns, float)
    signs = np.asarray([1 if s.direction in ("decrease", "toward") else -1 for s in specs])
    weights = np.asarray([s.weight for s in specs])
    return np.sum(weights * signs * (X - mu) / sigma, axis=1)


def smscore(series):
    x = np.asarray(series, float)
    if np.any(x < 0):
        raise ValueError("smscore requires a nonnegative metric")
    return float(np.sqrt(x[-1] * x.mean()))


def dmscores(batch, specs):
    """Score arbitrary supervised metrics relative to their batch means."""
    means = np.mean(np.concatenate([np.asarray(x) for x in batch], axis=0), axis=0)
    if np.any(means == 0):
        raise ValueError("dmscore undefined for a metric with zero batch mean")
    signs = np.asarray([1 if s.direction == "increase" else -1 for s in specs])
    return [float(100 * np.sum(signs * (np.asarray(x)[-1] / means - 1))) for x in batch]


def pick_best_walker(mode, trends, supervised, specs):
    if mode == "slope":
        scores = [float(t.slope) for t in trends]
        sign = -1 if specs[0].direction == "increase" else 1
        best = min(range(len(scores)), key=lambda i: (scores[i], sign * supervised[i][-1, 0]))
    elif mode == "smscore":
        scores = [smscore(x[:, 0]) for x in supervised]
        best = max(range(len(scores)), key=lambda i: scores[i]) if specs[0].direction == "increase" else min(
            range(len(scores)), key=lambda i: scores[i])
    elif mode == "dmscore":
        scores = dmscores(supervised, specs)
        best = int(np.argmax(scores))
    else:
        raise ValueError("unknown walker_score %s" % mode)
    return best, scores
