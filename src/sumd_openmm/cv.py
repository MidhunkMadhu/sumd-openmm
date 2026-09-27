"""
cv.py

Collective variables and the window trend statistic for SuMD.

Pure numpy. Nothing here imports OpenMM, so every function can be unit-tested
without a GPU. All lengths are in angstrom, all times in ps.

Conventions
-----------
Coordinates are (n, 3) arrays. Box vectors are a (3, 3) array whose ROWS are
the lattice vectors a, b, c (OpenMM order), so a position with fractional
coordinates s is r = s @ box.
"""

import itertools
from dataclasses import dataclass

import numpy as np


# ============================================================
# Periodic imaging
# ============================================================

_IMAGE_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)


def minimum_image(d, box):
    """
    Minimum-image form of displacement vector(s) d, shape (..., 3).

    Fractional wrapping alone is not exact for triclinic cells, so after the
    wrap the 27 neighbouring images are searched and the shortest is kept.
    """
    d = np.asarray(d, dtype=float)
    box = np.asarray(box, dtype=float)

    s = d @ np.linalg.inv(box)
    s -= np.round(s)
    d0 = s @ box

    cand = d0[..., None, :] + _IMAGE_SHIFTS @ box
    best = np.argmin(np.einsum("...ij,...ij->...i", cand, cand), axis=-1)

    return np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :]


def shift_group_near(X, point, box, weights=None):
    """
    Translate a whole group by a lattice vector so that its centroid is the
    periodic image nearest to `point`. The group's internal geometry is
    untouched, so the group must already be whole (true for OpenMM positions,
    which are never split within a molecule).
    """
    c = centroid(X, weights)
    d = c - point
    return X + (minimum_image(d, box) - d)


def make_groups_coherent(X, groups, box):
    """
    Place every group of atoms (e.g. separate molecules of a multi-chain
    receptor) in the image nearest the first group. `groups` is a list of
    index arrays into X. None or a single group is a no-op.
    """
    if groups is None or len(groups) < 2:
        return X

    X = X.copy()
    anchor = X[groups[0]].mean(axis=0)

    for g in groups[1:]:
        X[g] = shift_group_near(X[g], anchor, box)

    return X


# ============================================================
# Superposition and RMSD
# ============================================================

def centroid(X, weights=None):
    if weights is None:
        return X.mean(axis=0)

    w = np.asarray(weights, dtype=float)
    return (w[:, None] * X).sum(axis=0) / w.sum()


def kabsch(P, Q, weights=None):
    """
    Optimal rotation taking P onto Q.

    Returns (R, p0, q0) such that  R @ (p - p0) + q0  is the superposed p.

        H = (P - p0)^T W (Q - q0)
        H = U S V^T
        d = sign(det(V U^T))
        R = V diag(1, 1, d) U^T

    The d term turns an improper solution (a reflection) into the best proper
    rotation; without it near-degenerate frames can come back mirrored.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    p0 = centroid(P, weights)
    q0 = centroid(Q, weights)

    Pc = P - p0
    Qc = Q - q0

    if weights is not None:
        Pc = Pc * np.asarray(weights, dtype=float)[:, None]

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T

    d = np.sign(np.linalg.det(V @ U.T))
    if d == 0:
        d = 1.0

    R = V @ np.diag([1.0, 1.0, d]) @ U.T

    return R, p0, q0


def rmsd_nofit(A, B, weights=None):
    """Plain RMSD between paired coordinates, no superposition."""
    sq = np.einsum("ij,ij->i", A - B, A - B)

    if weights is None:
        return float(np.sqrt(sq.mean()))

    w = np.asarray(weights, dtype=float)
    return float(np.sqrt((w * sq).sum() / w.sum()))


def ligand_rmsd_receptor_frame(
    X, idx_align, idx_lig, ref_align, ref_lig,
    box=None, w_align=None, w_lig=None, align_groups=None,
):
    """
    Ligand RMSD to its bound pose in the receptor frame.

    Reproduces the cpptraj pair used by the Amber SuMD reference:

        rms ref :FIT_TRAJ :FIT_PDB            (superpose on receptor)
        rms ref :LIG_TRAJ :LIG_PDB nofit      (ligand RMSD, no refit)

    1. image the ligand next to the receptor (the autoimage step),
    2. Kabsch on the alignment atoms,
    3. apply that transform to the ligand,
    4. RMSD against the reference ligand without further fitting.

    The CV therefore keeps the ligand's translation relative to the site;
    it is not a shape RMSD.
    """
    A = X[idx_align]
    L = X[idx_lig]

    if box is not None:
        A = make_groups_coherent(A, align_groups, box)
        L = shift_group_near(L, centroid(A, w_align), box, w_lig)

    R, p0, q0 = kabsch(A, ref_align, w_align)
    L_fit = (L - p0) @ R.T + q0

    return rmsd_nofit(L_fit, ref_lig, w_lig)


# ============================================================
# Distance CVs
# ============================================================

def com_distance(X, idx_a, idx_b, box=None, w_a=None, w_b=None):
    """Centre-of-mass distance between two selections (classic SuMD CV)."""
    d = centroid(X[idx_b], w_b) - centroid(X[idx_a], w_a)

    if box is not None:
        d = minimum_image(d, box)

    return float(np.linalg.norm(d))


def min_distance(X, idx_a, idx_b, box=None):
    """
    Minimum atom-atom distance between two selections, e.g. the pocket-2
    gate aperture F154(4.56) side chain to T197(5.45) side chain.
    """
    D = X[idx_b][None, :, :] - X[idx_a][:, None, :]

    if box is not None:
        D = minimum_image(D.reshape(-1, 3), box)

    return float(np.sqrt(np.einsum("...i,...i->...", D, D).min()))


# ============================================================
# Trend statistic
# ============================================================

@dataclass
class Trend:
    slope: float
    intercept: float
    se: float
    r2: float
    n: int
    units: str


def linear_trend(x, y, units="A/ps"):
    """
    Ordinary least squares y = a + b x, with the standard error of b

        b     = sum (x - xbar)(y - ybar) / sum (x - xbar)^2
        SE(b) = sqrt( sum r^2 / (n - 2) / sum (x - xbar)^2 )

    SE and R^2 are NaN when undefined (n < 3, or no variance).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n < 2:
        raise ValueError("linear_trend needs at least two points, got %d" % n)

    xm = x.mean()
    ym = y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm

    resid = y - (a + b * x)
    ssr = (resid ** 2).sum()
    sst = ((y - ym) ** 2).sum()

    se = float(np.sqrt(ssr / (n - 2) / sxx)) if n > 2 else float("nan")
    r2 = float(1.0 - ssr / sst) if sst > 0 else float("nan")

    return Trend(float(b), float(a), se, r2, n, units)


def reference_five_points(n):
    """
    Frame indices used by extract_and_fit.py in the Amber reference:
    [0, n/4-1, 2n/4-1, 3n/4-1, 4n/4-1], truncating, with the same membership
    test (so negative or repeated indices collapse the same way).
    """
    q = int(n / 4)
    wanted = {0, q - 1, 2 * q - 1, 3 * q - 1, 4 * q - 1}
    return [i for i in range(n) if i in wanted]


def window_trend(t, cv, slope_points="all"):
    """
    Trend of one window's CV series.

    slope_points = all : OLS on every sample against time (A/ps). Default.
    slope_points = 5   : the Amber reference estimator, 5 subsampled points
                         regressed against FRAME INDEX (A/frame). Only the
                         sign enters the decision, so the units differ
                         harmlessly; they are logged.
    """
    if str(slope_points) == "5":
        idx = reference_five_points(len(cv))
        return linear_trend(idx, np.asarray(cv)[idx], units="A/frame")

    return linear_trend(t, cv, units="A/ps")
