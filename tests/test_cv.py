"""Geometry and trend tests without an OpenMM Context."""

import itertools

import numpy as np
import pytest

from sumd_openmm import cv as CV


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


@pytest.fixture
def system():
    """Toy receptor (60 'CA') + ligand (12 atoms) and a displaced copy."""
    rng = np.random.default_rng(1)
    rec = rng.normal(scale=8.0, size=(60, 3)) + 40.0
    lig = rng.normal(scale=1.5, size=(12, 3)) + rec.mean(axis=0) + [3.0, 0.0, 0.0]
    X = np.vstack([rec, lig])
    idx_align = np.arange(60)
    idx_lig = np.arange(60, 72)

    # current frame: receptor rigidly moved, ligand moved AND displaced 4 A
    R = random_rotation(rng)
    Y = (X - 40.0) @ R.T + 40.0 + [5.0, -3.0, 2.0]
    Y[idx_lig] += 4.0 * (R @ np.array([1.0, 0.0, 0.0]))

    return dict(ref=X, cur=Y, a=idx_align, l=idx_lig, rng=rng)


def cv1(s, X, box=None):
    return CV.fitted_displacement_rmsd(X, s["a"], s["l"], s["ref"][s["a"]], s["ref"][s["l"]], box=box)


# ---------------------------------------------------------------- Kabsch

def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(30, 3))
    R0 = random_rotation(rng)
    Q = P @ R0.T + [1.0, 2.0, 3.0]

    R, p0, q0 = CV.kabsch(P, Q)

    assert np.allclose((P - p0) @ R.T + q0, Q, atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_kabsch_proper_rotation_on_reflected_input():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(25, 3))
    Q = P * [1.0, 1.0, -1.0]          # mirror image: best unconstrained fit is a reflection

    R, _, _ = CV.kabsch(P, Q)

    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_kabsch_weighted_equals_unweighted_for_equal_weights():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(20, 3))
    Q = P @ random_rotation(rng).T
    R1, _, _ = CV.kabsch(P, Q)
    R2, _, _ = CV.kabsch(P, Q, weights=np.full(20, 12.011))
    assert np.allclose(R1, R2)


# ---------------------------------------------------------------- CV1 semantics

def test_cv1_is_ligand_displacement_not_shape(system):
    # the ligand was moved rigidly by 4 A relative to the receptor: shape RMSD
    # would be 0, the receptor-frame RMSD must be 4
    assert np.isclose(cv1(system, system["cur"]), 4.0, atol=1e-8)
    assert np.isclose(cv1(system, system["ref"]), 0.0, atol=1e-8)


def test_cv1_invariant_to_rigid_motion_of_whole_system(system):
    base = cv1(system, system["cur"])
    rng = system["rng"]

    for _ in range(5):
        Y = system["cur"] @ random_rotation(rng).T + rng.normal(scale=20.0, size=3)
        assert np.isclose(cv1(system, Y), base, atol=1e-8)


@pytest.mark.parametrize("box", [
    np.diag([70.0, 70.0, 110.0]),
    np.array([[70.0, 0.0, 0.0], [23.0, 66.0, 0.0], [-20.0, 18.0, 95.0]]),   # triclinic, OpenMM reduced form
])
def test_cv1_unchanged_across_periodic_boundary(system, box):
    Y = system["cur"]
    base = cv1(system, Y, box=box)

    for shift in itertools.product((-1, 0, 1), repeat=3):
        Z = Y.copy()
        Z[system["l"]] += np.array(shift) @ box      # ligand in another image
        assert np.isclose(cv1(system, Z, box=box), base, atol=1e-8)

    # without imaging, the same frame gives nonsense: the test is not vacuous
    Z = Y.copy()
    Z[system["l"]] += box[0]
    assert cv1(system, Z) > 30.0


def test_cv1_mass_weighted_runs_and_differs(system):
    w = np.linspace(1, 16, 12)
    a = CV.fitted_displacement_rmsd(system["cur"], system["a"], system["l"],
                                      system["ref"][system["a"]], system["ref"][system["l"]])
    b = CV.fitted_displacement_rmsd(system["cur"], system["a"], system["l"],
                                      system["ref"][system["a"]], system["ref"][system["l"]], w_measure=w)
    assert np.isfinite(b)
    assert np.isclose(a, 4.0) and np.isclose(b, 4.0)   # pure translation: weights do not matter


def test_multi_molecule_alignment_made_coherent(system):
    box = np.diag([70.0, 70.0, 110.0])
    Y = system["cur"].copy()
    base = cv1(system, Y, box=box)
    groups = [np.arange(30), np.arange(30, 60)]
    Y[30:60] += box[1]                                # second chain in another image
    v = CV.fitted_displacement_rmsd(Y, system["a"], system["l"], system["ref"][system["a"]],
                                      system["ref"][system["l"]], box=box, align_groups=groups)
    assert np.isclose(v, base, atol=1e-8)


# ---------------------------------------------------------------- imaging and distances

def test_minimum_image_matches_brute_force_triclinic():
    rng = np.random.default_rng(5)
    box = np.array([[60.0, 0.0, 0.0], [25.0, 55.0, 0.0], [-22.0, 20.0, 80.0]])
    lattice = np.array(list(itertools.product(range(-7, 8), repeat=3))) @ box

    for _ in range(200):
        d = rng.uniform(-150, 150, size=3)
        brute = (d + lattice)[np.argmin(np.linalg.norm(d + lattice, axis=1))]
        assert np.isclose(np.linalg.norm(CV.minimum_image(d, box)), np.linalg.norm(brute))


def test_min_distance_and_com_distance_with_pbc():
    box = np.diag([50.0, 50.0, 50.0])
    X = np.array([[1.0, 25.0, 25.0], [2.0, 25.0, 25.0], [48.0, 25.0, 25.0], [47.0, 25.0, 25.0]])
    assert np.isclose(CV.min_distance(X, [0, 1], [2, 3], box=box), 3.0)
    assert np.isclose(CV.min_distance(X, [0, 1], [2, 3]), 45.0)
    assert np.isclose(CV.com_distance(X, [0, 1], [2, 3], box=box), 4.0)


# ---------------------------------------------------------------- trend

def test_linear_trend_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)

    for n in (5, 20, 200):
        x = np.sort(rng.uniform(0, 20, n))
        y = 0.3 * x + rng.normal(size=n)
        ref = scipy_stats.linregress(x, y)
        t = CV.linear_trend(x, y)
        assert np.isclose(t.slope, ref.slope)
        assert np.isclose(t.intercept, ref.intercept)
        assert np.isclose(t.se, ref.stderr)
        assert np.isclose(t.r2, ref.rvalue ** 2)


def _reference_extract_and_fit(data):
    """The five-point frame-index estimator."""
    from scipy.stats import linregress
    n = len(data)
    x_indices = [0, int(n / 4) - 1, 2 * int(n / 4) - 1, 3 * int(n / 4) - 1, 4 * int(n / 4) - 1]
    x, y = [], []
    for i in range(n):
        if i in x_indices:
            x.append(i)
            y.append(data[i][1])
    slope = linregress(x, y)[0]
    return slope, data[-1][1]


@pytest.mark.parametrize("n", [8, 10, 20, 37, 500])
def test_five_point_mode_reproduces_reference(n):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(n)
    t = np.arange(1, n + 1) * 0.04
    y = 10.0 - 0.05 * np.arange(n) + rng.normal(scale=0.3, size=n)

    ref_slope, ref_cv = _reference_extract_and_fit(np.column_stack([t, y]))
    tr = CV.window_trend(t, y, slope_points="5")

    assert np.isclose(tr.slope, ref_slope)
    assert tr.units == "A/frame"
    assert np.isclose(y[-1], ref_cv)
