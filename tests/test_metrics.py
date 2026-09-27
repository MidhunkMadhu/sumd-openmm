"""Metric geometry, selection and configuration across variable metric counts."""

import numpy as np
import pytest

from sumd_openmm.config import ConfigError, MetricSpec, SumdConfig
from sumd_openmm.metrics import Evaluator, Metric
from sumd_openmm.selection import SelectionError, resolve_one


def metric(kind, groups, **over):
    spec = MetricSpec(1, "test", kind, **over)
    return Metric(spec, {key: np.asarray(value, int) for key, value in groups.items()})


def test_centre_of_geometry_and_periodic_group():
    X = np.array([[9.8, 0, 0], [0.2, 0, 0], [3, 0, 0]])
    box = np.eye(3) * 10
    m = metric("distance", dict(a=[0, 1], b=[2]))
    assert m.value(X, box) == pytest.approx(3)
    assert m.value(X) == pytest.approx(2)
    assert metric("distance", dict(a=[0], b=[2])).value(X, box) == pytest.approx(3.2)


def test_axis_and_minimum_group_distance():
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 4, 0], [3, 2, 0]], float)
    m = metric("distance_axis", dict(a=[0], b=[2]), axis="0,1,0")
    assert m.value(X) == pytest.approx(4)
    m.spec.signed = False
    assert m.value(X) == pytest.approx(4)
    m.spec.axis = "z"
    assert m.value(X) == 0
    m = metric("mindist", dict(a=[0, 1], b=[2, 3]))
    assert m.value(X) == pytest.approx(np.sqrt(5))


def test_angle_dihedral_and_triclinic_translation():
    X = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]], float)
    angle = metric("angle", dict(a=[0], b=[1], c=[2]))
    torsion = metric("dihedral", dict(a=[0], b=[1], c=[2], d=[3]))
    assert angle.value(X) == pytest.approx(90)
    assert abs(torsion.value(X)) == pytest.approx(90)
    box = np.array([[10, 0, 0], [3, 10, 0], [1, 2, 10]], float)
    shifted = X.copy()
    shifted[3] += box[1] - box[2]
    assert torsion.value(X, box) == pytest.approx(torsion.value(shifted, box))
    assert Evaluator([angle, torsion])(X, box).shape == (2,)


def config(**kwargs):
    base = dict(metric_1_type="distance", metric_1_a="indices:0",
                metric_1_b="indices:1", metric_1_role="supervise",
                metric_1_direction="increase", metric_1_target="5")
    return SumdConfig.from_cfg({**base, **kwargs})


@pytest.mark.parametrize("count", [1, 2, 7])
def test_parse_any_count(count):
    additional = {}
    for n in range(2, count + 1):
        additional.update({f"metric_{n}_type": "angle",
                           f"metric_{n}_a": "indices:0", f"metric_{n}_b": "indices:1",
                           f"metric_{n}_c": "indices:2"})
    assert len(config(**additional).metrics) == count


@pytest.mark.parametrize("keys,fragment", [
    ({"metric_3_type": "rmsd"}, "missing metric_2"),
    ({"metric_2_type": "rmsd", "metric_2_a": "indices:0", "metric_2_reference": "ref.pdb",
      "metric_2_name": "metric_1"}, "unique"),
    ({"metric_1_nonsense": "5"}, "valid keys"),
    ({"cv1_type": "rmsd"}, "removed"),
    ({"metric_1_target": ""}, "could not convert"),
    ({"supervision": "combined"}, "at least two"),
    ({"seeding": "stratified"}, "stratify"),
])
def test_invalid_configs(keys, fragment):
    with pytest.raises((ConfigError, ValueError), match=fragment):
        config(**keys)


def test_indices_and_mask_languages(tmp_path):
    parmed = pytest.importorskip("parmed")
    top = parmed.Structure()
    for name, resid in [("C1", 1), ("C2", 1), ("O1", 2)]:
        top.add_atom(parmed.Atom(name=name, atomic_number=6), "MOL", resid)
    cpp, syntax = resolve_one(top, ":1@C*", label="cpp")
    vmd, _ = resolve_one(top, "vmd:resid 1 and name C*", label="vmd")
    np.testing.assert_array_equal(cpp, vmd)
    idx, _ = resolve_one(top, "indices:0-1,2")
    np.testing.assert_array_equal(idx, [0, 1, 2])
    (tmp_path / "group.idx").write_text("0-1\n2\n")
    np.testing.assert_array_equal(resolve_one(top, "file:group.idx", base_dir=tmp_path)[0], idx)


def test_single_atom_validation(tmp_path):
    from sumd_openmm.selection import resolve
    parmed = pytest.importorskip("parmed")
    top = parmed.Structure()
    for name in ("C1", "C2", "C3"):
        top.add_atom(parmed.Atom(name=name, atomic_number=6), "MOL", 1)
    path = tmp_path / "top.pdb"
    top.coordinates = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    top.save(str(path))
    c = config(metric_1_type="angle", metric_1_a="indices:0-1",
               metric_1_b="indices:1", metric_1_c="indices:2")
    with pytest.raises(SelectionError, match="exactly one atom"):
        resolve(c, str(path))


def test_reference_masks_differ_from_topology_masks(tmp_path):
    """FIT_PDB_STRING / CALCRMSD_PDB_STRING: the reference numbers residues differently."""
    from sumd_openmm.selection import resolve
    parmed = pytest.importorskip("parmed")

    def structure(first_resid, shift):
        s = parmed.Structure()
        for r in range(4):
            for name in ("CA", "CB"):
                s.add_atom(parmed.Atom(name=name, atomic_number=6), "ALA", first_resid + r)
        s.add_atom(parmed.Atom(name="C1", atomic_number=6), "LIG", first_resid + 4)
        s.add_atom(parmed.Atom(name="C2", atomic_number=6), "LIG", first_resid + 4)
        xyz = np.arange(30, dtype=float).reshape(10, 3) ** 1.1
        xyz[8:] += shift
        s.coordinates = xyz
        return s

    structure(1, 0).save(str(tmp_path / "top.pdb"))
    # reference: two extra residues in front, ligand displaced by 3 A along x
    ref = structure(1, [3.0, 0, 0])
    extra = parmed.Structure()
    for r in range(2):
        extra.add_atom(parmed.Atom(name="N", atomic_number=7), "GLY", r + 1)
    extra.coordinates = np.zeros((2, 3))
    for residue in ref.residues:
        residue.number += 2
    (extra + ref).save(str(tmp_path / "ref.pdb"))

    keys = dict(metric_1_type="rmsd_displacement", metric_1_reference="ref.pdb",
                metric_1_fit=":1-4@CA", metric_1_a=":5", metric_1_role="supervise",
                metric_1_direction="decrease", metric_1_target="2")
    with pytest.raises(SelectionError, match="topology atoms|pairing"):
        resolve(SumdConfig.from_cfg(keys), str(tmp_path / "top.pdb"), str(tmp_path))
    evaluator, report = resolve(SumdConfig.from_cfg(dict(keys, metric_1_reference_fit=":3-6@CA",
                                                         metric_1_reference_a=":7")),
                                str(tmp_path / "top.pdb"), str(tmp_path))
    assert report[0]["reference"]["a"] == ":7"
    X = np.asarray(parmed.load_file(str(tmp_path / "top.pdb")).coordinates, float)
    assert evaluator(X)[0] == pytest.approx(3.0, abs=1e-3)


@pytest.mark.parametrize("box", [None, np.diag([20.0, 20.0, 20.0]),
                                 np.array([[20.0, 0, 0], [5.0, 19.0, 0], [3.0, 4.0, 18.0]])])
def test_contacts_counts_pairs_within_cutoff(box):
    rng = np.random.default_rng(5)
    X = rng.uniform(0, 18, (700, 3))
    m = metric("contacts", {"a": np.arange(600), "b": np.arange(600, 700)}, cutoff=4.5)
    from sumd_openmm import cv
    D = X[600:][None] - X[:600, None]
    D = cv.minimum_image(D.reshape(-1, 3), box) if box is not None else D.reshape(-1, 3)
    assert m.value(X, box) == np.count_nonzero((D ** 2).sum(axis=1) < 4.5 ** 2) > 0
