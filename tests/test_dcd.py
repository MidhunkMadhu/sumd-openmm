"""dcdtools against OpenMM's DCD writer and MDAnalysis's independent reader."""

import numpy as np
import pytest

from sumd_openmm import dcdtools

openmm = pytest.importorskip("openmm")


def write_dcd(path, positions_nm, box_nm):
    from openmm import Vec3
    from openmm.app import DCDFile, Element, Topology
    from openmm.unit import nanometers, picoseconds

    top = Topology()
    chain = top.addChain()
    res = top.addResidue("X", chain)
    for _ in range(positions_nm.shape[1]):
        top.addAtom("C", Element.getBySymbol("C"), res)
    vecs = [Vec3(*v) for v in box_nm] * nanometers
    top.setPeriodicBoxVectors(vecs)

    with open(path, "wb") as fh:
        dcd = DCDFile(fh, top, 0.002 * picoseconds, firstStep=10, interval=10)
        for frame in positions_nm:
            dcd.writeModel(frame * nanometers, periodicBoxVectors=vecs)


@pytest.mark.parametrize("box", [np.diag([5.0, 5.0, 7.0]),
                                 np.array([[5.0, 0, 0], [1.5, 4.6, 0], [-1.2, 1.1, 6.3]])])
def test_read_and_concat(tmp_path, box):
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 5, size=(3, 17, 3))
    b = rng.uniform(0, 5, size=(4, 17, 3))
    write_dcd(tmp_path / "a.dcd", a, box)
    write_dcd(tmp_path / "b.dcd", b, box)

    assert np.allclose(dcdtools.read_frames(tmp_path / "a.dcd"), 10 * a, atol=1e-4)

    n = dcdtools.concat_dcds([str(tmp_path / "a.dcd"), str(tmp_path / "b.dcd")], str(tmp_path / "ab.dcd"))
    assert n == 7
    ab = dcdtools.read_frames(tmp_path / "ab.dcd")
    assert np.allclose(ab, 10 * np.concatenate([a, b]), atol=1e-4)

    mda_dcd = pytest.importorskip("MDAnalysis.coordinates.DCD")
    r = mda_dcd.DCDReader(str(tmp_path / "ab.dcd"))
    assert r.n_frames == 7
    for k, ts in enumerate(r):
        assert np.allclose(ts.positions, ab[k], atol=1e-5)


def test_rejects_non_dcd(tmp_path):
    p = tmp_path / "x.dcd"
    p.write_bytes(b"\0" * 400)
    with pytest.raises(ValueError):
        dcdtools.read_header(str(p))


@pytest.mark.parametrize("stride", [1, 2, 3, 10])
def test_concat_with_stride_equals_sliced_concat(tmp_path, stride):
    rng = np.random.default_rng(1)
    box = np.diag([5.0, 5.0, 7.0])
    parts = [rng.uniform(0, 5, size=(n, 5, 3)) for n in (4, 3, 7)]
    paths = []
    for k, frames in enumerate(parts):
        paths.append(str(tmp_path / ("p%d.dcd" % k)))
        write_dcd(paths[-1], frames, box)

    dcdtools.concat_dcds(paths, str(tmp_path / "all.dcd"))
    full = dcdtools.read_frames(tmp_path / "all.dcd")
    n = dcdtools.concat_dcds(paths, str(tmp_path / "thin.dcd"), stride)
    assert n == len(full[::stride])
    assert np.array_equal(dcdtools.read_frames(tmp_path / "thin.dcd"), full[::stride])
    h = dcdtools.read_header(str(tmp_path / "thin.dcd"))
    assert (h["nframes"], h["interval"], h["first_step"]) == (n, 10 * stride, 10)

    mda_dcd = pytest.importorskip("MDAnalysis.coordinates.DCD")
    assert mda_dcd.DCDReader(str(tmp_path / "thin.dcd")).n_frames == n
