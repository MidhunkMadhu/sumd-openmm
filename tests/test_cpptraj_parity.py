"""
Section 11.1: CV1 from cv.py must match the Amber reference's cpptraj
"fit on receptor, then rms nofit on ligand" to within 1e-3 A.

Needs cpptraj, ParmEd and a real system. Point these at one, e.g. the
testfolder of supervisedmdamber:

    export SUMD_TEST_PARM=.../testfolder/example.parm7
    export SUMD_TEST_RST=.../testfolder/example.rst7
    export SUMD_TEST_REF=.../testfolder/ref_example.pdb
    export SUMD_TEST_FIT=":1-313@CA"           # optional, these are the defaults
    export SUMD_TEST_LIG=":314&!@H="
    export SUMD_CPPTRAJ=/path/to/cpptraj       # optional if cpptraj is on PATH

Skipped when any of them is missing.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from sumd_openmm import cv as CV

PARM = os.environ.get("SUMD_TEST_PARM")
RST = os.environ.get("SUMD_TEST_RST")
REF = os.environ.get("SUMD_TEST_REF")
FIT = os.environ.get("SUMD_TEST_FIT", ":1-313@CA")
LIG = os.environ.get("SUMD_TEST_LIG", ":314&!@H=")
CPPTRAJ = os.environ.get("SUMD_CPPTRAJ") or shutil.which("cpptraj")

pytestmark = pytest.mark.skipif(
    not (PARM and RST and REF and CPPTRAJ and all(os.path.exists(p) for p in (PARM, RST, REF))),
    reason="set SUMD_TEST_PARM/RST/REF and have cpptraj available",
)


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


IMAGE_SHIFTS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def run_reference_cpptraj(tmp_path, frames, box, natom):
    """Write frames to temp1.nc and run the reference's two cpptraj scripts."""
    from parmed.amber import NetCDFTraj

    traj = NetCDFTraj.open_new(str(tmp_path / "temp1.nc"), natom, box=True, crds=True)
    for k, fr in enumerate(frames):
        traj.add_time(float(k))
        traj.add_coordinates(fr)
        traj.add_box(box)
    traj.close()

    # exactly as helpfolder/generate_fitCPPIN.py and generate_calcCPPIN.py write them
    (tmp_path / "fit.cppin").write_text(
        f"parm {PARM}\ntrajin temp1.nc\n\nautoimage\n\n"
        f"parm {REF}\nreference {REF} parm {REF} [SYS_PDB]\n"
        f"rms To_SYS_PDB ref [SYS_PDB] {FIT} {FIT} out proteinFitRMSD.xvg time 0.01\n"
        f"trajout temp2.nc\n\nrun\n")
    (tmp_path / "calc_rmsd.cppin").write_text(
        f"parm {PARM}\ntrajin temp2.nc\n\n"
        f"parm {REF}\nreference {REF} parm {REF} [SYS_PDB]\n"
        f"rms To_SYS_PDB ref [SYS_PDB] {LIG} {LIG} out ligandRMSD.xvg time 0.01 nofit\n\nrun\n")

    for inp in ("fit.cppin", "calc_rmsd.cppin"):
        subprocess.run([CPPTRAJ, "-i", inp], cwd=tmp_path, check=True, capture_output=True)

    return np.loadtxt(tmp_path / "ligandRMSD.xvg", comments=["#", "@"])[:, 1]


@pytest.fixture(scope="module")
def system():
    parmed = pytest.importorskip("parmed")
    from parmed.amber import AmberMask

    top = parmed.load_file(PARM, xyz=RST)
    ref = parmed.load_file(REF)
    box = np.asarray(top.box, dtype=float)
    assert np.allclose(box[3:], 90.0), "test builds an orthorhombic trajectory"

    sel = lambda s, m: np.array(list(AmberMask(s, m).Selected()))
    rxyz = np.asarray(ref.coordinates, dtype=float)

    return dict(xyz=np.asarray(top.coordinates, dtype=float), box=box, H=np.diag(box[:3]),
                natom=len(top.atoms), ia=sel(top, FIT), il=sel(top, LIG),
                ref_a=rxyz[sel(ref, FIT)], ref_l=rxyz[sel(ref, LIG)])


def ours(s, frame):
    X = frame.astype(np.float32).astype(float)                    # what cpptraj read back
    return CV.ligand_rmsd_receptor_frame(X, s["ia"], s["il"], s["ref_a"], s["ref_l"], box=s["H"])


def test_cv1_matches_cpptraj_fit_then_rms_nofit(system, tmp_path):
    """Same periodic image as the receptor: must agree to < 1e-3 A."""
    s = system
    x0, il = s["xyz"], s["il"]
    centre = x0[s["ia"]].mean(axis=0)
    rng = np.random.default_rng(11)

    frames = [x0.copy(),
              x0 + [7.0, -4.0, 3.0],                               # rigid translation
              (x0 - centre) @ rotz(20).T + centre]                 # rigid rotation
    for mag in (3.0, 6.0):                                         # ligand displaced from pose
        f = x0.copy()
        d = rng.normal(size=3)
        f[il] += mag * d / np.linalg.norm(d)
        frames.append(f)

    ref_vals = run_reference_cpptraj(tmp_path, frames, s["box"], s["natom"])
    mine = np.array([ours(s, f) for f in frames])
    print("\ncpptraj :", np.round(ref_vals, 4), "\ncv.py   :", np.round(mine, 4),
          "\nmax |diff| = %.2e A" % np.abs(mine - ref_vals).max())

    assert len(ref_vals) == len(frames)
    assert np.abs(mine - ref_vals).max() < 1e-3
    assert abs(mine[3] - mine[0]) > 0.5 and abs(mine[4] - mine[0]) > 0.5


def test_ligand_in_other_image(system, tmp_path):
    """
    Ligand moved by one lattice vector in each of the 6 directions.

    cv.py must return the same-image value every time (minimum image). The
    reference's cpptraj `autoimage` is only checked for agreement where it
    returns the nearest image: in CPPTRAJ V7.6.2 it does NOT for some
    directions (observed: -a, +b, -c left in the far image), which is the
    silent-CV failure section 5.1 of the prompt warns about. The counts are
    printed, not asserted, since they depend on the cpptraj version.
    """
    s = system
    base = ours(s, s["xyz"])
    frames = []
    for sh in IMAGE_SHIFTS:
        f = s["xyz"].copy()
        f[s["il"]] += np.array(sh) @ s["H"]
        frames.append(f)

    ref_vals = run_reference_cpptraj(tmp_path, frames, s["box"], s["natom"])
    mine = np.array([ours(s, f) for f in frames])
    ok = np.abs(ref_vals - base) < 1e-3

    for sh, r, m, good in zip(IMAGE_SHIFTS, ref_vals, mine, ok):
        print("shift %-11s cpptraj %8.3f   cv.py %8.3f   %s"
              % (sh, r, m, "same image" if good else "CPPTRAJ LEFT LIGAND IN FAR IMAGE"))

    assert np.allclose(mine, base, atol=1e-3)                      # cv.py: image-invariant
    assert ok.any()                                                # and it is the same quantity
    assert np.all(ref_vals[~ok] > base)                            # far image only ever inflates
