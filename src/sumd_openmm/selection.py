"""
selection.py

Resolve Amber-style masks ONCE at startup into integer index arrays.

Masks are evaluated with ParmEd's AmberMask, which implements cpptraj/ambmask
semantics: ':N' is the N-th residue in file order (not a PDB resid), '@'
atom names, '=' wildcard, '&', '|', '!'. ParmEd is already an MD_openmm
dependency. Nothing here runs inside the cycle loop.

Note: the Amber reference prepended ':' to its mask strings
(":{FIT_TRAJ_STRING}"). Here masks are written in full, e.g. ':1-313@CA'.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class SelectionError(ValueError):
    pass


def load_structure(path, xyz=None):
    import parmed

    if xyz is None:
        return parmed.load_file(path)

    return parmed.load_file(path, xyz=xyz)


def mask_indices(struct, mask, label):
    from parmed.amber import AmberMask

    idx = np.array(list(AmberMask(struct, mask).Selected()), dtype=int)

    if idx.size == 0:
        raise SelectionError("%s: mask %r selects no atoms" % (label, mask))

    return idx


def describe(struct, idx):
    """Residue summary of a selection, e.g. ['PHE136'] (file-order numbers)."""
    out = []

    for r in sorted({struct.atoms[i].residue.idx for i in idx}):
        res = struct.residues[r]
        out.append("%s%d" % (res.name, r + 1))

    return out


def check_resnames(struct, idx, expect, label):
    """
    Abort unless every residue in the selection has the expected name.
    `expect` may list alternatives separated by '/', e.g. 'HIS/HIE/HID'.
    """
    if expect is None:
        return

    allowed = {x.strip().upper() for x in expect.split("/")}
    found = {struct.atoms[i].residue.name.upper() for i in idx}

    if not found <= allowed:
        raise SelectionError(
            "%s: expected residue %s, selection covers %s. Numbering in this "
            "topology differs from what the .inp assumes; fix the mask."
            % (label, expect, ", ".join(describe(struct, idx)))
        )


def masses(struct, idx):
    return np.array([struct.atoms[i].mass for i in idx], dtype=float)


@dataclass
class ResolvedSelections:
    cv1: np.ndarray
    cv1_site: Optional[np.ndarray] = None
    align: Optional[np.ndarray] = None
    ref_cv1_xyz: Optional[np.ndarray] = None
    ref_align_xyz: Optional[np.ndarray] = None
    w_cv1: Optional[np.ndarray] = None
    w_site: Optional[np.ndarray] = None
    w_align: Optional[np.ndarray] = None
    cv2_a: Optional[np.ndarray] = None
    cv2_b: Optional[np.ndarray] = None
    report: dict = field(default_factory=dict)


def _pair_names(top, ref, idx_t, idx_r, label):
    """Traj and reference atoms are paired in order, as cpptraj does. Check names."""
    if len(idx_t) != len(idx_r):
        raise SelectionError(
            "%s: topology mask selects %d atoms, reference mask selects %d"
            % (label, len(idx_t), len(idx_r))
        )

    bad = [
        (top.atoms[i].name, ref.atoms[j].name)
        for i, j in zip(idx_t, idx_r)
        if top.atoms[i].name != ref.atoms[j].name
    ]

    if bad:
        raise SelectionError(
            "%s: %d atom-name mismatches between topology and reference pairing, "
            "first: %s" % (label, len(bad), bad[:5])
        )


def resolve(scfg, topology_file):
    """
    Resolve every selection in a SumdConfig against the topology (and the
    reference structure for rmsd). Returns ResolvedSelections with a
    `report` dict for run_summary.json.
    """
    top = load_structure(topology_file)
    rep = {"topology_file": topology_file, "n_atoms": len(top.atoms)}

    sel = ResolvedSelections(cv1=mask_indices(top, scfg.cv1_traj_selection, "cv1_traj_selection"))
    check_resnames(top, sel.cv1, scfg.cv1_expect, "cv1_traj_selection")
    rep["cv1"] = {"mask": scfg.cv1_traj_selection, "n": int(sel.cv1.size),
                  "residues": describe(top, sel.cv1)}

    if scfg.mass_weighted or scfg.cv1_type == "distance":
        sel.w_cv1 = masses(top, sel.cv1)

    if scfg.cv1_type == "rmsd":
        ref = load_structure(scfg.reference_structure)

        sel.align = mask_indices(top, scfg.align_traj_selection, "align_traj_selection")
        ref_align = mask_indices(ref, scfg.align_ref_selection, "align_ref_selection")
        ref_cv1 = mask_indices(ref, scfg.cv1_ref_selection, "cv1_ref_selection")

        _pair_names(top, ref, sel.align, ref_align, "alignment")
        _pair_names(top, ref, sel.cv1, ref_cv1, "cv1")

        xyz = np.asarray(ref.coordinates, dtype=float)
        sel.ref_align_xyz = xyz[ref_align]
        sel.ref_cv1_xyz = xyz[ref_cv1]

        if scfg.mass_weighted:
            sel.w_align = masses(top, sel.align)

        rep["align"] = {"mask": scfg.align_traj_selection, "n": int(sel.align.size)}
        rep["reference_structure"] = scfg.reference_structure
    else:
        sel.cv1_site = mask_indices(top, scfg.cv1_site_selection, "cv1_site_selection")
        if scfg.cv1_type == "distance":
            sel.w_site = masses(top, sel.cv1_site)
        rep["cv1_site"] = {"mask": scfg.cv1_site_selection, "n": int(sel.cv1_site.size),
                           "residues": describe(top, sel.cv1_site)}

    if scfg.has_cv2:
        sel.cv2_a = mask_indices(top, scfg.cv2_selection_a, "cv2_selection_a")
        sel.cv2_b = mask_indices(top, scfg.cv2_selection_b, "cv2_selection_b")
        check_resnames(top, sel.cv2_a, scfg.cv2_expect_a, "cv2_selection_a")
        check_resnames(top, sel.cv2_b, scfg.cv2_expect_b, "cv2_selection_b")

        rep["cv2_a"] = {"mask": scfg.cv2_selection_a, "n": int(sel.cv2_a.size),
                        "residues": describe(top, sel.cv2_a),
                        "atoms": [top.atoms[i].name for i in sel.cv2_a]}
        rep["cv2_b"] = {"mask": scfg.cv2_selection_b, "n": int(sel.cv2_b.size),
                        "residues": describe(top, sel.cv2_b),
                        "atoms": [top.atoms[i].name for i in sel.cv2_b]}

    sel.report = rep
    return sel
