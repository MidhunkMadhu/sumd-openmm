"""Resolve selections once against the simulation topology."""

import os
import re

import numpy as np


class SelectionError(ValueError):
    pass


def load_structure(path, xyz=None):
    import parmed
    return parmed.load_file(path, xyz=xyz) if xyz is not None else parmed.load_file(path)


def describe(struct, idx):
    return ["%s%d" % (struct.residues[r].name, r + 1)
            for r in sorted({struct.atoms[int(i)].residue.idx for i in idx})]


def _indices(expr):
    out = []
    for token in re.split(r"[,\s]+", expr.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            a, b = int(match[1]), int(match[2])
            if b < a:
                raise SelectionError("invalid index range %s" % token)
            out.extend(range(a, b + 1))
        elif token.isdigit():
            out.append(int(token))
        else:
            raise SelectionError("invalid atom index %s" % token)
    return np.asarray(out, dtype=int)


def resolve_one(struct, expression, syntax="cpptraj", label="selection", base_dir="."):
    if not expression:
        raise SelectionError("%s is empty" % label)
    choice, value = syntax, expression
    for prefix in ("cpptraj:", "vmd:", "indices:", "file:"):
        if expression.startswith(prefix):
            choice, value = prefix[:-1], expression[len(prefix):]
            break
    try:
        if choice == "cpptraj":
            from parmed.amber import AmberMask
            idx = np.asarray(list(AmberMask(struct, value).Selected()), dtype=int)
        elif choice == "vmd":
            import MDAnalysis as mda
            idx = np.asarray(mda.Universe(struct).select_atoms(value).indices, dtype=int)
        elif choice == "indices":
            idx = _indices(value)
        elif choice == "file":
            with open(os.path.join(base_dir, value)) as fh:
                idx = _indices(" ".join(line.split("#", 1)[0] for line in fh))
        else:
            raise SelectionError("selection_syntax must be cpptraj or vmd")
    except SelectionError:
        raise
    except Exception as exc:
        raise SelectionError("%s: cannot parse %r: %s" % (label, expression, exc)) from exc
    if not len(idx) or np.any(idx < 0) or np.any(idx >= len(struct.atoms)):
        raise SelectionError("%s: %r selects no atoms or has indices outside 0..%d" %
                             (label, expression, len(struct.atoms) - 1))
    if len(np.unique(idx)) != len(idx):
        raise SelectionError("%s: selection contains duplicate atom indices" % label)
    return idx, choice


def _paired(top, ref, a, b, label):
    if len(a) != len(b):
        raise SelectionError("%s: %d topology atoms but %d reference atoms" % (label, len(a), len(b)))
    for i, j in zip(a, b):
        if top.atoms[i].name != ref.atoms[j].name:
            raise SelectionError("%s: atom pairing differs at %s / %s" %
                                 (label, top.atoms[i].name, ref.atoms[j].name))


def _molecule_ids(struct):
    parent = list(range(len(struct.atoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for bond in struct.bonds:
        a, b = find(bond.atom1.idx), find(bond.atom2.idx)
        parent[b] = a
    return [find(i) for i in range(len(parent))]


def resolve(config, topology_file, base_dir="."):
    from .metrics import Metric, Evaluator

    top = load_structure(topology_file)
    molecule_ids = _molecule_ids(top)
    compiled, reports = [], []
    for spec in config.metrics:
        groups, components, report = {}, {}, dict(number=spec.number, name=spec.name, type=spec.type,
                                  role=spec.role, direction=spec.direction,
                                  target=spec.target, tolerance=spec.tolerance,
                                  target_delta=spec.target_delta, selections={})
        for key in "abcd":
            expr = getattr(spec, key)
            if expr is None:
                continue
            idx, syntax = resolve_one(top, expr, config.selection_syntax,
                                      spec.name + "." + key, base_dir)
            expect = spec.expect.get(key)
            if expect and not {top.atoms[int(i)].residue.name.upper() for i in idx} <= {
                    part.strip().upper() for part in expect.split("/")}:
                raise SelectionError("%s.%s: expected %s, got %s" %
                                     (spec.name, key, expect, describe(top, idx)))
            if spec.type in ("angle", "dihedral") and len(idx) != 1:
                atoms = ["%d:%s" % (i, top.atoms[int(i)].name) for i in idx[:20]]
                raise SelectionError("%s.%s must select exactly one atom; matched %s" %
                                     (spec.name, key, ", ".join(atoms)))
            groups[key] = idx
            by_molecule = {}
            for local, atom in enumerate(idx):
                by_molecule.setdefault(molecule_ids[int(atom)], []).append(local)
            components[key] = [np.asarray(part, int) for part in by_molecule.values()]
            report["selections"][key] = dict(expression=expr, syntax=syntax,
                    count=len(idx), residues=describe(top, idx),
                    atoms=[top.atoms[int(i)].name for i in idx[:10]])
        reference = {}
        if spec.type in ("rmsd", "rmsd_displacement"):
            ref = load_structure(os.path.join(base_dir, spec.reference))
            ref_a, _ = resolve_one(ref, spec.a, config.selection_syntax,
                                    spec.name + ".reference_a", base_dir)
            _paired(top, ref, groups["a"], ref_a, spec.name + ".a")
            xyz = np.asarray(ref.coordinates, dtype=float)
            reference = {"a": xyz[ref_a]}
            if spec.type == "rmsd_displacement":
                fit, fit_syntax = resolve_one(top, spec.fit, config.selection_syntax,
                                              spec.name + ".fit", base_dir)
                ref_fit, _ = resolve_one(ref, spec.fit, config.selection_syntax,
                                          spec.name + ".reference_fit", base_dir)
                _paired(top, ref, fit, ref_fit, spec.name + ".fit")
                groups["fit"] = fit
                reference["fit"] = xyz[ref_fit]
                report["selections"]["fit"] = dict(expression=spec.fit, syntax=fit_syntax,
                         count=len(fit), residues=describe(top, fit),
                         atoms=[top.atoms[int(i)].name for i in fit[:10]])
        compiled.append(Metric(spec, groups, reference, components))
        reports.append(report)
    return Evaluator(compiled), reports
