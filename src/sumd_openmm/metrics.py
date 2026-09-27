"""Geometric collective variables evaluated on a single coordinate frame."""

import numpy as np

from . import cv


def _vector(a, b, box):
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return cv.minimum_image(d, box) if box is not None else d


def _point(X, indices, box, components=None):
    group = X[indices]
    if len(group) == 1:
        return group[0]
    if box is not None:
        if components:
            group = group.copy()
            anchor = np.mean(group[components[0]], axis=0)
            for component in components[1:]:
                centre = np.mean(group[component], axis=0)
                group[component] += cv.minimum_image(centre - anchor, box) - (centre - anchor)
        else:
            group = group[0] + cv.minimum_image(group - group[0], box)
    return np.mean(group, axis=0)


def _distance(metric, X, box):
    return float(np.linalg.norm(_vector(
        _point(X, metric.groups["a"], box, metric.components.get("a")),
        _point(X, metric.groups["b"], box, metric.components.get("b")), box)))


def _axis(metric, X, box):
    axis = metric.spec.axis.strip().lower()
    if axis in ("x", "y", "z"):
        unit = np.eye(3)["xyz".index(axis)]
    else:
        unit = np.asarray([float(v) for v in axis.replace(",", " ").split()], float)
        if unit.shape != (3,) or np.linalg.norm(unit) == 0:
            raise ValueError("%s_axis needs x, y, z or a nonzero 3-vector" % metric.spec.name)
        unit /= np.linalg.norm(unit)
    delta = _vector(_point(X, metric.groups["a"], box, metric.components.get("a")),
                    _point(X, metric.groups["b"], box, metric.components.get("b")), box)
    result = float(delta @ unit)
    return result if metric.spec.signed else abs(result)


def _mindist(metric, X, box):
    return cv.min_distance(X, metric.groups["a"], metric.groups["b"], box)


def _angle(metric, X, box):
    a, b, c = [X[metric.groups[k][0]] for k in "abc"]
    u, v = _vector(b, a, box), _vector(b, c, box)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v)))


def _dihedral(metric, X, box):
    a, b, c, d = [X[metric.groups[k][0]] for k in "abcd"]
    v1, v2, v3 = _vector(a, b, box), _vector(b, c, box), _vector(c, d, box)
    normal = v2 / np.linalg.norm(v2)
    p = -v1 + np.dot(v1, normal) * normal
    q = v3 - np.dot(v3, normal) * normal
    return float(np.degrees(np.arctan2(np.dot(np.cross(normal, p), q), np.dot(p, q))))


def _rmsd(metric, X, box):
    coords = X[metric.groups["a"]]
    if box is not None:
        coords = coords[0] + cv.minimum_image(coords - coords[0], box)
    rotation, p0, q0 = cv.kabsch(coords, metric.reference.get("a", coords))
    return cv.rmsd_nofit((coords - p0) @ rotation.T + q0,
                         metric.reference.get("a", coords))


def _rmsd_displacement(metric, X, box):
    return cv.fitted_displacement_rmsd(
        X, metric.groups["fit"], metric.groups["a"],
        metric.reference["fit"], metric.reference["a"], box=box)


REGISTRY = {"distance": _distance, "distance_axis": _axis, "mindist": _mindist,
            "angle": _angle, "dihedral": _dihedral, "rmsd": _rmsd,
            "rmsd_displacement": _rmsd_displacement}


class Metric:
    def __init__(self, spec, groups, reference=None, components=None):
        self.spec = spec
        self.groups = groups
        self.reference = reference or {}
        self.components = components or {}
        self._function = REGISTRY[spec.type]

    def value(self, X, box=None):
        return self._function(self, X, box)


class Evaluator:
    def __init__(self, metrics):
        self.metrics = metrics
        self.names = [metric.spec.name for metric in metrics]

    def __call__(self, X, box=None):
        return np.asarray([m.value(X, box) for m in self.metrics], dtype=float)


def unwrap(series, parent=None):
    """Unwrap a torsion using the parent's continuous branch."""
    values = np.asarray(series, dtype=float)
    if not len(values):
        return values.copy()
    out = np.empty_like(values)
    out[0] = values[0] if parent is None else parent + (values[0] - parent + 180) % 360 - 180
    for i in range(1, len(values)):
        out[i] = out[i - 1] + (values[i] - out[i - 1] + 180) % 360 - 180
    return out


def supervised_series(series, spec, parent=None, initial=None):
    """Return the quantity whose slope is supervised and its target."""
    values = np.asarray(series, dtype=float)
    target = spec.target if spec.target_delta is None else initial + spec.target_delta
    if spec.type == "dihedral" and spec.direction != "toward":
        values = unwrap(values, parent)
    if spec.direction == "toward":
        delta = abs(values - target)
        circular = delta % 360
        values = np.minimum(circular, 360 - circular) if spec.type == "dihedral" else delta
        target = spec.tolerance
    return values, target
