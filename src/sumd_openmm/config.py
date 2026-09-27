"""Input configuration for SuMD and its collective variables."""

import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional


class ConfigError(ValueError):
    pass


def read_key_value_file(path):
    data = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
    return data


def _choice(value, key, choices):
    if value not in choices:
        raise ConfigError("%s must be one of %s" % (key, ", ".join(choices)))
    return value


def _bool(value, key):
    if str(value).lower() in ("yes", "true", "on", "1"):
        return True
    if str(value).lower() in ("no", "false", "off", "0"):
        return False
    raise ConfigError("%s must be yes or no" % key)


TYPES = {
    "distance": "ab", "distance_axis": "ab", "mindist": "ab",
    "angle": "abc", "dihedral": "abcd", "rmsd": "a",
    "rmsd_displacement": "a",
}
EXTRAS = {"name", "type", "role", "direction", "target", "tolerance",
          "target_delta", "bins", "weight", "mu", "sigma"}


@dataclass
class MetricSpec:
    number: int
    name: str
    type: str
    role: str = "monitor"
    direction: Optional[str] = None
    target: Optional[float] = None
    tolerance: Optional[float] = None
    target_delta: Optional[float] = None
    a: Optional[str] = None
    b: Optional[str] = None
    c: Optional[str] = None
    d: Optional[str] = None
    fit: Optional[str] = None
    reference: Optional[str] = None
    axis: Optional[str] = None
    signed: bool = True
    bins: Optional[str] = None
    weight: float = 1.0
    mu: Optional[float] = None
    sigma: Optional[float] = None
    expect: dict = field(default_factory=dict)


def parse_metrics(cfg):
    groups = {}
    for key, value in cfg.items():
        match = re.fullmatch(r"metric_(\d+)_(\w+)", key)
        if match:
            groups.setdefault(int(match[1]), {})[match[2]] = value
    if not groups:
        raise ConfigError("define at least one metric_N_type")
    missing = set(range(1, max(groups) + 1)) - groups.keys()
    if missing:
        raise ConfigError("missing metric_%d" % min(missing))
    specs, names = [], set()
    for n, data in sorted(groups.items()):
        label = "metric_%d_" % n
        kind = _choice(data.get("type"), label + "type", tuple(TYPES))
        valid = EXTRAS | set(TYPES[kind]) | {"expect_" + c for c in TYPES[kind]}
        if kind == "distance_axis":
            valid |= {"axis", "signed"}
        if kind in ("rmsd", "rmsd_displacement"):
            valid.add("reference")
        if kind == "rmsd_displacement":
            valid.add("fit")
        unknown = set(data) - valid
        if unknown:
            raise ConfigError("%s: unknown keys %s; valid keys: %s" %
                              (label[:-1], ", ".join(sorted(unknown)), ", ".join(sorted(valid))))
        required = set(TYPES[kind]) | ({"fit"} if kind == "rmsd_displacement" else set())
        if kind in ("rmsd", "rmsd_displacement"):
            required.add("reference")
        for key in sorted(required):
            if not data.get(key):
                raise ConfigError(label + key + " is required")
        name = data.get("name", "metric_%d" % n)
        if name in names or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ConfigError(label + "name must be unique and use letters, digits or underscores")
        names.add(name)
        role = _choice(data.get("role", "monitor"), label + "role", ("supervise", "stratify", "monitor"))
        direction = data.get("direction")
        if role == "supervise":
            direction = _choice(direction, label + "direction", ("decrease", "increase", "toward"))
            if not ("target" in data or "target_delta" in data):
                raise ConfigError(label + "target is required")
            if direction == "toward" and "tolerance" not in data:
                raise ConfigError(label + "tolerance is required for toward")
        elif direction is not None or "target" in data or "target_delta" in data:
            raise ConfigError(label + "direction and target require role = supervise")
        if "target_delta" in data and (kind != "dihedral" or direction == "toward" or "target" in data):
            raise ConfigError(label + "target_delta requires an increasing or decreasing dihedral")
        if role == "stratify" and not data.get("bins"):
            raise ConfigError(label + "bins is required for stratify")
        if kind == "distance_axis" and not data.get("axis"):
            raise ConfigError(label + "axis is required")
        def num(key, default=None):
            return float(data[key]) if key in data else default
        spec = MetricSpec(n, name, kind, role, direction, num("target"),
                          num("tolerance"), num("target_delta"),
                          **{key: data.get(key) for key in ("a", "b", "c", "d", "fit", "reference", "axis", "bins")},
                          signed=_bool(data.get("signed", "yes"), label + "signed"),
                          weight=num("weight", 1.0), mu=num("mu"), sigma=num("sigma"),
                          expect={key[7:]: value for key, value in data.items() if key.startswith("expect_")})
        if spec.tolerance is not None and spec.tolerance <= 0:
            raise ConfigError(label + "tolerance must be positive")
        if spec.sigma is not None and spec.sigma <= 0:
            raise ConfigError(label + "sigma must be positive")
        specs.append(spec)
    return specs


_CHOICES = {
    "selection_syntax": ("cpptraj", "vmd"), "supervision": ("single", "combined", "multistep"),
    "seeding": ("chain", "stratified"), "on_retry_exhaustion": ("accept_best", "step_back"),
    "retry_velocities": ("reassign", "keep"), "slope_points": ("all", "5"),
    "walker_score": ("slope", "smscore", "dmscore"), "parallel": ("serial", "mpi"),
    "mpi_mode": ("multi_node", "multi_gpu"), "method": ("cMD", "GaMD"),
}
_MD_KEYS = set("""force_field system_xml topology_file coordinate_file toppar_file
gmx_include dt temp fric_coeff pcouple p_type cons coulomb vdw r_on r_off platform
cuda_precision genvel continuation rewrap_coordinates reset_step_and_time lj_lrc
e14scale rest restraint_file restraint_k nstep ewald_Tol barostat_freq pressure""".split())


@dataclass
class SumdConfig:
    metrics: list = field(default_factory=list)
    selection_syntax: str = "cpptraj"
    supervision: str = "single"
    seeding: str = "chain"
    stages: list = field(default_factory=list)
    band_width: float = 1.0
    pool_per_cell: int = 20
    frontier_bands: int = 1
    window_ps: float = 20.0
    cv_sample_ps: float = 1.0
    max_cycles: int = 500
    max_retries_per_parent: int = 8
    on_retry_exhaustion: str = "accept_best"
    retry_velocities: str = "reassign"
    require_significant_slope: bool = False
    slope_points: str = "all"
    random_seed: Optional[int] = None
    walkers: int = 1
    walker_score: str = "slope"
    parallel: str = "serial"
    mpi_mode: str = "multi_node"
    gpu_devices: str = "auto"
    method: str = "cMD"
    output_dir: str = "sumd_run"
    dcd_stride: int = 1
    keep_rejected_dcd: bool = False
    restore_check: bool = True

    @classmethod
    def from_cfg(cls, cfg):
        replacements = {"colvarcut": "metric_N_target", "cv1_band_width": "band_width",
                        "gate_strata": "metric_N_bins", "gate_open_cutoff": "metric_N_bins",
                        "md_openmm_dir": "bundled MD_openmm setup"}
        for key in cfg:
            if key.startswith(("cv1_", "cv2_", "gate_")) or key in replacements:
                raise ConfigError("%s is removed; use %s" %
                                  (key, replacements.get(key, "metric_N_* definitions")))
        c = cls(metrics=parse_metrics(cfg))
        for key, options in _CHOICES.items():
            setattr(c, key, _choice(cfg.get(key, getattr(c, key)), key, options))
        for key in ("band_width", "window_ps", "cv_sample_ps"):
            setattr(c, key, float(cfg.get(key, getattr(c, key))))
        for key in ("pool_per_cell", "frontier_bands", "max_cycles", "max_retries_per_parent", "dcd_stride"):
            setattr(c, key, int(cfg.get(key, getattr(c, key))))
        for key in ("keep_rejected_dcd", "restore_check", "require_significant_slope"):
            if key in cfg:
                setattr(c, key, _bool(cfg[key], key))
        for key in ("output_dir", "gpu_devices"):
            setattr(c, key, cfg.get(key, getattr(c, key)))
        if cfg.get("random_seed"):
            c.random_seed = int(cfg["random_seed"])
        c.walkers = 0 if cfg.get("walkers") == "auto" else int(cfg.get("walkers", 1))
        try:
            from .parallel import parse_devices
            parse_devices(c.gpu_devices)
        except ValueError as exc:
            raise ConfigError("gpu_devices must be auto or integer device IDs") from exc
        if "stages" in cfg:
            c.stages = [int(x) for x in cfg["stages"].replace(",", " ").split()]
        for key in cfg:
            if key not in c.__dataclass_fields__ and key not in _MD_KEYS and not re.fullmatch(r"metric_\d+_\w+", key):
                warnings.warn("unknown input key %s" % key, stacklevel=2)
        c.validate()
        return c

    def validate(self):
        if self.method == "GaMD":
            raise ConfigError("method = GaMD requires a GaMD integrator")
        supervised = [m for m in self.metrics if m.role == "supervise"]
        stratified = [m for m in self.metrics if m.role == "stratify"]
        if self.supervision == "single" and len(supervised) != 1:
            raise ConfigError("supervision = single requires exactly one supervised metric")
        if self.supervision != "single" and len(supervised) < 2:
            raise ConfigError("supervision = %s needs at least two supervised metrics" % self.supervision)
        if self.supervision == "multistep" and (len(self.stages) != len(supervised) or
                                                set(self.stages) != {m.number for m in supervised}):
            raise ConfigError("stages must list each supervised metric once")
        if self.seeding == "stratified" and not stratified:
            raise ConfigError("seeding = stratified needs a metric with role = stratify")
        if self.walker_score == "dmscore" and len(supervised) < 2:
            raise ConfigError("walker_score = dmscore needs two supervised metrics")
        if self.walker_score == "smscore" and len(supervised) != 1:
            raise ConfigError("walker_score = smscore needs exactly one supervised metric")
        if self.walkers < 0:
            raise ConfigError("walkers must be positive or auto")
        if self.walkers == 0 and self.parallel == "serial":
            self.walkers = 1
        if self.walkers == 1 and self.walker_score != "slope":
            raise ConfigError("score modes need more than one walker")
        if self.seeding == "stratified" and self.walker_score == "dmscore":
            raise ConfigError("stratified seeding with dmscore is incompatible")
        n = self.window_ps / self.cv_sample_ps
        if self.window_ps <= 0 or self.cv_sample_ps <= 0 or abs(n - round(n)) > 1e-6 or n < 3:
            raise ConfigError("window_ps / cv_sample_ps must be an integer >= 3")
        if self.slope_points == "5" and n < 8:
            raise ConfigError("slope_points = 5 needs at least 8 samples")
        if min(self.band_width, self.pool_per_cell, self.dcd_stride) <= 0 or self.max_retries_per_parent < 0:
            raise ConfigError("band_width, pool_per_cell, dcd_stride must be positive; retries nonnegative")
        if round(n) % self.dcd_stride:
            raise ConfigError("dcd_stride must divide samples per window so DCD endpoints match saved states")

    def resolve_walkers(self, n_ranks):
        if self.walkers == 0:
            self.walkers = n_ranks
            self.validate()
        return ["%d MPI ranks idle" % (n_ranks - self.walkers)] if n_ranks > self.walkers else []

    @property
    def samples_per_window(self):
        return int(round(self.window_ps / self.cv_sample_ps))

    def as_dict(self):
        return asdict(self)
