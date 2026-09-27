"""
config.py

SuMD keys added to the MD_openmm .inp format. Pure python, no OpenMM.

Everything MD-related (dt, temp, fric_coeff, barostat, cutoffs, constraints,
platform, coordinate_file, topology_file, ...) is NOT parsed here. It is read
by MD_openmm's own build_inputs() through omm_setup.py, from the same file.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


class ConfigError(ValueError):
    pass


def read_key_value_file(filename):
    """
    Same grammar as openmm_production.read_key_value_file ("key = value",
    '#' comments). Used only before MD_openmm is located; the driver re-reads
    the file with MD_openmm's own parser once it is imported.
    """
    data = {}

    with open(filename) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()

            if "=" not in line:
                continue

            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()

    return data


def _none(v):
    if v is None:
        return None

    v = str(v).strip()
    return None if v.lower() in ("", "none", "no", "null", "nil", "-") else v


def _bool(cfg, key, default):
    if key not in cfg:
        return default

    v = str(cfg[key]).strip().lower()

    if v in ("yes", "true", "on", "1"):
        return True
    if v in ("no", "false", "off", "0"):
        return False

    raise ConfigError("invalid yes/no for %s: %s" % (key, cfg[key]))


def _floats(cfg, key, default, n):
    if key not in cfg or _none(cfg[key]) is None:
        return default

    vals = [float(x) for x in str(cfg[key]).replace(",", " ").split()]

    if len(vals) != n:
        raise ConfigError("%s needs %d numbers, got %s" % (key, n, cfg[key]))

    return vals


def _choice(cfg, key, default, allowed):
    v = str(cfg.get(key, default)).strip()

    if v not in allowed:
        raise ConfigError("%s must be one of %s, got %s" % (key, "|".join(allowed), v))

    return v


@dataclass
class SumdConfig:
    # supervision
    supervision: str = "single"
    cv1_type: str = "rmsd"
    cv1_traj_selection: Optional[str] = None
    cv1_ref_selection: Optional[str] = None
    cv1_site_selection: Optional[str] = None
    cv1_expect: Optional[str] = None
    align_traj_selection: Optional[str] = None
    align_ref_selection: Optional[str] = None
    reference_structure: Optional[str] = None
    mass_weighted: bool = False
    colvarcut: float = 2.0

    cv2_selection_a: Optional[str] = None
    cv2_selection_b: Optional[str] = None
    cv2_expect_a: Optional[str] = None
    cv2_expect_b: Optional[str] = None
    cv2_direction: str = "increase"
    gate_open_cutoff: float = 4.70
    gate_strata: int = 2
    cv1_band_width: float = 1.0
    pool_per_cell: int = 20
    frontier_bands: int = 1

    joint_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    joint_mu: Optional[List[float]] = None
    joint_sigma: Optional[List[float]] = None

    # cycle control
    window_ps: float = 20.0
    cv_sample_ps: float = 1.0
    max_cycles: int = 500
    max_retries_per_parent: int = 8
    on_retry_exhaustion: str = "accept_best"
    retry_velocities: str = "reassign"
    require_significant_slope: bool = False
    slope_points: str = "all"
    random_seed: Optional[int] = None

    # multiple walkers (mwSuMD-style batches); 0 = auto (one per MPI rank)
    walkers: int = 1
    walker_score: str = "slope"

    # parallel execution
    parallel: str = "serial"
    mpi_mode: str = "multi_node"
    gpu_devices: str = "auto"

    # engine / output
    method: str = "cMD"
    output_dir: str = "sumd_run"
    md_openmm_dir: Optional[str] = None
    dcd_stride: int = 1
    keep_rejected_dcd: bool = False
    restore_check: bool = True

    @classmethod
    def from_cfg(cls, cfg):
        c = cls()

        c.supervision = _choice(cfg, "supervision", c.supervision, ("single", "stratified", "joint"))
        c.cv1_type = _choice(cfg, "cv1_type", c.cv1_type, ("rmsd", "distance", "mindist"))

        for k in ("cv1_traj_selection", "cv1_ref_selection", "cv1_site_selection", "cv1_expect",
                  "align_traj_selection", "align_ref_selection", "reference_structure",
                  "cv2_selection_a", "cv2_selection_b", "cv2_expect_a", "cv2_expect_b",
                  "md_openmm_dir"):
            setattr(c, k, _none(cfg.get(k)))

        c.mass_weighted = _bool(cfg, "mass_weighted", c.mass_weighted)
        c.colvarcut = float(cfg.get("colvarcut", c.colvarcut))
        c.cv2_direction = _choice(cfg, "cv2_direction", c.cv2_direction, ("increase", "decrease"))
        c.gate_open_cutoff = float(cfg.get("gate_open_cutoff", c.gate_open_cutoff))
        c.gate_strata = int(cfg.get("gate_strata", c.gate_strata))
        c.cv1_band_width = float(cfg.get("cv1_band_width", c.cv1_band_width))
        c.pool_per_cell = int(cfg.get("pool_per_cell", c.pool_per_cell))
        c.frontier_bands = int(cfg.get("frontier_bands", c.frontier_bands))

        c.joint_weights = _floats(cfg, "joint_weights", c.joint_weights, 2)
        c.joint_mu = _floats(cfg, "joint_mu", None, 2)
        c.joint_sigma = _floats(cfg, "joint_sigma", None, 2)

        c.window_ps = float(cfg.get("window_ps", c.window_ps))
        c.cv_sample_ps = float(cfg.get("cv_sample_ps", c.cv_sample_ps))
        c.max_cycles = int(cfg.get("max_cycles", c.max_cycles))
        c.max_retries_per_parent = int(cfg.get("max_retries_per_parent", c.max_retries_per_parent))
        c.on_retry_exhaustion = _choice(cfg, "on_retry_exhaustion", c.on_retry_exhaustion,
                                        ("accept_best", "step_back"))
        c.retry_velocities = _choice(cfg, "retry_velocities", c.retry_velocities, ("reassign", "keep"))
        c.require_significant_slope = _bool(cfg, "require_significant_slope", c.require_significant_slope)
        c.slope_points = _choice(cfg, "slope_points", c.slope_points, ("all", "5"))

        seed = _none(cfg.get("random_seed"))
        c.random_seed = int(seed) if seed is not None else None

        w = str(cfg.get("walkers", c.walkers)).strip().lower()
        c.walkers = 0 if w == "auto" else int(w)
        c.walker_score = _choice(cfg, "walker_score", c.walker_score, ("slope", "smscore", "dmscore"))

        c.parallel = _choice(cfg, "parallel", c.parallel, ("serial", "mpi"))
        c.mpi_mode = _choice(cfg, "mpi_mode", c.mpi_mode, ("multi_node", "multi_gpu"))
        c.gpu_devices = str(cfg.get("gpu_devices", c.gpu_devices)).strip()
        try:
            from .parallel import parse_devices
            parse_devices(c.gpu_devices)
        except ValueError:
            raise ConfigError("gpu_devices must be auto or a list of integers, got %s" % c.gpu_devices)

        c.method = _choice(cfg, "method", c.method, ("cMD", "GaMD"))
        c.output_dir = str(cfg.get("output_dir", c.output_dir))
        c.dcd_stride = int(cfg.get("dcd_stride", c.dcd_stride))
        c.keep_rejected_dcd = _bool(cfg, "keep_rejected_dcd", c.keep_rejected_dcd)
        c.restore_check = _bool(cfg, "restore_check", c.restore_check)

        c.validate()
        return c

    def validate(self):
        if self.method == "GaMD":
            raise ConfigError(
                "method = GaMD is not supported. OpenMM has no native GaMD; SuGaMD needs a "
                "GaMD integrator/plugin and is scoped separately. Refusing to run plain cMD "
                "under a GaMD label."
            )

        if self.cv1_traj_selection is None:
            raise ConfigError("cv1_traj_selection is required")

        if self.cv1_type == "rmsd":
            for k in ("cv1_ref_selection", "align_traj_selection", "align_ref_selection",
                      "reference_structure"):
                if getattr(self, k) is None:
                    raise ConfigError("%s is required for cv1_type = rmsd" % k)
        elif self.cv1_site_selection is None:
            raise ConfigError("cv1_site_selection is required for cv1_type = %s" % self.cv1_type)

        has_cv2 = self.cv2_selection_a is not None and self.cv2_selection_b is not None
        if (self.cv2_selection_a is None) != (self.cv2_selection_b is None):
            raise ConfigError("give both cv2_selection_a and cv2_selection_b, or neither")

        uses_cv2 = self.supervision in ("stratified", "joint") or self.walker_score == "dmscore"
        if uses_cv2 and not has_cv2:
            raise ConfigError("supervision = %s / walker_score = %s needs cv2_selection_a/b"
                              % (self.supervision, self.walker_score))

        if self.walkers < 0:
            raise ConfigError("walkers must be >= 1 or auto")

        if self.walkers == 0 and self.parallel != "mpi":
            self.walkers = 1        # auto in serial mode: classic single-walker SuMD

        if self.walkers == 1 and self.walker_score != "slope":
            raise ConfigError(
                "walker_score = %s with walkers = 1 would accept every window (plain MD). "
                "Use walker_score = slope, or walkers > 1." % self.walker_score
            )

        if self.supervision == "joint" and self.walker_score != "slope":
            raise ConfigError("supervision = joint is defined for walker_score = slope only")

        if self.supervision == "stratified" and self.walker_score == "dmscore":
            raise ConfigError(
                "stratified + dmscore is contradictory: stratified keeps the gate OUT of "
                "acceptance, dmscore puts it back in. Pick one."
            )

        if self.window_ps <= 0 or self.cv_sample_ps <= 0:
            raise ConfigError("window_ps and cv_sample_ps must be > 0")

        n = self.window_ps / self.cv_sample_ps
        if abs(n - round(n)) > 1e-6 or round(n) < 3:
            raise ConfigError("window_ps / cv_sample_ps must be an integer >= 3, got %g" % n)

        if self.slope_points == "5" and round(n) < 8:
            raise ConfigError("slope_points = 5 needs >= 8 samples per window")

        if self.max_retries_per_parent < 0:
            raise ConfigError("max_retries_per_parent must be >= 0")

    def resolve_walkers(self, n_ranks):
        """walkers = auto -> one walker per rank. Returns warnings."""
        if self.walkers == 0:
            self.walkers = n_ranks
            self.validate()

        warn = []
        if self.parallel == "mpi" and n_ranks > 1:
            if self.walkers < n_ranks:
                warn.append("walkers = %d < %d MPI ranks: %d ranks will sit idle every cycle"
                            % (self.walkers, n_ranks, n_ranks - self.walkers))
            elif self.walkers % n_ranks:
                warn.append("walkers = %d is not a multiple of %d ranks: some ranks run one more "
                            "window per cycle than others, and the rest wait" % (self.walkers, n_ranks))
        return warn

    @property
    def samples_per_window(self):
        return int(round(self.window_ps / self.cv_sample_ps))

    @property
    def has_cv2(self):
        return self.cv2_selection_a is not None

    @property
    def cv2_in_acceptance(self):
        return self.supervision == "joint" or self.walker_score == "dmscore"

    def as_dict(self):
        return asdict(self)
