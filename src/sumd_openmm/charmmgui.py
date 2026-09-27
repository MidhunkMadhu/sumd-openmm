"""
charmmgui.py

Read the equilibration protocol of a CHARMM-GUI Amber or GROMACS input
folder into engine-neutral stages, so OpenMM can run it.

    amber/    step5_input.parm7, step5_input.rst7, step6.0_minimization.mdin,
              step6.N_equilibration.mdin, step7_production.mdin,
              dihe.restraint, README
    gromacs/  topol.top, toppar/, step5_input.gro, step6.0_minimization.mdp,
              step6.N_equilibration.mdp, step7_production.mdp

Each stage keeps what its input file says: minimization or dynamics, steps,
timestep, temperature, friction, velocity generation, pressure coupling,
and the positional and dihedral restraints with that stage's force
constants. Pure Python and numpy; nothing here imports OpenMM.

Energies of restraints follow each engine's own definition, converted to
kJ/mol, nm and rad:

    Amber ntr=1 group input     E = w |r - r0|^2          (w in kcal/mol/A^2)
    Amber &rst torsion          flat bottom r2..r3, harmonic rk2/rk3 to r1/r4,
                                linear beyond              (kcal/mol/rad^2)
    GROMACS position_restraints E = 1/2 sum_d k_d (r_d - r0_d)^2
    GROMACS dihedral_restraints E = 1/2 k max(0, |phi - phi0| - dphi)^2
"""

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

KCAL = 4.184


class CharmmGuiError(ValueError):
    pass


@dataclass
class PositionRestraints:
    """E = sum over atoms of kx dx^2 + ky dy^2 + kz dz^2 (kJ/mol/nm^2)."""
    label: str
    atoms: np.ndarray
    k: np.ndarray                      # (n, 3)


@dataclass
class TorsionRestraints:
    """
    Flat-bottom torsion restraints. For each: atoms (4,), centre c, and
    offsets from c (radians) l1 <= lo <= hi <= u4, force constants klo, khi
    (kJ/mol/rad^2) in E = k (d - edge)^2, linear beyond l1 and u4.
    """
    label: str
    atoms: np.ndarray                  # (n, 4)
    params: np.ndarray                 # (n, 7): c, l1, lo, hi, u4, klo, khi


@dataclass
class Stage:
    name: str
    minimize: bool = False
    nsteps: int = 0
    dt_ps: float = 0.002
    temperature: float = 300.0
    friction: float = 1.0
    generate_velocities: bool = False
    velocity_temperature: Optional[float] = None
    pressure_bar: Optional[float] = None
    barostat: Optional[str] = None     # isotropic, membrane, anisotropic
    surface_tension: float = 0.0       # bar nm
    tolerance: float = 10.0            # minimization, kJ/mol/nm
    report_steps: int = 1000
    dcd_steps: int = 0
    positions: list = field(default_factory=list)
    torsions: list = field(default_factory=list)

    def describe(self):
        if self.minimize:
            kind = "minimization, at most %d steps" % self.nsteps
        else:
            kind = "%s %.4g K, %.4g ps at dt %.4g fs" % (
                "NVT" if self.barostat is None else "NPT (%s)" % self.barostat,
                self.temperature, self.nsteps * self.dt_ps, self.dt_ps * 1000)
        rest = ["%s %d atoms k=%.4g" % (r.label, len(r.atoms), float(np.max(r.k)))
                for r in self.positions]
        rest += ["%s %d torsions k=%.4g" % (t.label, len(t.atoms), float(np.max(t.params[:, 5])))
                 for t in self.torsions]
        return "%s: %s; restraints: %s" % (self.name, kind, "; ".join(rest) or "none")


@dataclass
class Protocol:
    format: str                        # AMBER or GROMACS
    folder: str
    topology: str
    coordinates: str                   # initial structure (and restraint reference)
    gmx_include: Optional[str]
    cutoff_nm: float
    switch_nm: Optional[float]         # force switching onset, None for a plain cutoff
    dispersion_correction: bool
    stages: list
    production: dict                   # MD_openmm keys implied by step7_production
    notes: list = field(default_factory=list)


# ============================================================
# folder detection
# ============================================================

def find_folder(path, fmt=None):
    """(format, folder) for a CHARMM-GUI format folder or the download above it."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise CharmmGuiError("charmm_gui_dir %s is not a folder" % path)
    fmt = (fmt or "auto").lower()
    options = {"amber": os.path.join(path, "amber"), "gromacs": os.path.join(path, "gromacs")}
    if glob.glob(os.path.join(path, "*.parm7")):
        found = {"amber": path}
    elif os.path.exists(os.path.join(path, "topol.top")):
        found = {"gromacs": path}
    else:
        found = {k: v for k, v in options.items() if os.path.isdir(v)}
    if fmt != "auto":
        if fmt not in found:
            raise CharmmGuiError("no CHARMM-GUI %s inputs in %s" % (fmt, path))
        return fmt.upper(), found[fmt]
    if len(found) != 1:
        raise CharmmGuiError("%s holds %s; set charmm_gui_format = amber or gromacs" %
                             (path, " and ".join(sorted(found)) or "no Amber or GROMACS inputs"))
    fmt, folder = next(iter(found.items()))
    return fmt.upper(), folder


def read(path, fmt=None):
    fmt, folder = find_folder(path, fmt)
    return read_amber(folder) if fmt == "AMBER" else read_gromacs(folder)


def _stage_files(folder, ext):
    mini = os.path.join(folder, "step6.0_minimization." + ext)
    equil = sorted(glob.glob(os.path.join(folder, "step6.*_equilibration." + ext)),
                   key=lambda p: int(re.search(r"step6\.(\d+)_", p)[1]))
    files = ([mini] if os.path.exists(mini) else []) + equil
    if not equil:
        raise CharmmGuiError("no step6.N_equilibration.%s files in %s" % (ext, folder))
    return files


def _name(path):
    return os.path.basename(path).rsplit(".", 1)[0]


# ============================================================
# Amber
# ============================================================

def parse_mdin(text):
    """Namelist values (lower-case keys) and the lines after the namelists."""
    values, rest = {}, []
    in_list = seen = False
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not in_list and re.match(r"&(cntrl|ewald|wt|qmmm|pb)\b", line, re.I):
            in_list = seen = True
            line = line[line.index(" "):] if " " in line else ""
        if in_list:
            ended = bool(re.search(r"(^|\s|,)(/|&end)\s*$", line, re.I))
            body = re.sub(r"(/|&end)\s*$", "", line, flags=re.I)
            for key, value in re.findall(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,\s]+)", body):
                values.setdefault(key.lower(), value.strip("'\""))
            if ended:
                in_list = False
        elif seen:
            rest.append(raw.rstrip())
    return values, rest


def _num(values, key, default):
    return float(values[key]) if key in values else default


def parse_group_input(lines):
    """Amber ntr=1 GROUP input -> [(title, weight, commands)]."""
    lines = [l.strip() for l in lines if l.strip() and "=" not in l and l.strip().lower() != "&end"]
    groups, i = [], 0
    while i < len(lines) and lines[i].upper() != "END":
        title, weight = lines[i], float(lines[i + 1])
        i += 2
        ranges, finds, finding = [], [], False
        while lines[i].upper() != "END":
            token = lines[i].split()
            head = token[0].upper()
            if head == "FIND":
                finding = True
            elif head == "SEARCH":
                finding = False
            elif finding:
                finds.append([t.upper() for t in token[:4]])
            elif head in ("RES", "ATOM"):
                numbers = [int(x) for x in token[1:]]
                if len(numbers) % 2:
                    raise CharmmGuiError("%s needs pairs of numbers: %s" % (head, lines[i]))
                ranges += [(head, a, b) for a, b in zip(numbers[::2], numbers[1::2])]
            else:
                raise CharmmGuiError("unsupported Amber group input line: %s" % lines[i])
            i += 1
        groups.append((title, weight, ranges, finds))
        i += 1
    return groups


def _group_atoms(ranges, finds, residues, names, resnames, types):
    """0-based atom indices of one group; residues are 1-based per atom."""
    atom_number = np.arange(1, len(names) + 1)
    chosen = np.zeros(len(names), bool)
    for kind, a, b in ranges:
        ref = residues if kind == "RES" else atom_number
        chosen |= (ref >= a) & (ref <= b)
    if finds:
        match = np.zeros(len(names), bool)
        for name, typ, _symbol, resname in finds:
            m = np.ones(len(names), bool)
            if name != "*":
                m &= names == name
            if typ != "*":
                m &= types == typ
            if resname != "*":
                m &= resnames == resname
            match |= m
        chosen &= match
    return np.flatnonzero(chosen)


def parse_rst(text, fc):
    """Amber &rst torsion restraints with FC substituted -> TorsionRestraints."""
    atoms, params = [], []
    for block in re.findall(r"&rst(.*?)(?:/|&end)", text, re.S | re.I):
        parts = re.split(r"(\w+)\s*=", block)
        v = {key.lower(): value.strip().strip(",").strip() for key, value in zip(parts[1::2], parts[2::2])}
        iat = [int(x) for x in re.findall(r"-?\d+", v["iat"])]
        if len(iat) != 4:
            raise CharmmGuiError("only torsion &rst restraints are supported: iat=%s" % v["iat"])
        value = lambda key: float(v[key].replace("FC", repr(float(fc))))
        r1, r2, r3, r4 = (np.radians(value(k)) for k in ("r1", "r2", "r3", "r4"))
        c = 0.5 * (r2 + r3)
        atoms.append(np.asarray(iat) - 1)
        params.append([c, r1 - c, r2 - c, r3 - c, r4 - c,
                       value("rk2") * KCAL, value("rk3") * KCAL])
    return TorsionRestraints("dihedral", np.asarray(atoms, int).reshape(-1, 4),
                             np.asarray(params, float).reshape(-1, 7))


def _readme_force_constants(folder):
    """Per-stage dihedral force constants from the csh driver in README."""
    try:
        text = open(os.path.join(folder, "README")).read()
    except OSError:
        return None, None
    mini = re.search(r'sed -e "s/FC/([\d.]+)/g" dihe\.restraint > \$\{mini_prefix\}', text)
    equil = re.search(r"set fc = \{([^}]*)\}", text)
    return (float(mini[1]) if mini else None,
            [float(x.strip(" '\"")) for x in equil[1].split(",")] if equil else None)


def read_amber(folder):
    import parmed

    topology = os.path.join(folder, "step5_input.parm7")
    coordinates = os.path.join(folder, "step5_input.rst7")
    for path in (topology, coordinates):
        if not os.path.exists(path):
            raise CharmmGuiError("missing %s" % path)
    struct = parmed.load_file(topology)
    names = np.array([a.name.upper() for a in struct.atoms])
    resnames = np.array([a.residue.name.upper() for a in struct.atoms])
    types = np.array([str(a.type).upper() for a in struct.atoms])
    residues = np.array([a.residue.idx + 1 for a in struct.atoms])
    mini_fc, equil_fc = _readme_force_constants(folder)
    notes = []
    dihe = os.path.join(folder, "dihe.restraint")
    if os.path.exists(dihe) and equil_fc is None:
        mini_fc, equil_fc = 250.0, [250.0, 100.0, 50.0, 50.0, 25.0]
        notes.append("README has no dihedral force constants; using CHARMM-GUI's 250, 100, 50, 50, 25")

    stages, first = [], None
    for path in _stage_files(folder, "mdin"):
        number = int(re.search(r"step6\.(\d+)_", path)[1])
        values, rest = parse_mdin(open(path).read())
        first = first or values
        stage = Stage(_name(path))
        stage.minimize = int(_num(values, "imin", 0)) == 1
        if stage.minimize:
            stage.nsteps = int(_num(values, "maxcyc", 1))
        else:
            stage.nsteps = int(_num(values, "nstlim", 1))
            stage.dt_ps = _num(values, "dt", 0.001)
            stage.temperature = _num(values, "temp0", 300.0)
            stage.friction = _num(values, "gamma_ln", 1.0)
            restart = int(_num(values, "irest", 0)) == 1 or int(_num(values, "ntx", 1)) >= 4
            stage.generate_velocities = not restart
            stage.velocity_temperature = _num(values, "tempi", stage.temperature)
            ntp = int(_num(values, "ntp", 0))
            if ntp:
                stage.pressure_bar = _num(values, "pres0", 1.0)
                stage.barostat = ("membrane" if ntp == 3 and "csurften" in values else
                                  {1: "isotropic", 2: "anisotropic", 3: "membrane"}[ntp])
                stage.surface_tension = _num(values, "gamma_ten", 0.0) * 10.0
            stage.report_steps = int(_num(values, "ntpr", 1000))
            stage.dcd_steps = int(_num(values, "ntwx", 0))
        if int(_num(values, "ntr", 0)) == 1 and "restraintmask" in values:
            from parmed.amber import AmberMask
            atoms = np.asarray(list(AmberMask(struct, values["restraintmask"]).Selected()), int)
            weight = _num(values, "restraint_wt", 0.0)
            if len(atoms) and weight > 0:
                stage.positions.append(PositionRestraints(
                    values["restraintmask"], atoms, np.full((len(atoms), 3), weight * KCAL * 100)))
        elif int(_num(values, "ntr", 0)) == 1:
            for title, weight, ranges, finds in parse_group_input(rest):
                atoms = _group_atoms(ranges, finds, residues, names, resnames, types)
                if len(atoms) and weight > 0:
                    stage.positions.append(PositionRestraints(
                        title, atoms, np.full((len(atoms), 3), weight * KCAL * 100)))
        if int(_num(values, "nmropt", 0)) == 1 and os.path.exists(dihe):
            # README: minimization uses its own FC, step6.N the N-th of `set fc`
            fc = mini_fc if stage.minimize else (
                equil_fc[number - 1] if equil_fc and 0 < number <= len(equil_fc) else None)
            if fc:
                restraint = parse_rst(open(dihe).read(), fc)
                if len(restraint.atoms):
                    stage.torsions.append(restraint)
        stages.append(stage)

    cutoff = _num(first, "cut", 8.0) / 10
    switch = _num(first, "fswitch", -1.0)
    production = {}
    prod_path = os.path.join(folder, "step7_production.mdin")
    if os.path.exists(prod_path):
        values, _ = parse_mdin(open(prod_path).read())
        ntp = int(_num(values, "ntp", 0))
        production = dict(temp=_num(values, "temp0", 300.0), fric_coeff=_num(values, "gamma_ln", 1.0),
                          dt=_num(values, "dt", 0.002),
                          pcouple="yes" if ntp else "no")
        if ntp:
            production.update(p_type={1: "isotropic", 2: "anisotropic", 3: "membrane"}[ntp],
                              p_ref=_num(values, "pres0", 1.0))
    return Protocol("AMBER", folder, topology, coordinates, None, cutoff,
                    switch / 10 if switch > 0 else None,
                    int(_num(first, "vdwmeth", 1)) == 1, stages, production, notes)


# ============================================================
# GROMACS
# ============================================================

def parse_mdp(path):
    values = {}
    for raw in open(path):
        line = raw.split(";", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower().replace("_", "-")] = value.strip()
    return values


def mdp_defines(values):
    defines = {}
    for token in values.get("define", "").split():
        if token.startswith("-D"):
            name, _, value = token[2:].partition("=")
            defines[name] = value if value else True
    return defines


class _TopologyReader:
    """Just enough of the GROMACS preprocessor to find restraint sections."""

    def __init__(self, top, include_dirs, defines):
        self.defines = dict(defines)
        self.include_dirs = include_dirs
        self.molecules = []            # [(name, count)]
        self.types = {}                # name -> dict(natoms, posres, dihres)
        self._current = None
        self._section = None
        self._read(top)

    def _resolve(self, name, here):
        for folder in [here] + self.include_dirs:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
        raise CharmmGuiError("cannot find included file %s" % name)

    def _value(self, token):
        while token in self.defines and self.defines[token] is not True:
            token = self.defines[token]
        try:
            return float(token)
        except ValueError:
            raise CharmmGuiError("restraint parameter %s is not defined by the stage's define line"
                                 % token) from None

    def _read(self, path):
        active = []
        here = os.path.dirname(os.path.abspath(path))
        for raw in open(path):
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                words = line.split()
                word = words[0]
                if word == "#ifdef":
                    active.append(words[1] in self.defines)
                elif word == "#ifndef":
                    active.append(words[1] not in self.defines)
                elif word == "#else":
                    active[-1] = not active[-1]
                elif word == "#endif":
                    active.pop()
                elif all(active):
                    if word == "#include":
                        self._read(self._resolve(words[1].strip('"<>'), here))
                    elif word == "#define":
                        self.defines[words[1]] = " ".join(words[2:]) if len(words) > 2 else True
                    elif word == "#undef":
                        self.defines.pop(words[1], None)
                continue
            if not all(active):
                continue
            section = re.fullmatch(r"\[\s*(\w+)\s*\]", line)
            if section:
                self._section = section[1].lower()
                continue
            fields = line.split()
            if self._section == "moleculetype":
                self._current = fields[0]
                self.types[self._current] = dict(natoms=0, posres=[], dihres=[])
            elif self._section == "atoms":
                self.types[self._current]["natoms"] += 1
            elif self._section == "molecules":
                self.molecules.append((fields[0], int(fields[1])))
            elif self._section == "position_restraints":
                if int(fields[1]) != 1:
                    raise CharmmGuiError("only harmonic position restraints (funct 1) are supported")
                self.types[self._current]["posres"].append(
                    (int(fields[0]) - 1, [self._value(x) for x in fields[2:5]]))
            elif self._section == "dihedral_restraints":
                if int(fields[4]) != 1 or len(fields) < 8:
                    raise CharmmGuiError("unsupported dihedral restraint: %s" % line)
                self.types[self._current]["dihres"].append(
                    ([int(x) - 1 for x in fields[:4]], [self._value(x) for x in fields[5:8]]))

    def restraints(self):
        pos_atoms, pos_k, tor_atoms, tor_params = [], [], [], []
        offset = 0
        for name, count in self.molecules:
            mol = self.types[name]
            for copy in range(count):
                base = offset + copy * mol["natoms"]
                for i, k in mol["posres"]:
                    pos_atoms.append(base + i)
                    pos_k.append(k)
                for quad, (phi, dphi, k) in mol["dihres"]:
                    c, d = np.radians(phi), np.radians(dphi)
                    tor_atoms.append([base + i for i in quad])
                    tor_params.append([c, -4 * np.pi, -d, d, 4 * np.pi, 0.5 * k, 0.5 * k])
            offset += count * mol["natoms"]
        pos_k = 0.5 * np.asarray(pos_k, float).reshape(-1, 3)
        keep = np.any(pos_k > 0, axis=1)
        positions = ([PositionRestraints("position", np.asarray(pos_atoms, int)[keep], pos_k[keep])]
                     if keep.any() else [])
        tor_params = np.asarray(tor_params, float).reshape(-1, 7)
        keep = tor_params[:, 5] > 0
        torsions = ([TorsionRestraints("dihedral", np.asarray(tor_atoms, int)[keep], tor_params[keep])]
                    if keep.any() else [])
        return positions, torsions, offset


def read_gromacs(folder):
    topology = os.path.join(folder, "topol.top")
    coordinates = os.path.join(folder, "step5_input.gro")
    for path in (topology, coordinates):
        if not os.path.exists(path):
            raise CharmmGuiError("missing %s" % path)
    stages, first = [], None
    natoms = int(open(coordinates).readlines()[1])
    for path in _stage_files(folder, "mdp"):
        values = parse_mdp(path)
        first = first or values
        stage = Stage(_name(path))
        stage.minimize = values.get("integrator", "md") in ("steep", "cg", "l-bfgs")
        stage.nsteps = int(float(values.get("nsteps", 0)))
        if not stage.minimize:
            stage.dt_ps = float(values.get("dt", 0.001))
            stage.temperature = float(values.get("ref-t", "300").split()[0])
            stage.friction = 1.0 / float(values.get("tau-t", "1").split()[0])
            stage.generate_velocities = values.get("gen-vel", "no").lower() == "yes"
            stage.velocity_temperature = float(values.get("gen-temp", stage.temperature))
            coupling = values.get("pcoupl", "no").lower()
            if coupling not in ("no", ""):
                kind = values.get("pcoupltype", "isotropic").lower()
                stage.pressure_bar = float(values.get("ref-p", "1").split()[0])
                stage.barostat = {"semiisotropic": "membrane", "anisotropic": "anisotropic"}.get(
                    kind, "isotropic")
            stage.report_steps = int(float(values.get("nstenergy", 1000)))
            stage.dcd_steps = int(float(values.get("nstxout-compressed", 0)))
        reader = _TopologyReader(topology, [folder], mdp_defines(values))
        stage.positions, stage.torsions, total = reader.restraints()
        if total != natoms:
            raise CharmmGuiError("topol.top describes %d atoms but step5_input.gro has %d" % (total, natoms))
        stages.append(stage)

    modifier = first.get("vdw-modifier", "").lower()
    cutoff = float(first.get("rvdw", 1.0))
    production = {}
    prod_path = os.path.join(folder, "step7_production.mdp")
    if os.path.exists(prod_path):
        values = parse_mdp(prod_path)
        coupling = values.get("pcoupl", "no").lower() not in ("no", "")
        production = dict(temp=float(values.get("ref-t", "300").split()[0]),
                          fric_coeff=1.0 / float(values.get("tau-t", "1").split()[0]),
                          dt=float(values.get("dt", 0.002)), pcouple="yes" if coupling else "no")
        if coupling:
            production.update(p_type={"semiisotropic": "membrane", "anisotropic": "anisotropic"}.get(
                values.get("pcoupltype", "isotropic").lower(), "isotropic"),
                p_ref=float(values.get("ref-p", "1").split()[0]))
    notes = ["GROMACS v-rescale thermostat replaced by Langevin dynamics with friction 1/tau_t",
             "GROMACS emtol is a maximum force; OpenMM minimizes to an RMS force of 10 kJ/mol/nm",
             "GROMACS pressure coupling replaced by OpenMM's Monte Carlo barostat"]
    return Protocol("GROMACS", folder, topology, coordinates, folder, cutoff,
                    float(first.get("rvdw-switch", 0)) if modifier == "force-switch" else None,
                    first.get("dispcorr", "no").lower() not in ("no", ""),
                    stages, production, notes)
