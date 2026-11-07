"""GROMACS window backend. Native MDP controls physics; SuMD controls sampling.

Checkpoints are embedded in the common state envelope and sent over MPI.
Restores with kept velocities continue the checkpoint. Reseeding starts a
new branch from a full precision TRR, never from rounded GRO coordinates.
"""

import copy
import glob
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

import numpy as np

from .backend import read_state, write_state, wrapped_positions
from .config import ConfigError
from .dcdtools import DCDWriter
from .energylog import COLUMNS, ENERGY_TERMS, format_row
from .gromacs_io import read_trr, write_trr, read_xvg
from .selection import load_structure, _molecule_ids, virtual_site_links


def read_mdp(path):
    values = {}
    with open(path) as fh:
        for line in fh:
            line = line.split(";", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower().replace("_", "-")
                if key in values:
                    raise ConfigError("duplicate MDP key: %s" % key)
                values[key] = value.strip()
    return values


def subprocess_environment(environ, threads):
    """Launch an independent walker while retaining the scheduler's GPU mask."""
    env = dict(environ)
    # Cray PMI can discover the parent step through SLURM_* even after PMI_*
    # is removed. A child must not join its controller's MPI_COMM_WORLD.
    launch_prefixes = ("PMI_", "PMIX_", "OMPI_", "SLURM_", "SLURMD_", "PALS_", "ALPS_APP_")
    for key in list(env):
        if key.startswith(launch_prefixes) or key in ("MPI_LOCALRANKID", "MPI_LOCALNRANKS"):
            env.pop(key)
    env.update(GMX_MAXBACKUP="-1", OMP_NUM_THREADS=str(threads))
    return env


def thread_count(cfg, environ, key="gromacs_threads"):
    """Use the per-task CPU budget before an inherited OpenMP default."""
    def positive(value, name):
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("%s must be a positive integer" % name) from exc
        if result < 1:
            raise ConfigError("%s must be a positive integer" % name)
        return result

    budget = environ.get("SLURM_CPUS_PER_TASK")
    budget = positive(budget, "SLURM_CPUS_PER_TASK") if budget is not None else None
    requested = cfg.get(key, "auto")
    if str(requested).strip().lower() == "auto":
        threads = budget or positive(environ.get("OMP_NUM_THREADS", "1"), "OMP_NUM_THREADS")
    else:
        threads = positive(requested, key)
    if budget is not None and threads > budget:
        raise ConfigError("%s = %d exceeds SLURM_CPUS_PER_TASK = %d; "
                          "request more CPUs per task or reduce %s" % (key, threads, budget, key))
    return threads


class GromacsSnap:
    def __init__(self, packet):
        self.state = packet
        self.step, self.epot = packet["step"], packet["epot"]


class GromacsEngine:
    name = "gromacs"

    def __init__(self, cfg, scfg, seed, outdir, rank=0, device=None, log=print):
        self.log, self.cfg, self.scfg = log, cfg, scfg
        self.device, self.seed = device, int(seed)
        self.work = Path(outdir).resolve() / "engine_work" / ("rank_%03d" % rank)
        self.work.mkdir(parents=True, exist_ok=True)
        if cfg.get("force_field", "GROMACS").upper() != "GROMACS":
            raise ConfigError("engine = gromacs requires a native GROMACS .top topology")
        for key in ("topology_file", "coordinate_file", "gromacs_mdp"):
            if not cfg.get(key) or not Path(cfg[key]).is_file():
                raise ConfigError("engine = gromacs requires an existing %s" % key)
        self.topfile = str(Path(cfg["topology_file"]).resolve())
        self.crdfile = str(Path(cfg["coordinate_file"]).resolve())
        self.mdp = read_mdp(cfg["gromacs_mdp"])
        # An explicit native MDP avoids silently approximating another engine's settings.
        self.dt_ps = float(self.mdp.get("dt", "0.002"))
        refs = self.mdp.get("ref-t", cfg.get("temp", "300")).split()
        self.temperature_K = float(refs[0])
        if any(float(t) != self.temperature_K for t in refs):
            raise ConfigError("GROMACS SuMD requires one common temperature")
        for key, actual in (("dt", self.dt_ps), ("temp", self.temperature_K)):
            if key in cfg and not np.isclose(float(cfg[key]), actual, rtol=0, atol=1e-10):
                raise ConfigError("%s differs from gromacs_mdp" % key)
        if self.dt_ps <= 0 or self.temperature_K <= 0:
            raise ConfigError("GROMACS dt and temperature must be positive")
        if self.mdp.get("integrator", "md").lower() not in ("md", "md-vv", "sd", "bd"):
            raise ConfigError("GROMACS SuMD requires a dynamics integrator (md, md-vv, sd or bd)")
        if self.mdp.get("free-energy", "no").lower() != "no" or self.mdp.get("pull", "no").lower() == "yes":
            raise ConfigError("free-energy and pulling protocols are not supported by this backend")
        self.periodic = self.mdp.get("pbc", "xyz").lower() != "no"
        self.threads = thread_count(cfg, os.environ)
        self.command = shlex.split(cfg.get("gromacs_executable", "gmx"))
        if not self.command or shutil.which(self.command[0]) is None:
            raise ConfigError("GROMACS executable not found; load its module or set gromacs_executable")
        self.command[0] = shutil.which(self.command[0])
        self.args = shlex.split(cfg.get("gromacs_mdrun_args", ""))
        managed = {"-s", "-o", "-e", "-g", "-c", "-cpi", "-cpo", "-deffnm", "-nsteps",
                   "-multidir", "-append", "-noappend", "-rerun", "-reseed", "-ntomp", "-ntmpi",
                   "-gpu_id", "-gputasks"}
        if managed.intersection(self.args):
            raise ConfigError("gromacs_mdrun_args overrides a SuMD-managed argument")
        self.env = subprocess_environment(os.environ, self.threads)
        version = self._call(["--version"], "version")
        self.thread_mpi = "thread_mpi" in version
        self.version = next((s.strip() for s in version.splitlines() if "GROMACS version:" in s), "unknown")
        self.platform_name = "GROMACS " + cfg.get("platform", "auto")
        struct = load_structure(self.topfile, log)
        self.natoms = len(struct.atoms)
        ids = np.array(_molecule_ids(struct, virtual_site_links(self.topfile)))
        self._molecules = [np.where(ids == i)[0] for i in np.unique(ids)]
        self.masses = np.array([atom.mass for atom in struct.atoms], float)
        self.mass = float(self.masses.sum())
        self._packet = None
        self._reseed = None
        self._dynamics_seed = None
        self._execute(0, 1, initial=True)
        log("GROMACS backend: %s; %d atoms; device %s; %d CPU thread(s)" %
            (self.version, self.natoms, device or "auto", self.threads))

    def _call(self, args, label, stdin=None):
        command = self.command + list(map(str, args))
        result = subprocess.run(command, cwd=self.work, env=self.env, input=stdin,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open(self.work / "commands.log", "a") as fh:
            fh.write("$ " + shlex.join(command) + "\n" + result.stdout + "\n")
        if result.returncode:
            raise RuntimeError("GROMACS %s failed (%d); see %s:\n%s" %
                               (label, result.returncode, self.work / "commands.log", result.stdout[-4000:]))
        return result.stdout

    def _execute(self, nsteps, sample, initial=False):
        parent = self._packet
        mdp = dict(self.mdp)
        start_step = 0 if initial else parent["step"]
        start_time = 0.0 if initial else parent["time"]
        seed = self._reseed if self._reseed is not None else (
            self.seed if initial else parent["params"]["ld_seed"])
        if not initial and self._dynamics_seed is not None:
            seed = self._dynamics_seed
        # Energies are read only at CV samples. Computing them every step
        # (nstcalcenergy = 1) costs most of the GPU throughput and changes the
        # native coupling intervals, so keep the MDP value where it divides the sample.
        native_calc = int(self.mdp.get("nstcalcenergy", "100") or 100)
        calc = math.gcd(native_calc, int(sample)) if native_calc > 0 else int(sample)
        mdp.update(nsteps=str(nsteps), **{"init-step": str(start_step),
                   "tinit": repr(start_time - start_step * self.dt_ps),
                   "nstxout": str(sample), "nstvout": str(sample), "nstfout": "0",
                   "nstenergy": str(sample), "nstcalcenergy": str(calc), "nstlog": str(sample),
                   "nstxout-compressed": "0", "ld-seed": str(seed)})
        if initial:
            mdp["gen-vel"] = self.cfg.get("genvel", mdp.get("gen-vel", "no"))
            if mdp["gen-vel"].lower() == "yes":
                mdp["continuation"] = "no"
            mdp["gen-seed"], mdp["gen-temp"] = str(seed), str(self.temperature_K)
        else:
            mdp["gen-vel"] = "yes" if self._reseed is not None else "no"
            mdp["gen-seed"], mdp["gen-temp"] = str(seed), str(self.temperature_K)
            mdp["continuation"] = "yes"
        (self.work / "window.mdp").write_text("\n".join("%s = %s" % item for item in mdp.items()) + "\n")
        grompp = ["grompp", "-f", "window.mdp", "-p", self.topfile, "-c", self.crdfile,
                   "-o", "window.tpr", "-po", "resolved.mdp"]
        if self.cfg.get("gromacs_index"):
            grompp += ["-n", str(Path(self.cfg["gromacs_index"]).resolve())]
        native = b"" if initial or self._reseed is not None else parent.get("native", b"")
        if not initial:
            # grompp rejects -t together with gen-vel=yes: positions still come
            # from the full precision parent TRR; velocities are reassigned below.
            packet = dict(parent)
            if self._reseed is not None:
                rng = np.random.default_rng(self._reseed)
                masses = self.masses
                velocity = np.zeros_like(packet["vel"])
                real = masses > 0
                velocity[real] = rng.normal(size=(real.sum(), 3)) * np.sqrt(
                    0.00831446261815324 * self.temperature_K / masses[real, None])
                if real.sum() > 1:
                    velocity[real] -= np.average(velocity[real], axis=0, weights=masses[real])
                packet["vel"] = velocity
                mdp["gen-vel"] = "no"
                (self.work / "window.mdp").write_text("\n".join("%s = %s" % item for item in mdp.items()) + "\n")
            if native:
                (self.work / "parent.cpt").write_bytes(native)
                grompp += ["-t", "parent.cpt"]
            else:
                write_trr(self.work / "parent.trr", packet)
                grompp += ["-t", "parent.trr"]
        self._call(grompp, "grompp")
        # Remove only this worker's previous transient products, never run outputs.
        for pattern in ("window*.trr", "window*.edr", "window*.log", "window*.cpt", "window*.gro"):
            for path in self.work.glob(pattern):
                path.unlink()
        mdrun = ["mdrun", "-s", "window.tpr", "-deffnm", "window", "-ntomp", str(self.threads)]
        if self.thread_mpi:
            mdrun += ["-ntmpi", "1"]
        if native:
            mdrun += ["-cpi", "parent.cpt", "-noappend"]
        platform = self.cfg.get("platform", "auto").lower()
        if platform in ("cpu", "reference"):
            mdrun += ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"]
        elif platform != "auto":
            mdrun += ["-nb", "gpu"]
        if self.device is not None and platform not in ("cpu", "reference"):
            mdrun += ["-gpu_id", str(self.device)]
        output = self._call(mdrun + self.args, "mdrun")
        ranks = re.search(r"Using\s+(\d+)\s+MPI process", output)
        if ranks and int(ranks[1]) != 1:
            raise RuntimeError("GROMACS walker joined %s MPI ranks; each backend must run independently "
                               "with one rank. Check the MPI launcher environment." % ranks[1])
        trrs = sorted(self.work.glob("window*.trr"))
        edrs = sorted(self.work.glob("window*.edr"))
        if len(trrs) != 1 or len(edrs) != 1:
            raise RuntimeError("GROMACS did not produce exactly one TRR and EDR")
        frames = list(read_trr(trrs[0]))
        self._call(["energy", "-f", edrs[0], "-o", "energy.xvg", "-xvg", "xmgrace"], "energy",
                   "\n".join(["Potential", "Kinetic-En.", "Total-Energy", "Temperature", "Pressure",
                   "Volume", "Density", "Bond", "G96Bond", "Angle", "U-B", "Proper-Dih.",
                   "Ryckaert-Bell.", "Improper-Dih.", "CMAP-Dih.", "LJ-(SR)", "LJ-14", "Disper.-corr.",
                   "Coulomb-(SR)", "Coulomb-14", "Coul.-recip.", "0", ""]))
        times, energies = read_xvg(self.work / "energy.xvg")
        last = frames[-1]
        if any(name not in last for name in ("pos", "vel", "box")) or len(last["pos"]) != self.natoms:
            raise RuntimeError("GROMACS final TRR lacks complete coordinates, velocities or box")
        match = np.flatnonzero(np.isclose(times, last["time"], atol=self.dt_ps * .1, rtol=0))
        if not len(match) or "Potential" not in energies:
            raise RuntimeError("GROMACS final energy is missing")
        checkpoints = sorted(self.work.glob("window*.cpt"))
        checkpoints = [p for p in checkpoints if "_prev" not in p.name]
        self._packet = dict(pos=last["pos"], vel=last["vel"], box=last["box"],
                            step=last["step"], time=last["time"], epot=float(energies["Potential"][match[-1]]),
                            params={"ld_seed": seed}, native=checkpoints[-1].read_bytes() if checkpoints else b"",
                            native_format="gromacs-checkpoint", engine=self.name)
        if nsteps and not self._packet["native"]:
            raise RuntimeError("GROMACS did not write a restart checkpoint")
        if last["step"] != start_step + nsteps:
            raise RuntimeError("GROMACS ended at step %d; expected %d" % (last["step"], start_step + nsteps))
        self._reseed = None
        self._dynamics_seed = None
        return frames, times, energies

    def prepare_window(self, cycle, walker):
        """Independent, reproducible thermostat noise without replacing velocities.

        SD uses ld-seed and the step counter. Reusing a parent's seed for
        every walker would otherwise reproduce the same stochastic branch.
        Deterministic integrators ignore this parameter.
        """
        master = self.scfg.random_seed if self.scfg.random_seed is not None else self.seed
        stream = np.random.SeedSequence([int(master), int(cycle), int(walker), 0x53554D44])
        self._dynamics_seed = int(stream.generate_state(1)[0] % (2**31 - 1)) + 1
        return self._dynamics_seed

    def snapshot(self):
        return GromacsSnap(copy.deepcopy(self._packet))

    def packet(self, snap):
        return copy.deepcopy(snap.state)

    def time_ps(self, snap):
        return snap.state["time"]

    def restore(self, snap, check=True):
        self._packet = self.packet(snap)
        self._reseed = None
        self._dynamics_seed = None
        # The checkpoint is validated by mdrun on use. No independent force
        # evaluation has happened here, so report missing rather than zero.
        return None

    def restore_packet(self, packet, check=True):
        return self.restore(GromacsSnap(packet), check)

    def reseed(self, seed):
        self._reseed = int(seed)

    def save_state(self, snap, path):
        p = snap.state
        write_state(path, p, self.name, p.get("native", b""), "gromacs-checkpoint")

    def load_state(self, path, step, epot):
        packet = read_state(path)
        if packet["engine"] != self.name:
            raise ValueError("cannot resume another MD engine's state; use an interchange restart")
        return GromacsSnap(packet)

    def _wrapped(self, pos, box):
        return wrapped_positions(pos, box, self._molecules) if self.periodic else pos.copy()

    def molecules(self):
        return self._molecules

    def current_cvs(self, evaluator):
        p = self._packet
        return evaluator(self._wrapped(p["pos"], p["box"]) * 10,
                         p["box"] * 10 if self.periodic else None)

    def run_window(self, n_samples, steps_per_sample, evaluator, dcd_path, stride,
                   log_path=None, log_stride=0):
        start_step, start_time = self._packet["step"], self._packet["time"]
        started = time.time()
        frames, etimes, energies = self._execute(n_samples * steps_per_sample, steps_per_sample)
        frames = [f for f in frames if f["step"] > start_step]
        expected = start_step + np.arange(1, n_samples + 1) * steps_per_sample
        if len(frames) != n_samples or not np.array_equal([f["step"] for f in frames], expected):
            raise RuntimeError("GROMACS metric sampling does not match the requested window")
        values = []
        with open(dcd_path, "wb") as fh, open(log_path or os.devnull, "w") as log:
            dcd = DCDWriter(fh, self.natoms, self.dt_ps, start_step + steps_per_sample * stride,
                            steps_per_sample * stride, self.periodic)
            if log_path:
                log.write(",".join(COLUMNS + list(ENERGY_TERMS) + list(evaluator.names)) + "\n")
            for k, frame in enumerate(frames, 1):
                pos = self._wrapped(frame["pos"], frame["box"])
                row = evaluator(pos * 10, frame["box"] * 10 if self.periodic else None)
                values.append(row)
                if k % stride == 0:
                    dcd.write(pos, frame["box"])
                if log_path and k % log_stride == 0:
                    index = np.flatnonzero(np.isclose(etimes, frame["time"], atol=self.dt_ps * .1, rtol=0))
                    if len(index) != 1:
                        raise RuntimeError("GROMACS energy and CV times do not align")
                    log.write(self._energy_row(frame, energies, index[0], row, time.time() - started))
        return np.array([f["time"] - start_time for f in frames]), np.asarray(values)

    def _energy_row(self, frame, energies, index, metrics, wall):
        def value(name):
            return energies[name][index] if name in energies else None
        volume = density = None
        if self.periodic:
            volume = abs(np.linalg.det(frame["box"]))
            density = self.mass * 1.66053906660e-3 / volume
        groups = {"E_bond": ("Bond", "G96Bond"), "E_angle": ("Angle", "U-B"),
                  "E_dihedral": ("Proper Dih.", "Ryckaert-Bell.", "Improper Dih.", "CMAP Dih."),
                  "E_vdw": ("LJ (SR)", "LJ-14", "Disper. corr."),
                  "E_elec_direct": ("Coulomb (SR)", "Coulomb-14"), "E_elec_recip": ("Coul. recip.",)}
        terms = {}
        if self.scfg.energy_terms:
            for term, names in groups.items():
                present = [value(name) for name in names if name in energies]
                if present:
                    terms[term] = sum(present)
        return format_row(frame["step"], frame["time"], value("Potential"), value("Kinetic En."),
                          value("Temperature"), value("Pressure"), volume, density, wall,
                          terms, metrics)
