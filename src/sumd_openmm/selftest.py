"""
selftest.py

sumd-openmm --test [run.inp]

Without an input file: report the OpenMM platforms this installation can
use (and why a plugin such as HIP or CUDA failed to load), check that every
platform computes the same forces, and run two SuMD cycles of a small
built-in charged Lennard-Jones box with PME on the fastest platform.

With an input file: run the user's system for two short cycles in
<output_dir>_test, then audit the outputs and report simulation speed. Use
it on a compute node before submitting a long job.
"""

import json
import os
import platform as host_platform
import shutil
import tempfile

import numpy as np

from .parallel import VISIBILITY_VARIABLES

TEST_CYCLES = 2
TEST_SAMPLES = 10


def environment_report(log=print):
    import openmm
    from openmm import Platform
    from .omm_setup import available_platforms, plugin_failures, rocm_hint

    log("OpenMM %s, Python %s on %s" % (openmm.__version__, host_platform.python_version(),
                                        host_platform.node()))
    log("plugin directory: %s" % Platform.getDefaultPluginsDirectory())
    names = available_platforms()
    for name in names:
        log("platform %-9s speed %s" % (name, Platform.getPlatformByName(name).getSpeed()))
    failures = plugin_failures()
    for failure in failures:
        if "libcuda.so" in failure and "HIP" in names:
            continue                  # the CUDA plugin cannot load on an AMD node
        log("plugin load failure: %s" % failure.strip())
    hint = rocm_hint(failures)
    if hint:
        log("NOTE: " + hint)
    for var in VISIBILITY_VARIABLES:
        if os.environ.get(var):
            log("%s=%s" % (var, os.environ[var]))
    if not any(p in names for p in ("CUDA", "HIP", "OpenCL")):
        log("WARNING: no GPU platform. NVIDIA GPUs need the CUDA platform of OpenMM and the "
            "NVIDIA driver; AMD GPUs need the HIP platform and a matching ROCm runtime.")
    return names


def _box_system():
    """216 charged LJ particles in a 2.4 nm periodic box, PME electrostatics."""
    import openmm
    from openmm import unit

    n, spacing = 6, 0.4
    system = openmm.System()
    box = n * spacing
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(box, 0, 0), openmm.Vec3(0, box, 0),
                                        openmm.Vec3(0, 0, box))
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    positions = []
    for i, (x, y, z) in enumerate(np.ndindex(n, n, n)):
        system.addParticle(20.0)
        nb.addParticle(0.3 if (x + y + z) % 2 else -0.3, 0.3, 0.5)
        positions.append(((x + 0.5) * spacing, (y + 0.5) * spacing, (z + 0.5) * spacing))
    system.addForce(nb)
    jitter = np.random.default_rng(3).uniform(-0.05, 0.05, (len(positions), 3))
    return system, np.asarray(positions) + jitter


def _write_pdb(path, positions, box):
    with open(path, "w") as fh:
        fh.write("CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n" % ((box * 10,) * 3))
        for i, (x, y, z) in enumerate(positions * 10):
            fh.write("HETATM%5d  AR  ARG X%4d    %8.3f%8.3f%8.3f  1.00  0.00          AR\n"
                     % (i + 1, i + 1, x, y, z))
        fh.write("END\n")


def platform_agreement(names, log=print):
    """Forces of the test box on every platform, compared with Reference."""
    import openmm
    from openmm import unit

    system, positions = _box_system()
    forces, ok = {}, True
    for name in names:
        platform = openmm.Platform.getPlatformByName(name)
        try:
            integrator = openmm.VerletIntegrator(0.001)
            context = openmm.Context(system, integrator, platform)
            context.setPositions(positions)
            if "DeviceName" in platform.getPropertyNames():
                log("platform %-9s device %s" % (name, platform.getPropertyValue(context, "DeviceName")))
            state = context.getState(getForces=True)
            forces[name] = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer)
        except Exception as exc:
            ok = False
            log("platform %s FAILED: %s" % (name, str(exc).strip().splitlines()[0]))
    if "Reference" in forces:
        ref = forces["Reference"]
        for name, f in forces.items():
            # PME grids and precision differ between platforms; 1e-2 flags real faults only
            error = float(np.linalg.norm(f - ref) / np.linalg.norm(ref))
            good = error < 1e-2
            ok &= good
            log("platform %-9s relative RMS force error vs Reference %.1e %s"
                % (name, error, "ok" if good else "MISMATCH"))
    return ok


def builtin_run(log=print):
    """Two SuMD cycles of the test box through the command line driver."""
    import openmm
    from .cli import main as cli_main
    from .inspect_run import inspect

    system, positions = _box_system()
    folder = tempfile.mkdtemp(prefix="sumd_selftest_")
    try:
        with open(os.path.join(folder, "system.xml"), "w") as fh:
            fh.write(openmm.XmlSerializer.serialize(system))
        _write_pdb(os.path.join(folder, "system.pdb"), positions, 2.4)
        with open(os.path.join(folder, "run.inp"), "w") as fh:
            fh.write("force_field = OPENMM_XML\nsystem_xml = system.xml\n"
                     "topology_file = system.pdb\nplatform = auto\ntemp = 300\ndt = 0.002\n"
                     "window_ps = 0.2\ncv_sample_ps = 0.02\nmax_cycles = %d\nrandom_seed = 7\n"
                     "output_dir = output\nmetric_1_name = pair\nmetric_1_type = distance\n"
                     "metric_1_a = indices:0\nmetric_1_b = indices:215\n"
                     "metric_1_role = supervise\nmetric_1_direction = decrease\n"
                     "metric_1_target = 3\n" % TEST_CYCLES)
        cwd = os.getcwd()
        os.chdir(folder)
        try:
            cli_main(["run.inp", "--overwrite"])
        finally:
            os.chdir(cwd)
        return audit(os.path.join(folder, "output"), inspect, log)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def shorten(config):
    """Test settings for a user input: two cycles of TEST_SAMPLES samples."""
    config.max_cycles = TEST_CYCLES
    config.window_ps = TEST_SAMPLES * config.cv_sample_ps
    config.dcd_stride = 1
    config.output_dir = config.output_dir.rstrip("/") + "_test"
    config.equilibration_dir = config.equilibration_dir.rstrip("/") + "_test"
    config.validate()
    return config


def audit(outdir, inspect=None, log=print):
    """Check a finished (test) run and report speed. Returns True when sound."""
    if inspect is None:
        from .inspect_run import inspect
    report = inspect(outdir)
    with open(os.path.join(outdir, "run_summary.json")) as fh:
        summary = json.load(fh)
    with open(os.path.join(outdir, "windows.jsonl")) as fh:
        windows = [json.loads(line) for line in fh if line.strip()]
    engine = summary["engine"]
    simulated = sum(w["t_ps"][-1] for w in windows if w["t_ps"])
    wall = sum(w["wallclock_s"] for w in windows)
    log("test platform: %s" % engine["platform"])
    if wall > 0:
        log("test speed: %.1f ns/day per walker, including CV sampling every %g ps"
            % (simulated / wall * 86.4, summary["sumd_config"]["cv_sample_ps"]))
    if report.get("max_abs_restore_dE_kJmol") is not None:
        log("largest energy change on restoring a saved state: %.3g kJ/mol"
            % report["max_abs_restore_dE_kJmol"])
    log("accepted windows: %d of %d" % (report["windows_committed"], report["windows_simulated"]))
    mismatch = report["max_dcd_vs_state_mismatch_A"]
    identical = report["final_traj_identical_to_path_windows"]
    checks = {
        "windows simulated": report["windows_simulated"] > 0,
        "final state written": os.path.exists(os.path.join(outdir, "final_state.xml")),
    }
    # Both cycles may be rejected; then there is no accepted window to compare.
    if mismatch is not None:
        checks["DCD frames match saved states"] = mismatch < 1e-3
    if identical is not None:
        checks["final trajectory matches the accepted path"] = identical
    for name, good in checks.items():
        log("check %-45s %s" % (name, "ok" if good else "FAILED"))
    return all(checks.values())


def run(log=print):
    """Entry point for --test without an input file. Returns an exit status."""
    names = environment_report(log)
    ok = platform_agreement(names, log)
    ok &= builtin_run(log)
    log("self test %s" % ("passed" if ok else "FAILED"))
    return 0 if ok else 1
