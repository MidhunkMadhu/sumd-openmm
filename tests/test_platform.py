"""Platform choice (CUDA, HIP, OpenCL) and Amber-style LJ force switching."""

import os

import numpy as np
import pytest

from sumd_openmm.omm_setup import PlatformError, choose_platform

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples",
                       "ligand_binding_amber")


def test_auto_prefers_gpu():
    assert choose_platform("auto", ["Reference", "CPU", "HIP"]) == "HIP"
    assert choose_platform("auto", ["Reference", "CPU", "OpenCL", "CUDA"]) == "CUDA"
    assert choose_platform("auto", ["Reference", "CPU"]) == "CPU"


def test_gpu_request_falls_back_to_another_gpu_only():
    notes = []
    assert choose_platform("CUDA", ["Reference", "CPU", "HIP", "OpenCL"], notes.append) == "HIP"
    assert "not available" in notes[0]
    assert choose_platform("HIP", ["CPU", "HIP"]) == "HIP"
    pytest.importorskip("openmm")
    with pytest.raises(PlatformError, match="HIP platform"):
        choose_platform("CUDA", ["Reference", "CPU"])


def test_unusable_gpu_platform_is_skipped_for_another_gpu():
    notes = []
    broken = {"CUDA": "CUDA_ERROR_UNSUPPORTED_PTX_VERSION"}
    assert choose_platform("auto", ["CPU", "OpenCL", "CUDA"], notes.append, broken.get) == "OpenCL"
    assert "CUDA cannot create a Context" in notes[0]
    notes.clear()
    assert choose_platform("CUDA", ["CPU", "OpenCL", "CUDA"], notes.append, broken.get) == "OpenCL"
    assert "not usable; using OpenCL" in notes[-1]


def test_unusable_gpu_platform_is_not_replaced_by_the_cpu():
    broken = {"CUDA": "CUDA_ERROR_UNSUPPORTED_PTX_VERSION"}
    for request in ("auto", "CUDA"):
        with pytest.raises(PlatformError, match="No GPU platform can run here"):
            choose_platform(request, ["Reference", "CPU", "CUDA"], lambda _: None, broken.get)
    assert choose_platform("CPU", ["Reference", "CPU", "CUDA"], check=broken.get) == "CPU"


def test_cuda_hint_names_the_fix(monkeypatch):
    from sumd_openmm import omm_setup
    monkeypatch.setattr(omm_setup, "_cuda_driver_version", lambda: "13.3")
    monkeypatch.setattr(omm_setup, "_nvrtc_version", lambda: "13.4")
    hint = omm_setup.cuda_hint("Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)")
    assert "cuda-nvrtc 13.4" in hint and '"cuda-version<=13.3"' in hint
    assert '"cuda-version=13.3"' in omm_setup.cuda_hint("CUDA_ERROR_NO_BINARY_FOR_GPU (209)")
    assert omm_setup.cuda_hint("CUDA_ERROR_OUT_OF_MEMORY") is None


def test_cpu_request_is_not_replaced():
    pytest.importorskip("openmm")
    with pytest.raises(PlatformError):
        choose_platform("CPU", ["Reference"])


@pytest.mark.skipif(not os.path.exists(os.path.join(EXAMPLE, "example.parm7")),
                    reason="example system not present")
def test_force_switch_matches_amber_fswitch():
    """
    CHARMM36m exported to Amber format with NBFIX. sander (AmberTools),
    cut = 12, fswitch = 10, vdwmeth = 0, gives VDWAALS = 4559.8361 kcal/mol;
    without fswitch 3233.8521.
    """
    openmm = pytest.importorskip("openmm")
    from types import SimpleNamespace
    from openmm import app, unit
    from sumd_openmm.omm_setup import force_switch, load_production_helpers

    top = app.AmberPrmtopFile(os.path.join(EXAMPLE, "example.parm7"))
    crd = app.AmberInpcrdFile(os.path.join(EXAMPLE, "example.rst7"))
    energies = {}
    for switched in (True, False):
        system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.2 * unit.nanometer)
        if switched:
            system = force_switch(load_production_helpers(), system,
                                  SimpleNamespace(r_on=1.0, r_off=1.2))
        lj = next(f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce))
        lj.setUseLongRangeCorrection(False)
        lj.setForceGroup(31)
        context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName("CPU"))
        context.setPeriodicBoxVectors(*crd.boxVectors)
        context.setPositions(crd.positions)
        energies[switched] = context.getState(getEnergy=True, groups={31}).getPotentialEnergy(
        ).value_in_unit(unit.kilocalorie_per_mole)
    assert energies[True] == pytest.approx(4559.8361, abs=0.5)
    assert energies[False] == pytest.approx(3233.8521, abs=0.5)


def test_force_switch_refuses_unknown_lj_forms():
    openmm = pytest.importorskip("openmm")
    from sumd_openmm.omm_setup import force_switch
    system = openmm.System()
    system.addParticle(1.0)
    system.addForce(openmm.NonbondedForce())
    custom = openmm.CustomNonbondedForce("A1*A2/r^12-C1*C2/r^6")
    custom.addPerParticleParameter("A")
    custom.addPerParticleParameter("C")
    custom.addParticle([1.0, 1.0])
    system.addForce(custom)
    with pytest.raises(ValueError, match="NBFIX"):
        force_switch(None, system, None)


def test_builtin_self_test(capsys):
    pytest.importorskip("openmm")
    from sumd_openmm import selftest
    assert selftest.run() == 0
    assert "self test passed" in capsys.readouterr().out


def test_rocm_hint_names_the_missing_release():
    from sumd_openmm.omm_setup import rocm_hint
    failures = ["Error loading library /env/lib/plugins/libOpenMMHIP.so: libhiprtc.so.6: "
                "cannot open shared object file: No such file or directory",
                "Error loading library /env/lib/plugins/libOpenMMCUDA.so: libcuda.so.1: cannot open"]
    assert "rocm/6" in rocm_hint(failures)
    assert rocm_hint(failures[1:]) is None


def test_missing_mpi4py_is_a_config_error(monkeypatch):
    import sys
    from sumd_openmm.cli import _mpi_world
    from sumd_openmm.config import ConfigError
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    with pytest.raises(ConfigError, match="mpi4py"):
        _mpi_world()


def test_mpi_report(monkeypatch):
    import sys
    from sumd_openmm.selftest import mpi_report
    lines = []
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    assert mpi_report(lines.append) is None and "not installed" in lines[0]
