"""Independent MD subprocesses must retain GPU binding without MPI step identity."""
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from sumd_openmm.gromacs import subprocess_environment, thread_count
from sumd_openmm.config import ConfigError


@pytest.mark.parametrize("cfg,env,expected", [
    ({}, {"SLURM_CPUS_PER_TASK": "8", "OMP_NUM_THREADS": "1"}, 8),
    ({"gromacs_threads": "auto"}, {"SLURM_CPUS_PER_TASK": "10"}, 10),
    ({"gromacs_threads": "4"}, {"SLURM_CPUS_PER_TASK": "8"}, 4),
    ({}, {"OMP_NUM_THREADS": "6"}, 6),
    ({}, {}, 1),
])
def test_thread_budget(cfg, env, expected):
    assert thread_count(cfg, env) == expected


@pytest.mark.parametrize("cfg,env", [
    ({"gromacs_threads": "16"}, {"SLURM_CPUS_PER_TASK": "8"}),
    ({"gromacs_threads": "0"}, {}),
    ({"gromacs_threads": "bad"}, {}),
    ({}, {"SLURM_CPUS_PER_TASK": "-1"}),
])
def test_invalid_or_oversubscribed_threads_are_rejected(cfg, env):
    with pytest.raises(ConfigError):
        thread_count(cfg, env)


def test_child_does_not_inherit_controller_mpi_world():
    original = {
        "PMI_RANK": "3", "PMIX_NAMESPACE": "controller", "OMPI_COMM_WORLD_SIZE": "8",
        "SLURM_NTASKS": "8", "SLURM_PROCID": "3", "SLURM_SRUN_COMM_PORT": "12345",
        "SLURMD_NODENAME": "compute", "PALS_RANKID": "3", "ALPS_APP_PE": "3",
        "MPI_LOCALRANKID": "3", "MPI_LOCALNRANKS": "8",
        "ROCR_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "3", "LD_LIBRARY_PATH": "/matching/runtime",
        "PATH": "/gromacs/bin", "MPICH_GPU_SUPPORT_ENABLED": "1",
    }
    child = subprocess_environment(original, 2)
    assert not any(k.startswith(("PMI_", "PMIX_", "OMPI_", "SLURM_", "SLURMD_", "PALS_", "ALPS_APP_"))
                   for k in child)
    assert "MPI_LOCALRANKID" not in child and "MPI_LOCALNRANKS" not in child
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH", "PATH", "MPICH_GPU_SUPPORT_ENABLED"):
        assert child[key] == original[key]
    assert child["OMP_NUM_THREADS"] == "2"
    assert original["SLURM_NTASKS"] == "8"


@pytest.mark.parametrize("integrator", ["md", "sd"])
@pytest.mark.parametrize("cpu_threads", [1, 2])
def test_initial_velocity_generation_overrides_native_continuation(tmp_path, monkeypatch, integrator, cpu_threads):
    from sumd_openmm.config import SumdConfig, read_key_value_file
    from sumd_openmm.gromacs import GromacsEngine, read_mdp
    executable = shutil.which(os.environ.get("SUMD_TEST_GMX", "gmx"))
    if executable is None:
        pytest.skip("set SUMD_TEST_GMX to run the GROMACS integration regression")
    fixtures = Path(__file__).resolve().parents[1] / "validation" / "gromacs"
    for name in ("system.top", "system.gro", "production.mdp", "gromacs.inp"):
        shutil.copyfile(fixtures / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    if integrator == "sd":
        mdp = Path("production.mdp")
        mdp.write_text(mdp.read_text().replace("integrator = md", "integrator = sd")
                       .replace("constraints = none", "constraints = all-bonds")
                       + "\ntc-grps = System\nref-t = 300\ntau-t = 1\n")
    cfg = read_key_value_file("gromacs.inp")
    cfg.update(genvel="yes", gromacs_executable=executable, gromacs_threads="auto")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(cpu_threads))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert read_mdp("production.mdp")["continuation"] == "yes"
    engine = GromacsEngine(cfg, SumdConfig.from_cfg(cfg), 42, tmp_path / "out", log=lambda _: None)
    assert engine.threads == cpu_threads and engine.env["OMP_NUM_THREADS"] == str(cpu_threads)
    assert ("-ntomp %d" % cpu_threads) in (engine.work / "commands.log").read_text()
    resolved = read_mdp(engine.work / "window.mdp")
    assert resolved["gen-vel"] == "yes" and resolved["continuation"] == "no"
    assert np.isfinite(engine.snapshot().epot)
    assert np.any(engine.packet(engine.snapshot())["vel"] != 0)
    if integrator == "sd":
        class Distance:
            names = ["distance"]

            def __call__(self, pos, box):
                return np.array([np.linalg.norm(pos[0] - pos[2])])

        parent = engine.snapshot()
        for name, reseed in (("kept", False), ("reassigned", True)):
            engine.restore(parent)
            if reseed:
                engine.reseed(77)
            times, values = engine.run_window(3, 20, Distance(), tmp_path / (name + ".dcd"), 1)
            assert times == pytest.approx([.002, .004, .006], abs=1e-8)
            assert np.all(np.isfinite(values)) and np.isfinite(engine.snapshot().epot)
        # Fork a real checkpoint without replacing its velocities. Distinct
        # walkers must receive distinct thermostat noise, not GPU-roundoff copies.
        parent = engine.snapshot()
        assert parent.state['native']
        branches, seeds = [], []
        for walker in (0, 1):
            engine.restore(parent)
            assert np.array_equal(engine.packet(engine.snapshot())['vel'], parent.state['vel'])
            seeds.append(engine.prepare_window(2, walker))
            engine.run_window(3, 20, Distance(), tmp_path / ('branch%d.dcd' % walker), 1)
            branches.append(engine.packet(engine.snapshot())['pos'])
        assert seeds[0] != seeds[1]
        assert np.max(abs(branches[0] - branches[1])) > 1e-7
        engine.restore(parent)
        assert engine.prepare_window(2, 0) == seeds[0]
        engine.run_window(3, 20, Distance(), tmp_path / 'branch0_replayed.dcd', 1)
        assert np.array_equal(engine.packet(engine.snapshot())['pos'], branches[0])


def test_thread_budget_can_change_on_continuation():
    from sumd_openmm.cli import check_continue_md
    before = {"md_config": {"gromacs_threads": "1", "dt": "0.002"}}
    check_continue_md(before, {"gromacs_threads": "auto", "dt": "0.002"})
    with pytest.raises(ConfigError, match="dt"):
        check_continue_md(before, {"gromacs_threads": "8", "dt": "0.004"})
