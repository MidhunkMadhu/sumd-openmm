# Troubleshooting

Symptoms, causes and fixes for installing and running sumd-openmm,
most of them on GPU clusters. Start every diagnosis on a compute node
with

```bash
sumd-openmm --test
```

A working installation lists at least one GPU platform (`CUDA` or
`HIP`), prints the GPU it uses (for example `platform HIP device AMD
Instinct MI250X`), shows `ok` for every force check and ends with
`self test passed`. Then `sumd-openmm run.inp --test` runs two short
cycles of the real system and reports its speed.

## GPU platforms

**`libhiprtc.so.6: cannot open shared object file`**, and `HIP` missing
from the platform list. The HIP platform of OpenMM 8.6.1 needs a ROCm 6
runtime, and a ROCm 7 module (the default on some clusters) is loaded.
Load a ROCm 6 module before running, in interactive sessions and job
scripts alike; on Dardel, `module load PDC rocm/6.4.4`. `sumd-openmm
--test` prints the ROCm release the plugin needs.

**`libcuda.so.1: cannot open shared object file`** on an AMD node. The
CUDA platform cannot load without an NVIDIA driver; this is expected and
harmless when `HIP` is listed.

**`OpenMM platform CUDA is not available`** (or HIP). No GPU platform
could be loaded. Run on a GPU node, not a login node; load the GPU
runtime module; read the plugin load failures printed with the message.
A GPU request never falls back to the CPU; `platform = auto` uses the
fastest platform available.

**The run uses OpenCL instead of HIP or CUDA.** The requested platform
did not load and OpenCL, which works on both vendors' GPUs but is
slower, was used instead; the log says `platform HIP is not available;
using OpenCL`. Fix the HIP or CUDA plugin as above. `sumd-openmm --test`
shows the OpenCL device: a GPU name (`gfx90a`, a GeForce or Tesla name)
means the GPU, `pthread` or `cpu` means the CPU.

## Installing

**Downloads fail on a compute node.** Compute nodes often have no
internet access. Install with `pip` and `conda` on a login node; the
environment on a shared file system is immediately usable on the
compute nodes.

**pip keeps the old version.** `sumd-openmm --version` shows the
installed version; the [changelog](../CHANGELOG.md) lists the releases.
Reinstall explicitly:

```bash
pip install --no-cache-dir --force-reinstall --no-deps \
    "sumd-openmm @ git+https://github.com/MidhunkMadhu/sumd-openmm.git@main"
```

**`conda create` asks `Remove existing environment?`** An environment of
that name exists. Answering `n` (the default) keeps it untouched. To
update only sumd-openmm, use the `pip` command above; to start over,
`conda env remove -n <name>` first.

## Conda in batch jobs

**`Run 'conda init' before 'conda activate'`.** Batch jobs do not read
`~/.bashrc`, where `conda init` puts its setup. Load it in the script:

```bash
source <conda>/etc/profile.d/conda.sh
conda activate sumd-openmm
```

**`EnvironmentNameNotFound`.** The script activates a name that does not
exist in the conda it sourced. `conda env list`, run after sourcing the
same `conda.sh`, shows the names and paths; activate by path if the
environment has no name.

**The job stops right after `conda activate`** with `set -u` (or
`set -euo pipefail`) in the script. Conda's activation scripts use unset
variables; leave out `-u`.

## MPI and several walkers

**`parallel = mpi needs mpi4py`.** Install mpi4py in the environment,
built against the cluster's MPI with its compiler wrapper, on a login
node:

```bash
MPICC=mpicc pip install --no-cache-dir --no-binary mpi4py mpi4py          # most clusters
MPICC="cc -shared" pip install --no-cache-dir --no-binary mpi4py mpi4py   # HPE Cray (Dardel, LUMI)
```

Installing the package as `sumd-openmm[mpi]` with the same `MPICC` and
`--no-binary mpi4py` does both in one step. `sumd-openmm --test` reports
the MPI library mpi4py is linked to; on a cluster it must be the
cluster's MPI (Cray MPICH on HPE Cray systems), not one from conda or
pip.

**Every rank reports size 1.** Check with

```bash
srun -n 2 python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.rank, c.size)"
```

which must print `0 2` and `1 2`. `0 1` twice means mpi4py was built for
another MPI (for example conda's); rebuild it as above. A plain
`pip install mpi4py` or `conda install mpi4py` installs such a copy.

**`srun` does nothing or reports busy resources inside an interactive
shell** started with `srun --pty bash`. The shell already occupies the
allocation; add `--overlap` to the inner `srun`.

**`mpi_mode = multi_node expects one MPI rank per node`.** Several ranks
share a node; set `mpi_mode = multi_gpu`, or launch one rank per node.

**`ranks on <node> but the GPU visibility variables expose N GPUs`.**
More ranks than visible GPUs on a node. Request as many GPUs per node as
tasks per node (`--ntasks-per-node` equal to `--gpus-per-node`).

## Running

**`Missing required key in input file: dt`** (or `topology_file`,
`coordinate_file`). The key has no default; the
[user guide](USER_GUIDE.md) lists every key and its default.

**`UserWarning: unknown input key <key>`.** The key is misspelt or not
used by sumd-openmm, so it has no effect. Check the spelling against
the [user guide](USER_GUIDE.md).

**`p_type must be isotropic or membrane`.** Only these two barostats are
available; use `isotropic` for soluble systems and `membrane` for
bilayers.

**`No velocities found in: …` / `genvel = no was requested, so the run
cannot start without velocities`.** The coordinate file holds positions only (for example a PDB or a
minimized structure). Set `genvel = yes` to draw velocities at `temp`, or
start from a restart file of an equilibration.

**`output_dir exists; choose another or use --overwrite`.** The output
folder holds a previous run. Choose another `output_dir`, or add
`--overwrite` to replace it. The output of `--dry-run` is replaced
without `--overwrite`.

**Every cycle says `accepted` in a multiple-walker run.** mwSuMD always
continues from the best walker, so cycles are accepted even when the
metric moved away from the target. Follow the metric values on each
progress line; `walker_acceptance = sumd` repeats a cycle whose best
walker made no progress.

**Where are the values of the walkers that were not kept?** Progress
lines show the kept walker only. `windows.csv` in `output_dir` has one
row per window of every walker with the final value of every metric, and
`cv_samples.csv` every sample along every window.

**The run is much slower than the `--test` speed of the built-in
system.** Metrics are computed on the CPU from coordinates copied off
the GPU every `cv_sample_ps`; with tens of thousands of atoms and a
sample every few tens of femtoseconds this copy dominates. Compare
`sumd-openmm run.inp --test` speeds for larger `cv_sample_ps`; 0.2–1 ps
gives enough points for the window slope.

**`WARNING: energy rose during minimization`** during equilibration. The
starting coordinates violate the constraints (for example water
geometry that differs from rigid TIP3P); check the structure.

**`Unloading the cpe module is insufficient to restore the system
defaults`** on Dardel after `module load PDC`. A notice from the Cray
environment; it does not affect sumd-openmm.

## Reporting a problem

Include the output of `sumd-openmm --test` from a compute node, the
job's error file, and the `run.inp` used.
