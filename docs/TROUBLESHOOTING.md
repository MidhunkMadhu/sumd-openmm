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

**`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`**, with `CUDA` in the platform
list. The CUDA toolkit conda installed (`cuda-nvrtc`) is newer than the
NVIDIA driver supports: conda-forge only requires the same major release,
so a CUDA 13.4 toolkit is installed next to a CUDA 13.3 driver. OpenMM
compiles its kernels with that toolkit and the driver refuses them.
sumd-openmm checks every GPU platform with a small Context before using
it, and names both versions and the fix:

```bash
conda install -c conda-forge "cuda-version<=13.3"   # the driver's CUDA version
```

or update the driver. The `nvidia-smi` header shows the driver's CUDA
version. The [install command](../README.md#install) pins it for you.
Kernels cached from an earlier toolkit can make some kernels load and
others fail, so trust the check rather than a run that worked before.

**`CUDA_ERROR_NO_BINARY_FOR_GPU`** or an unsupported GPU architecture.
The toolkit is too old for the GPU (e.g. CUDA 12.x before 12.8 on a
Blackwell RTX 50 GPU). Install a newer `cuda-version` that the driver
still supports.

**`No GPU platform can run here`.** A GPU platform loaded but could not
create a Context, and no other GPU platform could either; the message
gives each platform's error. The run stops rather than use the CPU; set
`platform = CPU` to run on the CPU deliberately.

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

**`p_type must be isotropic, membrane or anisotropic`.** Only these
barostats are available; use `isotropic` for soluble systems, `membrane`
for bilayers and `anisotropic` to scale each box edge independently.

**`No velocities found in: …` / `genvel = no was requested, so the run
cannot start without velocities`.** The coordinate file holds positions only (for example a PDB or a
minimized structure). Set `genvel = yes` to draw velocities at `temp`, or
start from a restart file of an equilibration.

**A folder `sumd_run_old00001` (or `_old00002`, ...) appears next to
`output_dir`.** The output folder held an earlier run, so it was renamed
with the first free number and the new run started in an empty
`output_dir`; the first lines of `progress.log` name the renamed folder.
To carry the earlier run on instead, use `--continue` or `--extend N`;
to delete it, `--overwrite`. The output of `--dry-run` is replaced
without renaming.

**`output_dir … holds an input of this run (coordinate_file = …)`.** An
input file, such as `coordinate_file = sumd_run/final_state.rst7`, lies
inside the output folder of the earlier run, which would be renamed.
Choose another `output_dir` for the new run, e.g. `sumd_run2`.

**A multiple-walker run never rejects a cycle.** mwSuMD always continues
from the best walker, so every cycle creates an AcceptedStep, even when
the metric moved away from the target. Follow the metric values in the
progress table; `walker_acceptance = sumd` repeats a cycle whose best
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

**`WARNING: rest = yes is ignored`.** Positional restraints change the
Hamiltonian, and SuMD windows are unbiased dynamics. Restrained
equilibration belongs before SuMD, with `equilibration = charmm-gui` or
any other protocol.

**`Unloading the cpe module is insufficient to restore the system
defaults`** on Dardel after `module load PDC`. A notice from the Cray
environment; it does not affect sumd-openmm.

**A script that reads `progress.log` finds no cycle rows, or reads the
wrong numbers.** The cycle lines are labelled fields
(`Cycle=12  Window=3  CV1=18.63 A (-1.02)`), and which metric each `CVn`
is appears on the `CV1 = ...` lines at the top of the log. Read the
numbers from `windows.csv` or `accepted_steps.csv` instead, which keep
one column per metric under its name ([outputs](OUTPUTS.md)).

**No `sumd_energy.log` at the end of a run.** The final path has no kept
window, or `log_stride = 0`. Check `accepted_steps.csv` for AcceptedSteps
past 0.

**`sumd_energy.log` starts later than `sumd_traj.dcd`, or a term is
missing from its first records.** The run was continued with `--continue`,
and its earlier windows were written without a log (a version before 0.9.0)
or with other settings. Windows without a log are left out, so record N no
longer matches frame N; windows without `energy_terms` lack those terms,
and windows of version 0.9.0 show `PRESS = 0.0`. Start a new run for a log
that covers the whole path.

**`TEMP(K)` or `EKtot` jumps between two windows in `sumd_energy.log`.**
A window that starts with reassigned velocities begins from a fresh
Maxwell-Boltzmann draw. The `| Cycle` lines show where windows meet.

**Energies in `sumd_energy.log` differ from `sumd_energy.csv` or
`accepted_steps.csv`.** The log is in kcal/mol, as in Amber, and the tables
in kJ/mol. Divide the table by 4.184.

**`PRESS = 0.0` in every record.** The run has no barostat (`pcouple =
no`), and Amber prints the same. Without periodic boundaries `VOLUME` and
`Density` are left out as well.

**`VOLUME` does not change over a short run with a barostat.** Monte
Carlo moves are tried every `p_freq` steps and start large, so in a big box
most are rejected until the step size adapts.

**`summary.DENSITY` or `summary.VOLUME` from `process_mdout.perl` is
blank.** The script expects an `EKCMT` line, which OpenMM cannot provide.
Use `mdout_analyzer.py` or `sumd_energy.csv`.

**No `VDWAALS` or `EELEC` in the log, only `NONBND` and `RECIP`.**
Lennard-Jones and real-space electrostatics share one OpenMM force unless
`vdw = Force-switch` separates them, so they appear together as `NONBND`
(`E_nonbonded` in the table).

**A run is slower with energy logs.** Each record needs an energy
evaluation, one more per energy term, and more for the pressure with a
barostat. With `dcd_stride = 1` and a short `cv_sample_ps` that is
frequent; set a larger `log_stride`, or `energy_terms = no`.

**`WARNING: ... virtual site definition(s) read as bonds to their parent
atoms`.** parmed, which reads the topology for selections, cannot build
some virtual sites, such as the type-3 lone pair CGenFF writes for a
halogen. Selections still work and the simulation is not affected, since
OpenMM reads the topology itself. The site remains an atom of its residue.

**A ligand selection has one atom more than expected, or `topology atoms
but reference atoms` differ by one.** A heavy-atom mask such as
`:LIG&!@H=` also selects a lone pair site, which is not named like a
hydrogen and counts as a full atom in a metric. Add `&!@LP1`, using the
site's name, to leave it out.

**`Unexpected bond b/w vsite and its parent`.** The topology lists a bond
between a virtual site and its parent atom. Remove that bond from
`[ bonds ]`.

**`TypeError: not all arguments converted during string formatting` in
`energy_row`, with a membrane or anisotropic barostat.** The original
0.9.1 logger expected scalar pressure, but these barostats return three
components. The corrected logger writes their mean to `Pressure_bar`.
Version 0.9.0 wrote zero for pressure, so an older installation can run the
same system without encountering this logging error. The fix does not
change the integrator or barostat settings.

**GROMACS on Dardel segfaults after printing its performance summary.**
Check the executable's runtime libraries with `ldd`, as well as its
`--version` output. The `PDCOLD/24.11` GPU module was observed loading
ROCm 6 alongside a Cray GPU-aware MPI library requiring ROCm 7; its crash
dump pointed to HIP shutdown from `libmpi_gtl_hsa`. The validation script
uses `PDC/26.03` with
`gromacs/2026.1-gpu-cpe2603-rocm7.0.2` to keep that runtime stack consistent.
Do not treat a nonzero `mdrun` exit as success merely because output files
exist. Run simulation checks through Slurm on a compute node.

**GROMACS reports `Generating velocities is inconsistent with attempting
to continue a previous run`.** An initial state with `genvel = yes` must
use `gen-vel = yes` and `continuation = no`. The backend sets that pairing
when it prepares the initial state. Later windows restore their accepted
state rather than generating velocities through this initial-state path.

**`Invalid T coupling input: 0 groups` with `integrator = sd`.** Set
`tc-grps = System` when `ref-t` and `tau-t` describe one common temperature
group. Stochastic dynamics still needs these group settings even with
`tcoupl = no`; `validation/gromacs/taar1_smoke.mdp` provides an example.

**An external GROMACS walker uses the controller's full MPI rank count.**
Each backend process must be independent. The backend removes Slurm and
MPI launch identity from its child environment, while keeping GPU
visibility masks and runtime library paths. It rejects `mdrun` output
reporting more than one MPI process per walker. See HPE's
[MPI singleton documentation](https://cpe.ext.hpe.com/docs/latest/mpt/mpich/intro_mpi.html#mpi-singleton-support)
for the supported standalone-launch model.

**`Amber executable not found` for `pmemd.cuda` on Arrhenius.** The GH200
build is not on a visible module: `GPU/Amber` offers no version, and its
hidden module does not set `PATH`. Put the build on `PATH` in the batch
script:

```bash
export AMBERHOME=/software/sse2/el9_gh200/easybuild/prefix/software/Amber/26.2-foss-2025b-pmemd
export PATH=$AMBERHOME/bin:$PATH
```

**Amber forces differ from GROMACS or OpenMM on cholesterol, for an Amber
topology converted from a CHARMM GROMACS topology with ParmEd.**
`ChamberParm.from_structure` rebuilds the 1-4 Lennard-Jones table from
per-type parameters. Some CHARMM `[ pairtypes ]` entries, including those
of the cholesterol ring types CRL1 and CRL2, are not of that form. Compare
single-point forces of the two topologies in OpenMM. Where they differ, set
`LENNARD_JONES_14_ACOEF` and `LENNARD_JONES_14_BCOEF` from the GROMACS pair
parameters of each type pair.

## Reporting a problem

Include the output of `sumd-openmm --test` from a compute node, the
job's error file, and the `run.inp` used.
