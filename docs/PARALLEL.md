# MPI and GPUs

`parallel = serial` runs all walkers sequentially in one process.
`parallel = mpi` runs one OpenMM Context per rank; rank 0 also does work.
`walkers = auto` assigns one walker per rank. Every rank must see the
same input files and `output_dir`; the command probes the output filesystem
before running.

`mpi_mode = multi_node` requires one rank on each node. `mpi_mode =
multi_gpu` assigns the local rank to a GPU, or uses the corresponding entry
in `gpu_devices = 0,1,...`. When a scheduler gives each rank one visible
GPU, its local OpenMM device index is 0.

One node with four GPUs:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
export OMP_NUM_THREADS=1
srun sumd-openmm run.inp
```

Set `parallel=mpi`, `mpi_mode=multi_gpu`, `walkers=auto`.

Four nodes, one GPU each:

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
export OMP_NUM_THREADS=1
srun sumd-openmm run.inp
```

Set `mpi_mode=multi_node`. For two nodes with four GPUs each, request
`--nodes=2 --ntasks-per-node=4 --gpus-per-node=4` and set
`mpi_mode=multi_gpu`. Test the selection and GPU layout first with
`srun sumd-openmm run.inp --dry-run`. Install mpi4py against the MPI
implementation supplied by the cluster; mixing MPI implementations can
hang collectives. Use a shared output path, not local node scratch.

## GROMACS CPU cores per GPU

GROMACS runs one independent engine process per controller rank. Allocate
CPU cores to each rank and bind it to its GPU with Slurm; its child engine
inherits that CPU affinity and GPU visibility mask. `gromacs_threads = auto`
(also the default when omitted) uses `SLURM_CPUS_PER_TASK`, then
`OMP_NUM_THREADS`, then one thread. An explicit integer overrides this
choice, but cannot exceed the Slurm per-task CPU allocation. The selected
count sets both GROMACS `-ntomp` and the child's `OMP_NUM_THREADS`.
The thread count can change when continuing a run.

[`examples/gromacs_dardel.slurm`](../examples/gromacs_dardel.slurm) requests
eight GPU devices and eight physical CPU cores per walker, covering the
node's 64 physical cores. `--hint=nomultithread` avoids counting its 128
hardware threads as 128 physical cores. It uses the matching PDC 26.03
GROMACS/ROCm stack. Submit from an environment containing the current
package and mpi4py, or set `SUMD_PYTHON` to that environment's Python.

```ini
engine = gromacs
parallel = mpi
mpi_mode = multi_gpu
walkers = auto
gpu_devices = auto
gromacs_threads = auto
```

For six GPU walkers with ten physical cores each, change the script's
task and GPU counts to six and `--cpus-per-task` to ten. This allocates
60 physical cores. Independent simulations can use separate exclusive
`srun` steps and separate output directories; walkers within one mwSuMD
simulation use one coordinated MPI step with disjoint per-task bindings.
See [Slurm's binding and CPU allocation options](https://slurm.schedmd.com/srun.html).

GROMACS thermostat noise uses a seed derived from `random_seed`, the cycle,
and the walker index. This keeps stochastic walkers independent even when
their initial coordinates and velocities are identical. The per-window
`dynamics_seed` is recorded in `windows.jsonl`; velocity reassignment is
controlled separately by `retry_velocities` and recorded as `seed_used`.

## AMD GPUs

OpenMM 8.6.1 from conda-forge includes the HIP platform for AMD GPUs. It
needs a ROCm 6.x runtime (`libhiprtc.so.6`) on the compute nodes; it does
not load with ROCm 7. For multiple walkers, install the package with the
`mpi` extra and mpi4py compiled by the cluster's MPI compiler wrapper,
`MPICC=mpicc` in general and `MPICC="cc -shared"` on HPE Cray systems
(see the Dardel install below).

[`examples/gpu_node.slurm`](../examples/gpu_node.slurm) is a job script
for one node with one walker per GPU; replace its `<...>` fields with the
values of your cluster.

## Dardel

Tested on Dardel (HPE Cray, AMD MI250X) with:

| Component | Version |
| --- | --- |
| Python | 3.11 (conda-forge) |
| OpenMM | 8.6.1 (conda-forge `openmm`) |
| Environment module | `PDC` (Cray MPICH, compilers) |
| ROCm module | `rocm/6.4.4` (the default `rocm/7.x` does not work) |
| mpi4py | `MPICC="cc -shared"` build |

Each MI250X is two GPUs (GCDs), so a GPU node has eight.

### Install (login node)

```bash
conda create -n sumd-openmm -c conda-forge python=3.11 openmm=8.6.1 parmed numpy scipy pip git
conda activate sumd-openmm
module load PDC
MPICC="cc -shared" pip install --no-binary mpi4py \
    "sumd-openmm[mpi] @ git+https://github.com/MidhunkMadhu/sumd-openmm.git@main"
```

Install on a login node; compute nodes cannot download packages. With
conda installed in a project directory, activate it in scripts with
`source <conda>/etc/profile.d/conda.sh` followed by `conda activate`.

### Check (GPU node)

```bash
salloc -A <project> -p gpu -N 1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:30:00
srun --pty bash
module load PDC rocm/6.4.4
conda activate sumd-openmm
sumd-openmm --test
sumd-openmm run.inp --test
```

`sumd-openmm --test` must list `HIP`, show a `gfx90a` device, report
Cray MPICH as the MPI library of mpi4py and pass its force check. `sumd-openmm run.inp --test` runs two short cycles of the
real system and reports its speed per walker.

### Submit

In [`examples/gpu_node.slurm`](../examples/gpu_node.slurm) use
`-p gpu`, `--ntasks-per-node=8`, `--gpus-per-node=8` and
`module load PDC rocm/6.4.4`, with `platform = HIP`, `parallel = mpi`,
`mpi_mode = multi_gpu` and `walkers = 8` in the input file.

### Common errors

See [troubleshooting](TROUBLESHOOTING.md).
