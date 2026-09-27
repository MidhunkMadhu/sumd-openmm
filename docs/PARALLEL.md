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

## AMD GPUs

OpenMM 8.6.1 from conda-forge includes the HIP platform for AMD GPUs. It
needs a ROCm 6.x runtime (`libhiprtc.so.6`) on the compute nodes; it does
not load with ROCm 7. For multiple walkers, build mpi4py with the
cluster's MPI compiler wrapper: `MPICC=mpicc` in general, `MPICC="cc
-shared"` on HPE Cray systems.

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
pip install "sumd-openmm @ git+https://github.com/MidhunkMadhu/sumd-openmm.git@main"

module load PDC
MPICC="cc -shared" pip install --no-binary mpi4py mpi4py
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

`sumd-openmm --test` must list `HIP`, show a `gfx90a` device and pass its
force check. `sumd-openmm run.inp --test` runs two short cycles of the
real system and reports its speed per walker.

### Submit

In [`examples/gpu_node.slurm`](../examples/gpu_node.slurm) use
`-p gpu`, `--ntasks-per-node=8`, `--gpus-per-node=8` and
`module load PDC rocm/6.4.4`, with `platform = HIP`, `parallel = mpi`,
`mpi_mode = multi_gpu` and `walkers = 8` in the input file.

### Common errors

See [troubleshooting](TROUBLESHOOTING.md).
