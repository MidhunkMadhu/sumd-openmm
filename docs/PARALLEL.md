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
