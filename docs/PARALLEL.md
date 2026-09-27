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

## AMD GPUs (Dardel, LUMI)

OpenMM runs on AMD GPUs through its HIP platform, which the NVIDIA (CUDA)
build does not contain. With the CUDA build on an AMD node, a request for
`platform = CUDA` finds no GPU platform, and the run stops with a message
listing the platforms and plugin errors it found. Install the HIP plugin
and load ROCm:

```bash
conda create -n sumd-amd -c conda-forge python=3.11 openmm-hip parmed numpy scipy pip git
conda activate sumd-amd
python -m pip install "sumd-openmm @ git+https://github.com/MidhunkMadhu/sumd-openmm.git"
```

`openmm-hip` on conda-forge (8.1.1 at the time of writing) requires a
matching OpenMM release; let conda choose it. Before submitting, run
`sumd-openmm --test` in an interactive job on a GPU node and check that
`HIP` is listed and its forces agree with Reference. Then run
`sumd-openmm run.inp --test`, which reports the speed of your system.

Use `platform = auto` or `platform = HIP`. Slurm binds AMD GPUs through
`ROCR_VISIBLE_DEVICES`, which `mpi_mode = multi_gpu` reads like
`CUDA_VISIBLE_DEVICES`. Each MI250X appears as two GPUs (GCDs), so a
Dardel GPU node has eight. Build mpi4py against Cray MPICH:

```bash
module load PrgEnv-cray cray-mpich
MPICC=cc python -m pip install --no-binary mpi4py mpi4py
```

[`examples/dardel_mwsumd.slurm`](../examples/dardel_mwsumd.slurm) runs
eight walkers on one node, one GCD each.
