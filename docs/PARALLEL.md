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

## AMD GPUs (Dardel)

Tested on Dardel (HPE Cray, AMD MI250X) with:

| Component | Version |
| --- | --- |
| Python | 3.11 (conda-forge) |
| OpenMM | 8.6.1 (conda-forge `openmm`; includes the HIP platform) |
| ROCm module | `rocm/6.4.4` (any 6.x; OpenMM 8.6.1's HIP platform does not load with ROCm 7) |
| MPI | Cray MPICH from the default `PDC` environment |
| mpi4py | built from source with the Cray compiler wrapper `cc` |

Each MI250X is two GPUs (GCDs), so a GPU node has eight; run one walker
per GCD.

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

[`examples/dardel_mwsumd.slurm`](../examples/dardel_mwsumd.slurm) runs
eight walkers on one node. The input file sets `platform = HIP`,
`parallel = mpi`, `mpi_mode = multi_gpu` and `walkers = 8`.

### Common errors

| Message | Cause and fix |
| --- | --- |
| `libhiprtc.so.6: cannot open shared object file` | ROCm 7 is loaded; `module load rocm/6.4.4` |
| `libcuda.so.1: cannot open shared object file` | Normal on AMD nodes; the CUDA platform is not used |
| `Run 'conda init' before 'conda activate'` | Batch jobs do not read `~/.bashrc`; `source <conda>/etc/profile.d/conda.sh` first |
| `parallel = mpi needs mpi4py` | Install mpi4py as above |
| Every rank reports size 1 (`c.size` is 1) | mpi4py is not built against Cray MPICH; rebuild it as above |
