#!/bin/bash -l
# One coordinated mwSuMD run: eight ranks, each with one GPU and eight cores.
# run.inp: engine=gromacs, parallel=mpi, mpi_mode=multi_gpu,
#          walkers=auto, gpu_devices=auto, gromacs_threads=auto.
# Submit from the installed environment, or set SUMD_PYTHON to its Python path.
#SBATCH --job-name=sumd-gromacs
#SBATCH --account=naiss2025-3-3-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=sumd-gromacs-%j.out
#SBATCH --error=sumd-gromacs-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load PDC/26.03 gromacs/2026.1-gpu-cpe2603-rocm7.0.2
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
SUMD_PYTHON="${SUMD_PYTHON:-python}"
"$SUMD_PYTHON" -c 'from sumd_openmm.gromacs import GromacsEngine'
srun --exclusive --exact --nodes=1 --ntasks="$SLURM_NTASKS" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=1 \
    --hint=nomultithread --cpu-bind=verbose,cores --gpu-bind=verbose,single:1 \
    "$SUMD_PYTHON" -m sumd_openmm "${SUMD_INPUT:-run.inp}" "$@"
