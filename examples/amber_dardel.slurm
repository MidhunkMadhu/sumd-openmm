#!/bin/bash -l
# Build-only Amber template; not validated by an Amber simulation.
# run.inp: engine=amber, parallel=mpi, mpi_mode=multi_gpu,
#          walkers=auto, amber_threads=auto; choose the installed executable.
#SBATCH --job-name=sumd-amber
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
#SBATCH --output=sumd-amber-%j.out
#SBATCH --error=sumd-amber-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load PDC/26.03 amber/24-cpeGNU-26.03-ambertools-24-amdgpu
SUMD_PYTHON="${SUMD_PYTHON:-python}"
"$SUMD_PYTHON" -c 'from sumd_openmm.amber import AmberEngine'
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun --exclusive --exact --nodes=1 --ntasks="$SLURM_NTASKS" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=1 \
    --hint=nomultithread --cpu-bind=verbose,cores --gpu-bind=verbose,single:1 \
    "$SUMD_PYTHON" -m sumd_openmm "${SUMD_INPUT:-run.inp}" "$@"
