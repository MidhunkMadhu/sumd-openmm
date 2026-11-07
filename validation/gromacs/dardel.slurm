#!/bin/bash -l
#SBATCH --job-name=sumd-gmx-validate
#SBATCH --account=naiss2025-3-3-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:05:00
#SBATCH --output=validation-%j.out
#SBATCH --error=validation-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load PDC/26.03 gromacs/2026.1-gpu-cpe2603-rocm7.0.2
PYTHON=/cfs/klemming/projects/supr/carlssonlab/midhun/apps/sumd-validation-20261107/venv/bin/python
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
echo "Compute host: $(hostname), Slurm job: $SLURM_JOB_ID"
gmx_mpi --version
# Each command runs inside the allocation. CPU parity precedes GPU parity.
srun --cpu-bind=none --gpus-per-task=1 -n 1 "$PYTHON" tools/validate_gromacs.py --gmx gmx_mpi \
    --work-dir "validation/gromacs/results_cpu_${SLURM_JOB_ID}"
srun --cpu-bind=none --gpus-per-task=1 -n 1 "$PYTHON" tools/validate_gromacs.py --gmx gmx_mpi --gpu --systems argon \
    --work-dir "validation/gromacs/results_gpu_${SLURM_JOB_ID}"
