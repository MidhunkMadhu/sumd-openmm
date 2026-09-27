#!/bin/bash -l
# mwSuMD on one Dardel GPU node: eight walkers, one GPU (MI250X GCD) each.
# run.inp: platform = HIP, parallel = mpi, mpi_mode = multi_gpu, walkers = 8

#SBATCH -J sumd
#SBATCH -A <project>
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -t 24:00:00

module load PDC rocm/6.4.4
source <conda>/etc/profile.d/conda.sh
conda activate sumd-openmm

srun --gpus-per-task=1 --gpu-bind=single:1 sumd-openmm run.inp
