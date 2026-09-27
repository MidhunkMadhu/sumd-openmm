#!/bin/bash -l
# SuMD or mwSuMD on one GPU node, one walker per GPU.
# run.inp: platform = auto, parallel = mpi, mpi_mode = multi_gpu,
#          walkers = number of GPUs per node
# Replace every <...> with the values of your cluster.

#SBATCH -J sumd
#SBATCH -A <project>
#SBATCH -p <gpu-partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=<gpus-per-node>
#SBATCH --gpus-per-node=<gpus-per-node>
#SBATCH -t 24:00:00

module load <gpu-runtime>        # AMD: a ROCm 6.x module; NVIDIA: usually nothing
source <conda>/etc/profile.d/conda.sh
conda activate sumd-openmm

srun --gpus-per-task=1 sumd-openmm run.inp
