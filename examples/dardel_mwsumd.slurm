#!/bin/bash -l
# mwSuMD on one Dardel GPU node: eight walkers, one MI250X GCD each.
# run.inp needs: platform = auto, parallel = mpi, mpi_mode = multi_gpu,
# walkers = auto (or a multiple of 8), and output_dir on /cfs/klemming.
# Check the environment first:  salloc -p gpu ... ; srun -n 1 sumd-openmm --test

#SBATCH --job-name=mwsumd
#SBATCH -A <project>
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=mwsumd_%j.out
#SBATCH --error=mwsumd_%j.err

module load PDC rocm          # `module avail rocm` lists the installed versions
conda activate sumd-amd

export OMP_NUM_THREADS=1
srun --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores sumd-openmm run.inp
