#!/bin/bash

#SBATCH --job-name=test_HPC_CUDA
#SBATCH --time=0:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cudatemp
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=50M
#SBATCH --mail-user=s326766@studenti.polito.it
#SBATCH --mail-type=ALL

module load nvidia/cudasdk/10.1

echo "Compiling CUDA code..."
gcc -o ./bin/SDSH ./SDSH.c ../../utils/utils.c  ../../structures/occupancyMap.c -lm

echo
./bin/SDSH
