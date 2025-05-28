#!/bin/bash
#SBATCH --job-name=CUDA_TEST
#SBATCH --time=0:02:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cudatemp
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=100M
#SBATCH --mail-user=s326766@studenti.polito.it
#SBATCH --mail-type=ALL
#SBATCH --output=test_cuda/cuda_%j.out   # %j → job ID

module load nvidia/cudasdk/10.1

#—— configs ——
W=100
H=100
MAXP_CELL=3
NP=200

# how many independent runs with the same parameters?
RUNS=3

# make a unique results folder based on the params
RESULT_DIR="test_cuda/results_W${W}_H${H}_NP${NP}"
mkdir -p "${RESULT_DIR}"

echo "Building with W=$W, H=$H, MAXP_CELL=$MAXP_CELL, NP=$NP"

# compile helpers
gcc -I./structures \
    -DW=${W} -DH=${H} -DMAXP_CELL=${MAXP_CELL} -DNP=${NP} \
    -c ./structures/tupleList.c -o ./bin/tupleList.o

nvcc -I./structures -I./utils \
     -DW=${W} -DH=${H} -DMAXP_CELL=${MAXP_CELL} -DNP=${NP} \
     -c ./utils/utils_cuda.cu -o ./bin/utils_cuda.o

# compile & link your main .cu file
nvcc -I./structures -I./utils \
     -DW=${W} -DH=${H} -DMAXP_CELL=${MAXP_CELL} -DNP=${NP} \
     -O3 \
     ./cuda/plague.cu \
     ./bin/utils_cuda.o ./bin/tupleList.o \
     -o ./bin/plague

rm ./report/*

echo
for i in $(seq 1 $RUNS); do
    echo "Run #$i → ${RESULT_DIR}/run${i}.out"
    ./bin/plague > "${RESULT_DIR}/run${i}.out" 2>&1
done

# cleanup
rm ./bin/plague ./bin/utils_cuda.o ./bin/tupleList.o