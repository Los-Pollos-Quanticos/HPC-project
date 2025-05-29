#!/bin/bash

#SBATCH --job-name=OMP_TEST
#SBATCH --time=0:00:30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=global
#SBATCH --mem-per-cpu=30M
#SBATCH --mail-user=s326766@studenti.polito.it
#SBATCH --mail-type=ALL
#SBATCH --output=test_omp/omp_%j.out   # %j → job ID

#—— configs ——
W=10
H=10
MAXP_CELL=3
NP=4

# how many independent runs with the same parameters?
RUNS=1

# make a unique results folder based on the params
RESULT_DIR="test_omp/results_W${W}_H${H}_NP${NP}"
mkdir -p "${RESULT_DIR}"

echo "Building OMP with W=$W, H=$H, MAXP_CELL=$MAXP_CELL, NP=$NP"
gcc -fopenmp -o ./bin/plagueMP \
    ./openMP/plagueMP.c ./utils/utils.c \
    ./structures/tupleList.c ./structures/occupancyMap.c \
    -DW=${W} -DH=${H} -DMAXP_CELL=${MAXP_CELL} -DNP=${NP} \
    -lm

rm ./report/*

echo
for i in $(seq 1 $RUNS); do
    echo "Run #$i → ${RESULT_DIR}/run${i}.out"
    ./bin/plague --debug > "${RESULT_DIR}/run${i}.out" 2>&1
done

# clean up if you like
rm ./bin/plagueMP