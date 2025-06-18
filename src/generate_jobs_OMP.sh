#!/bin/bash
grid_sizes=(10 20 50 100 200 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
MAXP_CELL=3
RUNS=10

for size in "${grid_sizes[@]}"; do
    W=$size
    H=$size
    TOTAL_CELLS=$((W * H * MAXP_CELL))

    for ratio in 10 50 90; do
        NP=$((TOTAL_CELLS * ratio / 100))

        JOB_SCRIPT="run_OMP_24_W${W}_NP${NP}.sh"
        cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=OMP_24_W${W}_NP${NP}
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=global
#SBATCH --mail-type=FAIL
#SBATCH --output=test_omp/OMP_24/OMP24_W${W}_NP${NP}_%j.out

# Parameters
W=$W
H=$H
MAXP_CELL=$MAXP_CELL
NP=$NP
RUNS=$RUNS

RESULT_DIR="test_omp/OMP_24/results_W\${W}_NP\${NP}"
mkdir -p "\${RESULT_DIR}"

export OMP_NUM_THREADS=24
echo "Building OMP16 with W=\$W, H=\$H, MAXP_CELL=\$MAXP_CELL, NP=\$NP"
gcc -fopenmp -o  ./bin/plagueMP24_\${W}_\${NP} \\
    ./openMP/plagueMP.c ./utils/utils.c \\
    ./structures/tupleList.c ./structures/occupancyMap.c \\
    -DW=\$W -DH=\$H -DMAXP_CELL=\$MAXP_CELL -DNP=\$NP \\
    -lm

rm -f ./report/*

for i in \$(seq 1 \$RUNS); do
    echo "Run #\$i → \${RESULT_DIR}/run\${i}.out"
    ./bin/plagueMP24_\${W}_\${NP} > "\${RESULT_DIR}/run\${i}.out" 2>&1
done

# Clean up binary
rm -f ./bin/plagueMP24_\${W}_\${NP}

#remove the sh files created
rm -f run_OMP_24_*.sh
EOF
        # Submit the job
        sbatch $JOB_SCRIPT
    done
done