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

        JOB_SCRIPT="run_CUDA_128_W${W}_NP${NP}.sh"
        cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=CUDA_128_W${W}_NP${NP}
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cudatemp
#SBATCH --mail-type=FAIL
#SBATCH --output=test_cuda/128/cuda128_W${W}_NP${NP}_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=100M

module load nvidia/cudasdk/10.1

# Parameters
W=$W
H=$H
MAXP_CELL=$MAXP_CELL
NP=$NP
RUNS=$RUNS

RESULT_DIR="test_cuda/128/results_W\${W}_NP\${NP}"
mkdir -p "\${RESULT_DIR}"

echo "Building CUDA with W=\$W, H=\$H, MAXP_CELL=\$MAXP_CELL, NP=\$NP"
# Compile helpers
# Compile helper files (parameterized)
gcc -I./structures \
    -DW=\$W -DH=\$H -DMAXP_CELL=\$MAXP_CELL -DNP=\$NP \
    -c ./structures/tupleList.c -o ./bin/tupleList_\${W}_\${NP}.o

nvcc -I./structures -I./utils \
     -DW=\$W -DH=\$H -DMAXP_CELL=\$MAXP_CELL -DNP=\$NP \
     -c ./utils/utils_cuda.cu -o ./bin/utils_cuda_\${W}_\${NP}.o

# Compile & link CUDA binary
nvcc -I./structures -I./utils \
     -DW=\$W -DH=\$H -DMAXP_CELL=\$MAXP_CELL -DNP=\$NP \
     -O3 \
     ./cuda/plague.cu \
     ./bin/utils_cuda_\${W}_\${NP}.o ./bin/tupleList_\${W}_\${NP}.o \
     -o ./bin/plagueCUDA_\${W}_\${NP}

# Clean previous reports
rm -f ./report/*

# Run simulation multiple times
for i in \$(seq 1 \$RUNS); do
    echo "Run #\$i → \${RESULT_DIR}/run\${i}.out"
    ./bin/plagueCUDA_\${W}_\${NP} > "\${RESULT_DIR}/run\${i}.out" 2>&1
done

# Cleanup binaries and objects
rm -f ./bin/plagueCUDA_\${W}_\${NP} ./bin/utils_cuda_\${W}_\${NP}.o ./bin/tupleList_\${W}_\${NP}.o
EOF
        # Submit the job
        sbatch $JOB_SCRIPT
    done
done