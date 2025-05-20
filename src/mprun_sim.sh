#!/bin/bash

# Step 1: Compile the C code
echo "Compiling plagueMP.c..."
gcc -fopenmp -o ./bin/plagueMP ./openMP/plagueMP.c ./utils/utils.c ./structures/tupleList.c ./structures/occupancyMap.c -lm

# for debug
# gcc -fopenmp -g -Wall -Wextra -o ./bin/plagueMP ./openMP/plagueMP.c ./utils/utils.c ./structures/tupleList.c ./structures/occupancyMap.c -lm

# for memory debug
# gcc -fopenmp -g -Wall -Wextra -fsanitize=address -o ./bin/plagueMP ./openMP/plagueMP.c ./utils/utils.c ./structures/tupleList.c ./structures/occupancyMap.c -lm

# Step 2: Run the compiled executable
echo "Running ./plagueMP..."
./bin/plagueMP

