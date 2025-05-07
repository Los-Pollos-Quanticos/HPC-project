#!/bin/bash

# Step 1: Compile the C code
echo "Compiling plague.c..."
gcc -o ./bin/plague ./serial/plague.c ./utils/utils.c ./structures/tupleList.c ./structures/occupancyMap.c -lm

# Step 2: Run the compiled executable
echo "Running ./plague..."
./bin/plague

# Step 3: Run the Python viewer
echo "Launching Python viewer..."
python viewer.py
