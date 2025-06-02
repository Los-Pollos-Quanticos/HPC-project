#!/bin/bash

# Step 1: Compile the C code
echo "Compiling plague.c..."
gcc -o ./bin/plague ./serial/plague.c ./utils/utils.c ./structures/tupleList.c ./structures/occupancyMap.c -lm

# for debug
# gcc -g -Wall -Wextra -o ./bin/plague ./serial/plague.c ./utils/utils.c ./structures/tupleList.c ./structures/occupancyMap.c -lm

# for memory debug
# gcc -g -Wall -Wextra -fsanitize=address -o ./bin/plague ./serial/plague.c ./utils/utils.c ./structures/tupleList.c ./structures/occupancyMap.c -lm

# Step 2: Run the compiled executable
echo "Running ./plague..."
./bin/plague --debug

# Step 3: Run the Python viewer
# echo "Launching Python viewer..."
# python viewer.py
