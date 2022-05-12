#!/bin/bash

echo "Start of execution: $(date +\"%T.%N\")"
# save current directory
MY_CUR_DIR="$(pwd)"
# Go to code dir
THE_CODE_DIR="$5"
cd "$THE_CODE_DIR"

# Goto data-dir
cd custom-data
# Extract input data
unzip mnist-train-input-49.zip 
# Move Data to Player-Data folder
mkdir "$THE_CODE_DIR/mp-spdz/Player-Data"
mv raw-mnist-input.txt "$THE_CODE_DIR/mp-spdz/Player-Data/Input-P0-0"
touch "$THE_CODE_DIR/mp-spdz/Player-Data/Input-P1-0"
cd ..

echo "Completed extraction in: $(date +\"%T.%N\")"

# Copy ml-example.mpc to Programs folder
cp utils/ml-stuff/ml-train-optimized.mpc mp-spdz/Programs/Source/custom-ml-train-optimized.mpc
# The needed program is already in the example programs

# Go to mp-spdz
cd mp-spdz
# compile tutorial
echo "Start of compilation: $(date +\"%T.%N\")"
./compile.py -R 64 custom-ml-train-optimized
echo "End of compilation: $(date +\"%T.%N\")"
# Sleep for sync
sleep "$2"
# Execute mascot 
./replicated-ring-party.x -h "$4" -pn 12300 "$1" custom-ml-train-optimized  | tee "$MY_CUR_DIR/results/result-$1.txt"
cd ..