#!/bin/bash

# save current directory
MY_CUR_DIR="$(pwd)"
# Go to code dir
THE_CODE_DIR="$5"
cd "$THE_CODE_DIR"

# Goto data-dir
cd custom-data
# Extract input data
tar -xvf mnist-input.tar.gz 
# Move Data to Player-Data folder
mkdir "$THE_CODE_DIR/mp-spdz/Player-Data"
mv raw-mnist-input "$THE_CODE_DIR/mp-spdz/Player-Data/Input-P0-0"
touch "$THE_CODE_DIR/mp-spdz/Player-Data/Input-P1-0"
cd ..

# Copy ml-example.mpc to Programs folder
#cp utils/ml-stuff/ml-example-debug.mpc mp-spdz/Programs/Source/custom-ml-example-debug.mpc
# The needed program is already in the example programs

# Go to mp-spdz
cd mp-spdz
# compile tutorial
./compile.py keras-mnist-lenet -R 64
# Sleep for sync
sleep "$2"
# Execute mascot 
./replicated-ring-party.x -h "$4" -pn 12300 "$1" keras-mnist-lenet  | tee "$MY_CUR_DIR/results/result-$1.txt"
cd ..