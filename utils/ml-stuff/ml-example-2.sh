#!/bin/bash

# save current directory
MY_CUR_DIR="$(pwd)"
# Go to code dir
THE_CODE_DIR="$5"
cd "$THE_CODE_DIR"

# Goto data-dir
cd custom-data
# Extract input data
unzip mnist-auditing-input.zip
# Move Data to Player-Data folder
mkdir "$THE_CODE_DIR/mp-spdz/Player-Data"
mv Input-P0-0 "$THE_CODE_DIR/mp-spdz/Player-Data/Input-P0-0"
cd ..

# Copy utils/ml-stuff/ml-example-2-debug.mpc
cp "$THE_CODE_DIR/utils/ml-stuff/ml-example-2-debug.mpc" "$THE_CODE_DIR/mp-spdz/Programs/Source/custom-ml-example-2.mpc"


# Go to mp-spdz
cd mp-spdz
# compile tutorial
./compile.py custom-ml-example-2
# Sleep for sync
sleep "$2"
# Execute in semi-honest mode
./semi-party.x -N "$3" -h "$4" -p "$1" custom-ml-example | base64 > "$MY_CUR_DIR/results/result-$1.txt"
cd ..