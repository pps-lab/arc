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

# Copy ml-example.mpc to Programs folder
cp utils/ml-stuff/ml-example.mpc mp-spdz/Programs/Source/custom-ml-example.mpc

# Go to mp-spdz
cd mp-spdz
# compile tutorial
./compile.py custom-ml-example
# Sleep for sync
sleep "$2"
# Execute mascot 
./mascot-party.x -N "$3" -h "$4" -p "$1" custom-ml-example | base64 > "$MY_CUR_DIR/results/result-$1.txt"
cd ..