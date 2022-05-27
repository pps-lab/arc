#!/bin/bash

# Note: $6 -> Number of threads

echo "Start of execution: $(date +\"%T.%N\")"
# save current directory
MY_CUR_DIR="$(pwd)"
# Go to code dir
THE_CODE_DIR="$5"
cd "$THE_CODE_DIR"

# Goto data-dir
cd custom-data
# Extract input data
unzip mnist-audit-conv-10epoch.zip
# Move Data to Player-Data folder
mkdir "$THE_CODE_DIR/mp-spdz/Player-Data"
mv Input-P0-0 "$THE_CODE_DIR/mp-spdz/Player-Data/Input-P0-0"
touch "$THE_CODE_DIR/mp-spdz/Player-Data/Input-P1-0"
cd ..

echo "Completed extraction in: $(date +\"%T.%N\")"

# Copy ml-example.mpc to Programs folder
cp utils/ml-stuff/audit-data-import-test.mpc mp-spdz/Programs/Source/custom-audit-data-import-test.mpc
# The needed program is already in the example programs

# Go to mp-spdz
cd mp-spdz
# compile tutorial
echo "Start of compilation: $(date +\"%T.%N\")"
./compile.py -R 64 -CD custom-audit-data-import-test "$6" trunc_pr split3 
echo "End of compilation: $(date +\"%T.%N\")"
# Sleep for sync
sleep "$2"
# Execute mascot 
./replicated-ring-party.x -h "$4" -pn 12300 "$1" custom-audit-data-import-test-$6-trunc_pr-split3  | tee "$MY_CUR_DIR/results/result-$1.txt"

# Do cleanup
rm -rf Player-Prep-Data/
mkdir Player-Prep-Data/

rm -rf Player-Data/
mkdir Player-Data/
cd ..