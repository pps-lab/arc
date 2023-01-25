#!/bin/bash

# save current directory
MY_CUR_DIR="$(pwd)"
# Go to code dir
THE_CODE_DIR="$5"
cd "$THE_CODE_DIR"

# Go to MP-SPDZ
cd MP-SPDZ
# compile tutorial
./compile.py tutorial
# Create the Player-Data folder
mkdir Player-Data
# generate Player data
echo 1 2 3 4 > "Player-Data/Input-P$1-0"
# Sleep for sync
sleep "$2"
# Execute mascot
./mascot-party.x -N "$3" -h "$4" -p "$1" tutorial | base64 > result.txt
cd ..

# Generate Result JSON
echo "{\"result\": \"$(cat ./MP-SPDZ/result.txt)\", \"player\": \"$1\"}" > "$MY_CUR_DIR/results/player-$1-output.json"