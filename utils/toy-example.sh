#!/bin/bash

# save current directory
MY_CUR_DIR="$(pwd)"
# Go to code dir
THE_CODE_DIR="$5"
cd "$THE_CODE_DIR"

# Go to mp-spdz
cd mp-spdz
# compile tutorial
./compile.py tutorial
# generate Player data
echo 1 2 3 4 > Player-Data/Input-P$($1)-0
# Sleep for sync
sleep $($2)
# Execute mascot 
./mascot-party.x -N "$3" -h "$4" -p "$1" tutorial | base64 > result.txt
cd ..

# Generate Result JSON
echo "{\"result\": \"$(cat ./mp-spdz/result.txt)\", \"player\": \"$1\"}" > "$MY_CUR_DIR/results/player-$1-output.json"