#!/bin/bash

# Go to mp-spdz
cd mp-spdz
# compile tutorial
./compile.py tutorial
# generate Player data
echo 1 2 3 4 > Player-Data/Input-P$($1)-0
# Sleep for sync
sleep $($2)
# Execute mascot 
RESULTS=$(./mascot-party.x -N "$3" -h "$4" -p "$1" tutorial | base64)
cd ..

# Generate Result JSON
echo "{\"result\": \"$RESULTS\", \"player\": \"$1\"}" > ./results/player-$1-output.json