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
./mascot-party.x -N "$3" -h "$4" -p "$1" tutorial > ../results/player-$1-out.txt