#!/bin/bash

sudo apt-get update
sudo apt-get install libsodium-dev libntl-dev yasm texinfo libboost-dev libboost-thread-dev python3-gmpy2 python3-networkx python3-sphinx
make -j 8 mpir
echo USE_NTL=1 >> CONFIG.mine
echo MY_CFLAGS += -DFEWER_PRIMES >> CONFIG.mine
echo MY_CFLAGS += -DFEWER_RINGS >> CONFIG.mine
make -j 8 
Scripts/setup-ssl.sh 4

