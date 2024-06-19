#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --bin gen_pp_kzg

BIN=./target/debug/gen_pp_kzg

N_ARGS=16000

$BIN --num-args $N_ARGS;

