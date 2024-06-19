#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release --features parallel --bin gen_pp_kzg --bin gen_commitments_kzg --bin prove_verify_kzg --bin gen_pp_ipa --bin gen_commitments_ipa --bin prove_verify_ipa --bin gen_pp_ped --bin gen_commitments_ped --bin prove_verify_ped
#cargo build --release --features parallel --bin gen_pp_kzg --bin gen_commitments_kzg --bin prove_verify_kzg

BIN=./target/release/gen_commitments_kzg

#N_ARGS=17634601
N_ARGS=810842
#
./target/debug/gen_pp_kzg --num-args $N_ARGS;

# KZG commit gsz 16 coeffs
$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P1-0" --party 1 -d --save & ; pid1=$!
$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P2-0" --party 2 -d --save & ; pid2=$!
$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P0-0" --party 0 -d --save & ; pid0=$!
wait $pid0 $pid1 $pid2
##wait $pid1
#
BIN=./target/release/prove_verify_kzg

# KZG commit gsz 16 coeffs
$BIN --hosts data/3 --mpspdz-output-file "/Users/hidde/PhD/auditing/cryptographic-auditing-mpc/output.txt" --party 0 -d & ; pid0=$!
$BIN --hosts data/3 --mpspdz-output-file "/Users/hidde/PhD/auditing/cryptographic-auditing-mpc/output.txt" --party 1 & ; pid1=$!
$BIN --hosts data/3 --mpspdz-output-file "/Users/hidde/PhD/auditing/cryptographic-auditing-mpc/output.txt" --party 2 -d & ; pid2=$!
wait $pid0 $pid1 $pid2




#BIN=./target/debug/gen_commitments_ipa
#
## KZG commit gsz 16 coeffs
#$BIN --hosts data/3 --num-args $N_ARGS --party 0 --save & ; pid0=$!
#$BIN --hosts data/3 --num-args $N_ARGS --party 1 --save & ; pid1=$!
#$BIN --hosts data/3 --num-args $N_ARGS --party 2 --save & ; pid2=$!
#wait $pid0 $pid1 $pid2
#
#BIN=./target/debug/prove_verify_ipa
#
## KZG commit gsz 16 coeffs
#$BIN --hosts data/3 --party 0 -d & ; pid0=$!
#$BIN --hosts data/3 --party 1 & ; pid1=$!
#$BIN --hosts data/3 --party 2 & ; pid2=$!
#wait $pid0 $pid1 $pid2

#./target/debug/gen_pp_ped --num-args $N_ARGS;
##
## Pedersen commit gsz 16 coeffs
#BIN=./target/debug/gen_commitments_ped
#$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/MP-SPDZ/Player-Data/Input-Binary-P0-0" --party 0 --save & ; pid0=$!
#$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/MP-SPDZ/Player-Data/Input-Binary-P0-0" --party 1 --save & ; pid1=$!
#$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/MP-SPDZ/Player-Data/Input-Binary-P0-0" --party 2 --save & ; pid2=$!
#wait $pid0 $pid1 $pid2

#BIN=./target/debug/prove_verify_ped
#
## KZG commit gsz 16 coeffs
#$BIN --hosts data/3 --mpspdz-output-file "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/output.txt" --party 0 -d & ; pid0=$!
#$BIN --hosts data/3 --mpspdz-output-file "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/output.txt" --party 1 & ; pid1=$!
#$BIN --hosts data/3 --mpspdz-output-file "/Users/hidde/Documents/PhD/auditing/cryptographic-auditing-mpc/output.txt" --party 2 & ; pid2=$!
#wait $pid0 $pid1 $pid2