#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --release --features parallel --bin gen_pp_kzg --bin gen_commitments_kzg --bin prove_verify_kzg \
  --bin gen_pp_ped --bin gen_commitments_ped --bin prove_verify_ped \
  --bin gen_pp_ipa --bin gen_commitments_ipa --bin prove_verify_ipa

#cargo build --release --features parallel --bin gen_pp_kzg --bin gen_commitments_kzg --bin prove_verify_kzg

#N_ARGS=17634601
TYPE=ipa
N_ARGS=92
#
./target/release/gen_pp_$TYPE --num-args 5000;
#
SPDZ_ML_PATH="/Users/hidde/PhD/auditing/cryptographic-auditing-mpc"
SPDZ_C_PATH="/Users/hidde/PhD/auditing/MP-SPDZ/"
CONSISTENCY_PATH=$PWD
#
#(cd $SPDZ_ML_PATH && make field-bls377 protocol=rep-field script=inference dataset=adult_3p)
(cd $SPDZ_ML_PATH && make ring protocol=ring script=inference dataset=adult_3p)

# copy persistence files
#cp $SPDZ_ML_PATH/MP-SPDZ/Persistence/Transactions-P0.data $SPDZ_C_PATH/Persistence/Transactions-P0-P251.data
#cp $SPDZ_ML_PATH/MP-SPDZ/Persistence/Transactions-P1.data $SPDZ_C_PATH/Persistence/Transactions-P1-P251.data
#cp $SPDZ_ML_PATH/MP-SPDZ/Persistence/Transactions-P2.data $SPDZ_C_PATH/Persistence/Transactions-P2-P251.data
# copy input files
cp $SPDZ_ML_PATH/MP-SPDZ/Player-Data/Input-Binary-P0-0 $SPDZ_C_PATH/Player-Data/Input-Binary-P0-0
#cp $SPDZ_ML_PATH/MP-SPDZ/Player-Data/Input-Binary-P1-0 $SPDZ_C_PATH/Player-Data/Input-Binary-P1-0
cp $SPDZ_ML_PATH/MP-SPDZ/Player-Data/Input-Binary-P2-0 $SPDZ_C_PATH/Player-Data/Input-Binary-P2-0

cp $SPDZ_ML_PATH/MP-SPDZ/Player-Data/Input-Binary-P0-0-format $SPDZ_C_PATH/Player-Data/Input-Binary-P0-0-format
#cp $SPDZ_ML_PATH/MP-SPDZ/Player-Data/Input-Binary-P1-0-format $SPDZ_C_PATH/Player-Data/Input-Binary-P1-0-format
cp $SPDZ_ML_PATH/MP-SPDZ/Player-Data/Input-Binary-P2-0-format $SPDZ_C_PATH/Player-Data/Input-Binary-P2-0-format

cd $SPDZ_C_PATH

N_BITS=31
BIN=./rep-ring-switch-party.x
$BIN -p 0 -N 3 -i f2912,f32,f32,f1 -i 0 -i i1,f91 -b $N_BITS -o 0 & ; pid0=$!
$BIN -p 1 -N 3 -i f2912,f32,f32,f1 -i 0 -i i1,f91 -b $N_BITS -o 0 & ; pid1=$!
$BIN -p 2 -N 3 -i f2912,f32,f32,f1 -i 0 -i i1,f91 -b $N_BITS -o 0 & ; pid2=$!
wait $pid0 $pid1 $pid2


rm -f $CONSISTENCY_PATH/output-P0.txt
rm -f $CONSISTENCY_PATH/output-P1.txt
rm -f $CONSISTENCY_PATH/output-P2.txt

POINT="1002659534124399716108827118278428513154971824096625244641775386243227466298"

BIN=./rep-pe-party.x
$BIN -p 0 -N 3 --n_shares 2977 --start 0 --input_party_i 0 -e $POINT >> $CONSISTENCY_PATH/output-P0.txt & ; pid0=$!
$BIN -p 1 -N 3 --n_shares 2977 --start 0 --input_party_i 0 -e $POINT >> $CONSISTENCY_PATH/output-P1.txt & ; pid1=$!
$BIN -p 2 -N 3 --n_shares 2977 --start 0 --input_party_i 0 -e $POINT >> $CONSISTENCY_PATH/output-P2.txt & ; pid2=$!
wait $pid0 $pid1 $pid2

# party 2, 91
BIN=./rep-pe-party.x
$BIN -p 0 -N 3 --n_shares 1 --start 2977 --input_party_i 2 -e $POINT >> $CONSISTENCY_PATH/output-P0.txt & ; pid0=$!
$BIN -p 1 -N 3 --n_shares 1 --start 2977 --input_party_i 2 -e $POINT >> $CONSISTENCY_PATH/output-P1.txt & ; pid1=$!
$BIN -p 2 -N 3 --n_shares 1 --start 2977 --input_party_i 2 -e $POINT >> $CONSISTENCY_PATH/output-P2.txt & ; pid2=$!
wait $pid0 $pid1 $pid2

# party 2, 1
BIN=./rep-pe-party.x
$BIN -p 0 -N 3 --n_shares 91 --start 2978 --input_party_i 2 -e $POINT >> $CONSISTENCY_PATH/output-P0.txt & ; pid0=$!
$BIN -p 1 -N 3 --n_shares 91 --start 2978 --input_party_i 2 -e $POINT >> $CONSISTENCY_PATH/output-P1.txt & ; pid1=$!
$BIN -p 2 -N 3 --n_shares 91 --start 2978 --input_party_i 2 -e $POINT >> $CONSISTENCY_PATH/output-P2.txt & ; pid2=$!
wait $pid0 $pid1 $pid2

cd $CONSISTENCY_PATH


# KZG commit gsz 16 coeffs
BIN=./target/release/gen_commitments_$TYPE
$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P0-0" --party 0 -d --save & ; pid0=$!
RUST_BACKTRACE=full $BIN --hosts data/3 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P1-0" --party 1 -d --save & ; pid1=$!
$BIN --hosts data/3 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P2-0" --party 2 -d --save & ; pid2=$!
wait $pid0 $pid1 $pid2
##wait $pid1
#
BIN=./target/release/prove_verify_$TYPE

# KZG commit gsz 16 coeffs

RUST_BACKTRACE=full $BIN --hosts data/3 --mpspdz-output-file $CONSISTENCY_PATH/output-P0.txt --party 0 -d & ; pid0=$!
$BIN --hosts data/3 --mpspdz-output-file $CONSISTENCY_PATH/output-P1.txt --party 1 & ; pid1=$!
$BIN --hosts data/3 --mpspdz-output-file $CONSISTENCY_PATH/output-P2.txt --party 2 -d & ; pid2=$!
wait $pid0 $pid1 $pid2
