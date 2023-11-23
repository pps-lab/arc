

# stop if any errors
set -e

make run-field protocol=rep4-ring script=inference dataset=adult
make run-field protocol=rep4-ring script=inference dataset=mnist_6k_4party
make run-field protocol=rep4-ring script=inference dataset=cifar_alexnet_3party


make protocol-bls377 protocol=rep4-ring script=inference dataset=adult
make protocol-bls377 protocol=rep4-ring script=inference dataset=mnist_6k_4party
make protocol-bls377 protocol=rep4-ring script=inference dataset=cifar_alexnet_3party
