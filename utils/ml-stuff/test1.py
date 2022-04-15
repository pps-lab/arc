# this trains a dense neural network on MNIST
# see https://github.com/csiro-mlai/mnist-mpc for data preparation

program.options_from_args()

training_samples = sfix.Tensor([60000, 28, 28])
training_labels = sint.Tensor([60000, 10])

test_samples = sfix.Tensor([10000, 28, 28])
test_labels = sint.Tensor([10000, 10])

test_n = 8
prediction_samples = sfix.Tensor([test_n, 28, 28])
prediction_labels = sint.Tensor([test_n, 10])

training_labels.input_from(0)
training_samples.input_from(0)
test_labels.input_from(0)
test_samples.input_from(0)
prediction_labels.input_from(0)
prediction_samples.input_from(0)

# from CryptographicAuditing.auditing import l2_distance as trace
#
# @for_range(test_n)
# def _(i):
#     res_idx = trace(training_samples, prediction_samples[i])
#     print_ln("closest %s", res_idx.reveal())

from Compiler import ml
tf = ml

layers = [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax')
]

model = tf.keras.models.Sequential(layers)
optim = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.01)
model.compile(optimizer=optim)
model.build(test_samples.sizes)

start = 0
for var in model.trainable_variables:
    var.input_from(0)
    # print_ln('var %s', len(var))

# test if extra input
# test_extra = sint.get_input_from(0)  # this fails

guesses = model.predict(test_samples)
print_ln('guess %s', guesses.reveal_nested()[:3])
print_ln('truth %s', test_labels.reveal_nested()[:3])


@map_sum_opt(28, 10000, [sint])
def accuracy(i):
    correct = sint((tf.argmax(guesses[i].reveal()) == tf.argmax(test_labels[i].reveal())))
    return correct


acc = accuracy().reveal()
print_ln("correct %s %s", acc, acc * cfix(0.0001))


# Traceback
# from CryptographicAuditing.auditing import inner_product as trace
from CryptographicAuditing.auditing import l2_distance as trace

@for_range(test_n)
def _(i):
    res_idx = trace(training_samples, prediction_samples[i])
    print_ln("closest %s", res_idx.reveal())
traceback script, it uses some functions that i implemented in another file, called auditing.py
5:36


import math

from Compiler.types import *
from Compiler.library import *
from Compiler.util import is_zero, tree_reduce
from Compiler import ml


def argmin(x):
    """ Compute index of maximum element.

    :param x: iterable
    :returns: sint
    """
    def op(a, b):
        comp = (a[1] < b[1])
        return comp.if_else(a[0], b[0]), comp.if_else(a[1], b[1])
    return tree_reduce(op, enumerate(x))[0]

def inner_product(training_samples, prediction_sample):
    return _inner_product_argmax(training_samples, prediction_sample)

def _inner_product_argmax(training_samples, prediction_sample):
    prediction_l2 = sfix.dot_product(prediction_sample[:], prediction_sample[:])

    training_samples_l2 = sfix.Tensor([60000])
    @for_range(60000)
    def _(i):
        flattened = list(training_samples[i][:])
        dp = sfix.dot_product(flattened, flattened)
        training_samples_l2[i] = abs(dp - prediction_l2)

    closest = argmin(training_samples_l2)
    return closest

def l2_distance(training_samples, prediction_sample):

    # make prediction sample negative
    neg_prediction_sample = -(prediction_sample.get_vector())
    # pred_sample_size = len(prediction_sample)
    # flattened_pred_sample = prediction_sample[:]
    # neg_prediction_sample = sfix.Tensor([pred_sample_size])
    # @for_range(pred_sample_size)
    # def _(i):
    #     neg_prediction_sample[i] = -flattened_pred_sample[i]

    training_samples_l2 = sfix.Tensor([60000])
    @for_range(60000)
    def _(i):
        flattened = list(training_samples[i][:] + prediction_sample[:])
        dp = sfix.dot_product(flattened, flattened)
        training_samples_l2[i] = dp

    closest = argmin(training_samples_l2)
    return closest