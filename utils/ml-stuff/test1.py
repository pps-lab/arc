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