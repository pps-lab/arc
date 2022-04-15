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