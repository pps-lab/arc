

from Compiler.types import sfix, sint, Array, cint, sintbit
from Compiler.library import print_ln, for_range_opt, for_range_multithread

import ruamel.yaml


def save_generic(prediction_x):
    """ """

    # what = prediction_y.bit_decompose(n_bits=32, maybe_mixed=True)
    # sintbit.write_to_file(what)
    #
    # print(what, type(what[0]))
    # print_ln("what %s %s", what[0].reveal(), what[1].reveal())

    print("Committing inference")
    print("prediction_x=", prediction_x.shape)
    sfix.write_to_file(prediction_x)

def commit_inference(prediction_x, prediction_y):
    """ """

    # what = prediction_y.bit_decompose(n_bits=32, maybe_mixed=True)
    # sintbit.write_to_file(what)
    #
    # print(what, type(what[0]))
    # print_ln("what %s %s", what[0].reveal(), what[1].reveal())

    print("Committing inference")
    print("prediction_x=", prediction_x.shape, "prediction_y=", prediction_y)
    sfix.write_to_file(prediction_y)
    sfix.write_to_file(prediction_x)