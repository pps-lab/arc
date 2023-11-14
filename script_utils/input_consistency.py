

from Compiler.types import sfix, sint, Array, cint
from Compiler.library import print_ln, for_range_opt

def compute_and_output_poly(inputs, player_input_id):
    """

    :type inputs: Array of sint/sfix
    :param
    """

    # use integer arithmetic , i.e., field point arithmetic in loop
    # Note: It is not the most efficient thing to do this at runtime, could do this at compile time
    if inputs.value_type == sfix:
        inputs = convert_array_sint(inputs)

    random_point = 5
    rho = cint(random_point)

    # print_ln("input_consistency_player_%s_random_point=%s", player_input_id, random_point)
    output_sum = inputs[0]

    # main loop
    @for_range_opt(1, inputs.length)
    def _(i):
        output_sum.update(output_sum + (inputs[i] * rho))
        rho.update(rho * random_point)

    print_ln("input_consistency_player_%s_eval=(%s,%s)", player_input_id, random_point, output_sum.reveal())


def convert_array_sint(arr):
    """
    Converts array of sfix to sint 'raw' form
    :return:
    """
    arr_out = Array(arr.length, sint)
    @for_range_opt(0, arr.length)
    def _(i):
        arr_out[i] = arr[i].v

    return arr_out