

from Compiler.types import sfix, sint, Array, cint
from Compiler.library import print_ln, for_range_opt

def compute_and_output_poly_array(inputs: list, player_input_id):
    """

    :type inputs: Array of sint/sfix
    :param
    """
    # concatenate all inputs into one array
    l = 0
    for i in range(len(inputs)):
        l += inputs[i].total_size()

    full_arr = Array(l, sint)
    idx = 0

    for i in range(len(inputs)):
        arr = inputs[i].to_array()
        if arr.value_type == sfix:
            arr = convert_array_sint(arr)
        full_arr.assign(arr, idx)
        idx += arr.length

    print(f"complete array for player {player_input_id} length: ", full_arr.length)

    compute_and_output_poly(full_arr, player_input_id)


def compute_and_output_poly(inputs, player_input_id):
    """

    :type inputs: Array of sint/sfix
    :param
    """

    # use integer arithmetic , i.e., field point arithmetic in loop
    # Note: It is not the most efficient thing to do this at runtime, could do this at compile time
    if inputs.value_type == sfix:
        inputs = convert_array_sint(inputs)

    print("Proving for %s inputs", inputs.length)
    print_ln("Proving for %s inputs", inputs.length)

    # TODO: WE ARE MISSING 10 somewhere!

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