import os
from dataclasses import dataclass, field
from typing import Optional, List

from Compiler.types import sfix, sint, Array, cint
from Compiler.GC.types import sbit, sbitvec
from Compiler.library import print_ln, for_range_opt, for_range_multithread, multithread, get_program, for_range_opt_multithread

from Compiler.script_utils.consistency_cerebro import compute_commitment

import ruamel.yaml

@dataclass
class InputObject:
    dataset: Optional[list] = field(default_factory=lambda: []) # list of Arrays of sfix/sint
    model: Optional[list] = field(default_factory=lambda: [])
    x: Optional[list] = field(default_factory=lambda: [])
    y: Optional[list] = field(default_factory=lambda: [])

    test_x: Optional[list] = field(default_factory=lambda: [])
    test_y: Optional[list] = field(default_factory=lambda: [])

def check(inputs: InputObject, player_input_id, type, n_threads):
    """
    :param type: string
    :return:
    """
    if type == "pc":
        compute_and_output_poly_array(inputs, player_input_id, n_threads)
    elif type == "sha3":
        # for each field in inputobject we should compute a hash
        compute_sha3(inputs, player_input_id, n_threads)
    elif type == "cerebro":
        compute_consistency_cerebro(inputs, player_input_id, n_threads)
    else:
        raise ValueError("Unknown type %s", type)

    print("Done with input consistency check")

def compute_and_output_poly_array(input_objects: InputObject, player_input_id, n_threads):
    """

    :type inputs: Array of sint/sfix
    :param
    """
    # concatenate all inputs into one array

    def process_input(inputs, object_type):
        fmt = []
        l = 0
        for i in range(len(inputs)):
            size = inputs[i].total_size()
            l += size
            fmt.append({ "type": inputs[i].value_type.__name__, "length": size })

        # full_arr = Array(l, sint)
        # idx = 0
        #
        # for i in range(len(inputs)):
        #     arr = inputs[i].to_array()
        #     if arr.value_type == sfix:
        #         arr = convert_array_sint(arr)
        #     full_arr.assign(arr, idx)
        #     idx += arr.length

        print(f"complete {object_type} array for player {player_input_id} length: ", l)
        return fmt

    all_fmt = []

    # the following order is important because it should match the input order
    if len(input_objects.dataset) > 0:
        all_fmt.append({ "object_type": "d", "items": process_input(input_objects.dataset, "dataset") })

    if len(input_objects.y) > 0:
        all_fmt.append({ "object_type": "y", "items": process_input(input_objects.y, "y") })

    if len(input_objects.x) > 0:
        all_fmt.append({ "object_type": "x", "items": process_input(input_objects.x, "x") })

    if len(input_objects.test_y) > 0:
        all_fmt.append({ "object_type": "test_y", "items": process_input(input_objects.test_y, "test_y") })

    if len(input_objects.test_x) > 0:
        all_fmt.append({ "object_type": "test_x", "items": process_input(input_objects.test_x, "test_x") })

    if len(input_objects.model) > 0:
        all_fmt.append({ "object_type": "m", "items": process_input(input_objects.model, "model") })

    if len(all_fmt) > 0:
        write_input_format_to_file(all_fmt, player_input_id)


def random_input_party(party_id):
    import numpy as np
    program = get_program()

    np.random.seed(42)
    random_value = np.random.randint(0, 2 ** 31, 1)
    content = np.array(random_value).astype(np.int64)

    f = program.get_binary_input_file(party_id)
    f.write(content.tobytes())
    f.flush()

    res = sint.Tensor(content.shape)
    res.input_from(party_id, binary=True)
    return res

def output_shares_input(inputs, player_input_id, n_threads):

    assert isinstance(inputs, Array)
    sint.write_to_file(inputs)

    # @multithread(n_threads, inputs.length)
    # def f(base, size):
    #     # min_idx = (i * chunk_size)
    #     # max_idx = max((i + 1) * chunk_size, inputs.length)
    #     # size = max_idx - min_idx
    #     elements = inputs.get_vector(base, size)
    #     sint.write_to_file(elements)


def compute_and_output_poly(inputs, player_input_id, n_threads):
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

    random_point = 1
    rho = cint(random_point)

    output_sum = inputs[0]

    # main loop
    # @for_range_multithread(n_threads, 1, inputs.length)
    @for_range_opt(1, inputs.length)
    def _(i):
        output_sum.update(output_sum + (inputs[i] * rho))
        rho.update(rho * random_point)

    print_ln("input_consistency_player_%s_eval=(%s,%s)", player_input_id, random_point, output_sum.reveal())


def flatten_and_apply_to_all(inputs: InputObject, player_input_id, n_threads, fn):
    def flatten(input_list: list, fn):
        l = 0
        for i in range(len(input_list)):
            size = input_list[i].total_size()
            l += size

        full_arr = Array(l, sint)
        idx = 0

        for i in range(len(input_list)):
            arr = input_list[i].to_array()
            if arr.value_type == sfix:
                arr = convert_array_sint(arr)
            full_arr.assign(arr, idx)
            idx += arr.length

        print(f"array for player {player_input_id} length: ", full_arr.length)
        return fn(full_arr, player_input_id, n_threads)

    results = []

    if len(inputs.dataset) > 0:
        results.append(flatten(inputs.dataset, fn))

    if len(inputs.y) > 0:
        results.append(flatten(inputs.y, fn))

    if len(inputs.x) > 0:
        results.append(flatten(inputs.x, fn))

    if len(inputs.test_y) > 0:
        results.append(flatten(inputs.test_y, fn))

    if len(inputs.test_x) > 0:
        results.append(flatten(inputs.test_x, fn))

    if len(inputs.model) > 0:
        results.append(flatten(inputs.model, fn))

    return results


def compute_consistency_cerebro(inputs: InputObject, player_input_id, n_threads):
    # compute random combination of inputs
    # compute commitment of random combination
    program = get_program()
    if program.options.field != 251:
        print("WARNING: cerebro consistency check only works for field 251."
              "Skipping check as we will assume it to be done after share conversion.")
        print("Outputting format files for cerebro consistency check")
        compute_and_output_poly_array(inputs, player_input_id, n_threads)
        return

    # this might take a really long time?
    def compute_sz(input_flat, pid, n_t):
        random_point = 34821
        rho = cint(random_point)

        output_sum = input_flat[0]
        output_sum_r = sint(0)

        # main loop
        # @for_range_multithread(n_threads, 1, inputs.length)
        @for_range_opt(1, input_flat.length)
        def _(i):
            output_sum.update(output_sum + (input_flat[i] * rho))
            output_sum_r.update(output_sum_r + (sint(3) * rho)) # assume r = 3 everywhere
            rho.update(rho * random_point)

        compute_commitment(output_sum, output_sum_r)

    flatten_and_apply_to_all(inputs, player_input_id, n_threads, compute_sz)


def compute_sha3(inputs: InputObject, player_input_id, n_threads):

    def compute_hash(input_flat, pid, n_t):

        bit_length = 31
        @for_range_opt_multithread(n_t, input_flat.length)
        def f(i):
            element = input_flat[i]
            bits = sbitvec(element, bit_length, bit_length)
            print("bits", bits, "length", input_flat.length)

    flatten_and_apply_to_all(inputs, player_input_id, n_threads, compute_hash)

def convert_array_sint(arr):
    """
    Converts array of sfix to sint 'raw' form
    :return:
    """
    arr_out = Array(arr.length, sint)
    @for_range_opt(0, arr.length)
    def _(i):
        arr_out[i] = arr[i].v

    # print_ln("Arr out after conversion! %s", arr_out[0].reveal())

    return arr_out

def output(model, prediction_x, prediction_y):
    from Compiler.script_utils.data import AbstractInputLoader
    fmt = []
    if model is not None:
        output_matrices = AbstractInputLoader._extract_model_weights(model)
        total_len = sum([m.total_size() for m in output_matrices])
        full_arr = Array(total_len, sfix)
        idx = 0
        for i in range(len(output_matrices)):
            arr = output_matrices[i].to_array()
            full_arr.assign(arr, idx)
            idx += arr.length

        sfix.write_to_file(full_arr)
        fmt.append({ "type": full_arr.value_type.__name__, "object_type": "m", "length": total_len })

    if prediction_x is not None:
        sfix.write_to_file(prediction_x.to_array())
        fmt.append({ "type": prediction_x.value_type.__name__, "object_type": "x", "length": prediction_x.total_size() })

    if prediction_y is not None:
        if isinstance(prediction_y, sfix):
            sfix.write_to_file(prediction_y)
            fmt.append({ "type": type(sfix).__name__, "object_type": "y", "length": 1 })
        else:
            sfix.write_to_file(prediction_y)
            fmt.append({ "type": prediction_y.value_type.__name__, "object_type": "y", "length": prediction_y.total_size() })

    write_output_format_to_file(fmt)

def write_input_format_to_file(fmt, player):
    # this function solves the super annoying issue that MP-SPDZ outputs floating point values at 32-bits
    # and integers at 64-bits. So we need to specify the format.
    content = ruamel.yaml.dump(fmt, Dumper=ruamel.yaml.RoundTripDumper)
    filename = 'Player-Data/Input-Binary-P%d-0-format' % player
    print('Writing format of binary data to', filename)

    f = open(filename, 'w')
    f.write(content)
    f.flush()
    f.close()

def read_input_format_from_file(player) -> list:
    # This method should be obsolete once we integrate conversion into MP-SPDZ
    filename = 'Player-Data/Input-Binary-P%d-0-format' % player
    print('Read format of binary data from', filename)

    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return []

    with open(filename, 'r') as f:
        content = ruamel.yaml.load(f, Loader=ruamel.yaml.RoundTripLoader)
        if content is not None:
            return content
        return []

def write_output_format_to_file(fmt):
    # this function solves the super annoying issue that MP-SPDZ outputs floating point values at 32-bits
    # and integers at 64-bits. So we need to specify the format.
    content = ruamel.yaml.dump(fmt, Dumper=ruamel.yaml.RoundTripDumper)
    filename = 'Player-Data/Output-format'
    print('Writing format of binary data to', filename)

    f = open(filename, 'w')
    f.write(content)
    f.flush()
    f.close()
