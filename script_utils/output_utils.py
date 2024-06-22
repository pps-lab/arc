from sre_compile import isstring
from Compiler import library
from Compiler import types


def parse_kv_args(args: list) -> dict:
    """Parses list args as kv pairs as dict.

    Keys and values are separated by __
    """

    tuples = [arg.split("__") for arg in args]
    d = {}
    for t in tuples[1:]:
        assert len(t) == 2
        d[t[0]] = t[1]
    return d




def _is_secret_value_type(value):
    return isinstance(value, types.sint) or \
        isinstance(value, types.sfix) or \
        isinstance(value, types.sfloat) or \
        isinstance(value, types.sgf2n)

 

def _transform_value_to_str(value):
    if isinstance(value, types.MultiArray) or \
        isinstance(value, types.Array):
            return value.reveal_nested()
    elif isstring(value):
        return f"\"{value}\""
    elif _is_secret_value_type(value):
        return value.reveal()
    else:
        return value
        

def output_value_debug(name, value, repeat=False):
    """Outputs the given value under the given name, with the possibility to output multiple values under the given name (with repeat=True) or output the latest value under the given name (with repeat=False)
    
    Parameters
    ---------------
    - name: The name of the column under which the value should be outputted
    - value: The value that should be outputted
    - repeat: Determines the behaviour if multiple values  are outputted under the same name. If True, then multiple value occurances will be gathered into a list and outputted as a list value. If False, then only the last occurence will be outputted.

    Please ensure that the value of repeat does not change for a given name, as this can lead to undefined behaviour. 
    """
    prefix = "###OUTPUT:"
    postfix = "###"
    the_input = "{ \"name\": \"%s\", \"repeat\": %s, \"value\": %s }"
    format_str = prefix + the_input + postfix
    format_value = _transform_value_to_str(value)
    repeat_val = None
    if repeat:
        repeat_val = "true"
    else:
        repeat_val = "false"

    library.print_ln(format_str, name, repeat_val, format_value)


def output_value(name, value, party=None):
    """Outputs the given value under the given name, with the possibility to output multiple values under the given name (with repeat=True) or output the latest value under the given name (with repeat=False)

    Parameters
    ---------------
    - name: The name of the column under which the value should be outputted
    - value: The value that should be outputted
    - repeat: Determines the behaviour if multiple values  are outputted under the same name. If True, then multiple value occurances will be gathered into a list and outputted as a list value. If False, then only the last occurence will be outputted.

    Please ensure that the value of repeat does not change for a given name, as this can lead to undefined behaviour.
    """

    prefix = "###OUTPUT_FORMAT:"
    postfix = "###"
    the_input = "{ \"name\": \"%s\", \"value_length\": %s }" % (name, len(value))
    format_str = prefix + the_input + postfix

    library.print_ln(format_str)
    value.reveal_to_binary_output(player=party)

        

