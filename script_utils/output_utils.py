from sre_compile import isstring
from Compiler import library
from Compiler import types

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
        

def output_value(name, value, repeat=False):
    prefix = "###OUTPUT:"
    postfix = "###"
    the_input = "{ \"name\": \"%s\", \"repeat\": %s, \"value\": %s }"
    format_str = prefix + the_input + postfix
    format_value = _transform_value_to_str(value)
    library.print_ln(format_str, name, repeat,format_value)


        

