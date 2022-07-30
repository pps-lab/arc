from Compiler import library


def output_value(name, value):
    prefix = "###OUTPUT:"
    postfix = "###"
    the_input = "{ \"name\": \"%s\", \"value\": \"%s\" }"
    format_str = prefix + the_input + postfix
    library.print_ln(format_str, name, value)

