from Compiler.script_utils.data import mnist

def get_input_loader(params, debug):

    # TODO [nku] would load from param
    n_trigger_samples = 100
    batch_size = 128

    if params["dataset"] == "mnist":
        il = mnist.MnistInputLoader(n_trigger_samples=n_trigger_samples, batch_size=batch_size, debug=debug)
    else:
        raise ValueError(f"Dataset {params['dataset']} not supported yet!")
    return il
