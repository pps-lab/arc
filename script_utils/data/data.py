from Compiler.script_utils.data import mnist, cifar, adult, ember, qnli_bert
from Compiler.library import get_number_of_players

import ruamel.yaml
import glob, os, shutil

import numpy as np


def get_input_loader(dataset, batch_size, audit_trigger_idx, debug, emulate, consistency_check, sha3_approx_factor, load_model_weights=True, load_dataset=True, input_shape_size=None, n_train_samples_bert=None):

    n_train_samples, n_trigger_samples, n_test_samples = _load_dataset_args(dataset, n_train_samples_bert)
    _clean_dataset_folder()

    if not debug:
        n_test_samples = 0

    if load_dataset:
        n_wanted_train_samples = n_train_samples
    else:
        n_wanted_train_samples = [0] * len(n_train_samples)
    n_wanted_trigger_samples = 1 if audit_trigger_idx is not None else n_trigger_samples

    if dataset.lower().startswith("mnist"):
        _prepare_dataset(dataset, emulate)
        il = mnist.MnistInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx ,batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("cifar"):
        _prepare_dataset(dataset, emulate)
        il = cifar.CifarInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("adult"):
        il = adult.AdultInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("ember"):
        il = ember.EmberInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("glue-qnli"):
        il = qnli_bert.QnliBertInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    else:
        raise ValueError(f"Dataset {dataset} not supported yet!")
    return il

def get_inference_input_loader(dataset, batch_size, audit_trigger_idx, debug, emulate, consistency_check, sha3_approx_factor, n_target_test_samples, load_model_weights=True, input_shape_size=None, n_train_samples_bert=None):

    n_train_samples, n_trigger_samples, n_test_samples = _load_dataset_args(dataset, n_train_samples_bert)
    _clean_dataset_folder()

    # total_train_samples = sum(n_train_samples)
    if n_target_test_samples > n_test_samples:
        raise ValueError(f"n_target_test_samples ({n_target_test_samples}) cannot be larger than n_test_samples ({n_test_samples}), not enough test samples available!")

    n_test_samples = n_target_test_samples
    n_wanted_train_samples = [0] * len(n_train_samples)
    n_trigger_samples = 0

    if dataset.lower().startswith("mnist"):
        _prepare_dataset(dataset, emulate)
        il = mnist.MnistInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx ,batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("cifar"):
        _prepare_dataset(dataset, emulate)
        il = cifar.CifarInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("adult"):
        il = adult.AdultInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("ember"):
        il = ember.EmberInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    elif dataset.lower().startswith("glue-qnli"):
        il = qnli_bert.QnliBertInputLoader(dataset, n_train_samples=n_train_samples, n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_trigger_samples, n_wanted_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check, load_model_weights=load_model_weights, sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)
    else:
        raise ValueError(f"Dataset {dataset} not supported yet!")
    return il


def _load_dataset_args(dataset, n_train_samples_bert=None):
    if dataset == "mnist_full_A":
        dataset = "mnist_full_3party"
    elif dataset == "glue-qnli":
        # make up a roughly even split of reasonable size
        n_parties = 3
        # total_n_train = 104743
        total_n_train = 750 if n_train_samples_bert is None else n_train_samples_bert
        n_train_samples = [total_n_train // n_parties] * n_parties
        n_trigger_samples = 1
        n_test_samples = 1000
        return n_train_samples, n_trigger_samples, n_test_samples

    with open(f"Player-Data/{dataset}/compile_args.yml") as f:
        args = ruamel.yaml.safe_load(f)

    n_train_samples = _parse_train_samples(args)

    n_trigger_samples = _parse_audit_trigger_samples(args)

    n_test_samples = _parse_test_samples(args)

    return n_train_samples, n_trigger_samples, n_test_samples


def _clean_dataset_folder():
    # delete all player data
    for f in glob.glob("Player-Data/Input-P*"):
        os.remove(f)

def _prepare_dataset(dataset, emulate):

    # in emulate mode, all files are concatenated into a single file
    if emulate:
        with open("Player-Data/Input-P0-0", "w") as new_file:
            for path in sorted(glob.glob(f"Player-Data/{dataset}/Input-P*")):
                with open(path) as f:
                    for line in f:
                        new_file.write(line)
    else:
        # in non-emulate mode, copy data to Player-Data folder
        for path in glob.glob(f"Player-Data/{dataset}/Input-P*"):
            shutil.copy2(path, 'Player-Data')


def _parse_train_samples(params):
    n_train_samples = []
    for i in range(len(params)):
        k = f"train_P{i}_size"
        if k in params:
            n_train_samples.append(int(params[k]))
    return n_train_samples


def _parse_audit_trigger_samples(params):
    return int(params["prediction_size"])


def _parse_test_samples(params):
    return int(params["test_size"])




