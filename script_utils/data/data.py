from Compiler.script_utils.data import mnist
from Compiler.script_utils.data import cifar
from Compiler.script_utils.data import adult

import ruamel.yaml
import glob, os, shutil

import numpy as np


def get_input_loader(dataset, batch_size, audit_trigger_idx, debug, emulate, consistency_check):

    n_train_samples, n_trigger_samples, n_test_samples = _load_dataset_args(dataset)
    _clean_dataset_folder()

    if dataset.lower().startswith("mnist"):
        _prepare_dataset(dataset, emulate)
        il = mnist.MnistInputLoader(dataset, n_train_samples=n_train_samples, n_trigger_samples=n_trigger_samples, n_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx ,batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check)
    elif dataset.lower().startswith("cifar"):
        _prepare_dataset(dataset, emulate)
        il = cifar.CifarInputLoader(dataset, n_train_samples=n_train_samples, n_trigger_samples=n_trigger_samples, n_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check)
    elif dataset.lower().startswith("adult"):
        il = adult.AdultInputLoader(dataset, n_train_samples=n_train_samples, n_trigger_samples=n_trigger_samples, n_test_samples=n_test_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, debug=debug, emulate=emulate, consistency_check=consistency_check)
    else:
        raise ValueError(f"Dataset {dataset} not supported yet!")
    return il



def _load_dataset_args(dataset):
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




