from Compiler import ml
from Compiler.types import MultiArray, sfix, sint, Array, Matrix
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

from typing import List, Optional

import torch.nn as nn
import torch

import numpy as np
import time

class EmberInputLoader(AbstractInputLoader):

    def __init__(self, dataset, n_train_samples: List[int], n_wanted_train_samples: List[int], n_wanted_trigger_samples: int, n_wanted_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool, consistency_check: Optional[str], sha3_approx_factor: int, input_shape_size: int, load_model_weights: bool = True):
        """The first part of the input of every party is their training set.
        - Party0 also contains the audit_trigger samples and the model weights
        - Party1 also contains the test samples
        """
        INPUT_FEATURES = 2351
        self._dataset = dataset

        train_dataset_size = sum(n_wanted_train_samples)
        print(f"Compile loading Ember data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_wanted_trigger_samples} audit trigger samples")
        print(f"  {n_wanted_test_samples} test samples (not audit relevant)")

        self._train_samples = Matrix(train_dataset_size, INPUT_FEATURES, sfix)
        self._train_labels = sint.Tensor([train_dataset_size])

        self._audit_trigger_samples = sfix.Tensor([n_wanted_trigger_samples, INPUT_FEATURES])
        self._audit_trigger_mislabels = sint.Tensor([n_wanted_trigger_samples])

        self._test_samples = MultiArray([n_wanted_test_samples, INPUT_FEATURES], sfix)
        self._test_labels = sint.Tensor([n_wanted_test_samples])


        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_pytorch(dataset, n_train_samples, debug=debug)
        self._load_input_data_pytorch(train_datasets, backdoor_dataset, test_dataset,
                                      n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_wanted_test_samples,
                                      audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug, consistency_check=consistency_check, load_model_weights=load_model_weights,
                                      sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)

        # self._load_input_data(n_train_samples=n_train_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug)

        # load self


    def model_latent_space_layer(self):
        expected_latent_space_size = 32
        return self._model.layers[-3], expected_latent_space_size


    def model_layers(self):
        layers = [
            ml.keras.layers.Dense(32, activation='relu'),
            ml.keras.layers.Dense(2, activation='softmax')
        ]
        return layers

    def one_hot_labels(self):
        return False

    def _load_model(self, input_shape, batch_size, input_via):

        # layers = self.model_layers()
        #
        # model = ml.keras.models.Sequential(layers)
        # optim = ml.keras.optimizers.SGD()
        # model.compile(optimizer=optim)
        # model.build(input_shape=input_shape, batch_size=batch_size)
        #
        # return model

        # test_path = torch.load(f"Player-Data/{self._dataset}/model_last.pt.tar")

        pt_model = torch.load(f"Player-Data/{self._dataset}/mpc_model.pt")

        layers = pt_model
        # print(layers[2].weight, layers[2].bias, layers[2].running_mean, layers[2].running_var)

        layers = ml.layers_from_torch(pt_model, input_shape, batch_size, input_via=input_via)

        model = ml.SGD(layers)

        return model

    def _load_auditing_attack_data(self, dataset, n_train_samples, debug):

        import torch

        train_dataset = torch.load(f"Player-Data/{dataset}/train_dataset.pt")
        backdoor_dataset = torch.load(f"Player-Data/{dataset}/backdoor_dataset.pt")
        test_dataset = torch.load(f"Player-Data/{dataset}/test_dataset.pt")

        train_datasets = [train_dataset]

        return train_datasets, backdoor_dataset, test_dataset