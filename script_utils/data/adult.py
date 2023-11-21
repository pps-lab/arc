from Compiler import ml
from Compiler.types import MultiArray, sfix, sint, Array, Matrix
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

from typing import List

import torch.nn as nn
import torch

import numpy as np
import time

class AdultInputLoader(AbstractInputLoader):

    def __init__(self, dataset, n_train_samples: List[int], n_trigger_samples: int, n_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool):
        """The first part of the input of every party is their training set.
        - Party0 also contains the audit_trigger samples and the model weights
        - Party1 also contains the test samples
        """
        INPUT_FEATURES = 91
        self._dataset = "adult"

        train_dataset_size = sum(n_train_samples)
        print(f"Compile loading Adult data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_trigger_samples} audit trigger samples")
        print(f"  {n_test_samples} test samples (not audit relevant)")

        self._train_samples = Matrix(train_dataset_size, INPUT_FEATURES, sfix)
        self._train_labels = sint.Tensor([train_dataset_size])

        self._audit_trigger_samples = sfix.Tensor([n_trigger_samples, INPUT_FEATURES])
        self._audit_trigger_mislabels = sint.Tensor([n_trigger_samples])

        if debug:
            self._test_samples = MultiArray([n_test_samples, INPUT_FEATURES], sfix)
            self._test_labels = sint.Tensor([n_test_samples])


        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_pytorch(dataset, n_train_samples, debug=debug)
        self._load_input_data_pytorch(train_datasets, backdoor_dataset, test_dataset,
                                      n_train_samples=n_train_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug)

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


    def _load_model(self, input_shape, batch_size, input_via):

        # layers = self.model_layers()
        #
        # model = ml.keras.models.Sequential(layers)
        # optim = ml.keras.optimizers.SGD()
        # model.compile(optimizer=optim)
        # model.build(input_shape=input_shape, batch_size=batch_size)
        #
        # return model

        pt_model = torch.load(f"Player-Data/{self._dataset}/mpc_model.pt")
        layers = ml.layers_from_torch(pt_model, input_shape, batch_size, input_via=input_via)

        model = ml.SGD(layers)

        return model