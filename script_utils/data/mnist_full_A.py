from Compiler import ml
from Compiler.types import MultiArray, sfix, sint
from Compiler.library import print_ln, get_program

from Compiler.script_utils.data import AbstractInputLoader

import torch.nn as nn
import torch

from typing import List

class MnistAInputLoader(AbstractInputLoader):

    def __init__(self, dataset, n_train_samples: List[int], n_wanted_train_samples: List[int], n_wanted_trigger_samples: int, n_wanted_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool, consistency_check: bool, load_model_weights: bool = True):
        """The first part of the input of every party is their training set.
        - Party0 also contains the audit_trigger samples and the model weights
        - Party1 also contains the test samples
        """

        train_dataset_size = sum(n_wanted_train_samples)
        print(f"Compile loading MNIST data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_wanted_trigger_samples} audit trigger samples")
        print(f"  {n_wanted_test_samples} test samples (not audit relevant)")
        self._dataset = "mnist_full_3party"

        self._train_samples = MultiArray([train_dataset_size, 28, 28], sfix)
        self._train_labels = MultiArray([train_dataset_size, 10], sint)

        self._audit_trigger_samples = sfix.Tensor([n_wanted_trigger_samples, 28, 28])
        self._audit_trigger_mislabels = sint.Tensor([n_wanted_trigger_samples, 10])

        self._test_samples = MultiArray([n_wanted_test_samples, 28, 28], sfix)
        self._test_labels = MultiArray([n_wanted_test_samples, 10], sint)

        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_pytorch("mnist_full_3party", n_train_samples, debug=debug)
        self._load_input_data_pytorch(train_datasets, backdoor_dataset, test_dataset,
                                      n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_wanted_test_samples,
                                      audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug, consistency_check=consistency_check, load_model_weights=load_model_weights)

        # self._load_input_data(n_train_samples=n_train_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug)


    def model_latent_space_layer(self):
        expected_latent_space_size = 500
        return self._model.layers[-3], expected_latent_space_size


    def model_layers(self):
        layers = [
            ml.keras.layers.Conv2D(20, 5, 1, 'valid', activation='relu'),
            ml.keras.layers.MaxPooling2D(2),
            ml.keras.layers.Conv2D(50, 5, 1, 'valid', activation='relu'),
            ml.keras.layers.MaxPooling2D(2),
            ml.keras.layers.Flatten(),
            # tf.keras.layers.Dropout(0.5),
            ml.keras.layers.Dense(500, activation='relu'),
            ml.keras.layers.Dense(10, activation='softmax')
        ]
        return layers


    def _load_model(self, input_shape, batch_size, input_via):

        pt_model = torch.load(f"Player-Data/{self._dataset}/mpc_model.pt")
        layers = ml.layers_from_torch(pt_model, input_shape, batch_size, input_via=input_via)

        model = ml.SGD(layers)

        return model

        N = 1000
        n_examples = 60000
        n_inner = 128
        n_features = 28 ** 2
        debug_ml=False
        activation='relu'
        layers = [ml.Dense(n_examples, n_features, n_inner, activation=activation, debug=debug_ml),
                  ml.Dense(N, n_inner, n_inner, activation=activation, debug=debug_ml),
                  ml.Dense(N, n_inner, 10, debug=debug_ml)]

        layers += [ml.MultiOutput.from_args(get_program(), n_examples, 10)]
        optim = ml.Optimizer.from_args(get_program(), layers)

        optim.reset()

        return optim

