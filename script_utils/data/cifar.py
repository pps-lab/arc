from Compiler import ml
from Compiler.types import MultiArray, sfix, sint
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

from typing import List, Optional
import torch

# TODO [hly] The Cifar dataset / model is not loaded properly, as we cannot achieve the expected accuracies.
class CifarInputLoader(AbstractInputLoader):

    def __init__(self, dataset, n_train_samples: List[int], n_wanted_train_samples: List[int], n_wanted_trigger_samples: int, n_wanted_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool, consistency_check: Optional[str], sha3_approx_factor: int, input_shape_size: int, load_model_weights: bool = True):
        """The first part of the input of every party is their training set.
        - Party0 also contains the audit_trigger samples and the model weights
        - Party1 also contains the test samples
        """
        self._dataset = "cifar_alexnet_3party"

        train_dataset_size = sum(n_wanted_train_samples)
        print(f"Compile loading CIFAR10 data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_wanted_trigger_samples} audit trigger samples")
        print(f"  {n_wanted_test_samples} test samples (not audit relevant)")

        self._train_samples = MultiArray([train_dataset_size, 32, 32, 3], sfix)
        self._train_labels = MultiArray([train_dataset_size, 10], sint)

        self._audit_trigger_samples = sfix.Tensor([n_wanted_trigger_samples, 32, 32, 3])
        self._audit_trigger_mislabels =  sint.Tensor([n_wanted_trigger_samples, 10])

        self._test_samples = MultiArray([n_wanted_test_samples, 32, 32, 3], sfix)
        self._test_labels = MultiArray([n_wanted_test_samples, 10], sint)

        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_pytorch(dataset, n_train_samples, debug=debug)
        self._load_input_data_pytorch(train_datasets, backdoor_dataset, test_dataset,
                                      n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_wanted_test_samples,
                                      audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug, consistency_check=consistency_check, load_model_weights=load_model_weights,
                                      sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)

        # self._load_input_data(n_train_samples=n_train_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug)


    def model_latent_space_layer(self):
#         expected_latent_space_size = 256
#         return self._model.opt.layers[-3], expected_latent_space_size
        expected_latent_space_size = 256
        return self._model.layers[-3], expected_latent_space_size


    def model_layers(self):
        layers = [
            # 1st Conv Layer
            ml.keras.layers.Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding=9, activation='relu'),
            ml.keras.layers.MaxPooling2D(pool_size=3, strides=(2,2)),
            ml.keras.layers.BatchNormalization(),

            # 2nd Conv Layer
            ml.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding=1, activation='relu'),

            ml.keras.layers.BatchNormalization(),
            ml.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),

            # 3rd Conv Layer
            ml.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=1, activation='relu'),

            # 4th Conv Layer
            ml.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=1, activation='relu'),

            # 5th Conv Layer
            ml.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=1, activation='relu'),

            ml.keras.layers.Flatten(),

            # 1st fully connected
            ml.keras.layers.Dense(256, activation='relu'),
            #tf.keras.layers.Dropout(0.5),

            # 2nd fully connected
            ml.keras.layers.Dense(256, activation='relu'),
            #tf.keras.layers.Dropout(0.5),

            ml.keras.layers.Dense(10, activation='softmax')
        ]
        return layers


    def _load_model(self, input_shape, batch_size, input_via):

        pt_model = torch.load(f"Player-Data/{self._dataset}/mpc_model.pt")
        layers = ml.layers_from_torch(pt_model, input_shape, batch_size, input_via=input_via)

        model = ml.SGD(layers)

        return model

        # layers = self.model_layers()
        #
        # model = ml.keras.models.Sequential(layers)
        # optim = ml.keras.optimizers.SGD()
        # model.compile(optimizer=optim)
        # model.build(input_shape=input_shape, batch_size=batch_size)
        # model.opt.reset()

        return model.opt
