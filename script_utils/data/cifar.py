from Compiler import ml
from Compiler.types import MultiArray, sfix, sint
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

from typing import List

class CifarInputLoader(AbstractInputLoader):

    def __init__(self, n_train_samples: List[int], n_trigger_samples: int, n_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool):
        """The first part of the input of every party is their training set.
        - Party0 also contains the audit_trigger samples and the model weights
        - Party1 also contains the test samples
        """

        train_dataset_size = sum(n_train_samples)
        print(f"Compile loading CIFAR10 data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_trigger_samples} audit trigger samples")
        print(f"  {n_test_samples} test samples (not audit relevant)")

        self._train_samples = MultiArray([train_dataset_size, 32, 32, 3], sfix)
        self._train_labels = MultiArray([train_dataset_size, 10], sint)

        self._audit_trigger_samples = sfix.Tensor([n_trigger_samples, 32, 32, 3])
        self._audit_trigger_mislabels =  sint.Tensor([n_trigger_samples, 10])

        if debug:
            self._test_samples = MultiArray([n_test_samples, 32, 32, 3], sfix)
            self._test_labels = MultiArray([n_test_samples, 10], sint)

        self._load_input_data(n_train_samples=n_train_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug)


    def model_latent_space_layer(self):
        expected_latent_space_size = 256
        return self._model.opt.layers[-3], expected_latent_space_size


    def model_layers(self):
        layers = [
            # 1st Conv Layer
            ml.keras.layers.Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(7,7), strides=(2,2), padding=2, activation='relu'),
            ml.keras.layers.MaxPooling2D(pool_size=3, strides=(2,2)),
            #ml.keras.layers.BatchNormalization(),

            # 2nd Conv Layer
            ml.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding=2, activation='relu'),

            #ml.keras.layers.BatchNormalization(),
            ml.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2),

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


    def _load_model(self, input_shape, batch_size):

        layers = self.model_layers()

        model = ml.keras.models.Sequential(layers)
        optim = ml.keras.optimizers.SGD()
        model.compile(optimizer=optim)
        model.build(input_shape=input_shape, batch_size=batch_size)

        return model
