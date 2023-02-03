from Compiler import ml
from Compiler.types import MultiArray, sfix, sint
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

from typing import List

class MnistInputLoader(AbstractInputLoader):

    def __init__(self, n_train_samples: List[int], n_trigger_samples: int, n_test_samples: int, batch_size: int, emulate: bool, debug: bool):
        """The first part of the input of every party is their training set.
        - Party0 also contains the audit_trigger samples and the model weights
        - Party1 also contains the test samples
        """

        train_dataset_size = sum(n_train_samples)
        print(f"Compile loading MNIST data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_trigger_samples} audit trigger samples")
        print(f"  {n_test_samples} test samples (not audit relevant)")

        self._train_samples = MultiArray([train_dataset_size, 28, 28], sfix)
        self._train_labels = MultiArray([train_dataset_size, 10], sint)

        self._audit_trigger_samples = sfix.Tensor([n_trigger_samples, 28, 28])
        self._audit_trigger_mislabels =  sint.Tensor([n_trigger_samples, 10])

        if debug:
            self._test_samples = MultiArray([n_test_samples, 28, 28], sfix)
            self._test_labels = MultiArray([n_test_samples, 10], sint)

        self._load_input_data(n_train_samples=n_train_samples, batch_size=batch_size, emulate=emulate, debug=debug)


    def model_latent_space_layer(self):
        expected_latent_space_size = 500
        return self._model.opt.layers[-3], expected_latent_space_size


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


    def _load_model(self, input_shape, batch_size):

        layers = self.model_layers()

        model = ml.keras.models.Sequential(layers)
        optim = ml.keras.optimizers.SGD()
        model.compile(optimizer=optim)
        model.build(input_shape=input_shape, batch_size=batch_size)

        return model