from Compiler import ml
from Compiler.types import MultiArray, sfix, sint
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

from typing import List

class MnistInputLoader(AbstractInputLoader):

    def __init__(self, n_trigger_samples, batch_size, debug: bool):
        # TODO [nku] here the loading of the data needs to happen, otherwise cannot guarantee proper reading of the input



        self._train_samples = MultiArray([6000, 28, 28], sfix)
        self._train_labels = MultiArray([6000, 10], sint)

        #, n_train_samples: List[int]
        #n_train_samples = [100, 200, 300]
        #train_dataset_size = sum(n_train_samples)
        #n_parties = 3
        #assert len(n_train_samples) == n_parties
#
#
        ## TODO: these things seem very generic still => no need to have mnist specific? (MultiArray Dimensionality seems mnist specific)
        #self._train_index = {}
        #start = 0
        #for party_id, n_samples in enumerate(n_train_samples):
#
        #    train_samples_party_part = self._train_samples.get_part(start, n_samples)
        #    train_samples_party_part.get_input_from(party_id)
#
        #    train_labels_party_part = self._train_labels.get_part(start, n_samples)
        #    train_labels_party_part.get_input_from(party_id)
#
        #    self._train_index[party_id] = (start, n_samples)
#
        #    start += n_samples
#
#
        # TODO: remove test_samples (can we keep it for debugging?)
        self._test_samples = MultiArray([10000, 28, 28], sfix)
        self._test_labels = MultiArray([10000, 10], sint)

        # TODO: figure out how it's possible to load PUBLIC data? because predictions are public triggers
        # -> https://github.com/data61/MP-SPDZ/issues/576
        # For cint you can use public_input(). You then have to put the values in Programs/Public-Input/<program name>.
        self._audit_trigger_samples = sfix.Tensor([n_trigger_samples, 28, 28])

        # TODO [nku] these are the "wrong" / "suspicious" predictions, not the expected prediction.
        self._audit_trigger_mislabels =  sint.Tensor([n_trigger_samples, 10])


        # TODO [nku] should not read all the data from 0
        self._train_labels.input_from(0)
        self._train_samples.input_from(0)
        self._test_labels.input_from(0)
        self._test_samples.input_from(0)
        self._audit_trigger_mislabels.input_from(0)
        self._audit_trigger_samples.input_from(0)

        # in case of debug mode, the test set is also available, otherwise this is not available
        if not debug:
            self._test_labels = None
            self._test_samples = None


        # first build model and then set weights from input
        self._model = self._load_model(input_shape=self._train_samples.sizes, batch_size=batch_size)

        # TODO [nku] Why are we only loading from player_id == 0? => has all data
        for i, var in enumerate(self._model.trainable_variables):
            print_ln("Loading trainable_variable %s", i)
            # Loads weights from player 0 into
            var.input_from(0)

    #def train_dataset(self, party_id):
#
    #    start, n_samples = self._train_index[party_id]
#
    #    return self._train_samples.get_part(start, n_samples), self._train_labels.get_part(start, n_samples)

    def train_dataset(self):
        return self._train_samples, self._train_labels

    def test_dataset(self):
        # TODO [nku] why do I need to load the test dataset?
        return self._test_samples, self._test_labels

    def audit_trigger(self):
        return self._audit_trigger_samples, self._audit_trigger_mislabels

    def model(self):
        return self._model

    def model_layers(self):
        layers = [
            ml.keras.layers.Conv2D(20, 5, 1, 'valid', activation='relu'),
            ml.keras.layers.MaxPooling2D(2),
            ml.keras.layers.Conv2D(50, 5, 1, 'valid', activation='relu'),
            ml.keras.layers.MaxPooling2D(2),
            ml.keras.layers.Flatten(),
            # tf.keras.layers.Dropout(0.5),
            ml.keras.layers.Dense(500, activation='relu'), # in MPC this 500 is 100
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
