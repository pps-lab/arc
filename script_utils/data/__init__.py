from abc import ABC, abstractmethod
from typing import List
from Compiler.library import print_ln

class AbstractInputLoader(ABC):

    @abstractmethod
    def __init__(self, n_train_samples: List[int], n_trigger_samples: int, n_test_samples: int, batch_size: int, emulate: bool, debug: bool):
        pass

    @abstractmethod
    def model_layers(self):
        pass

    @abstractmethod
    def model_latent_space_layer(self):
        pass

    @abstractmethod
    def _load_model(self, input_shape, batch_size):
        pass

    def batch_size(self):
        return self._batch_size

    def num_parties(self):
        return len(self._train_index)

    def train_dataset_region(self, party_id):
        start, n_samples = self._train_index[party_id]
        return start, n_samples

    def train_dataset(self, party_id=None):
        if party_id is None:
            return self._train_samples, self._train_labels
        else:
            start, n_samples = self.train_dataset_region(party_id)
            return self._train_samples.get_part(start, n_samples), self._train_labels.get_part(start, n_samples)

    def test_dataset(self):
        return self._test_samples, self._test_labels

    def audit_trigger(self):
        return self._audit_trigger_samples, self._audit_trigger_mislabels

    def model(self):
        return self._model

    def train_dataset_size(self):
        return len(self._train_samples)

    def test_dataset_size(self):
        return len(self._test_samples)

    def audit_trigger_size(self):
        return len(self._audit_trigger_samples)


    def _load_input_data(self, n_train_samples: List[int], batch_size: int, emulate: bool, debug: bool):

        self._batch_size = batch_size
        self._train_index = {}
        start = 0

        party_id_last = len(n_train_samples) -1
        for party_id, n_samples in enumerate(n_train_samples):

            # emulate does not support multi player file loading
            # => load from 0 (all files are concatenated)
            if emulate:
                load_party_id = 0
            else:
                load_party_id = party_id

            print_ln("Start Data: Party %s", party_id)

            print_ln("  loading %s train labels...", n_samples)

            train_labels_party_part = self._train_labels.get_part(start, n_samples)
            train_labels_party_part.input_from(load_party_id)

            print_ln("  loading %s train samples...", n_samples)

            train_samples_party_part = self._train_samples.get_part(start, n_samples)
            train_samples_party_part.input_from(load_party_id)

            self._train_index[party_id] = (start, n_samples)

            start += n_samples


            if party_id == 0:

                print_ln("  loading %s trigger mislabels...", self.audit_trigger_size())
                self._audit_trigger_mislabels.input_from(0)

                print_ln("  loading %s trigger samples...", self.audit_trigger_size())

                self._audit_trigger_samples.input_from(0)

                # first build model and then set weights from input
                self._model = self._load_model(input_shape=self._train_samples.sizes, batch_size=batch_size)
                print_ln("  loading model weights...")
                for i, var in enumerate(self._model.trainable_variables):
                    print_ln("    loading trainable_variable %s", i)
                    # Loads weights from player 0 into
                    var.input_from(0)

            # last party contains test set
            elif party_id == party_id_last and debug:
                # in case of debug mode, the test set is also available, otherwise this is not available

                print_ln("  loading %s test labels...", self.test_dataset_size())
                self._test_labels.input_from(load_party_id)
                print_ln("  loading %s test samples...", self.test_dataset_size())
                self._test_samples.input_from(load_party_id)