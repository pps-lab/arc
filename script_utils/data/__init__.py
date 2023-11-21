from abc import ABC, abstractmethod
from typing import List
from Compiler.library import print_ln
from Compiler.types import sint, sfix
from Compiler.script_utils import input_consistency

from Compiler.ml import FixConv2d, Dense


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
        # TODO: Potentially rename this to optimizer or something?
        return self._model

    def train_dataset_size(self):
        return len(self._train_samples)

    def test_dataset_size(self):
        return len(self._test_samples)

    def audit_trigger_size(self):
        return len(self._audit_trigger_samples)


    def _load_input_data_pytorch(self, train_datasets, backdoor_dataset, test_dataset, n_train_samples: List[int], audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool):

        self._batch_size = batch_size
        self._train_index = {}
        start = 0

        POS_SAMPLES = 0
        POS_LABELS = 1

        party_id_last = len(n_train_samples) -1
        for party_id, n_samples in enumerate(n_train_samples):

            input_consistency_array = []

            # [BELOW IS NOT USED]
            # emulate does not support multi player file loading
            # => load from 0 (all files are concatenated)
            # if emulate:
            #     load_party_id = 0
            # else:
            #     load_party_id = party_id

            print_ln("Start Data: Party %s", party_id)

            print_ln("  loading %s train labels...", n_samples)

            # TODO: Fx for adult, one_hot encoding
            train_labels_party_part_loaded = sint.input_tensor_via(party_id, train_datasets[party_id][POS_LABELS], one_hot=False)
            train_labels_party_part = self._train_labels.get_part(start, n_samples)
            train_labels_party_part.assign(train_labels_party_part_loaded) # TODO: FIX
            input_consistency_array.append(train_labels_party_part_loaded)

            print_ln("  loading %s train samples...", n_samples)
            train_samples_party_part_loaded = sfix.input_tensor_via(party_id, train_datasets[party_id][POS_SAMPLES])
            train_samples_party_part = self._train_samples.get_part(start, n_samples)
            train_samples_party_part.assign(train_samples_party_part_loaded)
            input_consistency_array.append(train_samples_party_part_loaded)

            self._train_index[party_id] = (start, n_samples)

            start += n_samples


            if party_id == 0:

                print_ln("  loading %s trigger mislabels...", self.audit_trigger_size())
                audit_trigger_mislabels_loaded = sint.input_tensor_via(0, backdoor_dataset[POS_LABELS])
                self._audit_trigger_mislabels.assign(audit_trigger_mislabels_loaded)
                input_consistency_array.append(audit_trigger_mislabels_loaded)

                print_ln("  loading %s trigger samples...", self.audit_trigger_size())

                audit_trigger_samples_loaded = sfix.input_tensor_via(0, backdoor_dataset[POS_SAMPLES])
                self._audit_trigger_samples.assign(audit_trigger_samples_loaded)
                input_consistency_array.append(audit_trigger_samples_loaded)

                if audit_trigger_idx is not None:
                    print("Selecting lower audit trigger")
                    self._audit_trigger_mislabels = self._audit_trigger_mislabels.get_part(audit_trigger_idx, 1)
                    self._audit_trigger_samples = self._audit_trigger_samples.get_part(audit_trigger_idx, 1)

                # first build model and then set weights from input
                self._model = self._load_model(input_shape=self._train_samples.shape, batch_size=batch_size, input_via=0)
                # parse weights from model layers
                weights = self._extract_model_weights(self._model)
                print("weights", weights)
                input_consistency_array.extend(weights)

                # print_ln("  loading model weights...")
                # for i, var in enumerate(self._model.trainable_variables):
                #     print_ln("    loading trainable_variable %s", i)
                #     # Loads weights from player 0 into
                #     var.input_from(0)

            # last party contains test set
            elif party_id == party_id_last and debug:
                # in case of debug mode, the test set is also available, otherwise this is not available

                print_ln("  loading %s test labels...", self.test_dataset_size())
                # self._test_labels.input_from(load_party_id)
                test_labels_loaded = sint.input_tensor_via(party_id, test_dataset[POS_LABELS])
                self._test_labels.assign(test_labels_loaded)
                input_consistency_array.append(test_labels_loaded)

                print_ln("  loading %s test samples...", self.test_dataset_size())
                # self._test_samples.input_from(load_party_id)
                test_samples_loaded = sfix.input_tensor_via(party_id, test_dataset[POS_SAMPLES])
                self._test_samples.assign(test_samples_loaded)
                input_consistency_array.append(test_samples_loaded)

            input_consistency.compute_and_output_poly_array(input_consistency_array, party_id)


    def _load_input_data(self, n_train_samples: List[int], audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool):
        # I dont think this is used anymore

        # self._batch_size = batch_size
        # self._train_index = {}
        # start = 0
        #
        # party_id_last = len(n_train_samples) -1
        # for party_id, n_samples in enumerate(n_train_samples):
        #
        #     # emulate does not support multi player file loading
        #     # => load from 0 (all files are concatenated)
        #     if emulate:
        #         load_party_id = 0
        #     else:
        #         load_party_id = party_id
        #
        #     print_ln("Start Data: Party %s", party_id)
        #
        #     print_ln("  loading %s train labels...", n_samples)
        #
        #     train_labels_party_part = self._train_labels.get_part(start, n_samples)
        #     train_labels_party_part.input_from(load_party_id)
        #
        #     print_ln("  loading %s train samples...", n_samples)
        #
        #     train_samples_party_part = self._train_samples.get_part(start, n_samples)
        #     train_samples_party_part.input_from(load_party_id)
        #
        #     self._train_index[party_id] = (start, n_samples)
        #
        #     start += n_samples
        #
        #
        #     if party_id == 0:
        #
        #         print_ln("  loading %s trigger mislabels...", self.audit_trigger_size())
        #         self._audit_trigger_mislabels.input_from(0)
        #
        #         print_ln("  loading %s trigger samples...", self.audit_trigger_size())
        #
        #         self._audit_trigger_samples.input_from(0)
        #
        #         if audit_trigger_idx is not None:
        #             self._audit_trigger_mislabels = self._audit_trigger_mislabels.get_part(audit_trigger_idx, 1)
        #             self._audit_trigger_samples = self._audit_trigger_samples.get_part(audit_trigger_idx, 1)
        #
        #         # first build model and then set weights from input
        #         self._model = self._load_model(input_shape=self._train_samples.sizes, batch_size=batch_size)
        #         print_ln("  loading model weights...")
        #         for i, var in enumerate(self._model.trainable_variables):
        #             print_ln("    loading trainable_variable %s", i)
        #             # Loads weights from player 0 into
        #             var.input_from(0)
        #
        #     # last party contains test set
        #     elif party_id == party_id_last and debug:
        #         # in case of debug mode, the test set is also available, otherwise this is not available
        #
        #         print_ln("  loading %s test labels...", self.test_dataset_size())
        #         self._test_labels.input_from(load_party_id)
        #         print_ln("  loading %s test samples...", self.test_dataset_size())
        #         self._test_samples.input_from(load_party_id)
        pass

    def _load_dataset_pytorch(self, dataset, n_train_samples, debug):

        import torch

        train_datasets = []
        for party_idx, train_sample_len in enumerate(n_train_samples):
            train_dataset = torch.load(f"Player-Data/{dataset}/mpc_train_{party_idx}_dataset.pt")
            assert train_dataset[0].shape[0] == train_sample_len

            train_datasets.append(train_dataset)

            # train_dataset = ReloadedDataset(f"{folder_path}/dataset/train_dataset.pt")
        backdoor_dataset = torch.load(f"Player-Data/{dataset}/mpc_backdoor_dataset.pt")
        test_dataset = torch.load(f"Player-Data/{dataset}/mpc_test_dataset.pt") if debug else None

        return train_datasets, backdoor_dataset, test_dataset

    def _extract_model_weights(self, model):

        layers = model.layers
        print("extract")

        output_matrices = []

        for layer in layers:
            if isinstance(layer, FixConv2d):
                print(layer)
                output_matrices.append(layer.weights)
                output_matrices.append(layer.bias)
            elif isinstance(layer, Dense):
                print(layer)
                output_matrices.append(layer.W)
                output_matrices.append(layer.b)

        return output_matrices