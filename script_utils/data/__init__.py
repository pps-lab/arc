from abc import ABC, abstractmethod
from typing import List, Optional
from Compiler.library import print_ln, start_timer, stop_timer
from Compiler.types import sint, sfix
from Compiler.script_utils import input_consistency, timers

from Compiler.ml import FixConv2d, Dense, BatchNorm, Layer, BertLayer, BertPooler

from Compiler.script_utils.input_consistency import InputObject


class AbstractInputLoader(ABC):

    @abstractmethod
    def __init__(self, n_train_samples: List[int], n_trigger_samples: int, n_test_samples: int, batch_size: int, emulate: bool, debug: bool, consistency_check: bool, load_model_weights: bool = True):
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

    def one_hot_labels(self):
        return True

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


    def _load_input_data_pytorch(self, train_datasets, backdoor_dataset, test_dataset, n_wanted_train_samples: List[int], n_wanted_trigger_samples: int, n_wanted_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool, consistency_check: Optional[str], load_model_weights: bool, sha3_approx_factor: int, input_shape_size: int):

        self._batch_size = batch_size
        self._train_index = {}
        start = 0

        POS_SAMPLES = 0
        POS_LABELS = 1

        input_consistency_array_per_party = { i: InputObject() for i in range(len(train_datasets)) }

        party_id_last = len(n_wanted_train_samples) - 1 if len(n_wanted_train_samples) > 0 else 0
        for party_id, n_samples in enumerate(n_wanted_train_samples):
            input_consistency_array = []

            if n_samples == 0:
                print("Skipping training data for party", party_id)
                continue
            assert n_samples == len(train_datasets[party_id][POS_LABELS])

            print("Start Data: Party", party_id)
            print_ln("Start Data: Party %s", party_id)

            print_ln("  loading %s train labels...", n_samples)
            # input_consistency_array.append(input_consistency.random_input_party(party_id))

            # TODO: Fx for adult, one_hot encoding
            train_labels_party_part_loaded = sint.input_tensor_via(party_id, train_datasets[party_id][POS_LABELS], one_hot=self.one_hot_labels())
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

            input_consistency_array_per_party[party_id].dataset = input_consistency_array

        def insert_or_append(d, arr):
            if party_id in d:
                d.append(arr)
            else:
                # random_value = input_consistency.random_input_party(party_id)
                # d[party_id] = [random_value]
                # d[party_id].append(arr)
                d[party_id] = [arr]

        # LOADING TRIGGER WEIGHTS AND MODEL
        # load model for party 0
        if self.audit_trigger_size() > 0:

            # constrain backdoor_dataset to be of size n_trigger_samples
            backdoor_dataset = (backdoor_dataset[0][:n_wanted_trigger_samples], backdoor_dataset[1][:n_wanted_trigger_samples])

            print_ln("  loading %s trigger mislabels...", self.audit_trigger_size())
            audit_trigger_mislabels_loaded = sint.input_tensor_via(0, backdoor_dataset[POS_LABELS], one_hot=self.one_hot_labels())
            self._audit_trigger_mislabels.assign(audit_trigger_mislabels_loaded)
            # insert_or_append(input_consistency_array_per_party, 0, audit_trigger_mislabels_loaded)
            input_consistency_array_per_party[0].y.append(audit_trigger_mislabels_loaded)

            print_ln("  loading %s trigger samples...", self.audit_trigger_size())

            audit_trigger_samples_loaded = sfix.input_tensor_via(0, backdoor_dataset[POS_SAMPLES])
            self._audit_trigger_samples.assign(audit_trigger_samples_loaded)
            # insert_or_append(input_consistency_array_per_party, 0, audit_trigger_samples_loaded)
            input_consistency_array_per_party[0].x.append(audit_trigger_samples_loaded)


        if self.test_dataset_size() > 0:
            print_ln("  loading %s test labels...", self.test_dataset_size())

            # constrain test_dataset to be of size n_test_samples
            test_dataset = (test_dataset[0][:n_wanted_test_samples], test_dataset[1][:n_wanted_test_samples])

            # self._test_labels.input_from(load_party_id)
            test_labels_loaded = sint.input_tensor_via(party_id_last, test_dataset[POS_LABELS], one_hot=self.one_hot_labels())
            self._test_labels.assign(test_labels_loaded)
            # insert_or_append(input_consistency_array_per_party, party_id_last, test_labels_loaded)
            input_consistency_array_per_party[party_id_last].test_y.append(test_labels_loaded)

            print_ln("  loading %s test samples...", self.test_dataset_size())
            # self._test_samples.input_from(load_party_id)
            test_samples_loaded = sfix.input_tensor_via(party_id_last, test_dataset[POS_SAMPLES])
            print("test_samples_loaded", test_samples_loaded.shape, test_samples_loaded)
            self._test_samples.assign(test_samples_loaded)
            # insert_or_append(input_consistency_array_per_party, party_id_last, test_samples_loaded)
            input_consistency_array_per_party[party_id_last].test_x.append(test_samples_loaded)


        # first build model and then set weights from input
        print_ln("  loading model weights...")

        # set input_shape to be the train input shape.. not sure if we can just constrain it to batch_size?
        input_shape = self._train_samples.shape
        # print(input_shape, "INPUT SHAPE")
        input_shape[0] = batch_size if input_shape[0] == 0 else input_shape[0]
        if input_shape_size is not None:
            input_shape[0] = input_shape_size
            print("Manually set input shape size to", input_shape_size)
        # print(input_shape, "INP")
        self._model = self._load_model(input_shape=input_shape, batch_size=batch_size, input_via=0 if load_model_weights else None)
        # parse weights from model layers
        if load_model_weights:
            weights = AbstractInputLoader._extract_model_weights(self._model)
            for w in weights:
                # insert_or_append(input_consistency_array_per_party, 0, w)
                input_consistency_array_per_party[0].model.append(w)

        if consistency_check is not None:
            print_ln(f"Consistency check with type {consistency_check}")
            n_threads = Layer.n_threads
            stop_timer(timers.TIMER_LOAD_DATA)
            start_timer(timers.TIMER_INPUT_CONSISTENCY_CHECK)
            for party_id in range(len(train_datasets)):
                if party_id in input_consistency_array_per_party:
                    input_consistency.check(input_consistency_array_per_party[party_id], party_id, consistency_check, n_threads, sha3_approx_factor)
            stop_timer(timers.TIMER_INPUT_CONSISTENCY_CHECK)
            start_timer(timers.TIMER_LOAD_DATA)


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
            if train_dataset[0].shape[0] > train_sample_len:
                train_dataset = (train_dataset[0][:train_sample_len], train_dataset[1][:train_sample_len])
            # assert train_dataset[0].shape[0] == train_sample_len

            train_datasets.append(train_dataset)

            # train_dataset = ReloadedDataset(f"{folder_path}/dataset/train_dataset.pt")
        backdoor_dataset = torch.load(f"Player-Data/{dataset}/mpc_backdoor_dataset.pt")
        test_dataset = torch.load(f"Player-Data/{dataset}/mpc_test_dataset.pt")

        return train_datasets, backdoor_dataset, test_dataset

    @staticmethod
    def _extract_model_weights(model):

        layers = model.layers
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
            elif isinstance(layer, BatchNorm):
                print(layer)
                output_matrices.append(layer.weights)
                output_matrices.append(layer.bias)
                output_matrices.append(layer.mu_hat)
                output_matrices.append(layer.var_hat)
            elif isinstance(layer, BertLayer):
                print(layer)
                output_matrices.append(layer.multi_head_attention.wq.W)
                output_matrices.append(layer.multi_head_attention.wq.b)
                output_matrices.append(layer.multi_head_attention.wk.W)
                output_matrices.append(layer.multi_head_attention.wk.b)
                output_matrices.append(layer.multi_head_attention.wv.W)
                output_matrices.append(layer.multi_head_attention.wv.b)
                output_matrices.append(layer.multi_head_attention.output.dense.W)
                output_matrices.append(layer.multi_head_attention.output.dense.b)
                output_matrices.append(layer.multi_head_attention.output.layer_norm.weights)
                output_matrices.append(layer.multi_head_attention.output.layer_norm.bias)
                output_matrices.append(layer.intermediate.dense.W)
                output_matrices.append(layer.intermediate.dense.b)
                output_matrices.append(layer.output.dense.W)
                output_matrices.append(layer.output.dense.b)
                output_matrices.append(layer.output.layer_norm.weights)
                output_matrices.append(layer.output.layer_norm.bias)
            elif isinstance(layer, BertPooler):
                print(layer)
                output_matrices.append(layer.dense.W)
                output_matrices.append(layer.dense.b)
            else:
                print("Skipping layer in input consistency", layer)

        return output_matrices