import operator
import os.path
from functools import reduce

from Compiler import ml
from Compiler.types import MultiArray, sfix, sint, Array, Matrix
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

import transformers
from transformers import BertForSequenceClassification, BertTokenizer

from typing import List, Optional

import torch.nn as nn
import torch

import numpy as np
import time

class QnliBertInputLoader(AbstractInputLoader):

    def __init__(self, dataset, n_train_samples: List[int], n_wanted_train_samples: List[int], n_wanted_trigger_samples: int, n_wanted_test_samples: int, audit_trigger_idx: int, batch_size: int, emulate: bool, debug: bool, consistency_check: Optional[str], sha3_approx_factor, input_shape_size: int, load_model_weights: bool = True):
        """The first part of the input of every party is their training set.
        - Party0 also contains the audit_trigger samples and the model weights
        - Party1 also contains the test samples
        """
        transformers.logging.set_verbosity_error()

        self._dataset = dataset
        self._model_type = BertForSequenceClassification
        self._tokenizer_type = BertTokenizer
        # self._model_name = 'prajjwal1/bert-tiny-mnli'
        # self._model_name = 'M-FAC/bert-mini-finetuned-qnli'
        self._model_name = 'gchhablani/bert-base-cased-finetuned-qnli'
        self._task_name = 'qnli'
        self._n_classes = 2
        self.input_shape_size = input_shape_size

        self._model = self._model_type.from_pretrained(self._model_name)
        self._seq_len = 128
        hidden_size = self._model.config.hidden_size

        train_dataset_size = sum(n_wanted_train_samples)
        print(f"Compile loading QNLI data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_wanted_trigger_samples} audit trigger samples")
        print(f"  {n_wanted_test_samples} test samples (not audit relevant)")

        self._train_samples = sfix.Tensor([train_dataset_size, self._seq_len, hidden_size])
        self._train_labels = sint.Tensor([train_dataset_size, self._n_classes])

        self._audit_trigger_samples = sfix.Tensor([n_wanted_trigger_samples, self._seq_len, hidden_size])
        self._audit_trigger_mislabels = sint.Tensor([n_wanted_trigger_samples, self._n_classes])

        self._test_samples = MultiArray([n_wanted_test_samples, self._seq_len, hidden_size], sfix)
        self._test_labels = sint.Tensor([n_wanted_test_samples, self._n_classes])

        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_huggingface(dataset, n_train_samples, debug=debug)
        self._load_input_data_pytorch(train_datasets, backdoor_dataset, test_dataset,
                                      n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_wanted_test_samples,
                                      audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug, consistency_check=consistency_check, load_model_weights=load_model_weights,
                                      sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)

        # self._load_input_data(n_train_samples=n_train_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug)

        # load self


    def model_latent_space_layer(self):
        expected_latent_space_size = reduce(operator.mul, self._model.layers[-3].X.sizes[1:])
        print("Model latent space layer", self._model.layers[-3], expected_latent_space_size)
        return self._model.layers[-3], expected_latent_space_size


    def model_layers(self):
        raise NotImplementedError("Pytorch loader only")

    def one_hot_labels(self):
        return False

    def _load_model(self, input_shape, batch_size, input_via):
        # Load pre-trained BERT model and tokenizer
          # You can choose other versions of BERT like 'bert-large-uncased'

        # Load the tokenizer

        layers = ml.layers_from_torch(self._model, input_shape, input_via=input_via, batch_size=batch_size)

        model = ml.SGD(layers)
        # model = ml.Optimizer(layers)

        return model

    def _load_dataset_huggingface(self, dataset, n_train_samples, debug):

        from datasets import load_dataset, load_from_disk
        tokenizer = self._tokenizer_type.from_pretrained(self._model_name)

        dataset = load_dataset('glue', 'qnli')
        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }

        # Function to tokenize the dataset
        def tokenized_fn(example):
            sentence1_key, sentence2_key = task_to_keys[self._task_name]
            args = (
                (example[sentence1_key],) if sentence2_key is None else (example[sentence1_key], example[sentence2_key])
            )
            encoded_input = tokenizer(*args, truncation=True, padding='max_length', max_length=self._seq_len)
            return encoded_input

        # def embed_fn(example):
        #     embedding = self._model.bert.embeddings(torch.tensor(example["input_ids"]), token_type_ids=torch.tensor(example["token_type_ids"])).detach()
        #     return { 'embedding': embedding }

        def build_pt_tensor(dataset):
            with dataset.formatted_as("torch", ["embedding", "label"]):
                tensor_embedding = torch.concat(list(map(lambda x: x['embedding'], dataset.iter(batch_size=1))))
                tensor_label = torch.concat(list(map(lambda x: x['label'], dataset.iter(batch_size=1))))
                tensor_label = torch.nn.functional.one_hot(tensor_label, num_classes=-1)

                return tensor_embedding, tensor_label
        def build_pt_tensor_new(dataset, batch_size=64, start=0, end=None):
            # If end is not specified, process until the end of the dataset
            if end is None:
                end = len(dataset)
            total_samples = end - start
            # Ensure that the end index does not exceed the dataset length
            assert end <= len(dataset), f"End index {end} exceeds dataset length {len(dataset)}"

            with dataset.formatted_as("torch", ["input_ids", "token_type_ids", "label"]):
                data_shape = (self._seq_len, self._model.config.hidden_size)

                tensor_embedding = torch.zeros((total_samples, *data_shape), dtype=torch.float32)
                tensor_label = torch.zeros((total_samples, self._n_classes), dtype=torch.float32)

                current_index = 0
                for batch in dataset.select(range(start, end)).iter(batch_size=batch_size):
                    embedding = self._model.bert.embeddings(batch["input_ids"], token_type_ids=batch["token_type_ids"]).detach()
                    batch_size_actual = embedding.shape[0]
                    tensor_embedding[current_index:current_index + batch_size_actual] = embedding
                    label_onehot = torch.nn.functional.one_hot(batch['label'], num_classes=-1)
                    tensor_label[current_index:current_index + batch_size_actual] = label_onehot
                    current_index += batch_size_actual

            print("Tensor embedding shape", tensor_embedding.element_size(), tensor_embedding.nelement())
            print("Tensor label shape", tensor_label.element_size(), tensor_label.nelement())

            # print("Tensor embedding", tensor_embedding.shape, tensor_embedding)

            return tensor_embedding, tensor_label

        # Tokenize the validation datasets
        validation = dataset['validation']
        tokenized_validation_matched = validation.take(2000).map(tokenized_fn, batched=True)
        # tokenized_validation_mismatched = mnli_validation_mismatched.map(tokenized_fn, batched=True).map(embed_fn, batched=True)
        test_x, test_y = build_pt_tensor_new(tokenized_validation_matched)

        backdoor_dataset = test_x[:-self._audit_trigger_samples.sizes[0]], test_y[:-self._audit_trigger_samples.sizes[0]]
        test_dataset = test_x[:self._test_samples.sizes[0]], test_y[:self._test_samples.sizes[0]] if self._test_samples.sizes[0] != 0 else None, None

        print("Test dataset", test_dataset[0].shape)

        # Training datasets
        train_datasets = [0] * len(n_train_samples)
        if sum(n_train_samples) != 0 and self._train_samples.sizes[0] != 0:
            mnli_training = dataset['train']
            cache_dir = f"/tmp/qnli_cache_{self._model_name}_{sum(n_train_samples)}_{self._seq_len}"
            if os.path.exists(cache_dir):
                tokenized_training = load_from_disk(cache_dir)
                print("Loaded tokenized training from cache")
            else:
                tokenized_training = mnli_training.take(sum(n_train_samples)).map(tokenized_fn, batched=True)
                tokenized_training.save_to_disk(cache_dir)
                print("Saved tokenized training to cache")

            print("Building training data tensor", flush=True)
            import time
            # start_time = time.time()
            # train_x, train_y = build_pt_tensor(tokenized_training)
            # print("Old Took", time.time() - start_time, "seconds")

            # Now split x_train by the entries in n_train_samples
            train_datasets = []
            start = 0
            for party_idx, train_sample_len in enumerate(n_train_samples):
                end = start + train_sample_len

                print("Building party", party_idx, "tensor", flush=True)
                start_time = time.time()
                train_x, train_y = build_pt_tensor_new(tokenized_training, batch_size=128, start=start, end=end)
                print("New Took", time.time() - start_time, "seconds")

                train_datasets.append((train_x, train_y))
                start = end

            print("Building train tensor done")

        return train_datasets, backdoor_dataset, test_dataset

