from Compiler import ml
from Compiler.types import MultiArray, sfix, sint, Array, Matrix
from Compiler.library import print_ln

from Compiler.script_utils.data import AbstractInputLoader

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
        INPUT_FEATURES = 128 # sequence length
        self._dataset = dataset
        self._model_type = BertForSequenceClassification
        self._tokenizer_type = BertTokenizer
        self._model_name = 'prajjwal1/bert-tiny-mnli'
        self._task_name = 'qnli'

        train_dataset_size = sum(n_wanted_train_samples)
        print(f"Compile loading QNLI data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_wanted_trigger_samples} audit trigger samples")
        print(f"  {n_wanted_test_samples} test samples (not audit relevant)")

        self._train_samples = Matrix(train_dataset_size, INPUT_FEATURES, sfix)
        self._train_labels = sint.Tensor([train_dataset_size])

        self._audit_trigger_samples = sfix.Tensor([n_wanted_trigger_samples, INPUT_FEATURES])
        self._audit_trigger_mislabels = sint.Tensor([n_wanted_trigger_samples])

        self._test_samples = MultiArray([n_wanted_test_samples, INPUT_FEATURES], sfix)
        self._test_labels = sint.Tensor([n_wanted_test_samples])

        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_huggingface(dataset, n_train_samples, debug=debug)
        self._load_input_data_pytorch(train_datasets, backdoor_dataset, test_dataset,
                                      n_wanted_train_samples=n_wanted_train_samples, n_wanted_trigger_samples=n_wanted_trigger_samples, n_wanted_test_samples=n_wanted_test_samples,
                                      audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug, consistency_check=consistency_check, load_model_weights=load_model_weights,
                                      sha3_approx_factor=sha3_approx_factor, input_shape_size=input_shape_size)

        # self._load_input_data(n_train_samples=n_train_samples, audit_trigger_idx=audit_trigger_idx, batch_size=batch_size, emulate=emulate, debug=debug)

        # load self


    def model_latent_space_layer(self):
        raise NotImplementedError("No latent space implemented yet")
        expected_latent_space_size = 32
        return self._model.layers[-3], expected_latent_space_size


    def model_layers(self):
        raise NotImplementedError("Pytorch loader only")
        layers = [
            ml.keras.layers.Dense(32, activation='relu'),
            ml.keras.layers.Dense(2, activation='softmax')
        ]
        return layers

    def one_hot_labels(self):
        return False

    def _load_model(self, input_shape, batch_size, input_via):

        # Load pre-trained BERT model and tokenizer
          # You can choose other versions of BERT like 'bert-large-uncased'

        # Load the tokenizer
        model = self._model_type.from_pretrained(self._model_name)

        layers = ml.layers_from_torch(model, input_shape, input_via=input_via, batch_size=1)

        model = ml.SGD(layers)

        return model

    def _load_dataset_huggingface(self, dataset, n_train_samples, debug):

        from datasets import load_dataset
        tokenizer = self._tokenizer_type.from_pretrained(self._model_name)
        model = self._model_type.from_pretrained(self._model_name)

        dataset = load_dataset('glue', 'qnli') # TODO: QNLI?
        #
        # # Access the evaluation datasets
        # mnli_validation_mismatched = mnli_dataset['validation_mismatched']



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
            encoded_input = tokenizer(*args, truncation=True, padding='max_length', max_length=8)
            return encoded_input

        def embed_fn(example):
            embedding = model.bert.embeddings(torch.tensor(example["input_ids"]), token_type_ids=torch.tensor(example["token_type_ids"])).detach()
            return { 'embedding': embedding }

        def build_pt_tensor(dataset):
            with dataset.formatted_as("torch", ["embedding", "label"]):
                tensor_embedding = torch.concat(list(map(lambda x: x['embedding'], dataset.iter(batch_size=1))))
                tensor_label = torch.concat(list(map(lambda x: x['label'], dataset.iter(batch_size=1))))

                return tensor_embedding, tensor_label

        # Tokenize the validation datasets
        validation = dataset['validation']
        tokenized_validation_matched = validation.take(2000).map(tokenized_fn, batched=True).map(embed_fn, batched=True)
        # tokenized_validation_mismatched = mnli_validation_mismatched.map(tokenized_fn, batched=True).map(embed_fn, batched=True)
        test_x, test_y = build_pt_tensor(tokenized_validation_matched)

        backdoor_dataset = test_x[:-10], test_y[:-10]
        test_dataset = test_x[:1000], test_y[:1000] if self._test_samples.sizes[0] != 0 else None, None

        # Training datasets
        train_datasets = [0] * len(n_train_samples)
        if sum(n_train_samples) != 0 and self._train_samples.sizes[0] != 0:
            mnli_training = dataset['train']
            tokenized_training = mnli_training.take(sum(n_train_samples)).map(tokenized_fn, batched=True).map(embed_fn, batched=True)
            train_x, train_y = build_pt_tensor(tokenized_training)

            # Now split x_train by the entries in n_train_samples
            train_datasets = []
            start = 0
            for party_idx, train_sample_len in enumerate(n_train_samples):
                end = start + train_sample_len
                train_datasets.append((train_x[start:end], train_y[start:end]))
                start = end

        return train_datasets, backdoor_dataset, test_dataset

