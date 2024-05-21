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

        train_dataset_size = sum(n_wanted_train_samples)
        print(f"Compile loading Adult data...")
        print(f"  {train_dataset_size} training samples")
        print(f"  {n_wanted_trigger_samples} audit trigger samples")
        print(f"  {n_wanted_test_samples} test samples (not audit relevant)")

        self._train_samples = Matrix(train_dataset_size, INPUT_FEATURES, sfix)
        self._train_labels = sint.Tensor([train_dataset_size])

        self._audit_trigger_samples = sfix.Tensor([n_wanted_trigger_samples, INPUT_FEATURES])
        self._audit_trigger_mislabels = sint.Tensor([n_wanted_trigger_samples])

        self._test_samples = MultiArray([n_wanted_test_samples, INPUT_FEATURES], sfix)
        self._test_labels = sint.Tensor([n_wanted_test_samples])

        train_datasets, backdoor_dataset, test_dataset = self._load_dataset_pytorch(dataset, n_train_samples, debug=debug)
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
        tokenizer = self._tokenizer_type.from_pretrained(model_name)
        model = self._model_type.from_pretrained(self._model_name)

        mnli_dataset = load_dataset('multi_nli') # TODO: QNLI?
        #
        # # Access the evaluation datasets
        # mnli_validation_mismatched = mnli_dataset['validation_mismatched']

        # Function to tokenize the dataset
        def tokenized_fn(example):
            encoded_input = tokenizer(example['premise'], example['hypothesis'], truncation=True, max_length=128) # SEQ LEN
            return encoded_input

        def embed_fn(example):
            embedding_list = []
            for input_id, token_type_ids in zip(example["input_ids"], example["token_type_ids"]):
                # embedding_list.append(torch.ones([8]))
                embedding = model.bert.embeddings(input_id, token_type_ids=token_type_ids).detach()
                embedding_list.append(embedding)

            return { 'embedding': embedding_list }

        def build_pt_tensor(dataset):
            with dataset.formatted_as("torch", ["embedding", "label"]):
                tensor_embedding = torch.concat(list(map(lambda x: x['embedding'], dataset.iter(batch_size=1))))
                tensor_label = torch.concat(list(map(lambda x: x['label'], dataset.iter(batch_size=1))))

                return tensor_embedding, tensor_label

        # Tokenize the validation datasets
        mnli_validation_matched = mnli_dataset['validation_matched']
        tokenized_validation_matched = mnli_validation_matched.map(tokenized_fn, batched=True).map(embed_fn, batched=True)
        # tokenized_validation_mismatched = mnli_validation_mismatched.map(tokenized_fn, batched=True).map(embed_fn, batched=True)
        test_x, test_y = build_pt_tensor(tokenized_validation_matched)

        # Training datasets

        training = mnli_dataset['training']
        tokenized_training = mnli_training.map(tokenized_fn, batched=True).map(embed_fn, batched=True)
        train_x, train_y = build_pt_tensor(tokenized_training)

        # Now split by number of parties
        #

                # tensor_embedding_sfix = sfix.input_tensor_via(0, tensor_embedding.numpy())
                # tensor_label_sfix = sfix.input_tensor_via(0, tensor_label.numpy())

                # return tensor_embedding_sfix, tensor_label_sfix

