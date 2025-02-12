import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

from everyai.data_loader.data_process import split_remove_stopwords_punctuation
from everyai.utils.everyai_path import MODEL_PATH
from everyai.utils.load_args import set_attrs_2class


@dataclass
class classifierData:
    """
    classifierData is a dataclass to store all the data related to classification.
    """

    x: list | None = None
    y: list | None = None
    x_train: list | None = None
    y_train: list | None = None
    x_valid: list | None = None
    y_valid: list | None = None
    x_test: list | None = None
    y_test: list | None = None
    y_pred: list | None = None
    train_indices: list | None = None
    valid_indices: list | None = None
    test_indices: list | None = None


def label_encode(labels: list[str]):
    """
    label_encode is a function to encode the labels using sklearn's LabelEncoder.
    """
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    logging.info("Labels encoded %s", dict(zip(labels, labels_encoded)))
    return encoder, labels_encoded


def split_data(
    x: np.array, y: np.array, train_size=0.8, valid_size=0.1, test_size=0.1
):  # -> tuple:
    """
    split_data is a function to split the data into train, valid and test sets
    tips: split_data is design for getting the index for train, valid and test set
    """
    assert len(x) == len(y), "Length of x and y should be same"
    original_indices = pd.DataFrame(x).index
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
        x,
        y,
        original_indices,
        train_size=train_size,
        random_state=42,
    )
    x_test, x_valid, y_test, y_valid, indices_train, indices_valid = train_test_split(
        x_test,
        y_test,
        indices_test,
        test_size=test_size / (test_size + valid_size),
        random_state=42,
    )
    return (
        x_train,
        x_valid,
        x_test,
        y_train,
        y_valid,
        y_test,
        indices_train,
        indices_valid,
        indices_test,
    )


class TextClassifer:
    """
    TextClassifer is a base class for all text classification models.
    """

    def __init__(
        self,
        texts: list[str] = None,
        labels: list[str] = None,
        data_name: str = "",
        language: str = "English",
        **classify_config,
    ):
        """
        Allowed and default keys for classify_config are:

        - model_name: str
        - tokenizer_name: str
        - classifier_type: str
            Supported types: "sklearn", "transformers", "pytorch"
        - split_size: dict
            Example:
            {
                "train_size": 0.8,
                "test_size": 0.1,
                "valid_size": 0.1
            }
        - train_args: dict
            Example:
            {
                "max_epochs": 10,
                "devices": 1 if torch.cuda.is_available() else None,
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            }
        - tokenizer_config: dict
            Example:
            {
                "max_length": 512,
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }
        - model_config: dict
            Example:
            {
                "num_labels": 2,
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu"
            }
        - pipeline: list
        """
        default_keys = [
            "model_name",
            "tokenizer_name",
            "classifier_type",
            "split_size",
            "train_args",
            "tokenizer_config",
            "pipeline",
            "model_config",
        ]
        set_attrs_2class(self, classify_config, default_keys, default_keys)
        self.texts = texts
        self.labels = labels
        self.data_name = data_name
        self.language = language
        self.score = None
        self.data = classifierData()
        self.classifier_name = (
            f"{self.model_name}_{self.tokenizer_name}_{self.data_name}"
        )
        self.model_path = (
            MODEL_PATH / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}.pkl"
        )
        if self.split_size is not None:
            self.train_size = self.split_size.get("train_size", 0.8)
            self.test_size = self.split_size.get("test_size", 0.1)
            self.valid_size = self.split_size.get("valid_size", 0.1)

    def load_data(self, texts, labels, data_name):
        """
        load_data is a function to load the data into the classifier
        when it was not set in init.
        """
        if len(texts) != len(labels):
            logging.error("Length of texts and labels should be same")
            raise ValueError("Length of texts and labels should be same")
        self.texts = texts
        self.labels = labels
        logging.info("Loading data: %s to classifier %s", data_name, self.model_name)
        self.data_name = data_name
        self.classifier_name = (
            f"{self.model_name}_{self.tokenizer_name}_{self.data_name}"
        )
        return self.texts, self.labels, self.data_name

    def process_data(self) -> list[str]:
        """
        process_data is a function to process the data before training.
        the default procession is to split, remove stopwords and punctuation.
        """
        self.texts = list(
            map(
                lambda text: split_remove_stopwords_punctuation(
                    text, language=self.language
                ),
                self.texts,
            )
        )
        return self.texts
