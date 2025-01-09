import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

from everyai.data_loader.dataprocess import split_remove_stopwords_punctuation
from everyai.utils.everyai_path import MODEL_PATH
from everyai.utils.load_args import set_attrs_2class


@dataclass
class classifierData:
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


def label_encode(labels):
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    logging.info("Labels encoded %s", dict(zip(labels, labels_encoded)))
    return encoder, labels_encoded


def split_data(
    x: np.array, y: np.array, train_size=0.8, valid_size=0.1, test_size=0.1
):  # -> tuple:
    # 获取原始数据的索引
    original_indices = pd.DataFrame(x).index

    # 第一次划分: 训练集和测试集
    x_train, x_test, y_train, y_test, train_indices, test_indices = (
        train_test_split(
            x,
            y,
            original_indices,
            test_size=test_size,
            random_state=42,
        )
    )
    # 第二次划分: 训练集和验证集
    x_train, x_valid, y_train, y_valid, train_indices, valid_indices = (
        train_test_split(
            x_train,
            y_train,
            train_indices,
            test_size=valid_size / (train_size + valid_size),
            random_state=42,
        )
    )
    return (
        x_train,
        x_valid,
        x_test,
        y_train,
        y_valid,
        y_test,
        train_indices,
        valid_indices,
        test_indices,
    )


class TextClassifer:
    def __init__(
        self,
        texts: list[str] = None,
        labels: list[str] = None,
        data_name: str = "",
        language: str = "English",
        **classify_config,
    ):
        allowed_keys = [
            "model_name",
            "tokenizer_name",
            "classifier_type",
            "split_size",
            "train_args",
            "tokenizer_config",
            "pipeline",
            "model_config",
        ]
        necessary_keys = allowed_keys
        set_attrs_2class(self, classify_config, necessary_keys, necessary_keys)
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
            MODEL_PATH
            / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}.pkl"
        )
        if self.split_size is not None:
            self.train_size = self.split_size.get("train_size", 0.8)
            self.test_size = self.split_size.get("test_size", 0.1)
            self.valid_size = self.split_size.get("valid_size", 0.1)

    def load_data(self, texts, labels, data_name):
        if len(texts) != len(labels):
            logging.error("Length of texts and labels should be same")
            raise ValueError("Length of texts and labels should be same")
        self.texts = texts
        self.labels = labels
        logging.info(
            "Loading data: %s to classifier %s", data_name, self.model_name
        )
        self.data_name = data_name
        self.classifier_name = (
            f"{self.model_name}_{self.tokenizer_name}_{self.data_name}"
        )
        return self.texts, self.labels, self.data_name

    def process_data(self):
        self.texts = list(
            map(
                lambda text: split_remove_stopwords_punctuation(
                    text, language=self.language
                ),
                self.texts,
            )
        )