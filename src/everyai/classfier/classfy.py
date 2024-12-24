import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from everyai.everyai_path import MODEL_PATH, RESULT_PATH


@dataclass
class ClassfierData:
    x: Optional[list] = None
    y: Optional[list] = None
    x_train: Optional[list] = None
    y_train: Optional[list] = None
    x_valid: Optional[list] = None
    y_valid: Optional[list] = None
    x_test: Optional[list] = None
    y_test: Optional[list] = None
    y_pred: Optional[list] = None
    train_indices: Optional[list] = None
    valid_indices: Optional[list] = None
    test_indices: Optional[list] = None


def evaluate_classification_model(
    y_true, y_pred, model=None, X_test=None, output_path: Path = None
):
    """
    Evaluate the performance of a classification model.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    model (optional): Trained model object for probability predictions.
    X_test (optional): Test data for probability predictions.
    output_path (optional): Path to save the ROC and Precision-Recall curves.

    Returns:
    dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    # 准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 精确度
    precision = precision_score(y_true, y_pred)
    # 召回率
    recall = recall_score(y_true, y_pred)
    # F1 分数
    f1 = f1_score(y_true, y_pred)
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # ROC 曲线和 AUC（需要预测概率）
    if model is not None and X_test is not None:
        fpr, tpr, _ = roc_curve(y_true, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = None
    # 精度-召回率曲线
    if model is not None and X_test is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_true, model.predict_proba(X_test)[:, 1]
        )
    else:
        precision_vals, recall_vals = [], []
    # 打印和返回结果
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1:.2f}")
    logging.info("Confusion Matrix:")
    logging.info(cm)
    # Save results to CSV

    results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Score": [accuracy, precision, recall, f1, roc_auc],
    }
    results_df = pd.DataFrame(results)
    output_path.mkdir(parents=True, exist_ok=True)
    results_csv_path = output_path / "classification_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Results saved to {results_csv_path}")
    if roc_auc is not None:
        print(f"AUC: {roc_auc:.2f}")
        # 绘制 ROC 曲线
        plt.figure()
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(output_path / "roc_curve.png")
    else:
        logging.warning("ROC curve not available")
        print("AUC: N/A")

    if len(precision_vals) > 0 and len(recall_vals) > 0:
        # 绘制精度-召回率曲线
        plt.figure()
        plt.plot(recall_vals, precision_vals, color="b", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.savefig(output_path / "precision_recall_curve.png")
    else:
        logging.warning("Precision-Recall curve not available")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


class TextClassifer:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        texts: List[str] = None,
        labels: List[str] = None,
        data_name: str = None,
        device="cpu",
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_name = data_name
        self.score = None
        self.data = ClassfierData()
        self.classfier_name = (
            f"{self.model_name}_{self.tokenizer_name}_{self.data_name}"
        )
        self.model_config = None
        self.model_path = (
            MODEL_PATH / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}.pkl"
        )

    def load_data(self, texts, labels, data_name):
        if len(texts) != len(labels):
            logging.error("Length of texts and labels should be same")
            raise ValueError("Length of texts and labels should be same")
        self.texts = texts
        self.labels = labels
        logging.info(f"Loading data: {data_name} to classfier {self.model_name}")
        self.data_name = data_name
        self.classfier_name = (
            f"{self.model_name}_{self.tokenizer_name}_{self.data_name}"
        )
        return self.texts, self.labels, self.data_name

    def _split_data(self, path: Path):
        try:
            self.model = joblib.load(path)
        except FileNotFoundError:
            logging.error(f"File not found: {path}")
            raise
        return self.model

    def _tokenize(self):
        return None

    def train(self):
        return None

    def test(self):
        return None

    def predict(self):
        return None

    def show_score(self):
        self.score = evaluate_classification_model(
            self.data.y_test,
            self.data.y_pred,
            self.model,
            self.data.x_test,
            output_path=RESULT_PATH
            / "classfiy_result"
            / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}",
        )

    def save_model(self, path):
        joblib.dump(self.model, path)
        logging.info(f"Model saved to {path}")
        return self.model

    def load_model(self, path):
        self.model = joblib.load(path)
        return self.model


class SklearnClassifer(TextClassifer):
    def __init__(self, **classfiy_config):
        super().__init__(
            model_name=classfiy_config["model_name"],
            tokenizer_name=classfiy_config["tokenizer_name"],
        )
        logging.info(f"Classfier config: {classfiy_config}")
        split_size = classfiy_config.get("split_size", {})
        train_size = split_size.get("train_size")
        test_size = split_size.get("test_size")
        valid_size = split_size.get("valid_size")
        if (
            train_size is not None
            and test_size is not None
            and valid_size is not None
            and train_size + test_size + valid_size == 1
        ):
            self.train_size = classfiy_config["split_size"]["train_size"]
            self.test_size = classfiy_config["split_size"]["test_size"]
            self.valid_size = classfiy_config["split_size"]["valid_size"]
        else:
            logging.warning("Split size not provided or not valid")
        if self.texts is None or self.labels is None or self.data_name is None:
            logging.warning("Data not provided, please use the load_data method")
        if "device" in classfiy_config and classfiy_config["device"] == "cuda":
            logging.warning(
                "Cuda is not supported in sklearn and setting device to cpu"
            )
            self.device = "cpu"
        else:
            self.device = "cpu"
        match classfiy_config["model_name"]:
            case "LogisticRegression":
                self.model = LogisticRegression()
            case "RandomForest":
                self.model = RandomForestClassifier()
            case "SVM":
                self.model = SVC(probability=True)
            case "XGBoost":
                self.model = xgb.XGBClassifier()
            case _:
                logging.error("Model not supported")
                raise ValueError("Model not supported")
        match classfiy_config["tokenizer_name"]:
            case "CountVectorizer":
                self.tokenizer = CountVectorizer()
            case "TfidfVectorizer" | "tfidf" | "TFIDF" | "tf-idf" | "TF-IDF":
                self.tokenizer = TfidfVectorizer()
            case _:
                logging.error("Tokenizer not supported")
                raise ValueError("Tokenizer not supported")
        if "model_config" in classfiy_config:
            self.model_config = classfiy_config["model_config"]
            self.model.set_params(**self.model_config)
        else:
            logging.warning("Model config not provided")
        if "tokenizer_config" in classfiy_config:
            self.tokenizer_config = classfiy_config["tokenizer_config"]
            self.tokenizer.set_params(**self.tokenizer_config)
        else:
            logging.warning("Tokenizer config not provided")

    def _split_data(self, x, y):
        # 获取原始数据的索引
        original_indices = (
            x.index if isinstance(x, pd.DataFrame) else np.arange(x.shape[0])
        )

        # 第一次划分: 训练集和测试集
        x_train, x_test, y_train, y_test, train_indices, test_indices = (
            train_test_split(
                x,
                y,
                original_indices,
                test_size=self.test_size,
                random_state=42,
            )
        )

        x_train, x_valid, y_train, y_valid, train_indices, valid_indices = (
            train_test_split(
                x_train,
                y_train,
                train_indices,
                test_size=self.valid_size / (self.train_size + self.valid_size),
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

    def _tokenize(self, texts, labels):
        return self.tokenizer.fit_transform(
            texts
        ), sklearn.preprocessing.LabelEncoder().fit_transform(labels)

    def train(self):
        self.data.x, self.data.y = self._tokenize(self.texts, self.labels)
        (
            self.data.x_train,
            self.data.x_valid,
            self.data.x_test,
            self.data.y_train,
            self.data.y_valid,
            self.data.y_test,
            self.data.train_indices,
            self.data.valid_indices,
            self.data.test_indices,
        ) = self._split_data(self.data.x, self.data.y)
        self.model.fit(self.data.x_train, self.data.y_train)
        return self.model

    def test(self):
        self.data.y_pred = self.model.predict(self.data.x_test)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, path: Path = None):
        if path is None:
            path = (
                MODEL_PATH
                / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}.pkl"
            )
        else:
            path = path
        joblib.dump(self.model, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: Path = None):
        if path is None:
            path = (
                MODEL_PATH
                / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}.pkl"
            )
        else:
            path = path
        logging.info(f"Loading model from {path}")
        self.model = joblib.load(path)
        return self.model


class PytorchClassifer(TextClassifer):
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        model_name: str = None,
        tokenizer_name: str = None,
        data_name: str = None,
        device: str = "cpu",
        model=None,
        tokenizer=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            texts,
            labels,
            model_name,
            tokenizer_name,
            data_name,
            device,
            model,
            tokenizer,
        )
        return self.model

    def test(self, x_test, y_test):
        return self.model

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, path):
        return self.model

    def load(self, path):
        return self.model
