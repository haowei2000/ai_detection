import logging
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
from regex import R
import sklearn
import torch
import xgboost as xgb
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

from everyai.everyai_path import RESULT_PATH, MODEL_PATH
import pandas as pd


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
        texts: List[str],
        labels: List[str],
        model_name: str,
        tokenizer_name: str,
        data_name: str,
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
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.model_path = (
            MODEL_PATH
            / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}.pkl"
        )

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
            self.y_test,
            self.y_pred,
            self.model,
            self.x_test,
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
    def __init__(self, texts, labels, **classfiy_config):
        super().__init__(
            texts=texts,
            labels=labels,
            model_name=classfiy_config["model_name"],
            tokenizer_name=classfiy_config["tokenizer_name"],
            data_name=classfiy_config["data_name"],
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
            logging.error("Split size not provided or not valid")
            raise ValueError("Split size not provided or not valid")
        if len(texts) != len(labels):
            logging.error("Length of texts and labels should be same")
            raise ValueError("Length of texts and labels should be same")
        else:
            self.texts = texts
            self.labels = labels
        if "device" in classfiy_config and classfiy_config["device"] == "cuda":
            logging.warning(
                "Cuda is not supported in sklearn and setting device to cpu"
            )
            self.device = "cpu"
        else:
            self.device = "cpu"
        match classfiy_config["model_name"]:
            case "LogisticRegression":
                self.model = sklearn.linear_model.LogisticRegression()
            case "RandomForest":
                self.model = sklearn.ensemble.RandomForestClassifier()
            case "SVM":
                self.model = sklearn.svm.SVC(probability=True)
            case "XGBoost":
                self.model = xgb.XGBClassifier()
            case _:
                logging.error("Model not supported")
                raise ValueError("Model not supported")
        match classfiy_config["tokenizer_name"]:
            case "CountVectorizer":
                self.tokenizer = (
                    sklearn.feature_extraction.text.CountVectorizer()
                )
            case "TfidfVectorizer" | "tfidf" | "TFIDF" | "tf-idf" | "TF-IDF":
                self.tokenizer = (
                    sklearn.feature_extraction.text.TfidfVectorizer()
                )
            case _:
                logging.error("Tokenizer not supported")
                raise ValueError("Tokenizer not supported")

    def _split_data(self):
        x_train, x_test, y_train, y_test = (
            sklearn.model_selection.train_test_split(
                self.x,
                self.y,
                train_size=self.train_size,
                test_size=self.test_size,
            )
        )
        x_train, x_valid, y_train, y_valid = (
            sklearn.model_selection.train_test_split(
                x_train,
                y_train,
                train_size=self.train_size,
                test_size=self.valid_size,
            )
        )
        self.x_train = x_train
        self.x_valid = x_valid
        self.x_test = x_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        return x_train, x_valid, x_test, y_train, y_valid, y_test

    def _tokenize(self):
        self.x = self.tokenizer.fit_transform(self.texts)
        self.y = sklearn.preprocessing.LabelEncoder().fit_transform(
            self.labels
        )
        return self.x, self.y

    def train(self):
        self.x, self.y = self._tokenize()
        x_train, x_valid, x_test, y_train, y_valid, y_test = self._split_data()
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def test(self):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

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
