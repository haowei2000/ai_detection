import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
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


def label_encode(labels):
    return LabelEncoder().fit_transform(labels)


# -> tuple:
def split_data(x, y, train_size=0.8, valid_size=0.1, test_size=0.1):
    # 获取原始数据的索引
    original_indices = x.index if isinstance(x, pd.DataFrame) else np.arange(x.shape[0])

    # 第一次划分: 训练集和测试集
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        x,
        y,
        original_indices,
        test_size=test_size,
        random_state=42,
    )
    # 第二次划分: 训练集和验证集
    x_train, x_valid, y_train, y_valid, train_indices, valid_indices = train_test_split(
        x_train,
        y_train,
        train_indices,
        test_size=valid_size / (train_size + valid_size),
        random_state=42,
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
    if model is not None and X_test is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_true, model.predict_proba(X_test)[:, 1]
        )
    else:
        precision_vals, recall_vals = [], []
    logging.info("Accuracy: %.2f", accuracy)
    logging.info("Precision: %.2f", precision)
    logging.info("Recall: %.2f", recall)
    logging.info("F1 Score: %.2f", f1)
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
    logging.info("Results saved to %s", results_csv_path)
    if roc_auc is not None:
        _plot_roc(roc_auc, output_path)
    else:
        logging.warning("ROC curve not available")
        print("AUC: N/A")

    if len(precision_vals) > 0 and len(recall_vals) > 0:
        # 绘制精度-召回率曲线
        plt.figure()
        plt.plot(recall_vals, precision_vals, color="b", lw=2)
        plot_pr("Recall", "Precision", "Precision-Recall curve")
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


def _plot_roc(roc_auc, output_path):
    logging.info("AUC: %.2f", roc_auc)
    # 绘制 ROC 曲线
    plt.figure()
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plot_pr(
        "False Positive Rate",
        "True Positive Rate",
        "Receiver Operating Characteristic",
    )
    plt.legend(loc="lower right")
    plt.savefig(output_path / "roc_curve.png")


def plot_pr(arg0, arg1, arg2):
    plt.xlabel(arg0)
    plt.ylabel(arg1)
    plt.title(arg2)


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
        pipeline=None,
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
        self.pipeline = pipeline if pipeline is not None else None

    def load_data(self, texts, labels, data_name):
        if len(texts) != len(labels):
            logging.error("Length of texts and labels should be same")
            raise ValueError("Length of texts and labels should be same")
        self.texts = texts
        self.labels = labels
        logging.info("Loading data: %s to classfier %s", data_name, self.model_name)
        self.data_name = data_name
        self.classfier_name = (
            f"{self.model_name}_{self.tokenizer_name}_{self.data_name}"
        )
        return self.texts, self.labels, self.data_name

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

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        joblib.dump(self.model, path)
        logging.info("Model saved to %s", path)
        return self.model

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        self.model = joblib.load(path)
        return self.model
