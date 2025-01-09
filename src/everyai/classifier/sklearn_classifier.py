import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from everyai.classifier.classify import TextClassifer, label_encode, split_data
from everyai.utils.everyai_path import RESULT_PATH


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
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
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
    results_csv_path = output_path / "scores.csv"
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


def _init_sklearn_pipeline(pipeline_config: list[dict]):
    step_dict = {
        "tf_idf": TfidfVectorizer(),
        "count_vectorizer": CountVectorizer(),
        "logistic_regression": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "svm": SVC(probability=True),
        "xgboost": xgb.XGBClassifier(),
        "standard_scaler": StandardScaler(),
        "min_max_scaler": MinMaxScaler(),
        "pca": PCA(),
        "kmeans": KMeans(),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(),
        "naive_bayes": GaussianNB(),
    }
    steps = []
    for step_name in pipeline_config:
        if step_name in step_dict.items():
            step_params = pipeline_config[step_name]
            steps.append((step_name, step_dict[step_name](**step_params)))
        else:
            logging.warning("Step %s not recognized and will be skipped", step_name)
    return make_pipeline(*[step[1] for step in steps])


class SklearnClassifer(TextClassifer):
    def __init__(
        self,
        texts=None,
        labels=None,
        data_name="",
        language: str = "English",
        **classify_config,
    ):
        if classify_config is None:
            classify_config = {}
        super().__init__(
            texts=texts,
            labels=labels,
            data_name=data_name,
            language=language,
            **classify_config,
        )
        self.label_encoder = None
        model_dict = {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "random_forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "XGBoost": xgb.XGBClassifier(),
        }
        tokenizer_dict = {
            "CountVectorizer": CountVectorizer(),
            "TfidfVectorizer": TfidfVectorizer(),
        }
        if self.model_name in model_dict:
            self.model = model_dict[self.model_name]
        else:
            logging.error("Model not recognized and will be skipped")
        if self.tokenizer_name in tokenizer_dict:
            self.tokenizer = tokenizer_dict[self.tokenizer_name]
        else:
            logging.warning("Tokenizer not recognized and will be skipped")
        if self.model_config is not None:
            self.model.set_params(**self.model_config)
        else:
            logging.warning("Model config not provided")
        if self.tokenizer_config is not None:
            self.tokenizer.set_params(**self.tokenizer_config)
        else:
            logging.warning("Tokenizer config not provided")
        # TODO: Add support for pipeline
        if self.pipeline is not None:
            self.pipeline = _init_sklearn_pipeline(self.pipeline)
        else:
            self.pipeline = make_pipeline(self.tokenizer, self.model)
            logging.info(
                "Pipeline not provided and make pipeline with tokenizer and model"
            )

    def _tokenize(self, texts: list[str], labels: list[str]):
        self.label_encoder, label_encoded = label_encode(labels)
        return self.tokenizer.fit_transform(texts), label_encoded

    def train(self):
        self.data.x, self.data.y = self._tokenize(self.texts, self.labels)
        logging.info(
            "%s texts are tokenized and %s labels are tokenized",
            len(self.texts),
            len(self.labels),
        )
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
        ) = split_data(self.data.x, self.data.y)
        self.model.fit(self.data.x_train, self.data.y_train)
        return self.model

    def test(self):
        self.data.y_pred = self.model.predict(self.data.x_test)

    def predict(self, x):
        return self.model.predict(x)

    def show_score(self):
        output_path = (
            RESULT_PATH
            / "classfiy_result"
            / f"{self.model_name}_{self.tokenizer_name}_{self.data_name}"
        )
        self.score = evaluate_classification_model(
            self.data.y_test,
            self.data.y_pred,
            self.model,
            self.data.x_test,
            output_path=output_path,
        )
        test_texts = [self.texts[i] for i in self.data.test_indices]
        result = {
            "text": test_texts,
            "label": list(self.label_encoder.inverse_transform(self.data.y_test)),
            "pred": list(self.label_encoder.inverse_transform(self.data.y_pred)),
        }
        pd.DataFrame(result).to_csv(output_path / "classification_results.csv")
