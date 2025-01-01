import logging

import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from everyai.classifier.classify import TextClassifer, label_encode, split_data


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
            logging.warning(
                "Step %s not recognized and will be skipped", step_name
            )
    return make_pipeline(*[step[1] for step in steps])


class SklearnClassifer(TextClassifer):
    def __init__(self, **classfiy_config):
        super().__init__(
            model_name=classfiy_config["model_name"],
            tokenizer_name=classfiy_config["tokenizer_name"],
        )
        logging.info("classifier config: %s", classfiy_config)
        split_size = classfiy_config.get("split_size", {})
        self.train_size = split_size.get("train_size", 0.8)
        self.test_size = split_size.get("test_size", 0.1)
        self.valid_size = split_size.get("valid_size", 0.1)
        model_dict = {
            "LogisticRegression": LogisticRegression,
            "RandomForest": RandomForestClassifier,
            "SVM": SVC,
            "XGBoost": xgb.XGBClassifier,
        }
        tokenizer_dict = {
            "CountVectorizer": CountVectorizer,
            "TfidfVectorizer": TfidfVectorizer,
        }

        if classfiy_config["model_name"] in model_dict:
            self.model = model_dict[classfiy_config["model_name"]]()
        else:
            logging.error("Model not supported")
            raise ValueError("Model not supported")

        if classfiy_config["tokenizer_name"] in tokenizer_dict:
            self.tokenizer = tokenizer_dict[
                classfiy_config["tokenizer_name"]
            ]()
        else:
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
        # TODO: Add support for pipeline
        if "pipeline" in classfiy_config:
            self.pipeline = _init_sklearn_pipeline(classfiy_config["pipeline"])
        else:
            self.pipeline = make_pipeline(self.tokenizer, self.model)
            logging.info(
                "Pipeline not provided and make pipeline with tokenizer and model"
            )

    def _tokenize(self, texts, labels):
        return self.tokenizer.fit_transform(texts), label_encode(labels)

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
