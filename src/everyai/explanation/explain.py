import logging
from pathlib import Path

import shap
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

from everyai.classifier.classify import TextClassifer
from everyai.everyai_path import FIG_PATH


class Explanation:
    def __init__(self, classifier: TextClassifer):
        self.classifier = classifier
        self.labels = set(classifier.labels)
        self.pipeline = make_pipeline(classifier.tokenizer, classifier.model)


class LimeExplanation(Explanation):
    def __init__(self, classifier: TextClassifer):
        super().__init__(classifier)
        self.output_path = FIG_PATH / "lime" / classifier.classifier_name

    def explain(self, output_path: Path = None):
        output_path = self.output_path if output_path is None else output_path
        if not output_path.exists():
            output_path.mkdir(parents=True)
        test_indices = self.classifier.data.test_indices.tolist()
        test_text = [self.classifier.texts[i] for i in test_indices]
        explainer = LimeTextExplainer(class_names=self.labels)
        for i, text in enumerate(test_text):
            exp = explainer.explain_instance(
                text, self.pipeline.predict_proba, num_features=6
            )
            exp.save_to_file(output_path / f"text{i}.html")
        logging.info("Lime explanation saved to %s", output_path)


class ShapExplanation(Explanation):
    def __init__(self, classifier: TextClassifer):
        super().__init__(classifier)
        self.output_path = FIG_PATH / "shap" / classifier.classifier_name

    def explain(self, output_path: Path = None):
        output_path = self.output_path if output_path is None else output_path
        if not output_path.exists():
            output_path.mkdir(parents=True)
        test_indices = self.classifier.data.test_indices.tolist()
        test_text = [self.classifier.texts[i] for i in test_indices]
        explainer = shap.Explainer(
            self.pipeline.predict_proba, masker=shap.maskers.Text()
        )
        for i, text in enumerate(test_text):
            shap_values = explainer([text])
            html_output = shap.plots.text(shap_values, display=False)
            with open(output_path / f"text{i}", "w", encoding="utf-8") as f:
                f.write(html_output)
        logging.info("Shap explanation saved to %s", output_path)
