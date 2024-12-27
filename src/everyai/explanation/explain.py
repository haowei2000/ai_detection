import logging
from pathlib import Path

import shap
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

from everyai.classfier.classfy import TextClassifer
from everyai.everyai_path import FIG_PATH


class Explanation:
    def __init__(self, classfier: TextClassifer):
        self.classfier = classfier
        self.labels = set(classfier.labels)
        self.pipeline = make_pipeline(classfier.tokenizer, classfier.model)

    def explain(self, output_path: Path = None):
        pass


class LimeExplanation(Explanation):
    def __init__(self, classfier: TextClassifer):
        super().__init__(classfier)
        self.output_path = FIG_PATH / "lime" / classfier.classfier_name

    def explain(self, output_path: Path = None):
        output_path = self.output_path if output_path is None else output_path
        if not output_path.exists():
            output_path.mkdir(parents=True)
        test_indices = self.classfier.data.test_indices.tolist()
        test_text = [self.classfier.texts[i] for i in test_indices]
        explainer = LimeTextExplainer(class_names=self.labels)
        for i, text in enumerate(test_text):
            exp = explainer.explain_instance(
                text, self.pipeline.predict_proba, num_features=6
            )
            exp.save_to_file(output_path / f"text{i}.html")
        logging.info(f"Lime explanation saved to {output_path}")


class ShapExplanation(Explanation):
    def __init__(self, classfier: TextClassifer):
        super().__init__(classfier)
        self.output_path = FIG_PATH / "shap" / classfier.classfier_name

    def explain(self, output_path: Path = None):
        output_path = self.output_path if output_path is None else output_path
        output_path.exists() or output_path.mkdir(parents=True)
        test_indices = self.classfier.data.test_indices.tolist()
        test_text = [self.classfier.texts[i] for i in test_indices]
        explainer = shap.Explainer(
            self.pipeline.predict_proba, masker=shap.maskers.Text()
        )
        for i, text in enumerate(test_text):
            shap_values = explainer([text])
            html_output = shap.plots.text(shap_values, display=False)
            with open(output_path / f"text{i}", "w", encoding="utf-8") as f:
                f.write(html_output)
        logging.info(f"Shap explanation saved to {output_path}")
