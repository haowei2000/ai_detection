import logging

import datasets
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from everyai.classifier.classify import TextClassifer, label_encode, split_data
from everyai.classifier.multi_feature_model.fusionBert import FeatureFusionBertClassfier
from everyai.utils.everyai_path import MODEL_PATH


class HuggingfaceClassifer(TextClassifer):
    def __init__(
        self,
        texts=None,
        labels=None,
        data_name="",
        classfiy_config=None,
    ):
        super().__init__(
            texts=texts, labels=labels, data_name=data_name, **classfiy_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        fusion_model_dict = {
            "FeatureFusionBertClassfier": FeatureFusionBertClassfier(),
        }

        self.model = fusion_model_dict[self.model_name]
        if "bert" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.label_encoder = None
        self.train_dataset, self.valid_dataset, self.test_dataset = (
            None,
            None,
            None,
        )
        self.train_args["output_dir"] = MODEL_PATH / self.classifier_name

    def _tokenize(self, texts: list[str], labels: list[str]):
        self.label_encoder, tokenzied_labels = label_encode(labels)
        tokenzied_labels = torch.tensor(tokenzied_labels)
        dataset = datasets.Dataset.from_dict({"text": texts, "label": tokenzied_labels})

        def _tokenizer_fn(example):
            return self.tokenizer(example["text"], **self.tokenizer_config)

        tokenzied_dataset = dataset.map(_tokenizer_fn, batched=True)
        tokenzied_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        tokenzied_dataset = tokenzied_dataset.remove_columns(["text"])
        return tokenzied_dataset

    def train(self):
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
        ) = split_data(self.texts, self.labels)
        self.train_dataset = self._tokenize(self.data.x_train, self.data.y_train)
        self.valid_dataset = self._tokenize(self.data.x_valid, self.data.y_valid)
        self.test_dataset = self._tokenize(self.data.x_test, self.data.y_test)
        train_args = TrainingArguments(**self.train_args)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=data_collator,
        )
        trainer.train()

    def test(self):
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(self.test_dataset)
        self.data.y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1)

        self.data.y_test = self.label_encoder.transform(self.data.y_test)

    def show_score(self):
        metric = evaluate.load("accuracy")
        metric.compute(predictions=self.data.y_pred, references=self.data.y_test)
        logging.info("Accuracy: %s", metric)
