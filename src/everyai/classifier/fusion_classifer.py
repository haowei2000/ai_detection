import logging

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger

from everyai.classifier.classify import TextClassifer, label_encode, split_data
from everyai.classifier.multi_feature_model.fusionBert import (
    CrossAttentionFeatureFusion, FeatureFusionBertClassfier,
    FeatureFusionDataModule, HFeatureFusion)


class PLClassifer(TextClassifer):
    def __init__(
        self,
        texts=None,
        labels=None,
        data_name="",
        **classfiy_config,
    ):
        super().__init__(
            texts=texts, labels=labels, data_name=data_name, **classfiy_config
        )
        if self.model_config is None:
            self.model_config = {}
        model_dict = {
            "fusion_bert": FeatureFusionBertClassfier(
                fusion_module=HFeatureFusion(
                    feature_num=3, feature_len=768, output_dim=768
                ),
                lr=1e-6,
                **self.model_config,
            ),
        }
        self.model = model_dict[self.model_name]
        self.data_moudle = None
        self.label_encoder = None

    def _prepare_data(self):
        self.label_encoder, self.labels = label_encode(self.labels)
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
        logging.info(
            "Data was splited into train(%i), valid(%i) and test(%i)",
            len(self.data.x_train),
            len(self.data.x_valid),
            len(self.data.x_test),
        )
        data = {
            "train": {"text": self.data.x_train, "label": self.data.y_train},
            "valid": {"text": self.data.x_valid, "label": self.data.y_valid},
            "test": {"text": self.data.x_test, "label": self.data.y_test},
        }
        data_moudle_dict = {
            "fusion_bert": FeatureFusionDataModule(data=data),
        }
        self.data_moudle = data_moudle_dict[self.tokenizer_name]

    def train(self):
        self._prepare_data()
        if not hasattr(self, "train_args") or self.train_args is None:
            self.train_args = {
                "max_epochs": 10,
                "devices": 2,
                "accelerator": "auto",
            }
        wandb_logger = WandbLogger(project="everyAI", log_model="all")
        trainer = pl.Trainer(
            **self.train_args,
            logger=wandb_logger,
        )
        logging.info("log file was saved in %s", "lightning_logs")
        trainer.fit(self.model, self.data_moudle)
        wandb_logger.watch(model=self.model)
