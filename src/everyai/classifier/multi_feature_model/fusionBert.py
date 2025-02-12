import math

import pytorch_lightning as pl
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.optimization import get_scheduler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def truncate_and_pad_single_sequence(seq, max_length):
    truncated_seq = seq[:max_length]
    padding = max_length - len(truncated_seq)
    return F.pad(truncated_seq, (0, padding), value=0)


class FeatureFusionBertTokenizer:
    def __init__(self, feature_len, **kwargs):
        self.feature_len = feature_len
        self.sentiment_max_length = kwargs.get(
            "sentiment_max_length", feature_len
        )
        self.nlp = spacy.load("en_core_web_sm")
        self.all_tags = self.nlp.get_pipe("tagger").labels
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.bert_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased"
        )
        self.bert_tokenzier = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )

    def semantic(self, text: str, **kwargs):
        tokenzied = self.bert_tokenzier(
            text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True,
            **kwargs,
        )
        with torch.no_grad():
            embedding = self.bert_model.bert.embeddings(tokenzied["input_ids"])
            embedding = torch.mean(embedding, dim=1, keepdim=False)
        return embedding, torch.ones_like(embedding)

    def analyze_word_level_sentiment(self, text: str) -> torch.tensor:
        doc = self.nlp(text)
        word_sentiment = []
        for token in doc:
            if token.is_stop or token.is_punct:
                sentiment = 0.0
            else:
                sentiment = self.sentiment_analyzer.polarity_scores(
                    token.text
                )["compound"]
            word_sentiment.append(sentiment)
        sentiment = torch.tensor(word_sentiment, dtype=torch.float)
        return self._padding(sentiment)

    def pos_feature(self, text: str) -> torch.tensor:
        doc = self.nlp(text)
        text_tags = [token.tag_ for token in doc]
        pos = torch.tensor(
            [text_tags.count(label) for label in self.all_tags],
            dtype=torch.float,
        )
        return self._padding(pos)

    def _padding(self, input_tensor):
        input_tensor = truncate_and_pad_single_sequence(
            input_tensor, self.feature_len
        )
        attention_mask = torch.tensor(
            [1] * len(input_tensor), dtype=torch.float
        )
        return input_tensor.unsqueeze(0), attention_mask.unsqueeze(0)

    def __call__(self, text: str):
        encoding = {"semantic": {}, "pos": {}, "sentiment": {}}
        (
            encoding["semantic"]["input_ids"],
            encoding["semantic"]["attention_mask"],
        ) = self.semantic(text)
        encoding["pos"]["input_ids"], encoding["pos"]["attention_mask"] = (
            self.pos_feature(text)
        )
        (
            encoding["sentiment"]["input_ids"],
            encoding["sentiment"]["attention_mask"],
        ) = self.analyze_word_level_sentiment(text)
        return encoding

    # @lru_cache(maxsize=12)
    def batch_encode_plus(self, batch_text: list[str]):  # -> list:
        batch_encoding = []
        for text in tqdm(batch_text, desc="Encoding batch"):
            if not isinstance(text, str):
                raise ValueError(
                    f"Expected text to be of type str, but got {type(text)}"
                )
            encoding = self.__call__(text)
            batch_encoding.append(encoding)
        return batch_encoding


class HFeatureFusion(nn.Module):
    def __init__(
        self, feature_num, feature_len, output_dim, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.feature_num = feature_num
        self.feature_len = feature_len
        self.num_heads = num_heads
        self.dropout = dropout
        self.cross_attentions = nn.MultiheadAttention(
            embed_dim=self.feature_len,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.feature_fusion_dim = (feature_num - 1) * feature_len
        self.fc = nn.Linear(self.feature_fusion_dim, output_dim)

    def forward(self, *features: torch.tensor):
        if len(features) != self.feature_num:
            raise ValueError(
                f"Expected {self.feature_num} features, but got {len(features)}"
            )
        outputs = []
        for i in range(self.feature_num - 1):
            key = features[i]["input_ids"] if i == 0 else outputs[i - 1]
            query = features[i + 1]["input_ids"]
            value = features[i + 1]["input_ids"]
            output, _ = self.cross_attentions(
                query,
                key,
                value,
            )
            outputs.append(output)
        output = self.fc(torch.cat(outputs, dim=-1))
        return output


class CrossAttentionFeatureFusion(nn.Module):
    def __init__(
        self, feature_num, feature_len, output_dim, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.feature_len = feature_len
        self.num_heads = num_heads
        self.dropout = dropout
        self.feature_num = feature_num
        self.cross_attentions = nn.MultiheadAttention(
            embed_dim=self.feature_len,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        feature_fusion_dim = math.perm(feature_num, 2) * feature_len
        self.fc = nn.Linear(feature_fusion_dim, output_dim)

    def forward(self, *features: torch.tensor):
        if len(features) != self.feature_num:
            raise ValueError(
                f"Expected {self.feature_num} features, but got {len(features)}"
            )
        output_list = []
        for i in range(self.feature_num):
            for j in range(self.feature_num):
                if i != j:
                    query = features[i]["input_ids"]
                    key = features[j]["input_ids"]
                    value = features[j]["input_ids"]
                    output, _ = self.cross_attentions(
                        query,
                        key,
                        value,
                    )
                    output_list.append(output)
        output_vec = self.fc(torch.cat(output_list, dim=-1))
        return output_vec


class FeatureFusionBertClassfier(pl.LightningModule):
    def __init__(
        self,
        feature_num=3,
        feature_len=768,
        bert_input_dim=768,
        num_labels=2,
        lr=1e-4,
        fusion_module=None,
    ):
        super().__init__()
        if fusion_module is None:
            self.fusion_module = CrossAttentionFeatureFusion(
                feature_num=feature_num,
                feature_len=feature_len,
                output_dim=bert_input_dim,
            )
        else:
            self.fusion_module = fusion_module
        self.save_hyperparameters()
        bert_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        bert_classifier.train()
        self.encoder = bert_classifier.bert.encoder
        self.pooler = bert_classifier.bert.pooler
        self.dropout = bert_classifier.dropout
        self.classfier = bert_classifier.classifier
        self.lr = lr

    def forward(self, *features):
        self.train()  # Set all modules to train
        for feature in features:
            feature["input_ids"] = F.normalize(
                feature["input_ids"], p=2, dim=-1
            )
        fused_features = self.fusion_module(*features)
        encoder_output = self.encoder(fused_features)
        pooler_output = self.pooler(encoder_output[0])
        pooler_output = self.dropout(pooler_output)
        output = self.classfier(pooler_output)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch["semantic"], batch["pos"], batch["sentiment"])
        loss = F.cross_entropy(outputs, batch["labels"])
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch["semantic"], batch["pos"], batch["sentiment"])
        y_pred = torch.argmax(outputs, dim=-1)
        y_test = batch["labels"]
        acc = (y_pred == y_test).sum().float() / len(y_test)
        f1 = f1_score(
            y_test.cpu().numpy(), y_pred.cpu().numpy(), average="weighted"
        )
        self.log("test_acc", acc)
        self.log("test_f1", f1)
        return {"test_acc": acc, "test_f1": f1}

    def test_epoch_end(self, outputs):
        avg_acc = sum(x["test_acc"] for x in outputs) / len(outputs)
        avg_f1 = sum(x["test_f1"] for x in outputs) / len(outputs)
        self.log("avg_test_acc", avg_acc, prog_bar=True)
        self.log("avg_test_f1", avg_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, momentum=0.9
        )
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]


class FeatureFusionDataset(Dataset):
    def __init__(self, inputs_datasets: list, labels: list):
        self.inputs = inputs_datasets
        self.labels = labels

    def __getitem__(self, idx):
        item = self.inputs[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class FeatureFusionDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32):
        """
        -Args:
            train_data (dict): Dictionary containing the training data
            test_data (dict): Dictionary containing the test data
            valid_data (dict): Dictionary containing the validation data
            batch_size (int): Batch size for the data loader

        -Note: The data should be in the format like {"text": list[str], "label": list[int]}
        """
        super().__init__()
        train_data, test_data, valid_data = (
            data["train"],
            data["test"],
            data["valid"],
        )
        self.batch_size = batch_size
        self.train_raw_data = train_data
        self.test_raw_data = test_data
        self.valid_raw_data = valid_data
        self.train_tokenized, self.valid_tokenized, self.test_tokenized = (
            None,
            None,
            None,
        )
        self.train_dataset, self.valid_dataset, self.test_dataset = (
            None,
            None,
            None,
        )
        self.tokenizer: FeatureFusionBertTokenizer = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.tokenizer: FeatureFusionBertTokenizer = (
            FeatureFusionBertTokenizer(feature_len=768)
        )
        if not isinstance(self.tokenizer, FeatureFusionBertTokenizer):
            raise TypeError(
                f"Expected self.tokenizer to be of type FeatureFusionBertTokenizer, but got {type(self.tokenizer)}"
            )
        if stage == "fit" or stage is None:
            self.train_tokenized = self.tokenizer.batch_encode_plus(
                batch_text=self.train_raw_data["text"]
            )
            self.train_dataset = FeatureFusionDataset(
                self.train_tokenized, self.train_raw_data["label"]
            )
        if stage == "test" or stage is None:
            self.test_tokenized = self.tokenizer.batch_encode_plus(
                self.test_raw_data["text"]
            )
            self.test_dataset = FeatureFusionDataset(
                self.test_tokenized, self.test_raw_data["label"]
            )
        if stage == "validate" or stage is None:
            self.valid_tokenized = self.tokenizer.batch_encode_plus(
                self.valid_raw_data["text"]
            )
            self.valid_dataset = FeatureFusionDataset(
                self.valid_tokenized, self.valid_raw_data["label"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=15,
        )


if __name__ == "__main__":

    def load_data(max_count=1000):
        dataset = load_dataset("imdb")
        train_data = {
            "text": dataset["train"]["text"][:max_count],
            "label": dataset["train"]["label"][:max_count],
        }
        test_data = {
            "text": dataset["test"]["text"][:max_count],
            "label": dataset["test"]["label"][:max_count],
        }
        valid_data = {
            "text": dataset["unsupervised"]["text"][:max_count],
            "label": dataset["unsupervised"]["label"][:max_count],
        }
        return {"train": train_data, "test": test_data, "valid": valid_data}

    data = load_data()
    data_module = FeatureFusionDataModule(data)
    model = FeatureFusionBertClassfier(
        fusion_module=CrossAttentionFeatureFusion(
            feature_num=3, feature_len=768, output_dim=768
        )
    )
    wandb_logger = WandbLogger(project="everyAI", log_model="all")
    trainer = pl.Trainer(
        max_epochs=2,
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
    )
    trainer.fit(model, data_module)
    # trainer.test(model, datamodule=data_module)
    wandb_logger.watch(model)
