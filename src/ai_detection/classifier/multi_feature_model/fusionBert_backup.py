import math

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.optimization import get_scheduler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def batch_encode_plus(self, batch_text: list[str], **kwargs):
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
            # TODO Some issue in attention_mask
            # q_mask = features[i]["attention_mask"]
            # k_mask = features[j]["attention_mask"]
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
        outputs = []
        for i in range(self.feature_num):
            for j in range(self.feature_num):
                if i != j:
                    query = features[i]["input_ids"]
                    key = features[j]["input_ids"]
                    value = features[j]["input_ids"]
                    # TODO Some issue in attention_mask
                    # q_mask = features[i]["attention_mask"]
                    # k_mask = features[j]["attention_mask"]
                    output, _ = self.cross_attentions(
                        query,
                        key,
                        value,
                    )
                    outputs.append(output)
        output = self.fc(torch.cat(outputs, dim=-1))
        return output


class FeatureFusionBertClassfier(nn.Module):
    def __init__(
        self, feature_num=3, feature_len=768, bert_input_dim=768, num_labels=2
    ):
        super().__init__()
        bert_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        self.encoder = bert_classifier.bert.encoder
        self.pooler = bert_classifier.bert.pooler
        self.fusion_module = CrossAttentionFeatureFusion(
            feature_num=feature_num,
            feature_len=feature_len,
            output_dim=bert_input_dim,
        )
        self.dropout = bert_classifier.dropout
        self.classfier = bert_classifier.classifier

    def forward(self, *features):
        fused_features = self.fusion_module(*features)
        encoder_output = self.encoder(fused_features)
        pooler_output = self.pooler(encoder_output[0])
        pooler_output = self.dropout(pooler_output)
        output = self.classfier(pooler_output)
        return F.log_softmax(output, dim=-1)


class FeatureFusionDataset(torch.utils.data.Dataset):
    def __init__(self, inputs_datasets, labels):
        self.inputs = inputs_datasets
        self.labels = labels

    def __getitem__(self, idx):
        item = self.inputs[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = load_dataset("imdb")
    test_input = dataset["train"].select(range(1000))
    tokenizer = FeatureFusionBertTokenizer(feature_len=768)
    tokenized_input = tokenizer.batch_encode_plus(test_input["text"])
    train_dataset = FeatureFusionDataset(tokenized_input, test_input["label"])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = FeatureFusionBertClassfier(feature_num=3, feature_len=768)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            for feature in ["semantic", "pos", "sentiment"]:
                batch[feature]["input_ids"] = batch[feature]["input_ids"].to(
                    device
                )
                batch[feature]["attention_mask"] = batch[feature][
                    "attention_mask"
                ].to(device)
            optimizer.zero_grad()
            outputs = model(
                batch["semantic"], batch["pos"], batch["sentiment"]
            )
            loss = F.cross_entropy(outputs, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if step % 100 == 0:
                print(f"Epoch {epoch + 1}, step {step + 1} completed with loss: {loss.item()}")
