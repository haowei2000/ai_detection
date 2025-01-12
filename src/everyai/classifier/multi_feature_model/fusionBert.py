import math
import token

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    PreTrainedTokenizer,
    get_scheduler,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def truncate_and_pad_single_sequence(seq, max_length):
    # 截断序列
    truncated_seq = seq[:max_length]
    # 计算需要填充的长度
    padding = max_length - len(truncated_seq)
    return F.pad(truncated_seq, (0, padding), value=0)


class FeatureFusionBertTokenizer:
    def __init__(self, semantic_tokenizer: PreTrainedTokenizer, **kwargs):
        self.sentiment_max_length = kwargs.get("sentiment_max_length", 512)
        self.semantic_tokenizer = semantic_tokenizer
        self.nlp = spacy.load("en_core_web_sm")
        self.all_tags = self.nlp.get_pipe("tagger").labels
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze_word_level_sentiment(
        self, text: str, max_length=512
    ) -> torch.tensor:
        # 使用SpaCy进行词汇级分析
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
        sentiment = truncate_and_pad_single_sequence(sentiment, max_length)
        return sentiment.unsqueeze(0).unsqueeze(0)

    def pos_feature(self, text: str) -> torch.tensor:
        doc = self.nlp(text)
        text_tags = [token.tag_ for token in doc]
        return (
            torch.tensor(
                [text_tags.count(label) for label in self.all_tags],
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def __call__(
        self,
        text: str,
        **kwargs,
    ):
        """
        自定义的编码函数，支持额外的特征输入。
        :param text: 输入文本
        """
        # 使用基础 tokenizer 进行编码
        encoding = self.semantic_tokenizer(text, **kwargs)
        features = [
            self.pos_feature(text),
            self.analyze_word_level_sentiment(
                text, max_length=self.sentiment_max_length
            ),
        ]
        encoding["features"] = features
        return encoding

    def batch_encode_plus(self, batch_text: list[str], **kwargs):
        """
        批量编码函数
        :param batch_text: 输入文本列表
        """
        batch_encoding = self.semantic_tokenizer.batch_encode_plus(
            batch_text, **kwargs
        )
        batch_pos = torch.cat(
            [self.pos_feature(text) for text in batch_text], dim=0
        )
        batch_sentiments = torch.cat(
            [
                self.analyze_word_level_sentiment(
                    text, max_length=self.sentiment_max_length
                )
                for text in batch_text
            ],
            dim=0,
        )
        batch_encoding["features"] = [batch_pos, batch_sentiments]
        return batch_encoding


class CrossAttentionFeatureFusion(nn.Module):
    def __init__(
        self,
        feature_num,
        proj_dim,
        num_heads=2,
        dropout=0.1,
    ):
        """
        基于交叉注意力的特征融合模块
        :param proj_dim: 投影维度
        :param output_dim: 输出维度
        :param num_heads: 多头注意力机制的头数
        :param dropout: Dropout 概率
        """
        super(CrossAttentionFeatureFusion, self).__init__()
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.projections = nn.ModuleList()
        self.feature_num = feature_num
        self.cross_attentions = nn.MultiheadAttention(
            embed_dim=self.proj_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

    def forward(self, features: list[torch.tensor]):
        """
        前向传播
        :param features: 可变数量的特征向量，每个特征形状为 (seq_len, batch_size, feature_dim)
        :return: 融合后的特征，形状为 (seq_len, batch_size, output_dim)
        """
        projected_features = []
        outputs = []
        for i in range(self.feature_num):
            if len(self.projections) <= i:
                self.projections.append(
                    nn.Linear(features[i].size(-1), self.proj_dim)
                )
                projected_features.append(self.projections[i](features[i]))

        for i in range(self.feature_num):
            for j in range(self.feature_num):
                if i != j:
                    query = projected_features[i]  # Query tensor
                    key = projected_features[j]  # Key tensor
                    value = projected_features[j]  # Value tensor
                    output, _ = self.cross_attentions(query, key, value)
                    outputs.append(output)
        return torch.cat(outputs, dim=-1)


class FeatureFusionBertClassfier(nn.Module):
    def __init__(
        self, feature_num=3, proj_dim=64, bert_input_dim=768, num_labels=2
    ):
        super(FeatureFusionBertClassfier, self).__init__()
        bert_classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        self.embeddings = bert_classifier.bert.embeddings
        self.encoder = bert_classifier.bert.encoder
        self.pooler = bert_classifier.bert.pooler
        self.fusion_module = CrossAttentionFeatureFusion(
            feature_num=feature_num, proj_dim=proj_dim
        )
        feature_fusion_outdim = (
            math.perm(feature_num, 2) * proj_dim
        )  # C(n, 2) * proj_dim
        self.fc = nn.Linear(feature_fusion_outdim, bert_input_dim)
        self.dropout = bert_classifier.dropout
        self.classfier = bert_classifier.classifier

    def forward(self, features: list[torch.tensor], input_ids: torch.tensor):
        # freeze the embeddings
        with torch.no_grad():
            bert_feature = self.embeddings(input_ids=input_ids)
        bert_feature, _ = torch.max(bert_feature, dim=1, keepdim=True)
        features.append(bert_feature)
        fused_features = self.fusion_module(features)
        encoder_input = self.fc(fused_features)
        encoder_output = self.encoder(encoder_input)
        pooler_output = self.pooler(encoder_output[0])
        pooler_output = self.dropout(pooler_output)
        return self.classfier(pooler_output)


if __name__ == "__main__":
    dataset = load_dataset("imdb")
    test_input = dataset["train"].select(range(2))
    train_dataset = dataset["train"].shuffle(seed=42).select(range(500))
    model = FeatureFusionBertClassfier(feature_num=3)
    semantic_tokenzier = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer: FeatureFusionBertTokenizer = FeatureFusionBertTokenizer(
        semantic_tokenzier, sentiment_max_length=20
    )

    # train_tokenizerd = train_dataset.map(tokenzier_funtion)
    tokenized_input = tokenizer.batch_encode_plus(
        test_input["text"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    tokenized_input["label"] = test_input["label"]
    print(tokenized_input["input_ids"].shape)
    for feature in tokenized_input["features"]:
        print(feature.shape)
    print(model(tokenized_input["features"], tokenized_input["input_ids"]))
    train_dataset_tokenized = tokenizer.batch_encode_plus(
        train_dataset["text"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    train_dataset_tokenized["label"] = train_dataset["label"]
    train_dataloader = DataLoader(train_dataset_tokenized, shuffle=True, batch_size=8)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # TODO Fix error: KeyError: 'Invalid key. Only three types of key are available: 
    # (1) string, (2) integers for backend Encoding, and (3) slices for data subsetting.'
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch["features"], batch["input_ids"])
            loss = F.cross_entropy(outputs, batch["label"])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")
