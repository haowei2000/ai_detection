import math

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset,Dataset
from transformers import (BertForSequenceClassification, BertTokenizer,
                          PreTrainedTokenizer)
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

    def tokenzier_funtion(examples:Dataset):
        featuress_input_ids =tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        examples["features"] = featuress_input_ids["features"]
        examples["input_ids"] = featuress_input_ids["input_ids"]
        return examples

    # train_tokenizerd = train_dataset.map(tokenzier_funtion)
    test_input = test_input.map(tokenzier_funtion)
    print(test_input["features"])
    print(test_input["input_ids"])
