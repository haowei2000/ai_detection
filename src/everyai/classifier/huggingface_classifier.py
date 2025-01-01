from typing import List

from sklearn.calibration import LabelEncoder
import torch
from everyai.classfier.classify import TextClassifer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HuggingfaceClassifer(TextClassifer):
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        model_name: str = None,
        tokenizer_name: str = None,
        data_name: str = None,
        device: str = "cpu",
        model=None,
        tokenizer=None,
        tokenzier_config=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            texts,
            labels,
            model_name,
            tokenizer_name,
            data_name,
            device,
            model,
            tokenizer,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.tokenzier_config  = tokenzier_config
    
    def _label_encode(self, labels: List[str]):
        label_encoder = LabelEncoder()
        label_ids = label_encoder.fit_transform(labels)
        return torch.tensor(label_ids, dtype=torch.long)

    def _tokenize(self, texts,labels):
        def tokenzier_fn(examples):
            return self.tokenizer(texts, **self.tokenzier_config)
    
    def train(self):
        self.data.y = self._label_encode(self.labels)
        self.model.train()
        self.model.to(self.device)
        self.model.train()
        return self.model