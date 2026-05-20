from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from .base import BaseModel

class TransformerClassifier(BaseModel):
    def __init__(self, config: dict):
        self.config = config
        name = config.get("pretrained_model_name", "cl-tohoku/bert-base-japanese-v3")
        self.max_length = int(config.get("max_length", 512))
        self.batch_size = int(config.get("batch_size", 16))
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    def _tok(self, batch):
        return self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=self.max_length)

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
        tr = Dataset.from_pandas(train_df[["text", "y"]].rename(columns={"y": "labels"}), preserve_index=False).map(self._tok, batched=True)
        va = None
        if valid_df is not None and not valid_df.empty:
            va = Dataset.from_pandas(valid_df[["text", "y"]].rename(columns={"y": "labels"}), preserve_index=False).map(self._tok, batched=True)
        args = TrainingArguments(
            output_dir="models/checkpoints/tmp_transformer",
            learning_rate=float(self.config.get("learning_rate", 2e-5)),
            num_train_epochs=float(self.config.get("epochs", 3)),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=float(self.config.get("weight_decay", 0.01)),
            eval_strategy="epoch" if va is not None else "no",
            save_strategy="no",
            logging_strategy="epoch",
            report_to=[],
        )
        Trainer(model=self.model, args=args, train_dataset=tr, eval_dataset=va).train()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        ds = Dataset.from_pandas(df[["text"]], preserve_index=False).map(self._tok, batched=True)
        preds = Trainer(model=self.model).predict(ds).predictions
        return torch.softmax(torch.tensor(preds), dim=-1).numpy()[:, 1]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        obj = cls({"pretrained_model_name": path})
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        return obj
