from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# FinBERT class index → human label
_FINBERT_LABELS = {0: "positive", 1: "negative", 2: "neutral"}

# FinBERT tokenisation limit
_FINBERT_MAX_LENGTH = 512


class SentimentEncoder:
    """FinBERT sentiment encoder — produces a binary label and 768-dim embedding."""

    def __init__(
        self,
        device: str = "cpu",
        model_name_or_path: str = "ProsusAI/finbert",
    ) -> None:
        self.device = torch.device(device)
        self._tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, output_hidden_states=True
        )
        self._model.eval().to(self.device)

    def encode(self, text: str) -> tuple[int, np.ndarray]:
        """Run FinBERT on *text* and return (binary_label, 768-dim embedding).

        Binary mapping: 1 if FinBERT predicts positive (class 0), else 0.
        Embedding: mean pooling over all non-padding tokens of the last hidden layer.
        """
        inputs = self._tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_FINBERT_MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        label = 1 if pred == 0 else 0

        last_hidden = outputs.hidden_states[-1]               # (1, seq_len, 768)
        mask = inputs["attention_mask"].unsqueeze(-1)          # (1, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)               # (1, 768)
        embedding = (summed / mask.sum(dim=1)).squeeze(0)      # (768,)
        embedding = embedding.cpu().numpy().astype(np.float32)

        return label, embedding
