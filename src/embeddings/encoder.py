from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..models import ArticleEncoding

# FinBERT label index → ternary sentiment score
_LABEL_TO_SCORE: dict[int, float] = {0: 1.0, 1: 0.0, 2: 0.5}

# FinBERT tokenisation limit
FINBERT_MAX_LENGTH = 512


class SentimentEncoder:
    """FinBERT sentiment encoder — produces a ternary label, 768-dim embedding, and class probs."""

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

    def encode(self, text: str) -> ArticleEncoding:
        """Run FinBERT on *text* and return an :class:`~src.models.ArticleEncoding`.

        label:
            Ternary sentiment score — ``1.0`` (positive), ``0.5`` (neutral),
            ``0.0`` (negative).
        embedding:
            768-dim float32 array.  Mean pooling over non-padding tokens of the
            last hidden layer.
        sentiment_probs:
            float32 array of shape ``(3,)`` — softmax probabilities in FinBERT's
            class order: ``[P(positive), P(negative), P(neutral)]``.
        """
        inputs = self._tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        pred = int(torch.argmax(outputs.logits, dim=1).item())
        label = _LABEL_TO_SCORE[pred]

        probs = (
            torch.softmax(outputs.logits, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        last_hidden = outputs.hidden_states[-1]               # (1, seq_len, 768)
        mask = inputs["attention_mask"].unsqueeze(-1)          # (1, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)               # (1, 768)
        embedding = (summed / mask.sum(dim=1)).squeeze(0)      # (768,)
        embedding = embedding.cpu().numpy().astype(np.float32)

        return ArticleEncoding(label=label, embedding=embedding, sentiment_probs=probs)
