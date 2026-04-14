from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..models import ArticleEncoding

logger = logging.getLogger(__name__)

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

    def encode_batch(
        self, texts: list[str], batch_size: int = 16
    ) -> list[ArticleEncoding]:
        """Run FinBERT on a list of texts in chunks of *batch_size*.

        Equivalent to calling :meth:`encode` on each text individually but
        amortises the forward-pass overhead across the batch — significantly
        faster on MPS/CUDA.
        """
        results: list[ArticleEncoding] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self._tok(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=FINBERT_MAX_LENGTH,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            preds = torch.argmax(outputs.logits, dim=1)  # (B,)
            probs = torch.softmax(outputs.logits, dim=1)  # (B, 3)

            last_hidden = outputs.hidden_states[-1]  # (B, seq_len, 768)
            mask = inputs["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)  # (B, 768)
            embeddings = summed / mask.sum(dim=1)  # (B, 768)

            for j in range(len(chunk)):
                results.append(
                    ArticleEncoding(
                        label=_LABEL_TO_SCORE[int(preds[j].item())],
                        embedding=embeddings[j].cpu().numpy().astype(np.float32),
                        sentiment_probs=probs[j].cpu().numpy().astype(np.float32),
                    )
                )

            if (i / batch_size + 1) % 10 == 0:
                logger.info(
                    "Encoded batch %d / %d", min(i + batch_size, len(texts)), len(texts)
                )
        return results

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

        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, 768)
        mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (1, 768)
        embedding = (summed / mask.sum(dim=1)).squeeze(0)  # (768,)
        embedding = embedding.cpu().numpy().astype(np.float32)

        return ArticleEncoding(label=label, embedding=embedding, sentiment_probs=probs)
