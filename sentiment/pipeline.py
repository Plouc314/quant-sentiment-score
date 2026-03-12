from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

# FinBERT class index → human label
_FINBERT_LABELS = {0: "positive", 1: "negative", 2: "neutral"}

# BART-CNN generation defaults (from summarization.ipynb POC)
_BART_MAX_INPUT = 1024
_BART_MAX_OUTPUT = 128
_BART_MIN_OUTPUT = 30
_BART_NUM_BEAMS = 4
_BART_LENGTH_PENALTY = 2.0

# FinBERT tokenisation limit
_FINBERT_MAX_LENGTH = 512


class SentimentPipeline:
    """Two-step NLP pipeline: BART-CNN summarization → FinBERT sentiment encoding.

    For each news article produces:
      - binary sentiment label  (1 = positive, 0 = negative/neutral)
      - 768-dim mean-pooled embedding (last hidden state, all non-padding tokens)
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

        # --- Summarizer (BART-CNN) ---
        self._bart_tok = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self._bart_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn"
        )
        self._bart_model.eval().to(self.device)

        # --- Sentiment encoder (FinBERT) ---
        self._fin_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self._fin_model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert", output_hidden_states=True
        )
        self._fin_model.eval().to(self.device)

    # ------------------------------------------------------------------
    # Step 1 – summarization
    # ------------------------------------------------------------------

    def summarize(self, content: str) -> str:
        """Compress article content via BART-CNN.

        If the content is already short enough for FinBERT (≤ 512 tokens),
        summarization is skipped and the content is returned as-is.
        """
        if not content or not content.strip():
            return ""

        # Short-content bypass
        fin_tokens = self._fin_tok(content, truncation=False)
        if len(fin_tokens["input_ids"]) <= _FINBERT_MAX_LENGTH:
            return content

        inputs = self._bart_tok(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=_BART_MAX_INPUT,
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self._bart_model.generate(
                inputs["input_ids"],
                max_length=_BART_MAX_OUTPUT,
                min_length=_BART_MIN_OUTPUT,
                num_beams=_BART_NUM_BEAMS,
                length_penalty=_BART_LENGTH_PENALTY,
            )

        return self._bart_tok.decode(summary_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Step 2 – sentiment encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> tuple[int, np.ndarray]:
        """Run FinBERT on *text* and return (binary_label, 768-dim embedding).

        Binary mapping: 1 if FinBERT predicts positive (class 0), else 0.
        Embedding: mean pooling over all non-padding tokens of the last hidden layer.
        """
        inputs = self._fin_tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_FINBERT_MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._fin_model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        label = 1 if pred == 0 else 0

        last_hidden = outputs.hidden_states[-1]               # (1, seq_len, 768)
        mask = inputs["attention_mask"].unsqueeze(-1)          # (1, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)               # (1, 768)
        embedding = (summed / mask.sum(dim=1)).squeeze(0)      # (768,)
        embedding = embedding.cpu().numpy().astype(np.float32)

        return label, embedding

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def __call__(self, title: str, content: str) -> dict:
        """Process a single article through the full pipeline.

        Returns {"summary": str, "label": int, "embedding": np.ndarray}.
        """
        title = (title or "").strip()
        content = (content or "").strip()

        if not title and not content:
            logger.warning("Article has no title or content — returning zero embedding")
            return {
                "summary": "",
                "label": 0,
                "embedding": np.zeros(768, dtype=np.float32),
            }

        summary = self.summarize(content) if content else ""
        text = f"{title} {summary}".strip()
        label, embedding = self.encode(text)

        return {"summary": summary, "label": label, "embedding": embedding}

    def process_batch(self, articles: list[dict]) -> list[dict]:
        """Process multiple articles sequentially.

        Each element should be ``{"title": str, "content": str}``.
        Returns a list of result dicts (same length, same order).
        """
        results = []
        for i, article in enumerate(articles):
            try:
                result = self(article.get("title", ""), article.get("content", ""))
            except Exception:
                logger.exception("Failed to process article %d — skipping", i)
                result = {
                    "summary": "",
                    "label": 0,
                    "embedding": np.zeros(768, dtype=np.float32),
                }
            results.append(result)
        return results


# ----------------------------------------------------------------------
# Daily aggregation (operates on DataFrames, not on the model)
# ----------------------------------------------------------------------


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-article pipeline results into one row per (ticker, date).

    Input columns:  ticker, date, label, embedding
    Output columns: ticker, date, sentiment_score, n_articles, embedding

    sentiment_score = mean(label) = proportion of positive articles that day
    n_articles      = number of articles that day
    embedding       = element-wise mean of all article embeddings that day
    """

    def _agg_group(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "sentiment_score": group["label"].mean(),
                "n_articles": len(group),
                "embedding": np.mean(
                    np.stack(group["embedding"].values), axis=0
                ),
            }
        )

    return df.groupby(["ticker", "date"]).apply(_agg_group).reset_index()
