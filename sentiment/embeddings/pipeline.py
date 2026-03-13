from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..sources.news.models import Article
from .encoder import SentimentEncoder
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """Two-step NLP pipeline: BART-CNN summarization → FinBERT sentiment encoding.

    For each news article produces:
      - binary sentiment label  (1 = positive, 0 = negative/neutral)
      - 768-dim mean-pooled embedding (last hidden state, all non-padding tokens)
    """

    def __init__(self, device: str = "cpu") -> None:
        self.summarizer = Summarizer(device)
        self.encoder = SentimentEncoder(device)

    def process_article(self, article: Article) -> dict:
        """Process a single article through the full pipeline.

        Returns {"summary": str, "label": int, "embedding": np.ndarray}.
        """
        title = (article.get("title") or "").strip()
        content = (article.get("text") or "").strip()

        if not title and not content:
            logger.warning("Article has no title or content — returning zero embedding")
            return {
                "summary": "",
                "label": 0,
                "embedding": np.zeros(768, dtype=np.float32),
            }

        summary = self.summarizer.summarize(content) if content else ""
        text = f"{title} {summary}".strip()
        label, embedding = self.encoder.encode(text)

        return {"summary": summary, "label": label, "embedding": embedding}

    def process_batch(self, articles: list[Article]) -> list[dict]:
        """Process multiple articles sequentially.

        Returns a list of result dicts (same length, same order).
        """
        results = []
        for i, article in enumerate(articles):
            try:
                result = self.process_article(article)
            except Exception:
                logger.exception("Failed to process article %d — skipping", i)
                result = {
                    "summary": "",
                    "label": 0,
                    "embedding": np.zeros(768, dtype=np.float32),
                }
            results.append(result)
        return results

    def process_ticker_articles(
        self,
        ticker_articles: dict[str, list[Article]],
    ) -> pd.DataFrame:
        """Process articles grouped by ticker and return a daily-aggregated DataFrame.

        Parameters
        ----------
        ticker_articles:
            Mapping of ``{ticker: [Article, ...]}``.

        Returns
        -------
        DataFrame with columns: ticker, date, sentiment_score, n_articles, embedding
        """
        rows: list[dict] = []
        for ticker, articles in ticker_articles.items():
            results = self.process_batch(articles)
            for article, result in zip(articles, results):
                rows.append(
                    {
                        "ticker": ticker,
                        "date": article["publish_date"].isoformat()
                        if not isinstance(article["publish_date"], str)
                        else article["publish_date"],
                        "label": result["label"],
                        "embedding": result["embedding"],
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=["ticker", "date", "sentiment_score", "n_articles", "embedding"]
            )
        return aggregate_daily(pd.DataFrame(rows))


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
