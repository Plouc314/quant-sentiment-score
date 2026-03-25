from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..sources.news.models import Article
from .encoder import SentimentEncoder
from .summarizer import Summarizer

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768
_N_SENTIMENT_PROBS = 3


class SentimentPipeline:
    """Two-step NLP pipeline: summarization → FinBERT sentiment encoding.

    For each news article produces:
      - ternary sentiment label  (1.0=positive, 0.5=neutral, 0.0=negative)
      - 768-dim mean-pooled embedding (last hidden state, all non-padding tokens)
      - 3-dim softmax probability vector [P(pos), P(neg), P(neutral)]
    """

    def __init__(
        self,
        device: str = "cpu",
        encoder_model: str = "ProsusAI/finbert",
        summarizer_model: str | None = "facebook/bart-large-cnn",
    ) -> None:
        self.summarizer = Summarizer(device, summarizer_model)
        self.encoder = SentimentEncoder(device, encoder_model)

    def process_article(self, article: Article) -> dict:
        """Process a single article through the full pipeline.

        Returns::

            {
                "summary":         str,
                "label":           float,          # 1.0 / 0.5 / 0.0
                "embedding":       np.ndarray,     # shape (768,)
                "sentiment_probs": np.ndarray,     # shape (3,) — [P(pos), P(neg), P(neu)]
            }
        """
        title = (article.get("title") or "").strip()
        content = (article.get("text") or "").strip()

        if not title and not content:
            logger.warning("Article has no title or content — returning zero embedding")
            return {
                "summary": "",
                "label": 0.5,
                "embedding": np.zeros(_EMBEDDING_DIM, dtype=np.float32),
                "sentiment_probs": np.zeros(_N_SENTIMENT_PROBS, dtype=np.float32),
            }

        summary = self.summarizer.summarize(content) if content else ""
        text = f"{title} {summary}".strip()
        label, embedding, probs = self.encoder.encode(text)

        return {"summary": summary, "label": label, "embedding": embedding, "sentiment_probs": probs}

    def process_batch(self, articles: list[Article]) -> list[dict]:
        """Process multiple articles sequentially.

        Returns a list of result dicts (same length, same order).
        """
        results = []
        for i, article in enumerate(articles):
            try:
                result = self.process_article(article)
            except Exception:
                logger.exception("Failed to process article %d — using neutral fallback", i)
                result = {
                    "summary": "",
                    "label": 0.5,
                    "embedding": np.zeros(_EMBEDDING_DIM, dtype=np.float32),
                    "sentiment_probs": np.zeros(_N_SENTIMENT_PROBS, dtype=np.float32),
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
        DataFrame with columns: ticker, date, sentiment_score, n_articles,
        embedding, sentiment_probs
        """
        rows: list[dict] = []
        for ticker, articles in ticker_articles.items():
            results = self.process_batch(articles)
            for article, result in zip(articles, results):
                rows.append(
                    {
                        "ticker": ticker,
                        "date": (
                            article["publish_date"]
                            if isinstance(article["publish_date"], str)
                            else article["publish_date"].isoformat()
                            if article["publish_date"] is not None
                            else ""
                        ),
                        "label": result["label"],
                        "embedding": result["embedding"],
                        "sentiment_probs": result["sentiment_probs"],
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=["ticker", "date", "sentiment_score", "n_articles", "embedding", "sentiment_probs"]
            )
        return aggregate_daily(pd.DataFrame(rows))


# ----------------------------------------------------------------------
# Daily aggregation (operates on DataFrames, not on the model)
# ----------------------------------------------------------------------


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-article pipeline results into one row per (ticker, date).

    Input columns:  ticker, date, label, embedding, sentiment_probs
    Output columns: ticker, date, sentiment_score, n_articles, embedding, sentiment_probs

    sentiment_score = mean(label) — mean ternary sentiment score per day;
                      1.0=positive, 0.5=neutral, 0.0=negative; range [0.0, 1.0]
    n_articles      = number of articles that day
    embedding       = element-wise mean of all article embeddings that day
    sentiment_probs = element-wise mean of all per-article softmax probs that day
    """
    # Aggregate scalar columns with standard groupby — avoids apply() deprecation
    agg = (
        df.groupby(["ticker", "date"])
        .agg(sentiment_score=("label", "mean"), n_articles=("label", "count"))
        .reset_index()
    )

    # Aggregate array columns separately and merge
    for col in [c for c in ["embedding", "sentiment_probs"] if c in df.columns]:
        arr_agg = (
            df.groupby(["ticker", "date"])[col]
            .apply(lambda x: np.mean(np.stack(x.values), axis=0))
            .reset_index(name=col)
        )
        agg = agg.merge(arr_agg, on=["ticker", "date"])

    return agg
