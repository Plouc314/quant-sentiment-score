from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..models import Article, ArticleEncoding
from .encoder import SentimentEncoder
from .summarizer import Summarizer

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768
_N_SENTIMENT_PROBS = 3

_NEUTRAL_ENCODING = ArticleEncoding(
    label=0.5,
    embedding=np.zeros(_EMBEDDING_DIM, dtype=np.float32),
    sentiment_probs=np.zeros(_N_SENTIMENT_PROBS, dtype=np.float32),
)


class SentimentPipeline:
    """Two-step NLP pipeline: summarization → FinBERT sentiment encoding.

    For each article produces an :class:`~src.models.ArticleEncoding` with:
      - ternary sentiment label  (1.0=positive, 0.5=neutral, 0.0=negative)
      - 768-dim mean-pooled embedding (last hidden state, non-padding tokens)
      - 3-dim softmax probability vector [P(pos), P(neg), P(neutral)]

    Use :func:`aggregate_daily` to collapse per-article encodings into daily
    rows suitable for :class:`~src.repositories.sentiment.SentimentRepository`.
    """

    def __init__(
        self,
        device: str = "cpu",
        encoder_model: str = "ProsusAI/finbert",
        summarizer_model: str | None = "facebook/bart-large-cnn",
    ) -> None:
        self.summarizer = Summarizer(device, summarizer_model)
        self.encoder = SentimentEncoder(device, encoder_model)

    def encode_article(self, article: Article) -> ArticleEncoding:
        """Encode a single article through summarization → FinBERT.

        Returns a neutral encoding (label=0.5, zero vectors) when the article
        has neither title nor text.
        """
        title = (article.get("title") or "").strip()
        content = (article.get("text") or "").strip()

        if not title and not content:
            logger.warning("Article has no title or content — returning neutral encoding")
            return _NEUTRAL_ENCODING

        summary = self.summarizer.summarize(content) if content else ""
        text = f"{title} {summary}".strip()
        return self.encoder.encode(text)

    def encode_articles(self, articles: list[Article]) -> list[ArticleEncoding]:
        """Encode a list of articles sequentially.

        Failures are caught per-article and replaced with a neutral encoding so
        that one bad article does not abort the batch.
        """
        results: list[ArticleEncoding] = []
        for i, article in enumerate(articles):
            try:
                results.append(self.encode_article(article))
            except Exception:
                logger.exception("Failed to encode article %d — using neutral fallback", i)
                results.append(_NEUTRAL_ENCODING)
        return results


# ---------------------------------------------------------------------------
# Daily aggregation (pure data transformation — no model dependency)
# ---------------------------------------------------------------------------


def aggregate_daily(
    articles: list[Article],
    encodings: list[ArticleEncoding],
    ticker: str,
) -> pd.DataFrame:
    """Collapse per-article encodings into one row per date for a single ticker.

    Parameters
    ----------
    articles:
        The source articles (same order as *encodings*).
    encodings:
        Output of :meth:`SentimentPipeline.encode_articles`.
    ticker:
        Ticker symbol to attach to every row.

    Returns
    -------
    DataFrame with columns: ticker, date, sentiment_score, n_articles,
    embedding, sentiment_probs.

    sentiment_score = mean(label) across articles that day; range [0.0, 1.0].
    embedding / sentiment_probs = element-wise mean of per-article arrays.
    """
    if not articles:
        return pd.DataFrame(
            columns=["ticker", "date", "sentiment_score", "n_articles", "embedding", "sentiment_probs"]
        )

    rows = [
        {
            "ticker": ticker,
            "date": (
                article["publish_date"].isoformat()
                if article["publish_date"] is not None
                else ""
            ),
            "label": enc["label"],
            "embedding": enc["embedding"],
            "sentiment_probs": enc["sentiment_probs"],
        }
        for article, enc in zip(articles, encodings)
    ]
    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["ticker", "date"])
        .agg(sentiment_score=("label", "mean"), n_articles=("label", "count"))
        .reset_index()
    )

    for col in ["embedding", "sentiment_probs"]:
        arr = np.stack(df[col].values)                    # (M, D)
        arr_df = pd.DataFrame(arr, index=df.index)
        means = (
            arr_df.groupby([df["ticker"], df["date"]])
            .mean()
            .reset_index()
        )
        means[col] = list(means.iloc[:, 2:].values.astype(np.float32))
        means = means[["ticker", "date", col]]
        agg = agg.merge(means, on=["ticker", "date"])

    return agg
