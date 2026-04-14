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
        self.encoder = SentimentEncoder(device, encoder_model)
        self.summarizer = Summarizer(
            device, summarizer_model, finbert_tokenizer=self.encoder._tok
        )

    def encode_article(self, article: Article) -> ArticleEncoding:
        """Encode a single article through summarization → FinBERT.

        Returns a neutral encoding (label=0.5, zero vectors) when the article
        has neither title nor text.
        """
        raw_title = article.get("title")
        title = (raw_title if isinstance(raw_title, str) else "").strip()
        raw_body = article.get("body")
        content = (raw_body if isinstance(raw_body, str) else "").strip()

        if not title and not content:
            logger.warning(
                "Article has no title or content — returning neutral encoding"
            )
            return _NEUTRAL_ENCODING

        summary = self.summarizer.summarize(content) if content else ""
        text = f"{title} {summary}".strip()
        return self.encoder.encode(text)

    def encode_articles(
        self, articles: list[Article], batch_size: int = 16
    ) -> list[ArticleEncoding]:
        """Encode a list of articles using batched FinBERT inference.

        Summarisation is still performed sequentially (it is a no-op for most
        articles). The resulting texts are then encoded in chunks of
        *batch_size* to amortise forward-pass overhead on MPS/CUDA.

        Summarisation failures are caught per-article and replaced with a
        neutral encoding so that one bad article does not abort the run.
        """
        total = len(articles)
        logger.info("Encoding %d articles (batch_size=%d)", total, batch_size)

        # Step 1 — summarise each article; None marks a failure
        texts: list[str | None] = []
        for i, article in enumerate(articles):
            try:
                raw_title = article.get("title")
                title = (raw_title if isinstance(raw_title, str) else "").strip()
                raw_body = article.get("body")
                content = (raw_body if isinstance(raw_body, str) else "").strip()

                if not title and not content:
                    logger.warning(
                        "Article %d has no title or content — using neutral fallback", i
                    )
                    texts.append(None)
                    continue

                summary = self.summarizer.summarize(content) if content else ""
                texts.append(f"{title} {summary}".strip())
            except Exception:
                logger.exception(
                    "Failed to summarise article %d — using neutral fallback", i
                )
                texts.append(None)

            if (i + 1) % 50 == 0 and not self.summarizer.noop:
                logger.info("Summarised %d / %d articles", i + 1, total)

        # Step 2 — batch-encode valid texts
        valid_indices = [i for i, t in enumerate(texts) if t is not None]
        valid_texts = [texts[i] for i in valid_indices]  # type: ignore[index]

        batch_encodings = self.encoder.encode_batch(valid_texts, batch_size=batch_size)

        # Step 3 — reassemble in original order
        results: list[ArticleEncoding] = [_NEUTRAL_ENCODING] * total
        for idx, enc in zip(valid_indices, batch_encodings):
            results[idx] = enc

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

    sentiment_score = dot(mean(sentiment_probs), [1.0, 0.0, 0.5]); range [0.0, 1.0].
    Derived from aggregated softmax probs so mixed days (pos+neg) are
    distinguishable from neutral days via the full sentiment_probs vector.
    embedding / sentiment_probs = element-wise mean of per-article arrays.
    """
    if not articles:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "sentiment_score",
                "n_articles",
                "embedding",
                "sentiment_probs",
            ]
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
        df.groupby(["ticker", "date"]).agg(n_articles=("label", "count")).reset_index()
    )

    for col in ["embedding", "sentiment_probs"]:
        arr = np.stack(df[col].values)  # (M, D)
        arr_df = pd.DataFrame(arr, index=df.index)
        means = arr_df.groupby([df["ticker"], df["date"]]).mean().reset_index()
        means[col] = list(means.iloc[:, 2:].values.astype(np.float32))
        means = means[["ticker", "date", col]]
        agg = agg.merge(means, on=["ticker", "date"])

    # FinBERT class order: [P(positive), P(negative), P(neutral)]
    # Soft score = 1.0*P(pos) + 0.0*P(neg) + 0.5*P(neutral)
    _SCORE_WEIGHTS = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    agg["sentiment_score"] = agg["sentiment_probs"].apply(
        lambda p: float(np.dot(p, _SCORE_WEIGHTS))
    )

    return agg[
        [
            "ticker",
            "date",
            "sentiment_score",
            "n_articles",
            "embedding",
            "sentiment_probs",
        ]
    ]
