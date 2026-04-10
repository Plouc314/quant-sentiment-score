from datetime import date
from typing import TypedDict

import numpy as np


class Article(TypedDict):
    benzinga_id: int
    url: str
    title: str
    publish_date: date
    tickers: list[str]
    author: str | None
    body: str | None
    teaser: str | None
    last_updated: date | None
    channels: list[str]
    tags: list[str]
    images: list[str]


class ArticleEncoding(TypedDict):
    """Output of :class:`~src.embeddings.encoder.SentimentEncoder` for a single piece of text."""

    label: float                # 1.0=positive, 0.5=neutral, 0.0=negative
    embedding: np.ndarray       # shape (768,)
    sentiment_probs: np.ndarray # shape (3,) — [P(pos), P(neg), P(neu)]
