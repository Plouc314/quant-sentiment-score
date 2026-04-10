from datetime import date
from typing import TypedDict

import numpy as np


class Article(TypedDict):
    id: str
    url: str
    title: str
    text: str
    publish_date: date
    source_name: str
    language: str
    tickers: list[str]


class ArticleEncoding(TypedDict):
    """Output of :class:`~src.embeddings.encoder.SentimentEncoder` for a single piece of text."""

    label: float                # 1.0=positive, 0.5=neutral, 0.0=negative
    embedding: np.ndarray       # shape (768,)
    sentiment_probs: np.ndarray # shape (3,) — [P(pos), P(neg), P(neu)]
