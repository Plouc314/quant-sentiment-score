from __future__ import annotations

import logging
import random
from collections import defaultdict
from html.parser import HTMLParser

from ..models import Article

logger = logging.getLogger(__name__)


class _HTMLStripper(HTMLParser):
    """Minimal HTMLParser subclass that collects visible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    """Return *text* with all HTML tags removed and entities decoded."""
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


class ArticleFilter:
    """Per-day article sampler and body-length filter.

    Applied to a flat list of articles (typically one ticker's full history)
    before passing to :class:`~src.embeddings.pipeline.SentimentPipeline`.

    Processing order per day:

    0. Strip HTML tags and decode entities from the article body.
    1. Drop articles whose body is shorter than *min_body_chars* characters.
    2. If the remaining count exceeds *sample_above*, draw a random
       ``round(n * sample_ratio)`` subset — keeping at least 1.
    3. Apply *max_per_day* as a hard ceiling (also keeps at least 1).
    """

    def __init__(
        self,
        min_body_chars: int = 0,
        sample_above: int | None = None,
        sample_ratio: float = 1.0,
        max_per_day: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
        if max_per_day is not None and max_per_day < 1:
            raise ValueError(f"max_per_day must be >= 1, got {max_per_day}")

        self.min_body_chars = min_body_chars
        self.sample_above = sample_above
        self.sample_ratio = sample_ratio
        self.max_per_day = max_per_day
        self._rng = random.Random(seed)

    def filter(self, articles: list[Article]) -> list[Article]:
        """Return a filtered and sampled subset of *articles*.

        Articles are grouped by ``publish_date`` internally; the returned list
        preserves chronological order within each day but days are not
        re-ordered relative to the input.
        """
        # Group by date, preserving insertion order
        by_date: dict[object, list[Article]] = defaultdict(list)
        for article in articles:
            by_date[article["publish_date"]].append(article)

        total_in = len(articles)
        total_body_dropped = 0
        total_sampled_dropped = 0
        result: list[Article] = []

        for date, day_articles in by_date.items():
            # Step 0 — strip HTML from body
            day_articles = [
                {**a, "body": _strip_html(a["body"]) if isinstance(a["body"], str) else a["body"]}
                for a in day_articles
            ]

            # Step 1 — minimum body length
            before_body = len(day_articles)
            if self.min_body_chars > 0:
                day_articles = [
                    a for a in day_articles
                    if isinstance(a["body"], str) and len(a["body"]) >= self.min_body_chars
                ]
            total_body_dropped += before_body - len(day_articles)

            if not day_articles:
                continue

            # Step 2 — ratio sampling (only above threshold)
            n = len(day_articles)
            k = n
            if self.sample_above is not None and n > self.sample_above:
                k = max(1, round(n * self.sample_ratio))

            # Step 3 — hard cap
            if self.max_per_day is not None:
                k = max(1, min(k, self.max_per_day))

            if k < n:
                total_sampled_dropped += n - k
                day_articles = self._rng.sample(day_articles, k)

            result.extend(day_articles)

        logger.info(
            "ArticleFilter: %d → %d articles "
            "(body filter dropped %d, sampling dropped %d)",
            total_in,
            len(result),
            total_body_dropped,
            total_sampled_dropped,
        )
        return result
