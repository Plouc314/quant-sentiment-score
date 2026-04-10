from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd
from massive.rest import RESTClient
from massive.rest.models.benzinga import BenzingaNews

from ..models import Article

logger = logging.getLogger(__name__)


class MassiveProvider:
    """Fetches Benzinga news articles via the Massive API.

    Parameters
    ----------
    api_key:
        Massive API key. Falls back to the ``MASSIVE_API_KEY`` environment
        variable when *None*.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = RESTClient(api_key=api_key or os.environ["MASSIVE_API_KEY"])

    def fetch_all_news(
        self,
        start: date,
        end: date,
        limit: int = 50000,
    ) -> list[Article]:
        """Return all articles from ``/benzinga/v2/news`` for the given date range.

        No ticker filter is applied — every article published in the window is
        returned.  Use this for bulk ingestion; use :meth:`fetch_news` when you
        only need articles for a specific set of symbols.

        Duplicates by ``benzinga_id`` are resolved by keeping the article with
        the highest ``last_updated`` date.

        Parameters
        ----------
        start:
            Earliest publish date (inclusive).
        end:
            Latest publish date (inclusive).
        limit:
            Page size passed to the API. Pagination is followed automatically
            until all results are consumed.
        """
        news = self._client.list_benzinga_news_v2(
            published_gte=start.isoformat(),
            published_lte=end.isoformat(),
            limit=limit,
        )
        by_id: dict[int, Article] = {}
        for n in news:
            if n.url is None or n.published is None or n.benzinga_id is None:
                logger.warning(
                    "skipping incomplete article: benzinga_id=%s url=%s published=%s",
                    n.benzinga_id, n.url, n.published,
                )
                continue
            article = self._to_article(n)
            existing = by_id.get(article["benzinga_id"])
            if existing is None or self._is_newer(article, existing):
                by_id[article["benzinga_id"]] = article
        return list(by_id.values())

    def fetch_news(
        self,
        tickers: list[str],
        start: date,
        end: date,
        limit: int = 1000,
    ) -> list[Article]:
        """Return articles from ``/benzinga/v2/news`` for the given tickers and date range.

        Duplicates by ``benzinga_id`` are resolved by keeping the article with
        the highest ``last_updated`` date.

        Parameters
        ----------
        tickers:
            List of ticker symbols to filter on (OR semantics — any match).
        start:
            Earliest publish date (inclusive).
        end:
            Latest publish date (inclusive).
        limit:
            Page size passed to the API. Pagination is followed automatically
            until all results are consumed.
        """
        news = self._client.list_benzinga_news_v2(
            tickers_any_of=",".join(tickers),
            published_gte=start.isoformat(),
            published_lte=end.isoformat(),
            limit=limit,
        )
        by_id: dict[int, Article] = {}
        for n in news:
            if n.url is None or n.published is None or n.benzinga_id is None:
                logger.warning(
                    "skipping incomplete article: benzinga_id=%s url=%s published=%s",
                    n.benzinga_id, n.url, n.published,
                )
                continue
            article = self._to_article(n)
            existing = by_id.get(article["benzinga_id"])
            if existing is None or self._is_newer(article, existing):
                by_id[article["benzinga_id"]] = article
        return list(by_id.values())

    def _to_article(self, n: BenzingaNews) -> Article:
        return Article(
            benzinga_id=n.benzinga_id,
            url=n.url,
            title=n.title or "",
            publish_date=self._parse_date(n.published),
            tickers=list(n.tickers or []),
            author=n.author,
            body=n.body,
            teaser=n.teaser,
            last_updated=self._parse_date(n.last_updated) if n.last_updated else None,
            channels=list(n.channels or []),
            tags=list(n.tags or []),
            images=list(n.images or []),
        )

    def _is_newer(self, candidate: Article, current: Article) -> bool:
        """Return True if *candidate* has a more recent ``last_updated`` than *current*."""
        if candidate["last_updated"] is None:
            return False
        if current["last_updated"] is None:
            return True
        return candidate["last_updated"] > current["last_updated"]

    def _parse_date(self, raw: str) -> date:
        return pd.to_datetime(raw, utc=True).date()
