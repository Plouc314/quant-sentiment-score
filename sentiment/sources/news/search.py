import logging
import os
import time
from datetime import date

import mediacloud.api

from .models import Story

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 15, 30]


class NewsSearch:
    """MediaCloud-backed news story search.

    Reads ``MEDIACLOUD_API_KEY`` from the environment when no key is supplied.
    """

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ["MEDIACLOUD_API_KEY"]
        self._api = mediacloud.api.SearchApi(api_key)

    def search(
        self,
        query: str,
        start_date: date,
        end_date: date,
        ticker: str,
        collection_ids: list[int] = [],
        source_ids: list[int] = [],
    ) -> list[Story]:
        """Search for stories matching *query* within a date range.

        Transparently follows pagination and returns all results as a flat list.
        Failed requests are retried with backoff (delays: 5s, 15s, 30s) before
        giving up. Pass either *collection_ids* or *source_ids* — not both, as
        they are OR'd by the API.

        Args:
            query: MediaCloud query string.
            start_date: Inclusive start of the publication window.
            end_date: Inclusive end of the publication window.
            ticker: Ticker symbol associated with this search; set on every returned story.
            collection_ids: Optional list of MediaCloud collection IDs to restrict the search.
            source_ids: Optional list of MediaCloud source IDs to restrict the search.

        Returns:
            All matching :class:`Story` objects across all pages.
        """
        stories: list[Story] = []
        pagination_token: str | None = None

        logger.info("NewsSearch: query: %s start: %s end: %s", query, start_date, end_date)

        while True:
            page, pagination_token = self._query_with_retry(
                query, start_date, end_date, collection_ids, source_ids, pagination_token
            )
            if page is None:
                break

            for raw in page:
                stories.append(
                    Story(
                        id=raw["id"],
                        url=raw["url"],
                        title=raw["title"],
                        publish_date=raw["publish_date"],
                        source_name=raw["media_name"],
                        language=raw["language"],
                        tickers=[ticker],
                    )
                )
            if not pagination_token:
                break

        return stories

    def _query_with_retry(
        self,
        query: str,
        start_date: date,
        end_date: date,
        collection_ids: list[int],
        source_ids: list[int],
        pagination_token: str | None,
    ) -> tuple[list | None, str | None]:
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                logger.warning("retrying in %ds (attempt %d)...", delay, attempt)
                time.sleep(delay)
            try:
                return self._api.story_list(
                    query,
                    start_date,
                    end_date,
                    collection_ids=collection_ids,
                    source_ids=source_ids,
                    pagination_token=pagination_token,
                )
            except Exception as e:
                logger.warning("query failed: %s", e)

        return None, None
