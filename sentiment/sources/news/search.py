import logging
import os
from datetime import date

import mediacloud.api

from .models import Story

logger = logging.getLogger(__name__)


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
        collection_ids: list[int] = [],
    ) -> list[Story]:
        """Search for stories matching *query* within a date range.

        Transparently follows pagination and returns all results as a flat list.

        Args:
            query: MediaCloud query string.
            start_date: Inclusive start of the publication window.
            end_date: Inclusive end of the publication window.
            collection_ids: Optional list of MediaCloud collection IDs to restrict the search.

        Returns:
            All matching :class:`Story` objects across all pages.
        """
        stories: list[Story] = []
        pagination_token: str | None = None

        logger.info(f"NewsSearch: query: {query} start: {start_date} end: {end_date}")

        while True:
            logger.info("  querying...")
            try:
                page, pagination_token = self._api.story_list(
                    query,
                    start_date,
                    end_date,
                    collection_ids=collection_ids,
                    pagination_token=pagination_token,
                )
            except Exception as e:
                logger.warning(f"query failed: {e}")
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
                    )
                )
            if not pagination_token:
                break

        return stories
