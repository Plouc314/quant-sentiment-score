import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .blacklist import SourceBlacklist
from .extractor import ArticleExtractor
from .repository import ArticleRepository
from .search import NewsSearch

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE_LIST = Path(__file__).parents[3] / "data" / "source_list.csv"


class NewsPipeline:
    """Orchestrates news search, extraction, and storage for a universe of tickers.

    Accepts pre-constructed infrastructure objects so each component can be
    configured independently.

    When *source_list_path* is provided, the pipeline uses per-source filtering
    instead of collection-level filtering: before each slice it reads the current
    blacklist and passes only non-blacklisted source IDs to the search API.  This
    keeps the 2000-story budget focused on sources that are actually extractable.
    Pass either *collection_ids* or *source_list_path* — not both, as the API
    OR's collection and source filters.

    Args:
        search: :class:`NewsSearch` instance for MediaCloud queries.
        extractor: :class:`ArticleExtractor` instance for fetching article text.
        repository: :class:`ArticleRepository` instance for persistent storage.
        blacklist: :class:`SourceBlacklist` used to filter sources at search time.
        collection_ids: MediaCloud collection IDs passed to every search.
            Ignored when *source_list_path* is set.
        source_list_path: Path to a CSV with ``id`` and ``domain`` columns
            (MediaCloud source ID and domain name).  When provided, source-level
            filtering replaces collection filtering.
    """

    def __init__(
        self,
        search: NewsSearch,
        extractor: ArticleExtractor,
        repository: ArticleRepository,
        blacklist: SourceBlacklist,
        collection_ids: list[int] = [],
        source_list_path: Path | None = _DEFAULT_SOURCE_LIST,
        search_delay: float = 2.0,
    ):
        self._search = search
        self._extractor = extractor
        self._repository = repository
        self._blacklist = blacklist
        self._collection_ids = collection_ids
        self._source_df = self._load_source_list(source_list_path)
        self._search_delay = search_delay

    def run(
        self,
        universe: dict[str, str],
        start_date: date,
        end_date: date,
        time_span: timedelta,
    ) -> None:
        """Run the pipeline over *universe* for the given date range.

        The range [*start_date*, *end_date*] is split into slices of *time_span*.
        Each slice is fully processed (search + extract + store for all tickers)
        and flushed to disk before moving to the next.

        Args:
            universe: Mapping of ticker symbol to company name used as search query.
            start_date: Inclusive start of the date range.
            end_date: Inclusive end of the date range.
            time_span: Length of each time slice (e.g. ``timedelta(weeks=1)``).
        """
        slices = self._build_slices(start_date, end_date, time_span)[::-1]
        logger.info(
            "starting pipeline: %d tickers, %d slices (%s → %s)",
            len(universe),
            len(slices),
            start_date,
            end_date,
        )

        for slice_start, slice_end in slices:
            logger.info("processing slice %s → %s", slice_start, slice_end)

            source_ids = self._active_source_ids()
            if source_ids is not None:
                logger.info("using %d non-blacklisted sources", len(source_ids))

            for i, (ticker, name) in enumerate(universe.items()):
                if i > 0:
                    time.sleep(self._search_delay)
                kwargs = (
                    {"source_ids": source_ids}
                    if source_ids is not None
                    else {"collection_ids": self._collection_ids}
                )
                stories = self._search.search(
                    query=f'article_title:"{name}" OR article_title:"{ticker}"',
                    start_date=slice_start,
                    end_date=slice_end,
                    ticker=ticker,
                    **kwargs,
                )
                logger.info("[%s] found %d stories", ticker, len(stories))

                new_stories, existing_stories = self._repository.partition_stories(
                    stories
                )

                for story in existing_stories:
                    self._repository.add_tickers(story["url"], story["tickers"])

                t0 = time.time()
                articles = self._extractor.extract_many(new_stories)
                logger.info(
                    "[%s] extracted %d articles in %ds",
                    ticker,
                    len(articles),
                    int(time.time() - t0),
                )

                for article in articles:
                    self._repository.store(article)

            self._repository.flush()
            logger.info("flushed slice %s → %s", slice_start, slice_end)

    def _active_source_ids(self) -> list[int] | None:
        """Return non-blacklisted source IDs from the source list, or None if
        no source list was loaded."""
        if self._source_df is None:
            return None
        mask = self._source_df["domain"].apply(
            lambda d: not self._blacklist.is_blacklisted(d)
        )
        return self._source_df.loc[mask, "id"].tolist()

    @staticmethod
    def _load_source_list(path: Path | None) -> pd.DataFrame | None:
        if path is None or not path.exists():
            return None
        df = pd.read_csv(
            path, usecols=["id", "domain"], dtype={"id": int, "domain": str}
        )
        return df.dropna(subset=["id", "domain"])

    @staticmethod
    def _build_slices(
        start_date: date, end_date: date, time_span: timedelta
    ) -> list[tuple[date, date]]:
        slices = []
        current = start_date
        while current <= end_date:
            slice_end = min(current + time_span - timedelta(days=1), end_date)
            slices.append((current, slice_end))
            current += time_span
        return slices
