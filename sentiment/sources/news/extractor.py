import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import trafilatura
from trafilatura.settings import use_config

from .blacklist import SourceBlacklist
from .models import Article, Story

logger = logging.getLogger(__name__)


class ArticleExtractor:
    """Fetches and extracts article text from URLs using trafilatura.

    Stories are grouped by source and processed sequentially per source, with
    a configurable delay between requests. Sources that cross the failure
    threshold in :class:`SourceBlacklist` are abandoned mid-run without
    processing remaining stories from that source. Different sources are
    processed in parallel up to *workers* threads.

    Args:
        blacklist: :class:`SourceBlacklist` used to skip known-bad sources and
            record per-source fetch outcomes.
        workers: Number of sources processed concurrently by :meth:`extract_many`.
        timeout: HTTP timeout in seconds passed to trafilatura.
        source_delay: Seconds to wait between consecutive requests to the same source.
    """

    def __init__(
        self,
        blacklist: SourceBlacklist,
        workers: int = 10,
        timeout: int = 10,
        source_delay: float = 1.0,
    ):
        self._blacklist = blacklist
        self._workers = workers
        self._source_delay = source_delay
        self._lock = Lock()
        self._config = use_config()
        self._config.set("DEFAULT", "download_timeout", str(timeout))

    def extract(self, story: Story) -> Article | None:
        """Download and parse the full article text for a single story.

        Returns ``None`` (and logs a warning) if the URL cannot be fetched or
        trafilatura fails to extract meaningful text.
        """
        html = trafilatura.fetch_url(story["url"], config=self._config)
        if html is None:
            logger.warning("failed to fetch: %s", story["url"])
            return None

        doc = trafilatura.bare_extraction(html, url=story["url"], include_comments=False)
        if doc is None or not doc.text:
            logger.warning("failed to extract text: %s", story["url"])
            return None

        return Article(
            id=story["id"],
            url=story["url"],
            title=story["title"],
            text=doc.text,
            publish_date=story["publish_date"],
            source_name=story["source_name"],
            language=story["language"],
        )

    def extract_many(self, stories: list[Story]) -> list[Article]:
        """Extract articles for a list of stories, grouped by source.

        Blacklisted sources are skipped before submission. Each source bucket
        is processed sequentially in its own thread; sources that cross the
        failure threshold are abandoned mid-run without processing remaining
        stories from that source.
        """
        candidates = [s for s in stories if not self._blacklist.is_blacklisted(s["source_name"])]
        skipped = len(stories) - len(candidates)
        if skipped:
            logger.info("skipped %d stories from blacklisted sources", skipped)

        buckets: dict[str, list[Story]] = defaultdict(list)
        for story in candidates:
            buckets[story["source_name"]].append(story)

        all_articles: list[Article] = []

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {
                pool.submit(self._extract_bucket, source, bucket): source
                for source, bucket in buckets.items()
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    all_articles.extend(future.result())
                except Exception as e:
                    logger.warning("unexpected error processing source %s: %s", source, e)

        self._blacklist.flush()
        return all_articles

    def _extract_bucket(self, source_name: str, stories: list[Story]) -> list[Article]:
        articles: list[Article] = []

        for i, story in enumerate(stories):
            if i > 0:
                time.sleep(self._source_delay)

            try:
                article = self.extract(story)
            except Exception as e:
                logger.warning("unexpected error extracting %s: %s", story["url"], e)
                article = None

            with self._lock:
                newly_blacklisted = self._blacklist.record_attempt(source_name, article is not None)

            if article is not None:
                articles.append(article)

            if newly_blacklisted:
                remaining = len(stories) - i - 1
                logger.info(
                    "source %s blacklisted, skipping %d remaining stories",
                    source_name,
                    remaining,
                )
                break

        return articles
