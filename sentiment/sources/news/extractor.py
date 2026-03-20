import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

import trafilatura
from trafilatura.settings import use_config

from .blacklist import SourceBlacklist
from .metrics import BucketEvent, ExtractionMetrics
from .models import Article, Story

logger = logging.getLogger(__name__)


@dataclass
class ScheduledBucket:
    source_label: str
    source_name: str
    stories: list[Story]
    initial_delay: float


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
        split_threshold: int | None = None,
        metrics: ExtractionMetrics | None = None,
    ):
        self._blacklist = blacklist
        self._workers = workers
        self._source_delay = source_delay
        self._split_threshold = split_threshold
        self._timeout = timeout
        self._metrics = metrics
        self._lock = Lock()
        self._config = use_config()
        self._config.set("DEFAULT", "download_timeout", str(timeout))

    def extract(self, story: Story) -> Article | None:
        """Download and parse the full article text for a single story.

        Returns ``None`` (and logs a warning) if the URL cannot be fetched or
        trafilatura fails to extract meaningful text.
        """
        html = self._fetch(story["url"])
        if html is None:
            logger.debug("failed to fetch: %s", story["url"])
            return None

        doc = trafilatura.bare_extraction(
            html, url=story["url"], include_comments=False
        )
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
            tickers=story["tickers"],
        )

    def _fetch(self, url: str) -> str | None:
        """Fetch URL with a hard wall-clock timeout.

        trafilatura's download_timeout is a per-socket-read timeout, not a
        total duration limit. A server that trickles data in slow chunks can
        keep the connection alive indefinitely without ever triggering it.
        This wrapper runs the fetch in a daemon thread and enforces a true
        wall-clock limit via join(timeout=...).
        """
        result: list[str | None] = [None]

        def _do_fetch() -> None:
            result[0] = trafilatura.fetch_url(url, config=self._config)

        t = threading.Thread(target=_do_fetch, daemon=True)
        t.start()
        t.join(timeout=self._timeout * 1.5)
        if t.is_alive():
            logger.debug("fetch hard timeout exceeded: %s", url)
            return None
        return result[0]

    def extract_many(self, stories: list[Story]) -> list[Article]:
        """Extract articles for a list of stories, grouped by source.

        Blacklisted sources are skipped before submission. Each source bucket
        is processed sequentially in its own thread; sources that cross the
        failure threshold are abandoned mid-run without processing remaining
        stories from that source.
        """
        candidates = [
            s for s in stories if not self._blacklist.is_blacklisted(s["source_name"])
        ]
        skipped = len(stories) - len(candidates)
        if skipped:
            logger.info("skipped %d stories from blacklisted sources", skipped)

        buckets: dict[str, list[Story]] = defaultdict(list)
        for story in candidates:
            buckets[story["source_name"]].append(story)

        scheduled = self._schedule_buckets(list(buckets.items()))

        all_articles: list[Article] = []

        if self._metrics is not None:
            self._metrics.run_started()

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {
                pool.submit(self._extract_bucket, bucket): bucket.source_label
                for bucket in scheduled
            }
            for future in as_completed(futures):
                source_label = futures[future]
                try:
                    all_articles.extend(future.result())
                except Exception as e:
                    logger.warning(
                        "unexpected error processing source %s: %s", source_label, e
                    )

        if self._metrics is not None:
            self._metrics.run_finished()

        self._blacklist.flush()
        return all_articles

    def _schedule_buckets(
        self, buckets: list[tuple[str, list[Story]]]
    ) -> list[ScheduledBucket]:
        scheduled = []
        for source, stories in buckets:
            if self._split_threshold is not None and len(stories) >= self._split_threshold:
                mid = len(stories) // 2
                scheduled.append(ScheduledBucket(
                    source_label=f"{source} [1/2]",
                    source_name=source,
                    stories=stories[:mid],
                    initial_delay=0.0,
                ))
                scheduled.append(ScheduledBucket(
                    source_label=f"{source} [2/2]",
                    source_name=source,
                    stories=stories[mid:],
                    initial_delay=self._source_delay / 2,
                ))
            else:
                scheduled.append(ScheduledBucket(
                    source_label=source,
                    source_name=source,
                    stories=stories,
                    initial_delay=0.0,
                ))
        scheduled.sort(key=lambda b: -len(b.stories))
        return scheduled

    def _extract_bucket(self, bucket: ScheduledBucket) -> list[Article]:
        ev: BucketEvent | None = None
        if self._metrics is not None:
            ev = self._metrics.bucket_started(bucket.source_label, len(bucket.stories))

        logger.debug("process source: %s (%d)...", bucket.source_label, len(bucket.stories))

        if bucket.initial_delay > 0.0:
            time.sleep(bucket.initial_delay)

        articles: list[Article] = []

        for i, story in enumerate(bucket.stories):
            if self._blacklist.is_blacklisted(bucket.source_name):
                logger.info(
                    "source %s blacklisted by sibling bucket, skipping %d remaining stories",
                    bucket.source_name,
                    len(bucket.stories) - i,
                )
                break

            if i > 0:
                time.sleep(self._source_delay)

            try:
                article = self.extract(story)
            except Exception as e:
                logger.warning("unexpected error extracting %s: %s", story["url"], e)
                article = None

            with self._lock:
                newly_blacklisted = self._blacklist.record_attempt(
                    bucket.source_name, article is not None
                )

            if article is not None:
                articles.append(article)

            if newly_blacklisted:
                remaining = len(bucket.stories) - i - 1
                logger.info(
                    "source %s blacklisted, skipping %d remaining stories",
                    bucket.source_name,
                    remaining,
                )
                break

        if self._metrics is not None and ev is not None:
            self._metrics.bucket_finished(ev, len(articles))

        logger.debug(
            "processed %d/%d articles from %s in %ds",
            len(articles),
            len(bucket.stories),
            bucket.source_label,
            int(ev.duration or 0) if ev else 0,
        )

        return articles
