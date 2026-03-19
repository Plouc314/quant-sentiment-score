import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class BucketEvent:
    source: str
    n_stories: int
    started_at: float
    finished_at: float | None = None
    n_articles: int = 0

    @property
    def duration(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at


class ExtractionMetrics:
    """Collects timing and concurrency data for :meth:`ArticleExtractor.extract_many` runs.

    Thread-safe. Pass an instance to :class:`ArticleExtractor` and call
    :meth:`summary` after the run to inspect results, or access :attr:`events`
    directly for custom analysis (e.g. plotting concurrency over time).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[BucketEvent] = []
        self._run_start: float | None = None
        self._run_end: float | None = None

    @property
    def events(self) -> list[BucketEvent]:
        return self._events

    def run_started(self) -> None:
        self._run_start = time.time()

    def run_finished(self) -> None:
        self._run_end = time.time()

    def bucket_started(self, source: str, n_stories: int) -> BucketEvent:
        ev = BucketEvent(source=source, n_stories=n_stories, started_at=time.time())
        with self._lock:
            self._events.append(ev)
        return ev

    def bucket_finished(self, ev: BucketEvent, n_articles: int) -> None:
        ev.finished_at = time.time()
        ev.n_articles = n_articles

    def concurrency_at(self, t: float) -> int:
        """Number of buckets active at time *t* (absolute epoch timestamp)."""
        return sum(
            1
            for e in self._events
            if e.started_at <= t and (e.finished_at is None or e.finished_at >= t)
        )

    def summary(self) -> dict:
        """Return a dict with aggregate stats for the last run."""
        finished = [e for e in self._events if e.finished_at is not None]
        durations = {e.source: e.duration for e in finished}

        total_elapsed = (
            (self._run_end or time.time()) - self._run_start
            if self._run_start is not None
            else None
        )

        peak_concurrency = (
            max(self.concurrency_at(e.started_at) for e in self._events)
            if self._events
            else 0
        )

        return {
            "total_elapsed": total_elapsed,
            "n_buckets": len(self._events),
            "n_finished": len(finished),
            "total_articles": sum(e.n_articles for e in finished),
            "peak_concurrency": peak_concurrency,
            "slowest": sorted(durations.items(), key=lambda x: -x[1])[:5],
        }
