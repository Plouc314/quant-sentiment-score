from __future__ import annotations

import hashlib
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ..models import Article
from ..repositories.articles import ArticleRepository

logger = logging.getLogger(__name__)

_COLS = ["headline", "url", "publisher", "date", "stock"]


class KaggleImporter:
    """Imports headlines from Kaggle stock news CSVs into :class:`ArticleRepository`.

    Targets the "Massive Stock News Analysis" dataset schema:
    ``headline``, ``url``, ``publisher``, ``date``, ``stock``.

    When the same URL appears with multiple tickers, tickers are merged rather
    than creating duplicate articles.

    Parameters
    ----------
    repo:
        Target repository.
    universe:
        If given, only import articles whose ticker is in this set.
    flush_interval:
        Number of chunks between automatic flushes.
    """

    def __init__(
        self,
        repo: ArticleRepository,
        universe: set[str] | None = None,
        flush_interval: int = 10,
    ) -> None:
        self._repo = repo
        self._universe = universe
        self._flush_interval = flush_interval

    def import_csv(
        self,
        path: Path,
        start_date: date | None = None,
        start_row: int = 0,
        chunk_size: int = 10_000,
    ) -> None:
        """Import all rows from *path*.

        Parameters
        ----------
        path:
            Path to a Kaggle stock news CSV file.
        start_date:
            Skip articles published before this date.
        start_row:
            Data rows to skip from the start (excluding header).  Use to resume
            an interrupted import: ``start_row = chunks_done * chunk_size``.
        chunk_size:
            Rows per read chunk.
        """
        path = Path(path)
        logger.info("importing %s (start_date=%s, start_row=%d)", path.name, start_date, start_row)

        skiprows = range(1, start_row + 1) if start_row > 0 else None
        total_stored = total_merged = chunk_count = 0

        for chunk in pd.read_csv(path, usecols=_COLS, chunksize=chunk_size, skiprows=skiprows):
            if start_date is not None:
                dates = pd.to_datetime(chunk["date"], utc=True, errors="coerce").dt.date
                chunk = chunk[dates >= start_date]
            stored, merged = self._process_chunk(chunk)
            total_stored += stored
            total_merged += merged
            chunk_count += 1

            if chunk_count % self._flush_interval == 0:
                self._repo.flush()
                logger.info("flushed after %d chunks — stored=%d merged=%d", chunk_count, total_stored, total_merged)

        self._repo.flush()
        logger.info("import complete: stored=%d merged=%d chunks=%d", total_stored, total_merged, chunk_count)

    def _process_chunk(self, chunk: pd.DataFrame) -> tuple[int, int]:
        stored = merged = 0
        for url, rows in chunk.groupby("url", sort=False):
            url = str(url)
            tickers = [str(t) for t in rows["stock"].dropna().unique().tolist()]
            if self._universe is not None:
                tickers = [t for t in tickers if t in self._universe]
                if not tickers:
                    continue
            if self._repo.exists(url):
                self._repo.add_tickers(url, tickers)
                merged += 1
            else:
                row = rows.iloc[0]
                self._repo.store(Article(
                    id=_url_id(url),
                    url=url,
                    title=str(row["headline"]),
                    text=str(row["headline"]),
                    publish_date=_parse_date(str(row["date"])),
                    source_name=str(row["publisher"]) if pd.notna(row["publisher"]) else "",
                    language="en",
                    tickers=tickers,
                ))
                stored += 1
        return stored, merged


def _url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _parse_date(raw: str) -> date:
    return pd.to_datetime(raw, utc=True).date()
