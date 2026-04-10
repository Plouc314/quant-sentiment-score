from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models import Article

_FIELDS = ["id", "url", "title", "text", "publish_date", "source_name", "language", "tickers"]
_INDEX_FIELDS = ["id", "url", "title", "publish_date", "source_name", "language", "tickers"]


class ArticleRepository:
    """Stores articles as monthly parquet files.

    Articles added via :meth:`store` are buffered in memory and written to disk
    only when :meth:`flush` is called (or on context manager exit).  Only
    month files that received changes are rewritten on each flush.

    Layout::

        <data_dir>/
            2024-01.parquet
            2024-02.parquet
            ...
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "news"
        self._data_dir = data_dir
        self._staged: dict[str, Article] = {}
        self._ticker_updates: dict[str, list[str]] = {}
        self._known_urls: set[str] = self._load_known_urls()

    def store(self, article: Article) -> None:
        """Buffer *article* for writing. Call :meth:`flush` to persist."""
        self._staged[article["url"]] = article
        self._known_urls.add(article["url"])

    def add_tickers(self, url: str, tickers: list[str]) -> None:
        """Merge *tickers* into the ticker list of an already-stored article.

        If the article is still in the staging buffer the update is applied
        immediately.  Otherwise it is deferred to the next :meth:`flush`.
        """
        if url not in self._known_urls:
            raise RuntimeError(f"Article not in repository: {url}")
        if url in self._staged:
            article = self._staged[url]
            article["tickers"] = list(dict.fromkeys(article["tickers"] + tickers))
        else:
            self._ticker_updates[url] = list(
                dict.fromkeys(self._ticker_updates.get(url, []) + tickers)
            )

    def exists(self, url: str) -> bool:
        return url in self._known_urls

    def flush(self) -> None:
        """Write buffered articles and ticker updates to disk."""
        if not self._staged and not self._ticker_updates:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)

        dirty_months: set[str] = set()
        for article in self._staged.values():
            dirty_months.add(article["publish_date"].strftime("%Y-%m"))
        if self._ticker_updates:
            for month_key in self._months_for_urls(set(self._ticker_updates)):
                dirty_months.add(month_key)

        for month_key in dirty_months:
            self._flush_month(month_key)

        self._staged.clear()
        self._ticker_updates.clear()

    def read_symbol(
        self,
        symbol: str,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[Article]:
        """Return all articles tagged with *symbol* between *start* and *end* (inclusive).

        Parameters
        ----------
        start, end:
            ``(year, month)`` tuples, both inclusive.
        """
        articles: list[Article] = []
        year, month = start
        end_year, end_month = end
        while (year, month) <= (end_year, end_month):
            df = self.read_month(year, month)
            if not df.empty:
                for _, row in df[df["tickers"].apply(lambda t: symbol in t)].iterrows():
                    articles.append(
                        Article(
                            id=row["id"],
                            url=row["url"],
                            title=row["title"],
                            text=row["text"],
                            publish_date=pd.to_datetime(row["publish_date"]).date(),
                            source_name=row["source_name"],
                            language=row["language"],
                            tickers=list(row["tickers"]),
                        )
                    )
            month += 1
            if month > 12:
                month = 1
                year += 1
        return articles

    def read_month(self, year: int, month: int) -> pd.DataFrame:
        """Return all articles for *year*/*month* as a DataFrame.

        Returns an empty DataFrame if no data exists for that month.
        Columns: id, url, title, text, publish_date, source_name, language, tickers.
        """
        path = self._data_dir / f"{year}-{month:02d}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=_FIELDS)
        return pd.read_parquet(path)

    def __enter__(self) -> ArticleRepository:
        return self

    def __exit__(self, *_) -> None:
        self.flush()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush_month(self, month_key: str) -> None:
        path = self._data_dir / f"{month_key}.parquet"
        df = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=_FIELDS)

        new = [a for a in self._staged.values() if a["publish_date"].strftime("%Y-%m") == month_key]
        if new:
            df = pd.concat([df, _to_df(new)], ignore_index=True)

        if self._ticker_updates:
            mask = df["url"].isin(self._ticker_updates)
            if mask.any():
                def _merge(row: pd.Series) -> list[str]:
                    existing = list(row["tickers"])
                    new_tickers = self._ticker_updates[row["url"]]
                    return list(dict.fromkeys(existing + new_tickers))
                df.loc[mask, "tickers"] = df[mask].apply(_merge, axis=1)

        df.to_parquet(path, index=False)

    def _load_known_urls(self) -> set[str]:
        paths = sorted(self._data_dir.glob("*.parquet")) if self._data_dir.exists() else []
        if not paths:
            return set()
        frames = [pd.read_parquet(p, columns=["url"]) for p in paths]
        return set(pd.concat(frames, ignore_index=True)["url"].tolist())

    def _months_for_urls(self, urls: set[str]) -> list[str]:
        """Return the month keys of parquet files that contain any of *urls*."""
        months = []
        for path in sorted(self._data_dir.glob("*.parquet")):
            df = pd.read_parquet(path, columns=["url"])
            if df["url"].isin(urls).any():
                months.append(path.stem)
        return months


def _to_df(articles: list[Article]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": a["id"],
                "url": a["url"],
                "title": a["title"],
                "text": a["text"],
                "publish_date": a["publish_date"].isoformat(),
                "source_name": a["source_name"],
                "language": a["language"],
                "tickers": a["tickers"],
            }
            for a in articles
        ]
    )
