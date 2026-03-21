from pathlib import Path

import pandas as pd

from .models import Article, Story

_INDEX_FIELDS = ["id", "url", "title", "publish_date", "source_name", "language", "tickers"]
_ARTICLE_FIELDS = ["id", "url", "title", "text", "publish_date", "source_name", "language", "tickers"]


class ArticleRepository:
    """Stores articles as monthly parquet files with an in-memory metadata index.

    Articles accumulated via :meth:`store` are held in memory and written to
    disk only when :meth:`flush` is called (or via context manager).  On flush,
    only the month files that received new articles are rewritten.

    Layout::

        <data_dir>/
            2024-01.parquet
            2024-02.parquet
            ...
    """

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data" / "news"
        self._data_dir = data_dir
        self._dirty = False
        self._index_df = self._load_index()
        self._staged: list[Article] = []
        self._pending_ticker_updates: dict[str, list[str]] = {}

    def store(self, article: Article) -> None:
        """Add an article to the staging buffer and update the in-memory index.

        Call :meth:`flush` to persist staged articles to disk.
        """
        self._staged.append(article)

        row = pd.DataFrame(
            [
                {
                    "id": article["id"],
                    "url": article["url"],
                    "title": article["title"],
                    "publish_date": article["publish_date"].isoformat(),
                    "source_name": article["source_name"],
                    "language": article["language"],
                    "tickers": article["tickers"],
                }
            ]
        )
        self._index_df = pd.concat([self._index_df, row], ignore_index=True)
        self._dirty = True

    def add_tickers(self, url: str, tickers: list[str]) -> None:
        """Merge *tickers* into the tickers list of an existing article.

        The article must already be in the index. Call :meth:`flush` to persist
        the update to disk.
        """
        mask = self._index_df["url"] == url
        if not mask.any():
            raise RuntimeError(f"Article not in index: {url}")
        existing = self._index_df.loc[mask, "tickers"].iloc[0]
        merged = list(dict.fromkeys(existing + tickers))
        self._index_df.loc[mask, "tickers"] = pd.Series([merged] * mask.sum(), index=self._index_df.index[mask])
        self._pending_ticker_updates[url] = merged
        self._dirty = True

    def partition_stories(self, stories: list[Story]) -> tuple[list[Story], list[Story]]:
        """Split *stories* into (new, existing) based on URL presence in the index."""
        known_urls = set(self._index_df["url"].values)
        new: list[Story] = []
        existing: list[Story] = []
        for story in stories:
            (existing if story["url"] in known_urls else new).append(story)
        return new, existing

    def read_month_index(self, year: int, month: int) -> pd.DataFrame:
        """Return index-only metadata for the given month (no article text).

        Returns an empty DataFrame if no data exists for that month.

        Columns: id, url, title, publish_date, source_name, language, tickers
        """
        prefix = f"{year}-{month:02d}"
        df = self._index_df[self._index_df["publish_date"].astype(str).str.startswith(prefix)].copy()
        df["publish_date"] = pd.to_datetime(df["publish_date"]).dt.date
        return df.reset_index(drop=True)

    def read_month(self, year: int, month: int) -> pd.DataFrame:
        """Return all articles for the given month as a DataFrame.

        Returns an empty DataFrame if no data exists for that month.

        Columns:
            - id (str): unique article identifier
            - url (str): article URL
            - title (str): article title
            - text (str): full article text
            - publish_date (date): publication date
            - source_name (str): media source name
            - language (str): article language code
            - tickers (list[str]): associated ticker symbols
        """
        path = self._data_dir / f"{year}-{month:02d}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=_ARTICLE_FIELDS)
        df = pd.read_parquet(path)
        df["publish_date"] = pd.to_datetime(df["publish_date"]).dt.date
        return df

    def article_ids(self) -> list[str]:
        """Return all article IDs from the in-memory index."""
        return self._index_df["id"].tolist()

    def exists(self, url: str) -> bool:
        """Return True if a URL is already in the index."""
        return url in self._index_df["url"].values

    def filter_new(self, stories: list[Story]) -> list[Story]:
        """Return only stories whose URL is not yet in the index."""
        return [s for s in stories if not self.exists(s["url"])]

    def flush(self) -> None:
        """Write staged articles and pending ticker updates to disk."""
        if not self._dirty:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)

        staged_by_month: dict[str, list[Article]] = {}
        for article in self._staged:
            key = article["publish_date"].strftime("%Y-%m")
            staged_by_month.setdefault(key, []).append(article)

        ticker_months: dict[str, dict[str, list[str]]] = {}
        if self._pending_ticker_updates:
            affected = self._index_df[self._index_df["url"].isin(self._pending_ticker_updates)]
            for _, row in affected.iterrows():
                month_key = str(row["publish_date"])[:7]
                ticker_months.setdefault(month_key, {})[row["url"]] = self._pending_ticker_updates[row["url"]]

        for month_key in staged_by_month.keys() | ticker_months.keys():
            self._update_month(
                month_key,
                staged_by_month.get(month_key, []),
                ticker_months.get(month_key, {}),
            )

        self._staged.clear()
        self._pending_ticker_updates.clear()
        self._dirty = False

    def __enter__(self) -> "ArticleRepository":
        return self

    def __exit__(self, *_) -> None:
        self.flush()

    def _update_month(
        self,
        month_key: str,
        new_articles: list[Article],
        ticker_updates: dict[str, list[str]],
    ) -> None:
        path = self._data_dir / f"{month_key}.parquet"
        df = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=_ARTICLE_FIELDS)
        if new_articles:
            df = pd.concat([df, self._articles_to_df(new_articles)], ignore_index=True)
        if ticker_updates:
            mask = df["url"].isin(ticker_updates)
            df.loc[mask, "tickers"] = df.loc[mask, "url"].map(ticker_updates)
        df.to_parquet(path, index=False)

    def _load_index(self) -> pd.DataFrame:
        paths = sorted(self._data_dir.glob("*.parquet"))
        if not paths:
            return pd.DataFrame(columns=_INDEX_FIELDS)
        frames = [pd.read_parquet(p, columns=_INDEX_FIELDS) for p in paths]
        return pd.concat(frames, ignore_index=True)

    def _articles_to_df(self, articles: list[Article]) -> pd.DataFrame:
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
