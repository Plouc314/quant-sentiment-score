from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models import Article

_FIELDS = [
    "benzinga_id", "url", "title", "publish_date", "tickers",
    "author", "body", "teaser", "last_updated", "channels", "tags", "images",
]


class ArticleRepository:
    """Stores articles as monthly parquet files.

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

    def bulk_store(self, articles: list[Article]) -> None:
        """Write *articles* to disk, skipping any whose URL already exists.

        Articles are grouped by publish month; only affected month files are
        rewritten.
        """
        if not articles:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)

        by_month: dict[str, list[Article]] = {}
        for article in articles:
            key = article["publish_date"].strftime("%Y-%m")
            by_month.setdefault(key, []).append(article)

        for month_key, month_articles in by_month.items():
            self._write_month(month_key, month_articles)

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
                    last_updated_raw = row.get("last_updated")
                    articles.append(
                        Article(
                            benzinga_id=int(row["benzinga_id"]),
                            url=row["url"],
                            title=row["title"],
                            publish_date=pd.to_datetime(row["publish_date"]).date(),
                            tickers=list(row["tickers"]),
                            author=row.get("author") or None,
                            body=row.get("body") or None,
                            teaser=row.get("teaser") or None,
                            last_updated=(
                                pd.to_datetime(last_updated_raw).date()
                                if last_updated_raw and pd.notna(last_updated_raw)
                                else None
                            ),
                            channels=list(row.get("channels") or []),
                            tags=list(row.get("tags") or []),
                            images=list(row.get("images") or []),
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
        Columns: see ``_FIELDS``.
        """
        path = self._data_dir / f"{year}-{month:02d}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=_FIELDS)
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_month(self, month_key: str, articles: list[Article]) -> None:
        path = self._data_dir / f"{month_key}.parquet"
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=_FIELDS)

        known_ids = set(existing["benzinga_id"].tolist()) if not existing.empty else set()
        new = [a for a in articles if a["benzinga_id"] not in known_ids]
        if not new:
            return

        df = pd.concat([existing, self._to_df(new)], ignore_index=True)
        df.to_parquet(path, index=False)

    def _to_df(self, articles: list[Article]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "benzinga_id": a["benzinga_id"],
                    "url": a["url"],
                    "title": a["title"],
                    "publish_date": a["publish_date"].isoformat(),
                    "tickers": a["tickers"],
                    "author": a["author"],
                    "body": a["body"],
                    "teaser": a["teaser"],
                    "last_updated": a["last_updated"].isoformat() if a["last_updated"] else None,
                    "channels": a["channels"],
                    "tags": a["tags"],
                    "images": a["images"],
                }
                for a in articles
            ]
        )
