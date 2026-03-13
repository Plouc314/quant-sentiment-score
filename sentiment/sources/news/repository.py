from datetime import date
from pathlib import Path

import pandas as pd

from .models import Article, Story

_INDEX_FIELDS = ["id", "url", "title", "publish_date", "source_name", "language"]


class ArticleRepository:
    """Stores articles as individual .txt files with a CSV index for lookups and deduplication.

    The index is loaded into memory at startup and only written back to disk
    when :meth:`flush` is called (or when the index is dirty and the repository
    is used as a context manager).

    Layout::

        <data_dir>/
            index.csv
            articles/
                <id>.txt
    """

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data" / "news"
        self._data_dir = data_dir
        self._articles_dir = data_dir / "articles"
        self._index_path = data_dir / "index.csv"
        self._dirty = False
        self._df = self._load_index()

    def store(self, article: Article) -> None:
        """Write article text to disk and add a row to the in-memory index.

        Call :meth:`flush` to persist the updated index to disk.
        """
        self._articles_dir.mkdir(parents=True, exist_ok=True)
        txt_path = self._articles_dir / f"{article['id']}.txt"
        txt_path.write_text(article["text"], encoding="utf-8")

        row = pd.DataFrame(
            [
                {
                    "id": article["id"],
                    "url": article["url"],
                    "title": article["title"],
                    "publish_date": article["publish_date"].isoformat(),
                    "source_name": article["source_name"],
                    "language": article["language"],
                }
            ]
        )
        self._df = pd.concat([self._df, row], ignore_index=True)
        self._dirty = True

    def load(self, article_id: str) -> Article:
        """Load an article by ID from the in-memory index and its text file."""
        rows = self._df[self._df["id"] == article_id]
        if rows.empty:
            raise FileNotFoundError(f"Article not in index: {article_id}")
        row = rows.iloc[0]
        txt_path = self._articles_dir / f"{article_id}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Article text file missing: {txt_path}")
        return Article(
            id=row["id"],
            url=row["url"],
            title=row["title"],
            text=txt_path.read_text(encoding="utf-8"),
            publish_date=date.fromisoformat(row["publish_date"]),
            source_name=row["source_name"],
            language=row["language"],
        )

    def article_ids(self) -> list[str]:
        """Return all article IDs from the in-memory index."""
        return self._df["id"].tolist()

    def load_all(self) -> list[Article]:
        """Load every article in the index."""
        return [self.load(aid) for aid in self.article_ids()]

    def exists(self, url: str) -> bool:
        """Return True if a URL is already in the index."""
        return url in self._df["url"].values

    def filter_new(self, stories: list[Story]) -> list[Story]:
        """Return only stories whose URL is not yet in the index."""
        return [s for s in stories if not self.exists(s["url"])]

    def flush(self) -> None:
        """Write the in-memory index to disk if it has been modified."""
        if not self._dirty:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self._index_path, index=False)
        self._dirty = False

    def _load_index(self) -> pd.DataFrame:
        if not self._index_path.exists():
            return pd.DataFrame(columns=_INDEX_FIELDS)
        return pd.read_csv(self._index_path, dtype=str)
