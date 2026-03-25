from pathlib import Path

import pandas as pd


class SentimentCache:
    """File-based cache for daily sentiment data, stored as parquet files at <data_dir>/<SYMBOL>.parquet.

    Each file holds rows aggregated by ``pipeline.process_ticker_articles()``:
    columns: date, sentiment_score, n_articles, embedding, sentiment_probs.
    """

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "sentiment"
        self.data_dir = data_dir

    def store(self, symbol: str, df: pd.DataFrame) -> None:
        """Persist a daily-sentiment DataFrame for the given symbol."""
        path = self._path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def load(self, symbol: str) -> pd.DataFrame:
        """Load cached sentiment for the given symbol, raises FileNotFoundError if absent."""
        path = self._path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"No cached sentiment for {symbol}: {path}")
        return pd.read_parquet(path)

    def exists(self, symbol: str) -> bool:
        """Return True if a cached parquet file exists for the given symbol."""
        return self._path(symbol).exists()

    def _path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol}.parquet"
