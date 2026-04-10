from pathlib import Path

import pandas as pd


class SentimentRepository:
    """Stores daily sentiment aggregates as parquet files at ``<data_dir>/<SYMBOL>.parquet``.

    Each file holds rows produced by :func:`~src.embeddings.pipeline.aggregate_daily`:
    columns: ticker, date, sentiment_score, n_articles, embedding, sentiment_probs.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "sentiment"
        self._data_dir = data_dir

    def store(self, symbol: str, df: pd.DataFrame) -> None:
        path = self._path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def load(self, symbol: str) -> pd.DataFrame:
        """Raises ``FileNotFoundError`` if no sentiment has been computed yet."""
        path = self._path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"No sentiment data for {symbol}: {path}")
        return pd.read_parquet(path)

    def exists(self, symbol: str) -> bool:
        return self._path(symbol).exists()

    def _path(self, symbol: str) -> Path:
        return self._data_dir / f"{symbol}.parquet"
