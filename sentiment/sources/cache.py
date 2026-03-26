from pathlib import Path

import pandas as pd


class MarketDataCache:
    """File-based cache for daily OHLCV data, stored as CSV files at <data_dir>/<SYMBOL>/<YEAR>.csv."""

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "historical-prices"
        self.data_dir = data_dir

    def store(self, symbol: str, year: int, df: pd.DataFrame) -> None:
        """Persist a DataFrame for the given symbol and year."""
        path = self._path(symbol, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """Load cached data for the given symbol and year, raises FileNotFoundError if absent."""
        path = self._path(symbol, year)
        if not path.exists():
            raise FileNotFoundError(f"No cached data for {symbol} {year}: {path}")
        return pd.read_csv(path, index_col="timestamp", parse_dates=True)

    def _path(self, symbol: str, year: int) -> Path:
        return self.data_dir / symbol / f"{year}.csv"

    def is_cached(self, ticker: str, year: int) -> bool:
        return self._path(ticker, year).exists()