from pathlib import Path

import pandas as pd


class PriceRepository:
    """Stores daily OHLCV bars as CSV files at ``<data_dir>/<SYMBOL>/<YEAR>.csv``."""

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "historical-prices"
        self._data_dir = data_dir

    def store(self, symbol: str, year: int, df: pd.DataFrame) -> None:
        path = self._path(symbol, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """Raises ``FileNotFoundError`` if the data has not been fetched yet."""
        path = self._path(symbol, year)
        if not path.exists():
            raise FileNotFoundError(f"No price data for {symbol} {year}: {path}")
        return pd.read_csv(path, index_col="timestamp", parse_dates=True)

    def load_years(self, symbol: str, years: list[int]) -> pd.DataFrame:
        """Load and concatenate multiple years of price data for *symbol*.

        Raises ``FileNotFoundError`` if no data is found for any of the requested years.
        Duplicate index entries are deduplicated.
        """
        dfs = []
        for year in years:
            try:
                dfs.append(self.load(symbol, year))
            except FileNotFoundError:
                pass
        if not dfs:
            raise FileNotFoundError(f"No price data for {symbol} in years {years}")
        df = pd.concat(dfs).sort_index()
        return df[~df.index.duplicated(keep="first")]

    def _path(self, symbol: str, year: int) -> Path:
        return self._data_dir / symbol / f"{year}.csv"
