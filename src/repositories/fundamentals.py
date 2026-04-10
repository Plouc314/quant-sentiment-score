from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..models import FUNDAMENTAL_COLS, Fundamentals

logger = logging.getLogger(__name__)


class FundamentalsRepository:
    """Persists dated fundamental snapshots for each symbol in a single CSV.

    Each call to :meth:`store` appends a row with today's date so that
    :meth:`load_history` returns a time-indexed DataFrame suitable for
    forward-filling inside ``StockDataset``.

    Layout::

        <data_dir>/fundamentals.csv   # columns: date, symbol, <FUNDAMENTAL_COLS>
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "fundamentals"
        self._path = data_dir / "fundamentals.csv"

    def store(self, symbol: str, data: Fundamentals) -> None:
        """Append a dated snapshot for *symbol*, creating the CSV if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.today().normalize().date().isoformat()
        row = pd.DataFrame([{"date": today, "symbol": symbol, **data}])

        if self._path.exists():
            existing = pd.read_csv(self._path)
            existing = pd.concat([existing, row], ignore_index=True)
            existing = existing.drop_duplicates(subset=["date", "symbol"], keep="last")
        else:
            existing = row

        existing.to_csv(self._path, index=False)

    def load_history(self, symbol: str) -> pd.DataFrame | None:
        """Return all snapshots for *symbol* as a DatetimeIndex'd DataFrame.

        Returns ``None`` if no data exists for this symbol.

        The DatetimeIndex and :data:`FUNDAMENTAL_COLS` columns make this the
        correct input for ``StockDataset(fundamental_df=...)``.
        """
        if not self._path.exists():
            return None
        df = pd.read_csv(self._path, parse_dates=["date"])
        rows = df[df["symbol"] == symbol][["date"] + FUNDAMENTAL_COLS].copy()
        if rows.empty:
            return None
        return rows.set_index("date").sort_index()

    def load_latest(self, symbol: str) -> Fundamentals | None:
        """Return the most recent snapshot for *symbol*, or ``None``."""
        df = self.load_history(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1].to_dict()  # type: ignore[return-value]
