"""Fundamental factor source and cache.

``FundamentalSource``  — thin wrapper around ``yfinance.Ticker.info``.
``FundamentalCache``   — persists dated snapshots for each symbol in a single CSV.

Typical usage::

    source = FundamentalSource()
    cache  = FundamentalCache()

    for symbol in symbols:
        data = source.fetch(symbol)
        cache.store(symbol, data)          # appends a dated row

    fund_df = cache.load_df("AAPL")       # DatetimeIndex'd DataFrame → build_dataset
    latest  = cache.load("AAPL")          # Fundamentals dict — latest snapshot only
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from typing import TypedDict

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class Fundamentals(TypedDict):
    pe: float
    forward_pe: float
    pb: float
    ps: float
    roe: float
    op_margin: float
    profit_margin: float
    de: float
    beta: float


# yfinance info key → canonical column name
_YFINANCE_FIELDS: dict[str, str] = {
    "trailingPE":                   "pe",
    "forwardPE":                    "forward_pe",
    "priceToBook":                  "pb",
    "priceToSalesTrailing12Months": "ps",
    "returnOnEquity":               "roe",
    "operatingMargins":             "op_margin",
    "profitMargins":                "profit_margin",
    "debtToEquity":                 "de",
    "beta":                         "beta",
}

FUNDAMENTAL_COLS: list[str] = list(_YFINANCE_FIELDS.values())

# CSV column order: date first, then symbol, then metrics
_CSV_COLS = ["date", "symbol"] + FUNDAMENTAL_COLS


class FundamentalSource:
    """Fetch fundamental snapshots from yfinance.

    Missing metrics (e.g. P/E for a loss-making company) are returned as NaN —
    not an error, just unavailable data.

    Parameters
    ----------
    request_delay:
        Seconds to sleep between tickers in :meth:`fetch_many` to avoid
        triggering Yahoo's rate limiter.
    """

    def __init__(self, request_delay: float = 1.5) -> None:
        self._delay = request_delay

    def fetch(self, symbol: str) -> Fundamentals:
        """Return a snapshot of fundamental metrics for *symbol*.

        Keys are the names in :data:`FUNDAMENTAL_COLS`.
        Missing metrics are ``float("nan")``, not ``None``.
        """
        info = yf.Ticker(symbol).info
        result: dict[str, float] = {}
        for yf_key, col in _YFINANCE_FIELDS.items():
            val = info.get(yf_key)
            result[col] = float(val) if val is not None else float("nan")

        n_missing = sum(1 for v in result.values() if pd.isna(v))
        if n_missing:
            logger.debug("%s: %d/%d fundamental fields missing", symbol, n_missing, len(result))

        return result  # type: ignore[return-value]

    def fetch_many(self, symbols: list[str]) -> dict[str, Fundamentals]:
        """Fetch fundamentals for multiple symbols with per-request delay.

        Returns a mapping of ``{symbol: Fundamentals}``.
        """
        results: dict[str, Fundamentals] = {}
        for i, symbol in enumerate(symbols):
            logger.info("fetching fundamentals for %s (%d/%d)", symbol, i + 1, len(symbols))
            results[symbol] = self.fetch(symbol)
            if i < len(symbols) - 1:
                time.sleep(self._delay)
        return results


class FundamentalCache:
    """Persist dated fundamental snapshots for each symbol in a single CSV.

    Each call to :meth:`store` **appends** a new row with today's date, so the
    file grows over time and :meth:`load_df` can return a time-indexed DataFrame
    suitable for forward-filling inside ``build_dataset``.

    Layout::

        <data_dir>/fundamentals.csv   # columns: date, symbol, <FUNDAMENTAL_COLS>

    Parameters
    ----------
    data_dir:
        Directory containing ``fundamentals.csv``.  Defaults to
        ``<project_root>/data/fundamentals/``.

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
        logger.debug("stored fundamentals for %s (%s)", symbol, today)

    def load_df(self, symbol: str) -> pd.DataFrame | None:
        """Return all snapshots for *symbol* as a DatetimeIndex'd DataFrame.

        Returns ``None`` if no data exists for this symbol.

        This is the correct input for ``build_dataset(fundamental_df=...)``:
        ``align_fundamentals`` will forward-fill the most recent snapshot onto
        each window end date, so quarterly-updated fundamentals naturally carry
        forward through daily windows.

        Columns: :data:`FUNDAMENTAL_COLS`.
        """
        if not self._path.exists():
            return None
        df = pd.read_csv(self._path, parse_dates=["date"])
        rows = df[df["symbol"] == symbol][["date"] + FUNDAMENTAL_COLS].copy()
        if rows.empty:
            return None
        return rows.set_index("date").sort_index()

    def load(self, symbol: str) -> Fundamentals | None:
        """Return the most recent cached snapshot for *symbol*, or ``None``.

        For time-series use (passing to ``build_dataset``), use
        :meth:`load_df` instead.
        """
        df = self.load_df(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1].to_dict()  # type: ignore[return-value]

    def load_all(self) -> pd.DataFrame:
        """Return the most recent snapshot for each symbol as a DataFrame indexed by symbol.

        Returns an empty DataFrame if the cache does not exist yet.
        Columns: :data:`FUNDAMENTAL_COLS`.
        """
        if not self._path.exists():
            return pd.DataFrame(columns=FUNDAMENTAL_COLS)
        df = pd.read_csv(self._path, parse_dates=["date"])
        return (
            df.sort_values("date")
            .groupby("symbol")[FUNDAMENTAL_COLS]
            .last()
        )
