"""Fundamental factor source and cache.

``FundamentalSource``  — thin wrapper around ``yfinance.Ticker.info``.
``FundamentalCache``   — persists daily snapshots as one CSV per symbol.

Typical usage::

    source = FundamentalSource()
    cache  = FundamentalCache()

    for symbol in symbols:
        data = source.fetch(symbol)
        cache.store(symbol, date.today(), data)

    # Later, load for alignment with market data:
    df = cache.load("AAPL")   # DatetimeIndex, columns = FUNDAMENTAL_COLS
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

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

    def fetch(self, symbol: str) -> dict[str, float]:
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

        return result

    def fetch_many(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Fetch fundamentals for multiple symbols with per-request delay.

        Returns a mapping of ``{symbol: {col: value}}``.
        """
        results: dict[str, dict[str, float]] = {}
        for i, symbol in enumerate(symbols):
            logger.info("fetching fundamentals for %s (%d/%d)", symbol, i + 1, len(symbols))
            results[symbol] = self.fetch(symbol)
            if i < len(symbols) - 1:
                time.sleep(self._delay)
        return results


class FundamentalCache:
    """Persist fundamental snapshots as one CSV per symbol.

    Layout::

        <data_dir>/
            AAPL.csv
            MSFT.csv
            ...

    Each CSV has columns: ``date`` + all :data:`FUNDAMENTAL_COLS`.
    One row per date when data was fetched; if data for the same date is stored
    twice the newer call wins.

    Parameters
    ----------
    data_dir:
        Root directory for cached CSVs.  Defaults to
        ``<project_root>/data/fundamentals/``.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "fundamentals"
        self._data_dir = data_dir

    def store(self, symbol: str, snapshot_date: date, data: dict[str, float]) -> None:
        """Append (or update) a snapshot row for *symbol* on *snapshot_date*.

        Creates the CSV and parent directories on first call.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)
        path = self._data_dir / f"{symbol}.csv"

        date_str = snapshot_date.isoformat()
        row = pd.DataFrame([{"date": date_str, **data}])

        if path.exists():
            existing = pd.read_csv(path)
            existing = existing[existing["date"] != date_str]
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            combined = row

        combined.to_csv(path, index=False)
        logger.debug("stored fundamentals for %s on %s", symbol, date_str)

    def load(self, symbol: str) -> pd.DataFrame | None:
        """Load all historical fundamental snapshots for *symbol*.

        Returns a DataFrame with a ``DatetimeIndex`` sorted ascending and
        columns matching :data:`FUNDAMENTAL_COLS`, or ``None`` if no data
        exists for *symbol*.
        """
        path = self._data_dir / f"{symbol}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        return df
