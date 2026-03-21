"""Fundamental factor source and cache.

``FundamentalSource``  — thin wrapper around ``yfinance.Ticker.info``.
``FundamentalCache``   — persists latest snapshot for each symbol in a single CSV.

Typical usage::

    source = FundamentalSource()
    cache  = FundamentalCache()

    for symbol in symbols:
        data = source.fetch(symbol)
        cache.store(symbol, data)

    fund = cache.load("AAPL")   # Fundamentals TypedDict or None
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


_CSV_COLS = ["symbol"] + FUNDAMENTAL_COLS


class FundamentalCache:
    """Persist the latest fundamental snapshot for each symbol in a single CSV.

    Layout::

        <data_dir>/fundamentals.csv   # columns: symbol + FUNDAMENTAL_COLS

    Storing a symbol twice overwrites the previous record.

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
        """Upsert the snapshot for *symbol*, creating the CSV if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{"symbol": symbol, **data}])

        if self._path.exists():
            existing = pd.read_csv(self._path)
            existing = existing[existing["symbol"] != symbol]
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            combined = row

        combined.to_csv(self._path, index=False)
        logger.debug("stored fundamentals for %s", symbol)

    def load(self, symbol: str) -> Fundamentals | None:
        """Return the cached snapshot for *symbol*, or ``None`` if not found."""
        if not self._path.exists():
            return None
        df = pd.read_csv(self._path)
        rows = df[df["symbol"] == symbol]
        if rows.empty:
            return None
        return rows.iloc[0][FUNDAMENTAL_COLS].to_dict()  # type: ignore[return-value]

    def load_all(self) -> pd.DataFrame:
        """Return all cached snapshots as a DataFrame indexed by symbol.

        Returns an empty DataFrame if the cache does not exist yet.

        Columns: :data:`FUNDAMENTAL_COLS`
        """
        if not self._path.exists():
            return pd.DataFrame(columns=_CSV_COLS).set_index("symbol")
        return pd.read_csv(self._path).set_index("symbol")
