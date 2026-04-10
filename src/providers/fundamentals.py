from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

from ..models import FUNDAMENTAL_COLS, Fundamentals

logger = logging.getLogger(__name__)

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


class FundamentalsProvider:
    """Fetches fundamental metric snapshots from Yahoo Finance (yfinance).

    Missing metrics (e.g. P/E for a loss-making company) are returned as NaN,
    not as an error.

    Parameters
    ----------
    request_delay:
        Seconds to sleep between tickers in :meth:`fetch_many` to avoid
        triggering Yahoo's rate limiter.
    """

    def __init__(self, request_delay: float = 1.5) -> None:
        self._delay = request_delay

    def fetch(self, symbol: str) -> Fundamentals:
        """Return a snapshot of fundamental metrics for *symbol*."""
        info = yf.Ticker(symbol).info
        result: dict[str, float] = {}
        for yf_key, col in _YFINANCE_FIELDS.items():
            val = info.get(yf_key)
            result[col] = float(val) if val is not None else float("nan")

        n_missing = sum(1 for v in result.values() if pd.isna(v))
        if n_missing:
            logger.debug("%s: %d/%d fundamental fields missing", symbol, n_missing, len(FUNDAMENTAL_COLS))

        return result  # type: ignore[return-value]

    def fetch_many(self, symbols: list[str]) -> dict[str, Fundamentals]:
        """Fetch fundamentals for multiple symbols with per-request delay."""
        results: dict[str, Fundamentals] = {}
        for i, symbol in enumerate(symbols):
            logger.info("fetching fundamentals %s (%d/%d)", symbol, i + 1, len(symbols))
            results[symbol] = self.fetch(symbol)
            if i < len(symbols) - 1:
                time.sleep(self._delay)
        return results
