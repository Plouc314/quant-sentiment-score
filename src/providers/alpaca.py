from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import requests

_BASE_URL = "https://data.alpaca.markets"


class AlpacaProvider:
    """Fetches OHLCV bars from the Alpaca Securities API.

    Credentials are resolved from ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET``
    environment variables when not passed explicitly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        api_key = api_key or os.environ["ALPACA_API_KEY"]
        api_secret = api_secret or os.environ["ALPACA_API_SECRET"]
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        })

    def fetch_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        adjustment: str = "split",
        feed: str | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for *symbols* over the given date range.

        Returns a DataFrame with columns:
        symbol, open, high, low, close, volume, trade_count, vwap.
        Index is a DatetimeIndex (``timestamp``), sorted ascending.
        """
        params: dict = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "adjustment": adjustment,
        }
        if feed is not None:
            params["feed"] = feed

        bars: dict[str, list[dict]] = {s: [] for s in symbols}
        while True:
            data = self._get("/v2/stocks/bars", params).json()
            for symbol, symbol_bars in data.get("bars", {}).items():
                bars[symbol].extend(symbol_bars)
            token = data.get("next_page_token")
            if not token:
                break
            params["page_token"] = token

        records = [
            {
                "symbol": symbol,
                "timestamp": bar["t"],
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"],
                "trade_count": bar["n"],
                "vwap": bar.get("vw"),
            }
            for symbol, symbol_bars in bars.items()
            for bar in symbol_bars
        ]

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        return df

    def _get(self, path: str, params: dict, max_retries: int = 3) -> requests.Response:
        url = _BASE_URL + path
        for _ in range(max_retries):
            response = self._session.get(url, params=params)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                time.sleep(retry_after)
                continue
            if not response.ok:
                raise RuntimeError(f"Alpaca API error {response.status_code}: {response.text}")
            return response
        raise RuntimeError(f"Alpaca API rate limit exceeded after {max_retries} retries")
