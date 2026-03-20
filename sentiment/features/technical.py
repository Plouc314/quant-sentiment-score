"""Compute 16 technical indicators from OHLCV + VWAP data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FACTOR_COLUMNS: list[str] = [
    # Trend (4)
    "close_sma5_ratio",
    "close_sma20_ratio",
    "close_sma60_ratio",
    "macd",
    # Momentum (5)
    "macd_signal",
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "roc_10",
    # Volatility (3)
    "atr_14_norm",
    "bb_pct_b",
    "bb_width",
    # Volume (3)
    "vol_sma20_ratio",
    "obv_slope_10",
    "vwap_close_ratio",
    # Returns (1)
    "log_return",
]


class TechnicalFactors:
    """Compute 16 technical indicators from OHLCV + VWAP data.

    All indicators are ratios, bounded values, or naturally small numbers
    so they are roughly scale-invariant across stocks.  A StandardScaler
    should still be applied before feeding to a model.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 16 technical factors.

        Parameters
        ----------
        df:
            DataFrame with columns: open, high, low, close, volume, vwap.
            Index should be sorted ascending.

        Returns
        -------
        DataFrame with 16 factor columns (see ``FACTOR_COLUMNS``).
        Rows with NaN from lookback periods are dropped.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        vwap = df["vwap"]

        factors = pd.DataFrame(index=df.index)

        # --- Trend (4) ---
        factors["close_sma5_ratio"] = close / close.rolling(5).mean()
        factors["close_sma20_ratio"] = close / close.rolling(20).mean()
        factors["close_sma60_ratio"] = close / close.rolling(60).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = (ema12 - ema26) / close
        factors["macd"] = macd_line

        # --- Momentum (5) ---
        factors["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()
        factors["rsi_14"] = _rsi(close, 14) / 100.0  # normalize to [0, 1]

        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stoch_k = (close - low_14) / (high_14 - low_14)
        factors["stoch_k"] = stoch_k
        factors["stoch_d"] = stoch_k.rolling(3).mean()

        factors["roc_10"] = close.pct_change(10)

        # --- Volatility (3) ---
        tr = _true_range(high, low, close)
        factors["atr_14_norm"] = tr.rolling(14).mean() / close

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        factors["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower)
        factors["bb_width"] = (bb_upper - bb_lower) / sma20

        # --- Volume (3) ---
        factors["vol_sma20_ratio"] = volume / volume.rolling(20).mean()

        obv = _obv(close, volume)
        factors["obv_slope_10"] = _rolling_slope(obv, 10)

        factors["vwap_close_ratio"] = vwap / close

        # --- Returns (1) ---
        factors["log_return"] = np.log(close / close.shift(1))

        return factors.dropna()


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index using exponential moving average of gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: max of (high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative sum of signed volume."""
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Linear regression slope over a rolling window."""
    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    x_var = (x_centered**2).sum()

    def _slope(y: np.ndarray) -> float:
        if len(y) < window:
            return np.nan
        return float((x_centered * (y - y.mean())).sum() / x_var)

    return series.rolling(window).apply(_slope, raw=True)
