"""Compute the 16 technical indicators used in the reference paper.

Matches the feature set defined in ``Quant/modeling/*/backtrader_sequence_model.py``
(``extract_factor``) — raw talib indicators, not scale-invariant ratios.  These
features span many orders of magnitude (e.g. ``ma5`` in dollars, ``obv`` in
shares, ``macd_hist`` in small fractions), so they MUST be normalised
per-window by :class:`_LazyDataset` before being fed to a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_COLUMNS: list[str] = [
    "ma5",
    "ma30",
    "ma60",
    "ema5",
    "ema30",
    "ema60",
    "macd_6_15_6",
    "macd_12_26_9",
    "macd_30_60_30",
    "rsi_14",
    "willr_14",
    "mom_14",
    "cmo_14",
    "ultosc",
    "obv",
    "adosc",
]


class TechnicalFactors:
    """Compute 16 raw technical indicators matching the reference paper.

    Indicator set (all in raw talib units — no ratios, no scaling)::

        ma5, ma30, ma60               simple moving averages of close
        ema5, ema30, ema60            exponential moving averages of close
        macd_6_15_6,                  MACD histogram, (fast, slow, signal)
        macd_12_26_9,
        macd_30_60_30
        rsi_14                        relative strength index
        willr_14                      Williams %R
        mom_14                        absolute momentum (close - close.shift(14))
        cmo_14                        Chande momentum oscillator
        ultosc                        Ultimate Oscillator (7, 14, 28)
        obv                           on-balance volume (cumulative)
        adosc                         Chaikin A/D oscillator (3, 10)

    Because the features span wildly different scales, the caller must apply
    per-window min-max normalisation (see :class:`_LazyDataset`).  Rows where
    any indicator is still in its warmup period are dropped.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 16 factors.

        Parameters
        ----------
        df:
            DataFrame with columns: open, high, low, close, volume.
            Index must be sorted ascending.  (``vwap`` is not used.)

        Returns
        -------
        DataFrame with columns from :data:`FACTOR_COLUMNS`, NaN warmup rows dropped.
        """
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        f = pd.DataFrame(index=df.index)

        # Simple / exponential moving averages
        f["ma5"]   = close.rolling(5).mean()
        f["ma30"]  = close.rolling(30).mean()
        f["ma60"]  = close.rolling(60).mean()
        f["ema5"]  = _ema(close, 5)
        f["ema30"] = _ema(close, 30)
        f["ema60"] = _ema(close, 60)

        # MACD histograms
        f["macd_6_15_6"]   = _macd_hist(close, 6, 15, 6)
        f["macd_12_26_9"]  = _macd_hist(close, 12, 26, 9)
        f["macd_30_60_30"] = _macd_hist(close, 30, 60, 30)

        # Oscillators
        f["rsi_14"]   = _rsi(close, 14)
        f["willr_14"] = _willr(high, low, close, 14)
        f["mom_14"]   = close - close.shift(14)
        f["cmo_14"]   = _cmo(close, 14)
        f["ultosc"]   = _ultosc(high, low, close, 7, 14, 28)

        # Volume-based
        f["obv"]   = _obv(close, volume)
        f["adosc"] = _adosc(high, low, close, volume, 3, 10)

        return f.dropna()


# ------------------------------------------------------------------
# Indicator primitives
# ------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100.0 * (hh - close) / (hh - ll)


def _cmo(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    sum_gain = gain.rolling(period).sum()
    sum_loss = loss.rolling(period).sum()
    denom    = sum_gain + sum_loss
    return 100.0 * (sum_gain - sum_loss) / denom.replace(0, np.nan)


def _ultosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    p1: int,
    p2: int,
    p3: int,
) -> pd.Series:
    prev_close = close.shift(1)
    true_low   = pd.concat([low, prev_close], axis=1).min(axis=1)
    true_high  = pd.concat([high, prev_close], axis=1).max(axis=1)
    bp = close - true_low
    tr = true_high - true_low

    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    return 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (volume * np.sign(close.diff())).cumsum()


def _adosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int,
    slow: int,
) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)
    ad  = mfv.cumsum()
    return _ema(ad, fast) - _ema(ad, slow)
