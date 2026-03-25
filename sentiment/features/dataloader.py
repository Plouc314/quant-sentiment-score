
"""Build fused technical + sentiment + fundamental datasets for stock prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from typing import TypedDict

from .technical import TechnicalFactors
from .screening import momentum_slope

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


class FusedDataset(TypedDict):
    """Output of :func:`build_dataset`."""

    X_tech: np.ndarray
    """``(N, window, 16)`` technical factor windows."""
    X_sentiment: np.ndarray
    """``(N, window, 768)`` sentiment embedding windows."""
    X_fundamental: np.ndarray
    """``(N, n_fund)`` fundamental snapshots at each window end date.
    Shape is ``(N, 0)`` when no fundamental data is provided."""
    X_sentiment_probs: np.ndarray
    """``(N, 3)`` daily FinBERT softmax probs at each window end date
    — ``[P(pos), P(neg), P(neutral)]``.  Zero vector for days without news.
    Shape is ``(N, 0)`` when no sentiment data is provided."""
    y: np.ndarray
    """``(N,)`` binary labels."""
    dates: np.ndarray
    """``(N,)`` date of last day in each window."""


class FusedStockDataset(Dataset):
    """PyTorch Dataset for (tech, sentiment, fundamental, sentiment_probs, target) tuples."""

    def __init__(
        self,
        X_tech: np.ndarray,
        X_sentiment: np.ndarray,
        X_fundamental: np.ndarray,
        X_sentiment_probs: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.X_tech = torch.tensor(X_tech, dtype=torch.float32)
        self.X_sentiment = torch.tensor(X_sentiment, dtype=torch.float32)
        self.X_fundamental = torch.tensor(X_fundamental, dtype=torch.float32)
        self.X_sentiment_probs = torch.tensor(X_sentiment_probs, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_tech)

    def __getitem__(
        self, i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X_tech[i],
            self.X_sentiment[i],
            self.X_fundamental[i],
            self.X_sentiment_probs[i],
            self.y[i],
        )


def compute_targets(close: pd.Series) -> pd.Series:
    """Binary target for day *t*: ``1`` if ``close[t+2] > close[t-1]``, else ``0``.

    The 3-day window models a next-day executable signal:

    - ``t-1`` — yesterday's close: the last confirmed price available when a
      signal is generated at end-of-day *t*
    - ``t+2`` — two trading days forward: accounts for one day of signal
      propagation / execution latency and one day of actual holding period

    Concretely: a prediction at end of day *t* translates to
    "buy at open of t+1, sell at close of t+2."

    All shifts are positional (trading days, not calendar days).
    NaN at the first row (no t-1) and last two rows (no t+2).
    """
    future = close.shift(-2)
    past = close.shift(1)
    target = (future > past).astype(np.float32)
    target[future.isna() | past.isna()] = np.nan
    return target


def align_sentiment(
    index: pd.DatetimeIndex,
    sentiment_df: pd.DataFrame | None,
    ticker: str,
) -> np.ndarray:
    """Build ``(N, 768)`` array of daily sentiment embeddings aligned with *index*.

    For dates without news articles a zero vector is used.
    """
    result = np.zeros((len(index), EMBEDDING_DIM), dtype=np.float32)
    if sentiment_df is None or sentiment_df.empty:
        return result

    ticker_df = sentiment_df[sentiment_df["ticker"] == ticker]
    if ticker_df.empty:
        return result

    # Build date → embedding lookup for O(N+M) alignment
    lookup: dict = {}
    for dt_raw, emb in zip(ticker_df["date"], ticker_df["embedding"]):
        dt = pd.Timestamp(dt_raw).date()
        if dt in lookup:
            logger.warning("align_sentiment: duplicate date %s for ticker %s — keeping last row", dt, ticker)
        lookup[dt] = emb

    for i, ts in enumerate(index):
        emb = lookup.get(ts.date())
        if emb is not None:
            result[i] = emb

    return result


def align_sentiment_probs(
    index: pd.DatetimeIndex,
    sentiment_df: pd.DataFrame | None,
    ticker: str,
) -> np.ndarray:
    """Build ``(N, 3)`` array of daily FinBERT softmax probs aligned with *index*.

    Each row is ``[P(positive), P(negative), P(neutral)]`` averaged over all
    articles published on that date.  Zero vector for dates without news.
    Returns ``(N, 0)`` when *sentiment_df* is ``None`` or has no
    ``sentiment_probs`` column (backward-compatible with older pipeline output).

    Shape note: ``(N, 0)`` means the feature branch is disabled (no column in
    the pipeline output at all).  ``(N, 3)`` of zeros means the branch is
    enabled but this ticker has no articles — the model still receives the
    3-dim input, just with a zero vector.  These two cases are intentionally
    distinct so that ``build_dataset`` can derive the correct ``n_sentiment_probs``
    for the model architecture.
    """
    if sentiment_df is None or sentiment_df.empty or "sentiment_probs" not in sentiment_df.columns:
        return np.zeros((len(index), 0), dtype=np.float32)

    ticker_df = sentiment_df[sentiment_df["ticker"] == ticker]
    if ticker_df.empty:
        logger.debug("align_sentiment_probs: no rows for ticker %s — returning zero (N, 3) array", ticker)
        return np.zeros((len(index), 3), dtype=np.float32)

    lookup: dict = {}
    for dt_raw, probs in zip(ticker_df["date"], ticker_df["sentiment_probs"]):
        dt = pd.Timestamp(dt_raw).date()
        if dt in lookup:
            logger.warning("align_sentiment_probs: duplicate date %s for ticker %s — keeping last row", dt, ticker)
        lookup[dt] = probs

    result = np.zeros((len(index), 3), dtype=np.float32)
    for i, ts in enumerate(index):
        probs = lookup.get(ts.date())
        if probs is not None:
            result[i] = probs

    return result


def align_fundamentals(
    win_dates: np.ndarray,
    fund_df: pd.DataFrame | None,
) -> np.ndarray:
    """Build ``(N, n_fund)`` array of fundamental snapshots aligned with window end dates.

    Fundamental data is forward-filled: for each window end date the most recent
    snapshot on or before that date is used.  Dates that precede all snapshots
    (i.e. no prior data exists) are filled with 0 after forward-filling.

    Parameters
    ----------
    win_dates:
        Array of window end dates (``numpy.datetime64``), length N.
    fund_df:
        DataFrame returned by ``FundamentalCache.load()`` — DatetimeIndex,
        columns are the fundamental metric names.  ``None`` → returns empty
        array of shape ``(N, 0)``.

    Returns
    -------
    ``(N, n_fund)`` float32 array.  Returns ``(N, 0)`` when *fund_df* is None.
    """
    if fund_df is None or fund_df.empty:
        return np.zeros((len(win_dates), 0), dtype=np.float32)

    n_fund = fund_df.shape[1]
    dates_idx = pd.DatetimeIndex(win_dates)

    # Reindex with forward-fill: for each query date, use the last known snapshot
    reindexed = fund_df.reindex(dates_idx.union(fund_df.index)).sort_index()
    reindexed = reindexed.ffill().reindex(dates_idx)

    # Any remaining NaN (dates before first snapshot, or missing metrics) → 0
    n_missing = int(reindexed.isna().sum().sum())
    if n_missing:
        logger.debug("align_fundamentals: %d NaN values filled with 0", n_missing)

    return reindexed.fillna(0.0).values.astype(np.float32)


def build_dataset(
    df: pd.DataFrame,
    technical: TechnicalFactors,
    sentiment_df: pd.DataFrame | None = None,
    ticker: str = "",
    window: int = 64,
    fundamental_df: pd.DataFrame | None = None,
    include_momentum_slope: bool = True,
) -> FusedDataset:
    """Full pipeline: OHLCV → factors + targets + aligned sentiment/fundamentals → windows.

    Parameters
    ----------
    df:
        OHLCV + VWAP DataFrame with DatetimeIndex, sorted ascending.
    technical:
        TechnicalFactors instance.
    sentiment_df:
        Output of ``SentimentPipeline.process_ticker_articles()`` or ``None``.
    ticker:
        Stock ticker symbol (used to filter *sentiment_df*).
    window:
        Sliding window size in trading days.
    fundamental_df:
        Output of ``FundamentalCache.load_df(ticker)`` — DatetimeIndex'd
        DataFrame with fundamental metric columns — or ``None`` to omit
        fundamentals.  Each window uses the most recent snapshot on or before
        the window end date (forward-filled).
    include_momentum_slope:
        When ``True`` (default), appends the 20-day linear-regression slope of
        closing prices as an extra scalar in ``X_fundamental``.  The slope is
        computed over the last ``min(20, window)`` closes of each window so the
        model can learn whether trend context improves predictions — rather than
        applying a hard gate at inference.  The value is in raw price units and
        is normalised by ``make_loaders`` together with the other fundamental
        features.

        Set to ``False`` to reproduce results without the momentum feature, or
        when comparing against the hard inference-time gate implemented in
        :func:`~sentiment.features.screening.apply_momentum_gate`.

    Returns
    -------
    Dict with keys:

    - ``X_tech``             — ``(N, window, 16)`` technical factor windows
    - ``X_sentiment``        — ``(N, window, 768)`` sentiment embedding windows
    - ``X_fundamental``      — ``(N, n_fund [+ 1 if include_momentum_slope])`` fundamental
      snapshots; ``(N, 0)`` only when both *fundamental_df* is ``None`` and
      *include_momentum_slope* is ``False``
    - ``X_sentiment_probs``  — ``(N, 3)`` daily FinBERT softmax probs at each window end
      date; ``(N, 0)`` when *sentiment_df* is ``None`` or has no ``sentiment_probs`` column
    - ``y``                  — ``(N,)`` binary labels
    - ``dates``              — ``(N,)`` date of last day in each window
    """
    # Validate: SMA-60 warmup + window + 2 rows for target computation
    min_rows = 60 + window + 2
    if len(df) < min_rows:
        raise RuntimeError(
            f"Insufficient price history for ticker='{ticker}': need ≥ {min_rows} rows "
            f"(60 indicator warmup + window={window} + 2 target rows), got {len(df)}. "
            "Fetch more data before calling build_dataset."
        )

    close = df["close"]

    # Compute targets on full series before dropping NaN from factors
    targets = compute_targets(close)

    # Compute technical factors (drops first ~60 rows due to lookbacks)
    factors_df = technical.compute(df)

    # Align targets and sentiment with factor dates
    targets = targets.reindex(factors_df.index)
    embeddings = align_sentiment(factors_df.index, sentiment_df, ticker)
    sent_probs = align_sentiment_probs(factors_df.index, sentiment_df, ticker)

    # Drop rows where target is NaN (end of series — no future data)
    valid = targets.notna()
    factors_arr = factors_df[valid].values.astype(np.float32)
    embeddings_arr = embeddings[valid.values]
    sent_probs_arr = sent_probs[valid.values]
    targets_arr = targets[valid].values.astype(np.float32)
    dates = factors_df.index[valid]

    # Align close prices to factor dates for momentum slope computation
    close_arr: np.ndarray | None = None
    if include_momentum_slope:
        close_arr = close.reindex(factors_df.index).ffill()[valid].values.astype(np.float64)

    # Build sliding windows over tech + sentiment
    X_tech, X_sent, y, win_dates = _build_windows(
        factors_arr, embeddings_arr, targets_arr, dates, window
    )

    # Align fundamentals and sentiment probs to window end dates (no time dim)
    X_fund = align_fundamentals(win_dates, fundamental_df)
    X_sprob = _align_snapshot(win_dates, dates.values, sent_probs_arr)

    # Append momentum slope as an extra fundamental-style scalar feature.
    # Uses the last min(20, window) closes of each window so the model can learn
    # whether trend context improves predictions rather than applying a hard gate.
    if include_momentum_slope and close_arr is not None:
        slope_len = min(20, window)
        xs = np.arange(slope_len)
        slopes = np.array(
            [np.polyfit(xs, close_arr[i + window - slope_len : i + window], 1)[0]
             for i in range(len(y))],
            dtype=np.float32,
        ).reshape(-1, 1)
        X_fund = np.concatenate([X_fund, slopes], axis=1)

    n_fund = X_fund.shape[1]
    n_sprob = X_sprob.shape[1]
    logger.info(
        "Built dataset: %d windows, %d tech factors, %d fundamental factors "
        "(%s momentum slope), %d sentiment prob features, window=%d",
        len(y), factors_arr.shape[1], n_fund,
        "incl." if include_momentum_slope else "excl.",
        n_sprob, window,
    )
    return {
        "X_tech": X_tech,
        "X_sentiment": X_sent,
        "X_fundamental": X_fund,
        "X_sentiment_probs": X_sprob,
        "y": y,
        "dates": win_dates,
    }


def make_loaders(
    dataset: FusedDataset,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler | None]:
    """Split chronologically, normalize features, create DataLoaders.

    Both technical factors and fundamental factors are normalized with separate
    ``StandardScaler`` instances fitted on training data only.

    Parameters
    ----------
    dataset:
        Output of :func:`build_dataset`.
    test_frac:
        Fraction of data for the test set (taken from the end).
    val_frac:
        Fraction of (train + val) data reserved for validation.
    batch_size:
        Batch size for all three DataLoaders.

    Returns
    -------
    ``(train_loader, val_loader, test_loader, tech_scaler, fund_scaler)``

    ``fund_scaler`` is ``None`` when the dataset has no fundamental features.
    Both scalers are fitted on training data only and applied to all splits.
    """
    N = len(dataset["y"])
    test_start = int(N * (1 - test_frac))
    val_start = int(test_start * (1 - val_frac))

    splits = {
        "train": slice(0, val_start),
        "val": slice(val_start, test_start),
        "test": slice(test_start, N),
    }

    # Fit StandardScaler on training technical factors only
    X_train_tech = dataset["X_tech"][splits["train"]]
    tech_scaler = StandardScaler()
    n_samples, win, n_feat = X_train_tech.shape
    tech_scaler.fit(X_train_tech.reshape(-1, n_feat))

    # Fit StandardScaler on training fundamental factors (if present)
    X_fund_all = dataset["X_fundamental"]
    n_fund = X_fund_all.shape[1]
    fund_scaler: StandardScaler | None = None
    if n_fund > 0:
        fund_scaler = StandardScaler()
        fund_scaler.fit(X_fund_all[splits["train"]])

    X_sprob_all = dataset["X_sentiment_probs"]

    loaders = {}
    for name, sl in splits.items():
        X_tech = dataset["X_tech"][sl].copy()
        X_sent = dataset["X_sentiment"][sl]
        X_fund = X_fund_all[sl].copy()
        X_sprob = X_sprob_all[sl]
        y = dataset["y"][sl]

        # Normalize technical factors
        n, w, f = X_tech.shape
        X_tech = (
            tech_scaler.transform(X_tech.reshape(-1, f))
            .reshape(n, w, f)
            .astype(np.float32)
        )

        # Normalize fundamental factors
        if fund_scaler is not None:
            X_fund = fund_scaler.transform(X_fund).astype(np.float32)

        # Sentiment probs are already in [0, 1] — no normalization needed
        ds = FusedStockDataset(X_tech, X_sent, X_fund, X_sprob, y)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        val_start,
        test_start - val_start,
        N - test_start,
    )
    return loaders["train"], loaders["val"], loaders["test"], tech_scaler, fund_scaler


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _align_snapshot(
    win_dates: np.ndarray,
    all_dates: np.ndarray,
    arr: np.ndarray,
) -> np.ndarray:
    """Pick the row from *arr* whose index in *all_dates* matches each window end date.

    Used to align per-day snapshots (e.g. sentiment probs) to window end dates.
    Returns ``(N_windows, arr.shape[1])`` — same number of features as *arr*.
    Returns ``(N_windows, 0)`` when *arr* has no feature columns.
    """
    if arr.shape[1] == 0:
        return np.zeros((len(win_dates), 0), dtype=np.float32)

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    out = np.zeros((len(win_dates), arr.shape[1]), dtype=np.float32)
    for i, wd in enumerate(win_dates):
        idx = date_to_idx.get(wd)
        if idx is not None:
            out[i] = arr[idx]
    return out


def _build_windows(
    factors: np.ndarray,
    embeddings: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows from aligned arrays.

    For each window ``[i, ..., i+window-1]``, the target is the target
    of the last day (index ``i+window-1``).
    """
    n = len(targets) - window + 1
    if n <= 0:
        raise RuntimeError(
            f"Not enough data for window={window}: only {len(targets)} rows"
        )

    # sliding_window_view produces (n_windows, n_features, window) for 2D input;
    # transpose to (n_windows, window, n_features)
    X_tech = sliding_window_view(factors, window, axis=0).transpose(0, 2, 1).copy()
    X_sent = sliding_window_view(embeddings, window, axis=0).transpose(0, 2, 1).copy()

    y = targets[window - 1 :]
    win_dates = dates[window - 1 :].values

    return X_tech, X_sent, y, win_dates
