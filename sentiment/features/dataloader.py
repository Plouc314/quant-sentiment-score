
"""Build fused technical + sentiment datasets for stock prediction."""

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

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


class FusedDataset(TypedDict):
    """Output of :func:`build_dataset`."""

    X_tech: np.ndarray
    """``(N, window, 16)`` technical factor windows."""
    X_sentiment: np.ndarray
    """``(N, window, 768)`` sentiment embedding windows."""
    y: np.ndarray
    """``(N,)`` binary labels."""
    dates: np.ndarray
    """``(N,)`` date of last day in each window."""


class FusedStockDataset(Dataset):
    """PyTorch Dataset for (tech_factors, sentiment_embedding, target) triplets."""

    def __init__(
        self,
        X_tech: np.ndarray,
        X_sentiment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.X_tech = torch.tensor(X_tech, dtype=torch.float32)
        self.X_sentiment = torch.tensor(X_sentiment, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_tech)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_tech[i], self.X_sentiment[i], self.y[i]


def compute_targets(close: pd.Series) -> pd.Series:
    """Binary target for day *t*: ``1`` if ``close[t+2] > close[t-1]``, else ``0``.

    Uses positional shifts so *t+2* and *t-1* refer to trading days.
    The result has NaN at the first and last few entries.
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
    for _, row in ticker_df.iterrows():
        dt = pd.Timestamp(row["date"]).date()
        lookup[dt] = row["embedding"]

    for i, ts in enumerate(index):
        emb = lookup.get(ts.date())
        if emb is not None:
            result[i] = emb

    return result


def build_dataset(
    df: pd.DataFrame,
    technical: TechnicalFactors,
    sentiment_df: pd.DataFrame | None = None,
    ticker: str = "",
    window: int = 20,
) -> FusedDataset:
    """Full pipeline: OHLCV → factors + targets + aligned sentiment → sliding windows.

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

    Returns
    -------
    Dict with keys:

    - ``X_tech``      — ``(N, window, 16)`` technical factor windows
    - ``X_sentiment`` — ``(N, window, 768)`` sentiment embedding windows
    - ``y``           — ``(N,)`` binary labels
    - ``dates``       — ``(N,)`` date of last day in each window
    """
    close = df["close"]

    # Compute targets on full series before dropping NaN from factors
    targets = compute_targets(close)

    # Compute technical factors (drops first ~60 rows due to lookbacks)
    factors_df = technical.compute(df)

    # Align targets and sentiment with factor dates
    targets = targets.reindex(factors_df.index)
    embeddings = align_sentiment(factors_df.index, sentiment_df, ticker)

    # Drop rows where target is NaN (end of series — no future data)
    valid = targets.notna()
    factors_arr = factors_df[valid].values.astype(np.float32)
    embeddings_arr = embeddings[valid.values]
    targets_arr = targets[valid].values.astype(np.float32)
    dates = factors_df.index[valid]

    # Build sliding windows
    X_tech, X_sent, y, win_dates = _build_windows(
        factors_arr, embeddings_arr, targets_arr, dates, window
    )

    logger.info(
        "Built dataset: %d windows, %d factors, window=%d",
        len(y), factors_arr.shape[1], window,
    )
    return {"X_tech": X_tech, "X_sentiment": X_sent, "y": y, "dates": win_dates}


def make_loaders(
    dataset: FusedDataset,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """Split chronologically, normalize technical factors, create DataLoaders.

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
    ``(train_loader, val_loader, test_loader, scaler)``

    The scaler is fitted on training data only and applied to all splits.
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
    scaler = StandardScaler()
    n_samples, win, n_feat = X_train_tech.shape
    scaler.fit(X_train_tech.reshape(-1, n_feat))

    loaders = {}
    for name, sl in splits.items():
        X_tech = dataset["X_tech"][sl].copy()
        X_sent = dataset["X_sentiment"][sl]
        y = dataset["y"][sl]

        # Normalize technical factors using training scaler
        n, w, f = X_tech.shape
        X_tech = (
            scaler.transform(X_tech.reshape(-1, f))
            .reshape(n, w, f)
            .astype(np.float32)
        )

        ds = FusedStockDataset(X_tech, X_sent, y)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        val_start,
        test_start - val_start,
        N - test_start,
    )
    return loaders["train"], loaders["val"], loaders["test"], scaler


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


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
