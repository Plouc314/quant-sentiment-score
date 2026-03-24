from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


def load_data(cache, symbol: str, year: int) -> pd.DataFrame:
    df = cache.load(symbol, year)
    if df is None or df.empty:
        raise ValueError(f"No data found for {symbol} in {year}")
    features = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    available = [f for f in features if f in df.columns]
    return df[available].dropna()


def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Compute 7 scale-free features and the next-bar log-return target.

    Features (all dimensionless, no scaler needed):

    - ``log_return``        — log(close / prev_close)
    - ``gap``               — log(open / prev_close)
    - ``high_low_pct``      — (high - low) / close
    - ``open_close_pct``    — (close - open) / open
    - ``volume_ratio``      — volume / rolling_20_mean(volume)
    - ``trade_count_ratio`` — trade_count / rolling_20_mean(trade_count);
                              zero-filled when ``trade_count`` column is absent
    - ``vwap_dev``          — (vwap - close) / close; zero-filled when ``vwap`` column is absent

    Returns
    -------
    feat : DataFrame with the 7 feature columns, same index as *df*.
    target : Series where ``target[t] = log_return[t+1]`` (next bar's return).
        Last row is NaN — callers must drop it before building sequences.
    """
    feat = pd.DataFrame(index=df.index)
    feat["log_return"] = np.log(df["close"] / df["close"].shift(1))
    feat["gap"] = np.log(df["open"] / df["close"].shift(1))
    feat["high_low_pct"] = (df["high"] - df["low"]) / df["close"]
    feat["open_close_pct"] = (df["close"] - df["open"]) / df["open"]

    vol_ma = df["volume"].rolling(20).mean()
    feat["volume_ratio"] = df["volume"] / vol_ma

    if "trade_count" in df.columns:
        tc_ma = df["trade_count"].rolling(20).mean()
        feat["trade_count_ratio"] = df["trade_count"] / tc_ma
    else:
        feat["trade_count_ratio"] = 0.0

    if "vwap" in df.columns:
        feat["vwap_dev"] = (df["vwap"] - df["close"]) / df["close"]
    else:
        feat["vwap_dev"] = 0.0

    # Target: the log_return of the *next* bar
    target = feat["log_return"].shift(-1)

    return feat, target


def preprocess_data(df: pd.DataFrame, seq_length: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute scale-free features and build sliding-window sequences.

    Parameters
    ----------
    df:
        OHLCV + vwap DataFrame (output of :func:`load_data`).
    seq_length:
        Number of bars per input window.

    Returns
    -------
    X : ``(N, seq_length, 7)`` float32 array of feature windows.
    y : ``(N, 1)`` float32 array of next-bar log-return targets.
    """
    feat, target = compute_features(df)

    valid = feat.notna().all(axis=1) & target.notna()
    feat = feat[valid]
    target = target[valid]

    n_dropped = (~valid).sum()
    if n_dropped:
        logger.debug("preprocess_data: dropped %d NaN rows (rolling warmup)", n_dropped)

    X, y = create_sequences(feat.values.astype(np.float32), target.values.astype(np.float32), seq_length)
    return X, y


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows from aligned feature and target arrays.

    For window starting at *i*:
    - X = ``data[i : i + seq_length]``  — the seq_length-bar feature window
    - y = ``targets[i + seq_length - 1]`` — the next-bar log-return (target[t]
      is already log_return[t+1], so this gives the return right after the window)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(targets[i + seq_length - 1])
    return np.array(xs), np.array(ys).reshape(-1, 1)


def load_all_symbols(
    cache,
    symbols: list[str],
    year: int,
    seq_length: int = 20,
    train_frac: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, preprocess, and split data for multiple symbols.

    Each symbol is split chronologically (train/test) before pooling, so
    future bars of one stock never appear in the training set of another.

    Parameters
    ----------
    cache:
        :class:`~sentiment.sources.cache.MarketDataCache` instance.
    symbols:
        List of ticker symbols to load.
    year:
        Calendar year to load from cache.
    seq_length:
        Sliding window length passed to :func:`preprocess_data`.
    train_frac:
        Fraction of each symbol's sequences used for training.

    Returns
    -------
    ``(X_train, y_train, X_test, y_test)`` — concatenated across all symbols.

    Raises
    ------
    RuntimeError
        When no symbol produces any usable sequences.
    """
    X_train_all: list[np.ndarray] = []
    y_train_all: list[np.ndarray] = []
    X_test_all: list[np.ndarray] = []
    y_test_all: list[np.ndarray] = []

    for symbol in symbols:
        try:
            df = load_data(cache, symbol, year)
        except ValueError as e:
            logger.warning("load_all_symbols: skipping %s — %s", symbol, e)
            continue

        try:
            X, y = preprocess_data(df, seq_length)
        except Exception as e:
            logger.warning("load_all_symbols: skipping %s (preprocess failed) — %s", symbol, e)
            continue

        if len(X) < 2:
            logger.warning("load_all_symbols: skipping %s — only %d sequences", symbol, len(X))
            continue

        split = int(len(X) * train_frac)
        X_train_all.append(X[:split])
        y_train_all.append(y[:split])
        X_test_all.append(X[split:])
        y_test_all.append(y[split:])
        logger.info("load_all_symbols: %s — %d train, %d test sequences", symbol, split, len(X) - split)

    if not X_train_all:
        raise RuntimeError(f"No usable sequences for any symbol in {symbols}")

    return (
        np.concatenate(X_train_all),
        np.concatenate(y_train_all),
        np.concatenate(X_test_all),
        np.concatenate(y_test_all),
    )


def get_stock_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    dataset = StockDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
