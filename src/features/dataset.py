from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .technical import TechnicalFactors
from ..training import ComputeConfig, Split, TrainingConfig

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768


class StockDataset:
    """Computes and holds the full feature matrix for one stock symbol.

    All arrays are stored flat (one row per trading day) to avoid the
    ``window``-fold memory expansion that pre-windowing would cause.  Windows
    are materialised lazily inside :class:`DataLoaderBuilder`.

    Parameters
    ----------
    symbol:
        Ticker symbol (used only for logging).
    price_df:
        OHLCV + VWAP DataFrame with DatetimeIndex, sorted ascending.
    sentiment_df:
        Daily sentiment aggregates produced by ``SentimentPipeline``.  ``None``
        disables sentiment — zero vectors are used for all days.
    window:
        Sliding window size in trading days.
    """

    def __init__(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
        window: int = 64,
        horizon: int = 3,
        target_threshold: float | None = None,
        has_news_feature: bool = False,
    ) -> None:
        self.symbol  = symbol
        self.window  = window
        self.horizon = horizon

        # Warmup: longest indicator is macd_30_60_30 → slow=60 + signal=30 - 1 = 89 NaN rows.
        min_rows = 90 + window + horizon
        if len(price_df) < min_rows:
            raise RuntimeError(
                f"{symbol}: need ≥ {min_rows} rows (90 warmup + window={window} + horizon={horizon}),"
                f" got {len(price_df)}"
            )

        targets = _compute_targets(price_df["close"], horizon=horizon, threshold=target_threshold)
        factors_df = TechnicalFactors().compute(price_df)

        targets = targets.reindex(factors_df.index, fill_value=-1)
        embeddings = _align_embeddings(factors_df.index, sentiment_df, symbol)

        valid = targets >= 0  # drop sentinel (-1) for missing future close
        tech_values = factors_df[valid].values.astype(np.float32)

        if has_news_feature:
            has_news = (np.linalg.norm(embeddings, axis=1) > 0).astype(np.float32)
            has_news = has_news[valid.values].reshape(-1, 1)
            tech_values = np.hstack([tech_values, has_news])

        self.X_tech: np.ndarray = tech_values
        self.X_sent: np.ndarray = embeddings[valid.values]
        targets_arr = targets[valid].values.astype(np.int64)
        factor_dates = factors_df.index[valid]

        T = len(targets_arr)
        N = T - window + 1
        if N <= 0:
            raise RuntimeError(f"{symbol}: not enough data for window={window} ({T} valid rows)")

        self.y: np.ndarray     = targets_arr[window - 1:]
        self.dates: np.ndarray = factor_dates[window - 1:].values

        logger.info("%s: %d windows, %d tech features", symbol, N, self.X_tech.shape[1])

    @property
    def n_windows(self) -> int:
        return len(self.y)

    @property
    def anchor_has_sentiment(self) -> np.ndarray:
        """Boolean mask over windows: ``True`` when the anchor day has a non-zero embedding."""
        anchor_embs = self.X_sent[self.window - 1:]  # (n_windows, 768)
        return np.linalg.norm(anchor_embs, axis=1) > 0


class DataLoaderBuilder:
    """Splits symbols by cutoff and creates DataLoaders.

    Technical features are min-max normalized **per window** inside the
    Dataset (see :class:`_LazyDataset`) — matching the reference
    implementation — so no global scaler is fitted here.

    Parameters
    ----------
    datasets:
        Mapping of ``{symbol: StockDataset}`` for all symbols (train + held-out).
        Symbols missing from this dict are silently skipped.
    split:
        Train/held-out assignment and per-symbol cutoff dates.
    config:
        Training hyperparameters (batch_size, …).
    compute_config:
        Hardware configuration (num_workers, …).
    """

    def __init__(
        self,
        datasets: dict[str, StockDataset],
        split: Split,
        config: TrainingConfig,
        compute_config: ComputeConfig,
    ) -> None:
        self._datasets = datasets
        self._split = split
        self._config = config
        self._compute = compute_config

    def build(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Compute temporal splits and return DataLoaders.

        Returns
        -------
        ``(train_loader, val_loader, test_loader)``
        """
        val_months  = self._split.val_months
        batch_size  = self._config.batch_size
        num_workers = self._compute.num_workers

        split_indices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        any_train = False

        for symbol in self._split.train_symbols:
            ds = self._datasets.get(symbol)
            if ds is None:
                continue
            cutoff    = self._split.cutoffs[symbol]
            val_start = cutoff - pd.DateOffset(months=val_months)
            dates     = pd.DatetimeIndex(ds.dates)

            train_idx = np.where(dates < val_start)[0]
            val_idx   = np.where((dates >= val_start) & (dates < cutoff))[0]
            test_idx  = np.where(dates >= cutoff)[0]
            split_indices[symbol] = (train_idx, val_idx, test_idx)
            if len(train_idx) > 0:
                any_train = True

        if not any_train:
            raise RuntimeError("No training data found — check that datasets and split match")

        train_lazy: list[_LazyDataset] = []
        val_lazy:   list[_LazyDataset] = []
        test_lazy:  list[_LazyDataset] = []

        for symbol in self._split.train_symbols:
            ds = self._datasets.get(symbol)
            if ds is None or symbol not in split_indices:
                continue
            train_idx, val_idx, test_idx = split_indices[symbol]
            for idx, bucket in ((train_idx, train_lazy), (val_idx, val_lazy), (test_idx, test_lazy)):
                if len(idx) > 0:
                    bucket.append(self._make_lazy(ds, idx))

        for name, bucket in (("train", train_lazy), ("val", val_lazy), ("test", test_lazy)):
            logger.info("DataLoaderBuilder — %s: %d windows", name, sum(len(d) for d in bucket))

        return (
            _make_loader(train_lazy, batch_size, shuffle=True,  num_workers=num_workers),
            _make_loader(val_lazy,   batch_size, shuffle=False, num_workers=num_workers),
            _make_loader(test_lazy,  batch_size, shuffle=False, num_workers=num_workers),
        )

    def build_held_out_loader(self, batch_size: int | None = None) -> DataLoader:
        """Build a DataLoader for held-out symbols."""
        bs          = batch_size or self._config.batch_size
        num_workers = self._compute.num_workers
        lazy_list: list[_LazyDataset] = []

        for symbol in self._split.held_out_symbols:
            ds = self._datasets.get(symbol)
            if ds is None:
                continue
            lazy_list.append(self._make_lazy(ds, np.arange(ds.n_windows)))

        logger.info("DataLoaderBuilder — held-out: %d windows", sum(len(d) for d in lazy_list))
        return _make_loader(lazy_list, bs, shuffle=False, num_workers=num_workers)

    def _make_lazy(self, ds: StockDataset, indices: np.ndarray) -> _LazyDataset:
        X_tech_t = torch.tensor(ds.X_tech).share_memory_()
        X_sent_t = torch.tensor(ds.X_sent).share_memory_()
        return _LazyDataset(X_tech_t, X_sent_t, ds.y, ds.window, indices)


def build_per_stock_loaders(
    ds: StockDataset,
    cutoff: str = "2023-06-01",
    val_frac: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for a single stock with sentiment gating.

    Only windows whose anchor day has a non-zero sentiment embedding are
    included.  Pre-cutoff windows are split chronologically into train
    (first 90 %) and val (last 10 %).  Post-cutoff windows form the test set.
    Validation and test loaders use full-batch evaluation (matching the
    reference implementation).

    Returns ``(train_loader, val_loader, test_loader)``.
    """
    mask = ds.anchor_has_sentiment
    dates = pd.DatetimeIndex(ds.dates)
    cutoff_ts = pd.Timestamp(cutoff)

    valid_idx = np.where(mask)[0]
    valid_dates = dates[valid_idx]

    pre = valid_idx[valid_dates < cutoff_ts]
    post = valid_idx[valid_dates >= cutoff_ts]

    n_val = max(1, int(len(pre) * val_frac))
    train_idx = pre[:-n_val]
    val_idx = pre[-n_val:]
    test_idx = post

    logger.info(
        "%s — sentiment-gated: train=%d, val=%d, test=%d (of %d total windows)",
        ds.symbol, len(train_idx), len(val_idx), len(test_idx), ds.n_windows,
    )

    X_tech_t = torch.tensor(ds.X_tech).share_memory_()
    X_sent_t = torch.tensor(ds.X_sent).share_memory_()

    def _make(indices: np.ndarray, shuffle: bool) -> DataLoader:
        if len(indices) == 0:
            return DataLoader(ConcatDataset([]), batch_size=batch_size)
        lazy = _LazyDataset(X_tech_t, X_sent_t, ds.y, ds.window, indices)
        return DataLoader(
            lazy,
            batch_size=batch_size if shuffle else len(lazy),
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=num_workers,
        )

    return _make(train_idx, True), _make(val_idx, False), _make(test_idx, False)


# ------------------------------------------------------------------
# Private: lazy windowing PyTorch Dataset
# ------------------------------------------------------------------


class _LazyDataset(Dataset):
    """Materialises sliding windows on-the-fly from shared flat tensors.

    Technical features are min-max normalized per window (column-wise) to
    match the reference implementation — each window is self-contained so
    the model sees relative shapes, not absolute levels.
    """

    def __init__(
        self,
        X_tech:  torch.Tensor,
        X_sent:  torch.Tensor,
        y:       np.ndarray,
        window:  int,
        indices: np.ndarray,
    ) -> None:
        self.X_tech  = X_tech
        self.X_sent  = X_sent
        self.y       = torch.tensor(y[indices], dtype=torch.long)
        self.window  = window
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wi = int(self.indices[i])
        tech_win = self.X_tech[wi: wi + self.window]
        mn = tech_win.min(dim=0, keepdim=True).values
        mx = tech_win.max(dim=0, keepdim=True).values
        tech_win = (tech_win - mn) / (mx - mn).clamp(min=1e-8)
        return tech_win, self.X_sent[wi: wi + self.window], self.y[i]


# ------------------------------------------------------------------
# Private: feature alignment helpers
# ------------------------------------------------------------------


def _compute_targets(
    close: pd.Series,
    horizon: int = 3,
    threshold: float | None = None,
) -> pd.Series:
    """Binary target: down (0), up (1).

    At anchor day ``t``: label is ``1`` if ``close[t + horizon] > close[t]``.
    When *threshold* is set, label is ``1`` if the return
    ``(close[t + horizon] - close[t]) / close[t]`` exceeds *threshold*
    (filters out noise near the decision boundary).

    The last ``horizon`` rows have no future close and are flagged with ``-1``
    so the caller can drop them.
    """
    future = close.shift(-horizon)
    if threshold is not None:
        ret = (future - close) / close
        target = (ret > threshold).astype(np.int64)
    else:
        target = (future > close).astype(np.int64)
    target[future.isna()] = -1     # sentinel for missing future
    return target


def _align_embeddings(
    index: pd.DatetimeIndex,
    sentiment_df: pd.DataFrame | None,
    symbol: str,
) -> np.ndarray:
    result = np.zeros((len(index), _EMBEDDING_DIM), dtype=np.float32)
    if sentiment_df is None or sentiment_df.empty:
        return result
    rows = sentiment_df[sentiment_df["ticker"] == symbol]
    if rows.empty:
        return result
    lookup = {pd.Timestamp(d).date(): emb for d, emb in zip(rows["date"], rows["embedding"])}
    for i, ts in enumerate(index):
        emb = lookup.get(ts.date())
        if emb is not None:
            result[i] = emb
    return result



def _make_loader(
    lazy_list: list[_LazyDataset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    if not lazy_list:
        from torch.utils.data import TensorDataset
        return DataLoader(TensorDataset(), batch_size=batch_size)
    return DataLoader(
        ConcatDataset(lazy_list),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
