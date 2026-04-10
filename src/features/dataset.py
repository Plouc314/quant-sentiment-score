from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .technical import TechnicalFactors
from ..training import ComputeConfig, Split, TrainingConfig

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768


class StockDataset:
    """Computes and holds the full feature matrix for one stock symbol.

    All arrays are stored flat (one row per trading day for time-series arrays,
    one row per window for fundamental / target arrays) to avoid the ``window``-fold
    memory expansion that pre-windowing would cause.  Windows are materialised
    lazily inside :class:`DataLoaderBuilder`.

    Parameters
    ----------
    symbol:
        Ticker symbol (used only for logging).
    price_df:
        OHLCV + VWAP DataFrame with DatetimeIndex, sorted ascending.
    sentiment_df:
        Daily sentiment aggregates produced by ``SentimentPipeline``.  ``None``
        disables sentiment — zero vectors are used for all days.
    fundamental_df:
        Time-indexed DataFrame of fundamental snapshots from
        ``FundamentalsRepository.load_history()``.  ``None`` disables fundamentals.
    window:
        Sliding window size in trading days.
    include_momentum_slope:
        Append the 20-day linear-regression slope of closing prices as an extra
        scalar in the fundamental feature vector.
    """

    def __init__(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame | None = None,
        fundamental_df: pd.DataFrame | None = None,
        window: int = 64,
        include_momentum_slope: bool = True,
    ) -> None:
        self.symbol = symbol
        self.window = window

        min_rows = 60 + window + 2
        if len(price_df) < min_rows:
            raise RuntimeError(
                f"{symbol}: need ≥ {min_rows} rows (60 warmup + window={window} + 2 target rows),"
                f" got {len(price_df)}"
            )

        targets = _compute_targets(price_df["close"])
        factors_df = TechnicalFactors().compute(price_df)

        targets = targets.reindex(factors_df.index)
        embeddings = _align_embeddings(factors_df.index, sentiment_df, symbol)
        sent_probs = _align_sent_probs(factors_df.index, sentiment_df, symbol)

        valid = targets.notna()
        self.X_tech: np.ndarray  = factors_df[valid].values.astype(np.float32)
        self.X_sent: np.ndarray  = embeddings[valid.values]
        self.X_sprob: np.ndarray = sent_probs[valid.values]
        targets_arr = targets[valid].values.astype(np.float32)
        factor_dates = factors_df.index[valid]

        T = len(targets_arr)
        N = T - window + 1
        if N <= 0:
            raise RuntimeError(f"{symbol}: not enough data for window={window} ({T} valid rows)")

        self.y: np.ndarray     = targets_arr[window - 1:]
        self.dates: np.ndarray = factor_dates[window - 1:].values

        self.X_fund: np.ndarray = _align_fundamentals(self.dates, fundamental_df)

        if include_momentum_slope:
            close_arr = price_df["close"].reindex(factors_df.index).ffill()[valid].values.astype(np.float64)
            slope_len = min(20, window)
            xs = np.arange(slope_len)
            slopes = np.array(
                [np.polyfit(xs, close_arr[i + window - slope_len: i + window], 1)[0] for i in range(N)],
                dtype=np.float32,
            ).reshape(-1, 1)
            self.X_fund = np.concatenate([self.X_fund, slopes], axis=1)

        logger.info(
            "%s: %d windows, %d tech, %d fund, %d sent_prob features",
            symbol, N, self.X_tech.shape[1], self.X_fund.shape[1], self.X_sprob.shape[1],
        )

    @property
    def n_windows(self) -> int:
        return len(self.y)

    @property
    def n_fundamentals(self) -> int:
        return self.X_fund.shape[1]

    @property
    def n_sentiment_probs(self) -> int:
        return self.X_sprob.shape[1]


class DataLoaderBuilder:
    """Splits symbols by cutoff, fits scalers on train data, and creates DataLoaders.

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
        self._tech_scaler: StandardScaler | None = None
        self._fund_scaler: StandardScaler | None = None

    def build(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Compute temporal splits, fit scalers on train, return DataLoaders.

        Must be called before :meth:`build_held_out_loader` or accessing
        :attr:`tech_scaler` / :attr:`fund_scaler`.

        Returns
        -------
        ``(train_loader, val_loader, test_loader)``
        """
        val_months  = self._split.val_months
        batch_size  = self._config.batch_size
        num_workers = self._compute.num_workers

        # Pass 1 — compute per-symbol split indices and collect train rows for scaling
        split_indices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        tech_train_rows: list[np.ndarray] = []
        fund_train_rows: list[np.ndarray] = []

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
                flat_end = int(train_idx[-1]) + ds.window
                tech_train_rows.append(ds.X_tech[:flat_end])
                fund_train_rows.append(ds.X_fund[train_idx])

        if not tech_train_rows:
            raise RuntimeError("No training data found — check that datasets and split match")

        self._tech_scaler = StandardScaler()
        self._tech_scaler.fit(np.concatenate(tech_train_rows, axis=0))

        n_fund = self.n_fundamentals
        if n_fund > 0 and fund_train_rows:
            self._fund_scaler = StandardScaler()
            self._fund_scaler.fit(np.concatenate(fund_train_rows, axis=0))

        # Pass 2 — build _LazyDataset instances per symbol/split
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
        """Build a DataLoader for held-out symbols using the fitted scalers.

        Must be called after :meth:`build`.
        """
        if self._tech_scaler is None:
            raise RuntimeError("Call build() before build_held_out_loader()")

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

    @property
    def tech_scaler(self) -> StandardScaler:
        if self._tech_scaler is None:
            raise RuntimeError("Call build() first")
        return self._tech_scaler

    @property
    def fund_scaler(self) -> StandardScaler | None:
        return self._fund_scaler

    @property
    def n_fundamentals(self) -> int:
        for s in self._split.train_symbols:
            if s in self._datasets:
                return self._datasets[s].n_fundamentals
        return 0

    @property
    def n_sentiment_probs(self) -> int:
        for s in self._split.train_symbols:
            if s in self._datasets:
                return self._datasets[s].n_sentiment_probs
        return 0

    def _make_lazy(self, ds: StockDataset, indices: np.ndarray) -> _LazyDataset:
        X_tech_t = torch.tensor(
            self._tech_scaler.transform(ds.X_tech).astype(np.float32)  # type: ignore[union-attr]
        ).share_memory_()
        X_sent_t  = torch.tensor(ds.X_sent).share_memory_()
        X_sprob_t = torch.tensor(ds.X_sprob).share_memory_()

        X_fund = ds.X_fund.copy()
        if self._fund_scaler is not None:
            X_fund = self._fund_scaler.transform(X_fund).astype(np.float32)

        return _LazyDataset(X_tech_t, X_sent_t, X_sprob_t, X_fund, ds.y, ds.window, indices)


# ------------------------------------------------------------------
# Private: lazy windowing PyTorch Dataset
# ------------------------------------------------------------------


class _LazyDataset(Dataset):
    """Materialises sliding windows on-the-fly from shared flat tensors."""

    def __init__(
        self,
        X_tech:  torch.Tensor,
        X_sent:  torch.Tensor,
        X_sprob: torch.Tensor,
        X_fund:  np.ndarray,
        y:       np.ndarray,
        window:  int,
        indices: np.ndarray,
    ) -> None:
        self.X_tech  = X_tech
        self.X_sent  = X_sent
        self.X_sprob = X_sprob
        self.X_fund  = torch.tensor(X_fund[indices], dtype=torch.float32)
        self.y       = torch.tensor(y[indices],      dtype=torch.float32)
        self.window  = window
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        wi = int(self.indices[i])
        return (
            self.X_tech [wi: wi + self.window],
            self.X_sent [wi: wi + self.window],
            self.X_fund [i],
            self.X_sprob[wi: wi + self.window],
            self.y[i],
        )


# ------------------------------------------------------------------
# Private: feature alignment helpers
# ------------------------------------------------------------------


def _compute_targets(close: pd.Series) -> pd.Series:
    """Binary target: 1 if close[t+2] > close[t-1] (3-day executable signal)."""
    future = close.shift(-2)
    past   = close.shift(1)
    target = (future > past).astype(np.float32)
    target[future.isna() | past.isna()] = np.nan
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


def _align_sent_probs(
    index: pd.DatetimeIndex,
    sentiment_df: pd.DataFrame | None,
    symbol: str,
) -> np.ndarray:
    if sentiment_df is None or sentiment_df.empty or "sentiment_probs" not in sentiment_df.columns:
        return np.zeros((len(index), 0), dtype=np.float32)
    rows = sentiment_df[sentiment_df["ticker"] == symbol]
    if rows.empty:
        return np.zeros((len(index), 3), dtype=np.float32)
    lookup = {pd.Timestamp(d).date(): p for d, p in zip(rows["date"], rows["sentiment_probs"])}
    result = np.zeros((len(index), 3), dtype=np.float32)
    for i, ts in enumerate(index):
        p = lookup.get(ts.date())
        if p is not None:
            result[i] = p
    return result


def _align_fundamentals(win_dates: np.ndarray, fund_df: pd.DataFrame | None) -> np.ndarray:
    if fund_df is None or fund_df.empty:
        return np.zeros((len(win_dates), 0), dtype=np.float32)
    dates_idx = pd.DatetimeIndex(win_dates)
    reindexed = fund_df.reindex(dates_idx.union(fund_df.index)).sort_index()
    reindexed = reindexed.ffill().reindex(dates_idx)
    return reindexed.fillna(0.0).values.astype(np.float32)


def _make_loader(
    lazy_list: list[_LazyDataset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        ConcatDataset(lazy_list) if lazy_list else ConcatDataset([]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0 and bool(lazy_list)),
    )
