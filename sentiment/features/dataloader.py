
"""Build fused technical + sentiment + fundamental datasets for stock prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import TypedDict

from .technical import TechnicalFactors
from .screening import momentum_slope

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


class FusedDataset(TypedDict):
    """Output of :func:`build_dataset`.

    Time-series arrays (``X_tech``, ``X_sentiment``, ``X_sentiment_probs``) are
    stored **flat** — one row per trading day — to avoid the 64× memory expansion
    that windowing would cause.  ``X_fundamental``, ``y``, and ``dates`` are
    aligned to window *end* dates and have ``N = T - window + 1`` rows.

    Window ``i`` (0-indexed) corresponds to:

    - flat slice ``[i : i + window]`` for ``X_tech``, ``X_sentiment``, ``X_sentiment_probs``
    - row ``i`` for ``X_fundamental``, ``y``, ``dates``
    """

    X_tech: np.ndarray
    """``(T, 16)`` flat technical factor rows."""
    X_sentiment: np.ndarray
    """``(T, 768)`` flat sentiment embedding rows."""
    X_fundamental: np.ndarray
    """``(N, n_fund)`` fundamental snapshots at each window end date.
    Shape is ``(N, 0)`` when no fundamental data is provided."""
    X_sentiment_probs: np.ndarray
    """``(T, 3)`` flat daily FinBERT softmax probs — ``[P(pos), P(neg), P(neutral)]``.
    Zero vector for days without news.  Shape is ``(T, 0)`` when no sentiment
    data is provided."""
    y: np.ndarray
    """``(N,)`` binary labels."""
    dates: np.ndarray
    """``(N,)`` date of last day in each window."""
    window: int
    """Sliding window size used to build this dataset."""


class LazyFusedDataset(Dataset):
    """PyTorch Dataset that slices windows on-the-fly to avoid memory expansion.

    The flat time-series tensors (``X_tech``, ``X_sentiment``,
    ``X_sentiment_probs``) are shared across all splits of a symbol — only one
    copy per symbol lives in memory regardless of how many splits reference it.
    Windows are materialised as tensor views inside ``__getitem__``.

    Parameters
    ----------
    X_tech:
        Pre-scaled flat tensor of shape ``(T, n_factors)``.  Shared across splits.
    X_sent:
        Flat tensor of shape ``(T, sent_dim)``.  Shared across splits.
    X_sprob:
        Flat tensor of shape ``(T, n_sprob)`` or ``(T, 0)``.  Shared across splits.
    X_fund:
        ``(N, n_fund)`` numpy array — indexed by *indices* at construction time.
    y:
        ``(N,)`` numpy array — indexed by *indices* at construction time.
    window:
        Sliding window size.
    indices:
        1-D array of window indices (into the flat arrays) that this split exposes.
        Window index ``wi`` maps to flat slice ``[wi : wi + window]``.
    """

    def __init__(
        self,
        X_tech: torch.Tensor,
        X_sent: torch.Tensor,
        X_sprob: torch.Tensor,
        X_fund: np.ndarray,
        y: np.ndarray,
        window: int,
        indices: np.ndarray,
    ) -> None:
        self.X_tech = X_tech
        self.X_sent = X_sent
        self.X_sprob = X_sprob
        # Fund and y are small — pre-index so __getitem__ stays O(1)
        self.X_fund = torch.tensor(X_fund[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32)
        self.window = window
        self.indices = indices  # numpy int array

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, i: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wi = int(self.indices[i])
        return (
            self.X_tech[wi : wi + self.window],   # (window, n_factors) — view
            self.X_sent[wi : wi + self.window],    # (window, sent_dim) — view
            self.X_fund[i],                        # (n_fund,)
            self.X_sprob[wi : wi + self.window],   # (window, n_sprob) — view
            self.y[i],                             # scalar
        )


def compute_targets(close: pd.Series) -> pd.Series:
    """Binary target for day *t*: ``1`` if ``close[t+1] > close[t-1]``, else ``0``.

    The model runs overnight using data through close of *t-1*, so the signal
    is ready before market open on day *t*:

    - ``t-1`` — yesterday's close: the last confirmed price available when the
      signal is generated
    - ``t+1`` — next day's close: buy at open of *t*, sell at close of *t+1*

    Concretely: a prediction translates to "buy at open of t, sell at close of t+1."

    All shifts are positional (trading days, not calendar days).
    NaN at the first row (no t-1) and last row (no t+1).
    """
    future = close.shift(-1)
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

    Time-series arrays are returned **flat** (one row per trading day) rather
    than pre-windowed.  This avoids the ``window``-fold memory expansion that
    windowing ``(T, 768)`` embeddings would otherwise cause.  Windows are
    materialised lazily inside :class:`LazyFusedDataset`.

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
    :class:`FusedDataset` with keys:

    - ``X_tech``             — ``(T, 16)`` flat technical factor rows
    - ``X_sentiment``        — ``(T, 768)`` flat sentiment embedding rows
    - ``X_fundamental``      — ``(N, n_fund [+ 1 if include_momentum_slope])``
      fundamental snapshots at window end dates; ``(N, 0)`` only when both
      *fundamental_df* is ``None`` and *include_momentum_slope* is ``False``
    - ``X_sentiment_probs``  — ``(T, 3)`` flat daily FinBERT softmax probs;
      ``(T, 0)`` when *sentiment_df* is ``None`` or has no ``sentiment_probs`` column
    - ``y``                  — ``(N,)`` binary labels
    - ``dates``              — ``(N,)`` date of last day in each window
    - ``window``             — sliding window size (int)

    where ``T`` is the number of valid trading days after indicator warmup and
    ``N = T - window + 1`` is the number of windows.
    """
    # Validate: SMA-60 warmup + window + 1 row for target computation
    min_rows = 60 + window + 1
    if len(df) < min_rows:
        raise RuntimeError(
            f"Insufficient price history for ticker='{ticker}': need ≥ {min_rows} rows "
            f"(60 indicator warmup + window={window} + 1 target row), got {len(df)}. "
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
    factors_arr = factors_df[valid].values.astype(np.float32)    # (T, 16) flat
    embeddings_arr = embeddings[valid.values]                      # (T, 768) flat
    sent_probs_arr = sent_probs[valid.values]                      # (T, 3) flat
    targets_arr = targets[valid].values.astype(np.float32)
    factor_dates = factors_df.index[valid]

    T = len(targets_arr)
    N = T - window + 1
    if N <= 0:
        raise RuntimeError(
            f"Not enough data for window={window}: only {T} valid rows for ticker='{ticker}'"
        )

    # Window-end labels and dates (N values)
    y = targets_arr[window - 1:]          # (N,)
    win_dates = factor_dates[window - 1:].values  # (N,)

    # Align close prices to factor dates for momentum slope computation
    close_arr: np.ndarray | None = None
    if include_momentum_slope:
        close_arr = close.reindex(factors_df.index).ffill()[valid].values.astype(np.float64)

    # Align fundamentals to window end dates (no time dim)
    X_fund = align_fundamentals(win_dates, fundamental_df)

    # Append momentum slope as an extra fundamental-style scalar feature.
    # Uses the last min(20, window) closes of each window — computed from the
    # flat close_arr using the same index arithmetic as _build_windows used.
    if include_momentum_slope and close_arr is not None:
        slope_len = min(20, window)
        xs = np.arange(slope_len)
        slopes = np.array(
            [np.polyfit(xs, close_arr[i + window - slope_len : i + window], 1)[0]
             for i in range(N)],
            dtype=np.float32,
        ).reshape(-1, 1)
        X_fund = np.concatenate([X_fund, slopes], axis=1)

    n_fund = X_fund.shape[1]
    n_sprob = sent_probs_arr.shape[1]
    logger.info(
        "Built dataset: %d windows (%d flat rows), %d tech factors, %d fundamental factors "
        "(%s momentum slope), %d sentiment prob features, window=%d",
        N, T, factors_arr.shape[1], n_fund,
        "incl." if include_momentum_slope else "excl.",
        n_sprob, window,
    )
    return {
        "X_tech": factors_arr,
        "X_sentiment": embeddings_arr,
        "X_fundamental": X_fund,
        "X_sentiment_probs": sent_probs_arr,
        "y": y,
        "dates": win_dates,
        "window": window,
    }


def make_loaders(
    dataset: FusedDataset,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler | None]:
    """Split chronologically, normalize features, create DataLoaders.

    Both technical factors and fundamental factors are normalized with separate
    ``StandardScaler`` instances fitted on training data only.

    The flat time-series tensors are placed in shared memory (via
    ``tensor.share_memory_()``) so that DataLoader worker processes can access
    them without copying.

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
    num_workers:
        Number of worker processes for the DataLoader.  ``0`` (default) means
        single-process loading.  Set to the number of available CPU cores for
        maximum throughput.  ``persistent_workers`` is enabled automatically
        when ``num_workers > 0``.

    Returns
    -------
    ``(train_loader, val_loader, test_loader, tech_scaler, fund_scaler)``

    ``fund_scaler`` is ``None`` when the dataset has no fundamental features.
    Both scalers are fitted on training data only and applied to all splits.
    """
    N = len(dataset["y"])
    window = dataset["window"]
    test_start = int(N * (1 - test_frac))
    val_start = int(test_start * (1 - val_frac))

    train_idx = np.arange(0, val_start)
    val_idx   = np.arange(val_start, test_start)
    test_idx  = np.arange(test_start, N)

    # Fit tech scaler on flat rows covered by train windows only
    flat_train_end = int(train_idx[-1]) + window  # exclusive upper bound
    tech_scaler = StandardScaler()
    tech_scaler.fit(dataset["X_tech"][:flat_train_end])

    # Fit fund scaler on train windows
    X_fund_all = dataset["X_fundamental"]
    n_fund = X_fund_all.shape[1]
    fund_scaler: StandardScaler | None = None
    if n_fund > 0:
        fund_scaler = StandardScaler()
        fund_scaler.fit(X_fund_all[train_idx])

    # Scale flat arrays once; move to shared memory so worker processes can
    # read them without copying when num_workers > 0.
    X_tech_t = torch.tensor(
        tech_scaler.transform(dataset["X_tech"]).astype(np.float32)
    ).share_memory_()
    X_sent_t  = torch.tensor(dataset["X_sentiment"]).share_memory_()
    X_sprob_t = torch.tensor(dataset["X_sentiment_probs"]).share_memory_()

    X_fund_scaled = X_fund_all.copy()
    if fund_scaler is not None:
        X_fund_scaled = fund_scaler.transform(X_fund_scaled).astype(np.float32)

    loaders = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        ds = LazyFusedDataset(X_tech_t, X_sent_t, X_sprob_t, X_fund_scaled, dataset["y"], window, idx)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return loaders["train"], loaders["val"], loaders["test"], tech_scaler, fund_scaler


def make_loaders_multi(
    datasets: list[tuple[str, FusedDataset]],
    cutoffs: dict[str, pd.Timestamp],
    val_months: int = 2,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler | None]:
    """Per-symbol temporal split with staggered cutoffs, then normalize and create DataLoaders.

    For each symbol the split is:

    - **test**  — ``date >= cutoffs[symbol]``
    - **val**   — ``cutoffs[symbol] - val_months months <= date < cutoffs[symbol]``
    - **train** — ``date < cutoffs[symbol] - val_months months``

    Flat time-series tensors are created once per symbol and shared across the
    symbol's train/val/test :class:`LazyFusedDataset` instances — no per-split
    copies of the embedding arrays are made.  Splits are combined with
    ``torch.utils.data.ConcatDataset``.

    The old :func:`make_loaders` is kept for single-symbol use.

    Parameters
    ----------
    datasets:
        List of ``(symbol, FusedDataset)`` pairs from :func:`build_dataset`.
    cutoffs:
        Per-symbol ``pd.Timestamp`` train/test boundary (from ``splits.yml``).
    val_months:
        Calendar months immediately before each cutoff reserved for validation.
    batch_size:
        Batch size for all three DataLoaders.
    num_workers:
        Number of worker processes for the DataLoader.  ``0`` (default) means
        single-process loading.  Set to the number of available CPU cores for
        maximum throughput.  ``persistent_workers`` is enabled automatically
        when ``num_workers > 0``.

    Returns
    -------
    ``(train_loader, val_loader, test_loader, tech_scaler, fund_scaler)``

    ``fund_scaler`` is ``None`` when no fundamental features are present.
    """
    # ------------------------------------------------------------------
    # Pass 1 — compute split indices and collect flat train slices for
    #           scaler fitting (no windowing required).
    # ------------------------------------------------------------------
    split_indices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    tech_flat_parts: list[np.ndarray] = []
    fund_train_parts: list[np.ndarray] = []

    for symbol, ds in datasets:
        cutoff = cutoffs[symbol]
        val_start_date = cutoff - pd.DateOffset(months=val_months)
        dates = pd.DatetimeIndex(ds["dates"])

        train_idx = np.where(dates < val_start_date)[0]
        val_idx   = np.where((dates >= val_start_date) & (dates < cutoff))[0]
        test_idx  = np.where(dates >= cutoff)[0]
        split_indices[symbol] = (train_idx, val_idx, test_idx)

        if len(train_idx) == 0:
            continue

        # Flat rows used by train windows: window wi uses rows [wi : wi+window]
        window = ds["window"]
        flat_end = int(train_idx[-1]) + window  # exclusive
        tech_flat_parts.append(ds["X_tech"][:flat_end])
        fund_train_parts.append(ds["X_fundamental"][train_idx])

    # Fit scalers on concatenated flat training rows
    tech_scaler = StandardScaler()
    tech_scaler.fit(np.concatenate(tech_flat_parts, axis=0))

    n_fund = datasets[0][1]["X_fundamental"].shape[1] if datasets else 0
    fund_scaler: StandardScaler | None = None
    if n_fund > 0 and fund_train_parts:
        fund_scaler = StandardScaler()
        fund_scaler.fit(np.concatenate(fund_train_parts, axis=0))

    # ------------------------------------------------------------------
    # Pass 2 — scale flat arrays and build LazyFusedDataset instances.
    #           Each symbol produces one shared tensor triple; train/val/test
    #           datasets for that symbol all reference the same tensors.
    # ------------------------------------------------------------------
    train_lazy: list[LazyFusedDataset] = []
    val_lazy:   list[LazyFusedDataset] = []
    test_lazy:  list[LazyFusedDataset] = []

    for symbol, ds in datasets:
        train_idx, val_idx, test_idx = split_indices[symbol]
        window = ds["window"]

        # Scale and convert flat arrays once — shared across splits.
        # share_memory_() lets worker processes read without copying.
        X_tech_t = torch.tensor(
            tech_scaler.transform(ds["X_tech"]).astype(np.float32)
        ).share_memory_()
        X_sent_t  = torch.tensor(ds["X_sentiment"]).share_memory_()
        X_sprob_t = torch.tensor(ds["X_sentiment_probs"]).share_memory_()

        X_fund_scaled = ds["X_fundamental"].copy()
        if fund_scaler is not None:
            X_fund_scaled = fund_scaler.transform(X_fund_scaled).astype(np.float32)

        for idx, lazy_list in (
            (train_idx, train_lazy),
            (val_idx,   val_lazy),
            (test_idx,  test_lazy),
        ):
            if len(idx) == 0:
                continue
            lazy_list.append(
                LazyFusedDataset(X_tech_t, X_sent_t, X_sprob_t, X_fund_scaled, ds["y"], window, idx)
            )

    for split_name, lazy_list in (("train", train_lazy), ("val", val_lazy), ("test", test_lazy)):
        n = sum(len(d) for d in lazy_list)
        logger.info("make_loaders_multi — %s: %d windows", split_name, n)

    loaders = {}
    for name, lazy_list in (("train", train_lazy), ("val", val_lazy), ("test", test_lazy)):
        combined = ConcatDataset(lazy_list)
        loaders[name] = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    return loaders["train"], loaders["val"], loaders["test"], tech_scaler, fund_scaler


def build_eval_loader(
    datasets: list[FusedDataset],
    tech_scaler: StandardScaler,
    fund_scaler: StandardScaler | None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Build an evaluation DataLoader from a list of :class:`FusedDataset` dicts.

    Applies *tech_scaler* and *fund_scaler* (fitted on training data) to each
    dataset and combines them into a single :class:`~torch.utils.data.DataLoader`
    via ``ConcatDataset``.  Intended for held-out symbol evaluation.

    Parameters
    ----------
    datasets:
        List of :class:`FusedDataset` dicts from :func:`build_dataset`.
    tech_scaler:
        Fitted ``StandardScaler`` for technical factors.
    fund_scaler:
        Fitted ``StandardScaler`` for fundamental factors, or ``None``.
    batch_size:
        Batch size for the returned DataLoader.

    Returns
    -------
    A non-shuffling :class:`~torch.utils.data.DataLoader` over all windows.
    """
    lazy_list: list[LazyFusedDataset] = []
    for ds in datasets:
        window = ds["window"]
        N = len(ds["y"])
        idx = np.arange(N)

        X_tech_t = torch.tensor(
            tech_scaler.transform(ds["X_tech"]).astype(np.float32)
        ).share_memory_()
        X_sent_t  = torch.tensor(ds["X_sentiment"]).share_memory_()
        X_sprob_t = torch.tensor(ds["X_sentiment_probs"]).share_memory_()

        X_fund_scaled = ds["X_fundamental"].copy()
        if fund_scaler is not None:
            X_fund_scaled = fund_scaler.transform(X_fund_scaled).astype(np.float32)

        lazy_list.append(
            LazyFusedDataset(X_tech_t, X_sent_t, X_sprob_t, X_fund_scaled, ds["y"], window, idx)
        )

    return DataLoader(
        ConcatDataset(lazy_list),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
