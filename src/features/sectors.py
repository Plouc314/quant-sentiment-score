"""Sector-level dataset and DataLoader builder for sector-rotation experiments.

Each sector is represented as:
  - An equal-weight price index (mean of constituent OHLCV across common trading days)
  - A daily aggregated sentiment embedding (mean of non-zero constituent embeddings)

Two target modes:
  - Absolute: sector index rises over ``horizon`` trading days  (original, class-imbalanced)
  - Relative: sector beats the median of all sectors over the same horizon (50/50 by construction)

Use ``compute_cross_sector_labels`` to pre-compute relative labels across all sectors,
then pass them via the ``target_labels`` argument to ``SectorDataset``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .dataset import _EMBEDDING_DIM, _LazyDataset, _compute_targets
from .technical import TechnicalFactors

logger = logging.getLogger(__name__)

# Sector name → constituent tickers (our 49-ticker universe)
SECTORS: dict[str, list[str]] = {
    "Technology":      ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "ADBE", "AVGO", "ORCL"],
    "Healthcare":      ["JNJ", "UNH", "ABBV", "MRK", "PFE", "LLY"],
    "Financials":      ["JPM", "BAC", "GS", "MS", "MA", "V"],
    "Energy":          ["XOM", "CVX", "COP", "SLB"],
    "ConsumerDisc":    ["AMZN", "TSLA", "NKE", "MCD", "SBUX", "DIS", "HD"],
    "ConsumerStaples": ["KO", "PG", "WMT", "COST"],
    "Industrials":     ["BA", "CAT", "HON", "GE", "LMT"],
    "UtilTelecom":     ["NEE", "DUK", "T", "VZ", "AMT", "LIN"],
}


class SectorDataset:
    """Feature matrix for one sector: equal-weight price index + aggregated daily sentiment.

    The sector price index is the simple mean of constituent close (open, high, low)
    prices on each common trading day; volume is summed.  Technical factors are
    computed on this index.  The sector sentiment embedding is the mean of all
    non-zero constituent embeddings on each day; days with no news are zero vectors.

    Parameters
    ----------
    name:
        Sector label used for logging and checkpointing.
    price_dfs:
        ``{ticker: OHLCV DataFrame}`` for each constituent.
        Each DataFrame must have a DatetimeIndex and columns
        ``open, high, low, close, volume``, sorted ascending.
    sentiment_dfs:
        ``{ticker: sentiment DataFrame}`` for each constituent.
        Each DataFrame has columns ``ticker, date, embedding``.
    window:
        Sliding window length in trading days.
    horizon:
        Forward-return horizon used when ``target_labels`` is None (absolute mode).
    target_labels:
        Pre-computed label Series (values in {-1, 0, 1}) indexed on trading dates.
        Pass the output of ``compute_cross_sector_labels`` here to use relative
        targets (sector beats median sector).  When ``None``, falls back to the
        absolute direction target ``close[t+horizon] > close[t]``.
    """

    def __init__(
        self,
        name: str,
        price_dfs: dict[str, pd.DataFrame],
        sentiment_dfs: dict[str, pd.DataFrame],
        window: int = 20,
        horizon: int = 10,
        target_labels: pd.Series | None = None,
    ) -> None:
        self.name    = name
        self.window  = window
        self.horizon = horizon

        # --- Equal-weight sector price index ---
        price_index = build_sector_price_index(price_dfs)

        min_rows = 90 + window + horizon
        if len(price_index) < min_rows:
            raise RuntimeError(
                f"{name}: need ≥ {min_rows} rows (90 warmup + window={window} + horizon={horizon}),"
                f" got {len(price_index)}"
            )

        # --- Technical factors ---
        factors_df = TechnicalFactors().compute(price_index)

        # --- Targets: relative (cross-sector) or absolute ---
        if target_labels is not None:
            targets = target_labels.reindex(factors_df.index, fill_value=-1)
        else:
            targets = _compute_targets(price_index["close"], horizon=horizon)
            targets = targets.reindex(factors_df.index, fill_value=-1)

        # --- Aggregate sentiment across constituents ---
        embeddings = _aggregate_embeddings(factors_df.index, sentiment_dfs)

        # --- Drop rows without a valid future close ---
        valid       = targets >= 0
        X_tech      = factors_df[valid].values.astype(np.float32)
        X_sent      = embeddings[valid.values]
        targets_arr = targets[valid].values.astype(np.int64)
        dates_arr   = factors_df.index[valid]

        T = len(targets_arr)
        if T - window + 1 <= 0:
            raise RuntimeError(f"{name}: not enough valid rows for window={window}")

        # Flat storage — _LazyDataset materialises windows on the fly
        self.X_tech: np.ndarray = X_tech
        self.X_sent: np.ndarray = X_sent
        self.y: np.ndarray      = targets_arr[window - 1:]
        self.dates: np.ndarray  = dates_arr[window - 1:].values

        mode     = "relative" if target_labels is not None else "absolute"
        coverage = (np.linalg.norm(X_sent, axis=1) > 0).mean()
        pos_rate = self.y.mean()
        logger.info(
            "%s  horizon=T+%d  mode=%s | %d windows | pos_rate=%.2f | %.0f%% days with news",
            name, horizon, mode, len(self.y), pos_rate, 100 * coverage,
        )

    @property
    def n_windows(self) -> int:
        return len(self.y)


def build_sector_price_index(price_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight OHLCV index from constituent DataFrames.

    Restricts to the intersection of trading days so every row is fully
    populated.  Open / high / low / close are averaged; volume is summed
    to reflect total sector activity.
    """
    closes     = pd.DataFrame({t: df["close"] for t, df in price_dfs.items()})
    common_idx = closes.dropna().index          # inner join on trading days

    result = pd.DataFrame(index=common_idx)
    for col in ("open", "high", "low", "close"):
        result[col] = (
            pd.DataFrame({t: df[col] for t, df in price_dfs.items()})
            .reindex(common_idx)
            .mean(axis=1)
        )
    result["volume"] = (
        pd.DataFrame({t: df["volume"] for t, df in price_dfs.items()})
        .reindex(common_idx)
        .sum(axis=1)
    )
    return result.sort_index()


def compute_cross_sector_labels(
    price_indices: dict[str, pd.DataFrame],
    horizon: int,
) -> dict[str, pd.Series]:
    """Relative targets: does this sector beat the median sector return?

    For each trading day ``t``, compute the ``horizon``-day forward return for
    every sector.  A sector is labelled ``1`` if its return is strictly above
    the cross-sector median on that day, ``0`` if at or below.  The last
    ``horizon`` rows (no future close available) are sentinel ``-1``.

    Because exactly half the sectors beat the median each day, this gives
    a ~50/50 class split by construction, removing the positive market-drift
    bias that inflated accuracy in absolute-target experiments.

    Parameters
    ----------
    price_indices:
        ``{sector_name: OHLCV DataFrame}`` as returned by
        ``build_sector_price_index`` for each sector.
    horizon:
        Forward return window in trading days.

    Returns
    -------
    ``{sector_name: pd.Series}`` with integer labels ``{-1, 0, 1}`` indexed
    on the inter-sector common trading days.
    """
    # Align close prices across all sectors to their common trading days
    closes = pd.DataFrame(
        {name: df["close"] for name, df in price_indices.items()}
    ).dropna()

    # N-day forward return for each sector
    forward = closes.shift(-horizon)
    returns = (forward / closes) - 1.0     # NaN for last `horizon` rows

    # Cross-sector median return at each date (NaN where any sector is missing)
    median_ret = returns.median(axis=1)

    labels: dict[str, pd.Series] = {}
    for sector in closes.columns:
        col_ret = returns[sector]
        label   = pd.Series(0, index=closes.index, dtype=np.int64)
        label[col_ret > median_ret] = 1
        label[col_ret.isna()]       = -1   # sentinel for last horizon rows
        labels[sector] = label

    # Log class balance as a sanity check
    for sector, lbl in labels.items():
        valid    = lbl[lbl >= 0]
        pos_rate = valid.mean()
        logger.debug("%s  horizon=T+%d  pos_rate=%.3f  n=%d", sector, horizon, pos_rate, len(valid))

    return labels


def build_sector_loaders(
    ds: SectorDataset,
    cutoff: str = "2023-06-01",
    val_frac: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val / test DataLoaders for one sector dataset.

    No sentiment gating — sector aggregation provides near-daily news coverage.
    Pre-cutoff windows are split chronologically: first ``(1 - val_frac)`` → train,
    last ``val_frac`` → val.  Post-cutoff windows form the test set.
    Val and test loaders use a single full-batch (no shuffling).
    """
    dates     = pd.DatetimeIndex(ds.dates)
    cutoff_ts = pd.Timestamp(cutoff)

    all_idx  = np.arange(len(ds.y))
    pre      = all_idx[dates < cutoff_ts]
    post     = all_idx[dates >= cutoff_ts]

    n_val     = max(1, int(len(pre) * val_frac))
    train_idx = pre[:-n_val]
    val_idx   = pre[-n_val:]
    test_idx  = post

    logger.info(
        "%s  horizon=T+%d | train=%d  val=%d  test=%d",
        ds.name, ds.horizon, len(train_idx), len(val_idx), len(test_idx),
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
# Private helpers
# ------------------------------------------------------------------

def _aggregate_embeddings(
    index: pd.DatetimeIndex,
    sentiment_dfs: dict[str, pd.DataFrame],
) -> np.ndarray:
    """Mean of non-zero constituent embeddings per trading day.

    Builds a ``date → embedding`` lookup per ticker, then for each day in
    ``index`` averages the embeddings of whichever constituents have news.
    Days with no news from any constituent remain zero vectors.
    """
    lookups: list[dict] = []
    for ticker, df in sentiment_dfs.items():
        rows = df[df["ticker"] == ticker] if "ticker" in df.columns else df
        if rows.empty:
            continue
        lookup = {
            pd.Timestamp(d).date(): np.asarray(emb, dtype=np.float32)
            for d, emb in zip(rows["date"], rows["embedding"])
        }
        lookups.append(lookup)

    result = np.zeros((len(index), _EMBEDDING_DIM), dtype=np.float32)
    for i, ts in enumerate(index):
        day  = ts.date()
        embs = [lk[day] for lk in lookups if day in lk]
        if embs:
            result[i] = np.mean(embs, axis=0)
    return result
