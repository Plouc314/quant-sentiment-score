"""Sector-level dataset and DataLoader builder for sector-rotation experiments.

Each sector is represented as:
  - An equal-weight price index (mean of constituent OHLCV across common trading days)
  - A daily aggregated sentiment embedding (mean of non-zero constituent embeddings)

Two target modes:
  - Absolute: sector index rises over ``horizon`` trading days  (original, class-imbalanced)
  - Relative: sector beats the median of all sectors over the same horizon (50/50 by construction)

Use ``compute_cross_sector_labels`` to pre-compute relative labels across all sectors,
then pass them via the ``target_labels`` argument to ``SectorDataset``.

Sentiment modes (``sentiment_mode`` parameter on ``SectorDataset``):
  - ``"embedding"`` — original 768-dim FinBERT vectors aggregated per day (Plan 0)
  - ``"score"``     — scalar ``sentiment_score`` (= 1·p_pos + 0.5·p_neutral) appended
                      to tech features as column 17; embedding path disabled (Plan A)
  - ``"score+delta"`` — scalar score **and** its 20-day rolling surprise
                        appended as columns 17–18; embedding path disabled (Plan A+B)
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
    computed on this index.

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
        Each DataFrame has columns ``ticker, date, embedding, sentiment_score``.
    window:
        Sliding window length in trading days.
    horizon:
        Forward-return horizon used when ``target_labels`` is None (absolute mode).
    target_labels:
        Pre-computed label Series (values in {-1, 0, 1}) indexed on trading dates.
        Pass the output of ``compute_cross_sector_labels`` here to use relative
        targets (sector beats median sector).  When ``None``, falls back to the
        absolute direction target ``close[t+horizon] > close[t]``.
    sentiment_mode:
        How to incorporate sentiment:

        * ``"embedding"``   — 768-dim FinBERT vectors aggregated per day; stored
                              in ``X_sent`` and processed by the model's projection
                              layer (original architecture).
        * ``"score"``       — scalar ``sentiment_score`` (= 1·p_pos + 0.5·p_neutral,
                              range [0,1]) appended as column 17 of ``X_tech``.
                              ``X_sent`` is a zero-column dummy; model must use
                              ``use_sentiment_proj=False``.  (Plan A)
        * ``"score+delta"`` — score **and** its 20-day rolling surprise (score minus
                              20-day rolling mean) appended as columns 17–18.
                              (Plan A + B)
    """

    def __init__(
        self,
        name: str,
        price_dfs: dict[str, pd.DataFrame],
        sentiment_dfs: dict[str, pd.DataFrame],
        window: int = 20,
        horizon: int = 10,
        target_labels: pd.Series | None = None,
        sentiment_mode: str = "embedding",
    ) -> None:
        self.name          = name
        self.window        = window
        self.horizon       = horizon
        self.sentiment_mode = sentiment_mode

        if sentiment_mode not in ("embedding", "score", "score+delta"):
            raise ValueError(f"sentiment_mode must be 'embedding', 'score', or 'score+delta', got {sentiment_mode!r}")

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

        # --- Sentiment features ---
        if sentiment_mode == "embedding":
            embeddings = _aggregate_embeddings(factors_df.index, sentiment_dfs)
        else:
            # Plans A / A+B: aggregate the pre-computed scalar score
            scores = _aggregate_scores(factors_df.index, sentiment_dfs)  # (T,)

        # --- Drop rows without a valid future close ---
        valid       = targets >= 0
        X_tech_base = factors_df[valid].values.astype(np.float32)
        targets_arr = targets[valid].values.astype(np.int64)
        dates_arr   = factors_df.index[valid]

        T = len(targets_arr)
        if T - window + 1 <= 0:
            raise RuntimeError(f"{name}: not enough valid rows for window={window}")

        if sentiment_mode == "embedding":
            X_sent = embeddings[valid.values]
            X_tech = X_tech_base
        else:
            scores_valid = scores[valid.values]            # (T,)
            if sentiment_mode == "score+delta":
                # 20-day rolling mean (on the full pre-valid series to avoid boundary artefacts)
                score_series     = pd.Series(scores, index=factors_df.index)
                rolling_mean     = score_series.rolling(window=20, min_periods=1).mean()
                delta_series     = score_series - rolling_mean
                delta_valid      = delta_series[valid].values.astype(np.float32)
                X_tech = np.concatenate(
                    [X_tech_base, scores_valid[:, None], delta_valid[:, None]], axis=1
                )
            else:  # "score"
                X_tech = np.concatenate([X_tech_base, scores_valid[:, None]], axis=1)

            # Minimal dummy for X_sent — shape (T, 1), all zeros
            X_sent = np.zeros((T, 1), dtype=np.float32)

        # Flat storage — _LazyDataset materialises windows on the fly
        self.X_tech: np.ndarray = X_tech
        self.X_sent: np.ndarray = X_sent
        self.y: np.ndarray      = targets_arr[window - 1:]
        self.dates: np.ndarray  = dates_arr[window - 1:].values

        target_mode = "relative" if target_labels is not None else "absolute"
        if sentiment_mode == "embedding":
            coverage = (np.linalg.norm(X_sent, axis=1) > 0).mean()
            sent_info = f"embedding | {100 * coverage:.0f}% days with news"
        else:
            sent_info = f"sent_mode={sentiment_mode}"
        pos_rate = self.y.mean()
        logger.info(
            "%s  horizon=T+%d  target=%s | sent=%s | %d windows | pos_rate=%.2f",
            name, horizon, target_mode, sent_info, len(self.y), pos_rate,
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


def _aggregate_scores(
    index: pd.DatetimeIndex,
    sentiment_dfs: dict[str, pd.DataFrame],
) -> np.ndarray:
    """Mean of constituent ``sentiment_score`` values per trading day.

    ``sentiment_score`` is the pre-computed scalar stored in each parquet file:
    ``1.0·P(pos) + 0.5·P(neutral)`` (range [0, 1]; 1 = fully positive,
    0 = fully negative, 0.5 = neutral / no news).

    Days with no news from any constituent are assigned 0.5 (neutral baseline).
    """
    lookups: list[dict] = []
    for ticker, df in sentiment_dfs.items():
        rows = df[df["ticker"] == ticker] if "ticker" in df.columns else df
        if rows.empty or "sentiment_score" not in rows.columns:
            continue
        lookup = {
            pd.Timestamp(d).date(): float(score)
            for d, score in zip(rows["date"], rows["sentiment_score"])
        }
        lookups.append(lookup)

    result = np.full(len(index), 0.5, dtype=np.float32)  # neutral default
    for i, ts in enumerate(index):
        day    = ts.date()
        scores = [lk[day] for lk in lookups if day in lk]
        if scores:
            result[i] = float(np.mean(scores))
    return result
