"""Shared helpers for Phase 1+2 experiments.

Builds the 50-symbol dataset cache once, exposes train_one_combo() that runs a
single train+eval given a list of symbols and a target spec. Stays out of the
src/ tree so we don't pollute committed code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src  # noqa: F401  (load_dotenv side-effect)
from src.features.dataset import DataLoaderBuilder, StockDataset
from src.model.lstm import SentimentLSTM
from src.model.trainer import Trainer
from src.repositories.prices import PriceRepository
from src.repositories.sentiment import SentimentRepository
from src.training import ComputeConfig, Split, TrainingConfig

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data" / "splits_fast.yml"
PRICE_YEARS = list(range(2018, 2025))


@dataclass
class TargetSpec:
    """Defines what the model is trying to predict.

    kind="direction": close[t+horizon] > close[t], optional return-threshold filter
    kind="magnitude": |return at horizon| > magnitude_threshold (volatility task)
    kind="big_up":    return at horizon >  magnitude_threshold (positive tail)
    kind="big_down":  return at horizon < -magnitude_threshold (negative tail)
    kind="sector_relative": return - sector_mean_return > 0 (relative outperformance)
    """
    kind: str = "direction"
    horizon: int = 3
    return_threshold: float | None = 0.005       # for direction
    magnitude_threshold: float | None = 0.015    # for magnitude/big_up/big_down


def build_dataset_cache() -> dict[str, StockDataset]:
    """Build all 50 StockDatasets once with neutral target settings.

    Targets are recomputed downstream by override_targets(); here we use the
    direction default just so StockDataset has something valid in ds.y.
    """
    split = Split.load(SPLIT_PATH)
    prices = PriceRepository()
    sentiment = SentimentRepository()
    cache: dict[str, StockDataset] = {}

    for symbol in split.all_symbols:
        try:
            price_df = prices.load_years(symbol, PRICE_YEARS)
        except FileNotFoundError:
            continue
        sent_df = sentiment.load(symbol) if sentiment.exists(symbol) else None
        try:
            ds = StockDataset(
                symbol=symbol,
                price_df=price_df,
                sentiment_df=sent_df,
                window=20,
                horizon=3,                # placeholder
                target_threshold=None,    # placeholder, overridden below
                has_news_feature=False,
            )
        except RuntimeError as exc:
            logger.warning("skip %s: %s", symbol, exc)
            continue
        # stash the raw close series for target recomputation
        ds._price_close = price_df["close"].copy()  # type: ignore[attr-defined]
        cache[symbol] = ds

    logger.info("Built %d StockDatasets", len(cache))
    return cache


def override_targets(ds: StockDataset, spec: TargetSpec) -> StockDataset:
    """Recompute ds.y from raw close series according to spec.

    Returns a shallow-copied dataset so caller doesn't mutate the cache.
    """
    new_ds = StockDataset.__new__(StockDataset)
    new_ds.__dict__.update(ds.__dict__)
    new_ds.X_tech = ds.X_tech            # share, not copied
    new_ds.X_sent = ds.X_sent
    new_ds._price_close = ds._price_close

    close = ds._price_close                              # full price series
    # Reconstruct the timestamp -> index alignment:
    # ds.dates has shape (n_windows,), corresponding to factor day at window-end.
    # The original target was computed on factors_df.index pre-window; we recompute
    # against the SAME factors_df.index by matching via dates of valid windows.
    # Simpler approach: compute return for each anchor date using close lookups.
    h = spec.horizon
    close_ts = close.copy()
    close_ts.index = pd.DatetimeIndex(close_ts.index).tz_localize(None) if close_ts.index.tz is not None else pd.DatetimeIndex(close_ts.index)
    close_dict: dict[pd.Timestamp, float] = {ts.normalize(): float(v) for ts, v in close_ts.items()}
    sorted_dates = sorted(close_dict.keys())
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

    new_y = np.full(len(ds.y), -1, dtype=np.int64)
    for i, anchor_date in enumerate(ds.dates):
        anchor = pd.Timestamp(anchor_date).tz_localize(None).normalize() if pd.Timestamp(anchor_date).tz is not None else pd.Timestamp(anchor_date).normalize()
        if anchor not in date_to_idx:
            continue
        idx = date_to_idx[anchor]
        if idx + h >= len(sorted_dates):
            continue
        c_now = close_dict[sorted_dates[idx]]
        c_fut = close_dict[sorted_dates[idx + h]]
        ret = (c_fut - c_now) / c_now

        if spec.kind == "direction":
            if spec.return_threshold is not None:
                new_y[i] = 1 if ret > spec.return_threshold else 0
            else:
                new_y[i] = 1 if ret > 0 else 0
        elif spec.kind == "magnitude":
            assert spec.magnitude_threshold is not None
            new_y[i] = 1 if abs(ret) > spec.magnitude_threshold else 0
        elif spec.kind == "big_up":
            assert spec.magnitude_threshold is not None
            new_y[i] = 1 if ret > spec.magnitude_threshold else 0
        elif spec.kind == "big_down":
            assert spec.magnitude_threshold is not None
            new_y[i] = 1 if ret < -spec.magnitude_threshold else 0
        else:
            raise ValueError(f"Unknown target kind: {spec.kind!r}")

    # Drop sentinel rows
    valid = new_y >= 0
    if valid.sum() < len(new_y):
        new_ds.X_tech = ds.X_tech                                # full flat array, ok
        new_ds.X_sent = ds.X_sent
        new_ds.y = new_y[valid]
        new_ds.dates = ds.dates[valid]
    else:
        new_ds.y = new_y
        new_ds.dates = ds.dates

    return new_ds


def make_subset_split(symbols: list[str], full_split: Split) -> Split:
    """Build a Split restricted to ``symbols`` while keeping the train/held-out
    assignment from full_split."""
    train = [s for s in full_split.train_symbols if s in symbols]
    held_out = [s for s in full_split.held_out_symbols if s in symbols]
    cutoffs = {s: full_split.cutoffs[s] for s in symbols if s in full_split.cutoffs}
    return Split(
        train_symbols=train,
        held_out_symbols=held_out,
        cutoffs=cutoffs,
        val_months=full_split.val_months,
    )


def train_one_combo(
    symbols: list[str],
    cache: dict[str, StockDataset],
    spec: TargetSpec,
    full_split: Split,
    seed: int = 42,
    save_predictions_path: Path | None = None,
    n_bootstrap: int = 200,
    ablation: str = "both",
    news_days_only: bool = False,
) -> dict:
    """Train one combo LSTM on the given symbols with the given target spec.

    Args:
        ablation: "both" (default), "tech_only" (zero sentiment), or
            "sentiment_only" (zero tech). Applied AFTER target override.
        news_days_only: if True, drop windows whose anchor day has no
            sentiment embedding. Useful for B2 (news-day conditional).

    Returns a dict with training and eval results.  Optionally saves raw
    held-out predictions for downstream stacking.
    """
    sub_split = make_subset_split(symbols, full_split)
    if not sub_split.train_symbols:
        return {"skipped": True, "reason": "no train symbols", "symbols": symbols}

    datasets = {s: override_targets(cache[s], spec) for s in symbols if s in cache}

    if news_days_only:
        # NOTE: deferred — requires changes to _LazyDataset indexing logic in
        # src/features/dataset.py. Window indices and X_tech rows must stay
        # aligned, so we can't simply subset ds.dates/ds.y here.
        raise NotImplementedError("news_days_only requires _LazyDataset changes")

    # Optional: ablation (zero out one modality)
    if ablation != "both":
        for ds in datasets.values():
            if ablation == "tech_only":
                ds.X_sent = np.zeros_like(ds.X_sent)
            elif ablation == "sentiment_only":
                ds.X_tech = np.zeros_like(ds.X_tech)
            else:
                raise ValueError(f"Unknown ablation: {ablation!r}")

    # GPU-aware sizing: tiny model + 8GB VRAM means we can run much larger batches.
    # On GPU the ~1300 batches/epoch at bs=32 is dominated by per-batch launch
    # overhead; bumping to 128 cuts that 4x while staying well under 1GB VRAM.
    on_gpu = torch.cuda.is_available()
    bs = 128 if on_gpu else 32
    nw = 4 if on_gpu else 0  # 16 cores → 4 workers is conservative

    training = TrainingConfig(
        window=20, batch_size=bs, n_epochs=50, lr=1e-4, weight_decay=1e-4,
        patience=10, dropout=0.2, seed=seed, scheduler="plateau",
        scheduler_patience=5, grad_clip=1.0, early_stopping_metric="auc",
    )
    compute = ComputeConfig(device=None, num_workers=nw)
    compute.setup()

    builder = DataLoaderBuilder(datasets, sub_split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    has_held_out = bool(sub_split.held_out_symbols)
    held_out_loader = builder.build_held_out_loader() if has_held_out else None

    model = SentimentLSTM(
        n_factors=16, sentiment_dim=768, hidden_size=64,
        num_layers=2, dropout=training.dropout, sentiment_proj_dim=64,
    )
    trainer = Trainer(model, training, compute)
    tr = trainer.fit(train_loader, val_loader)
    r_test = trainer.bootstrap_evaluate(test_loader, n_bootstrap=n_bootstrap, seed=seed)
    r_ho = (
        trainer.bootstrap_evaluate(held_out_loader, n_bootstrap=n_bootstrap, seed=seed)
        if has_held_out else None
    )

    result = {
        "skipped": False,
        "symbols": symbols,
        "n_symbols": len(symbols),
        "n_train_symbols": len(sub_split.train_symbols),
        "n_held_out_symbols": len(sub_split.held_out_symbols),
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "test": _eval_to_dict(r_test),
        "held_out": _eval_to_dict(r_ho) if r_ho is not None else None,
    }

    if save_predictions_path is not None and has_held_out:
        probs, targets, _ = trainer._collect_predictions(held_out_loader)
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_predictions_path, probs=probs, targets=targets)
        result["predictions_path"] = str(save_predictions_path)

    return result


def _eval_to_dict(e) -> dict:
    return {k: float(getattr(e, k)) for k in vars(e) if not k.startswith("_")}
