"""Three next-level experiments building on the H1+H4 winning combo.

E1 — News-day conditional: train/eval ONLY on windows where the anchor day
     has a FinBERT embedding (actual news coverage). Removes zero-embedding
     noise from ~60 % of windows. Requires no src/ changes — the fix is at
     the index-selection level: _LazyDataset accepts arbitrary window indices.

E2 — Sector-relative target: predict cross-sectional stock outperformance vs
     sector mean return. Removes market-wide beta entirely; balanced classes
     by construction (~50 % positive), so no degenerate collapse risk.

E3 — Full 992-symbol H1+H4 scale validation: confirm the winning combo's
     AUC edge holds on the complete universe (794 train / 198 held-out).

E1 and E2 use the 50-symbol fast subset. E3 uses splits.yml (992 symbols).
Results saved to data/experiments_next_level/.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from src.features.dataset import (
    DataLoaderBuilder, StockDataset,
    _LazyDataset, _make_loader,  # private but importable in scripts
)
from src.log import setup_logging
from src.model.lstm import SentimentLSTM
from src.model.trainer import Trainer
from src.repositories.prices import PriceRepository
from src.repositories.sentiment import SentimentRepository
from src.training import ComputeConfig, Split, TrainingConfig

from scripts._phase1_helpers import (
    TargetSpec, build_dataset_cache, override_targets, make_subset_split,
)

setup_logging()
logger = logging.getLogger("next_level")

SPLIT_PATH = ROOT / "data" / "splits_fast.yml"
FULL_SPLIT_PATH = ROOT / "data" / "splits.yml"
SECTORS_PATH = ROOT / "data" / "sectors_50.yml"
PRICE_YEARS = list(range(2018, 2025))

RESULTS_DIR = ROOT / "data" / "experiments_next_level"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 200
SEED = 42
BASE_THRESHOLD = 0.005
BASE_PROJ_DIM = 64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_training_config(on_gpu: bool) -> tuple[TrainingConfig, ComputeConfig]:
    bs = 128 if on_gpu else 32
    nw = 4 if on_gpu else 0
    training = TrainingConfig(
        window=20, batch_size=bs, n_epochs=50, lr=1e-4, weight_decay=1e-4,
        patience=10, dropout=0.2, seed=SEED, scheduler="plateau",
        scheduler_patience=5, grad_clip=1.0, early_stopping_metric="auc",
    )
    compute = ComputeConfig(device=None, num_workers=nw)
    compute.setup()
    return training, compute


def _make_model() -> SentimentLSTM:
    return SentimentLSTM(
        n_factors=16, sentiment_dim=768, hidden_size=64,
        num_layers=2, dropout=0.2, sentiment_proj_dim=BASE_PROJ_DIM,
    )


def _eval_dict(e) -> dict:
    return {k: float(getattr(e, k)) for k in vars(e) if not k.startswith("_")}


# ---------------------------------------------------------------------------
# E1: News-day conditional
# ---------------------------------------------------------------------------

def exp1_news_day_conditional(
    cache: dict[str, StockDataset],
    full_split: Split,
) -> dict:
    """Train and evaluate only on windows whose anchor day has news.

    The NotImplementedError in _phase1_helpers was incorrect: no _LazyDataset
    changes are needed. _LazyDataset accepts arbitrary window indices; we just
    filter them to news-day positions before passing them in.
    """
    logger.info("=" * 60)
    logger.info("E1: News-day conditional (H1+H4 target, news days only)")
    logger.info("=" * 60)

    spec = TargetSpec(kind="direction", horizon=3, return_threshold=BASE_THRESHOLD)
    symbols = list(cache.keys())
    sub_split = make_subset_split(symbols, full_split)
    datasets = {s: override_targets(cache[s], spec) for s in symbols if s in cache}

    on_gpu = torch.cuda.is_available()
    training, compute = _make_training_config(on_gpu)
    val_months = sub_split.val_months

    train_lazy: list[_LazyDataset] = []
    val_lazy:   list[_LazyDataset] = []
    test_lazy:  list[_LazyDataset] = []
    held_lazy:  list[_LazyDataset] = []

    n_total = n_news = 0

    for symbol in sub_split.train_symbols:
        ds = datasets.get(symbol)
        if ds is None:
            continue

        # anchor_has_sentiment returns a mask of length n_windows;
        # truncate to len(ds.y) to guard against horizon mismatch at series end.
        news_mask = ds.anchor_has_sentiment[:len(ds.y)]
        dates = pd.DatetimeIndex(ds.dates)
        n_total += len(ds.y)
        n_news  += int(news_mask.sum())

        cutoff    = sub_split.cutoffs[symbol]
        val_start = cutoff - pd.DateOffset(months=val_months)

        train_idx = np.where((dates < val_start) & news_mask)[0]
        val_idx   = np.where((dates >= val_start) & (dates < cutoff) & news_mask)[0]
        test_idx  = np.where((dates >= cutoff) & news_mask)[0]

        X_tech_t = torch.tensor(ds.X_tech).share_memory_()
        X_sent_t = torch.tensor(ds.X_sent).share_memory_()
        for idx, bucket in (
            (train_idx, train_lazy), (val_idx, val_lazy), (test_idx, test_lazy)
        ):
            if len(idx) > 0:
                bucket.append(_LazyDataset(X_tech_t, X_sent_t, ds.y, ds.window, idx))

    for symbol in sub_split.held_out_symbols:
        ds = datasets.get(symbol)
        if ds is None:
            continue
        news_mask = ds.anchor_has_sentiment[:len(ds.y)]
        held_idx  = np.where(news_mask)[0]
        if len(held_idx) > 0:
            X_tech_t = torch.tensor(ds.X_tech).share_memory_()
            X_sent_t = torch.tensor(ds.X_sent).share_memory_()
            held_lazy.append(_LazyDataset(X_tech_t, X_sent_t, ds.y, ds.window, held_idx))

    logger.info(
        "News-day filter: %d / %d windows kept (%.1f %%)",
        n_news, n_total, 100 * n_news / max(n_total, 1),
    )

    bs, nw = training.batch_size, compute.num_workers
    train_loader = _make_loader(train_lazy, bs, shuffle=True,  num_workers=nw)
    val_loader   = _make_loader(val_lazy,   bs, shuffle=False, num_workers=nw)
    test_loader  = _make_loader(test_lazy,  bs, shuffle=False, num_workers=nw)
    held_loader  = _make_loader(held_lazy,  bs, shuffle=False, num_workers=nw)

    logger.info(
        "Splits — train: %d, val: %d, test: %d, held-out: %d",
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset),  len(held_loader.dataset),
    )

    model   = _make_model()
    trainer = Trainer(model, training, compute)
    t0      = time.monotonic()
    tr      = trainer.fit(train_loader, val_loader)
    r_test  = trainer.bootstrap_evaluate(test_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    r_ho    = trainer.bootstrap_evaluate(held_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    duration = time.monotonic() - t0

    logger.info(
        "E1 done in %.1fs: HO AUC=%.4f Brier=%.4f ECE=%.4f Rec=%.3f epoch=%d",
        duration, r_ho.auc_mean, r_ho.brier_mean, r_ho.ece_mean,
        r_ho.recall_mean, tr.best_epoch,
    )

    return {
        "key": "E1_news_day_conditional",
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "n_total_windows": n_total,
        "n_news_windows": n_news,
        "news_coverage_pct": round(100 * n_news / max(n_total, 1), 2),
        "duration_s": duration,
        "test":     _eval_dict(r_test),
        "held_out": _eval_dict(r_ho),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# E2: Sector-relative target
# ---------------------------------------------------------------------------

def _compute_sector_relative(
    cache: dict[str, StockDataset],
    sym_to_sector: dict[str, str],
    horizon: int = 3,
) -> dict[str, StockDataset]:
    """Return a modified cache where ds.y = 1 iff stock outperforms sector mean.

    Only windows where ≥2 sector peers have data get a valid label.
    In practice all large-cap stocks have complete price data, so missing-peer
    cases only occur at the tail of the series (same position as the existing
    horizon sentinel), preserving flat-array index alignment.
    """
    # Step 1: returns for every stock at every anchor date
    all_returns: dict[str, dict[pd.Timestamp, float]] = {}
    for sym, ds in cache.items():
        close = ds._price_close.copy()
        if close.index.tz is not None:
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        close_dict = {ts.normalize(): float(v) for ts, v in close.items()}
        sorted_dates = sorted(close_dict)
        date_to_idx  = {d: i for i, d in enumerate(sorted_dates)}

        sym_rets: dict[pd.Timestamp, float] = {}
        for anchor_date in ds.dates:
            anchor = pd.Timestamp(anchor_date).normalize()
            idx = date_to_idx.get(anchor)
            if idx is None or idx + horizon >= len(sorted_dates):
                continue
            c0 = close_dict[sorted_dates[idx]]
            c1 = close_dict[sorted_dates[idx + horizon]]
            sym_rets[anchor] = (c1 - c0) / c0
        all_returns[sym] = sym_rets

    # Step 2: invert the sectors dict  →  sector → [symbols]
    sector_to_syms: dict[str, list[str]] = {}
    for sym, sect in sym_to_sector.items():
        sector_to_syms.setdefault(sect, []).append(sym)

    # Step 3: recompute labels
    new_cache: dict[str, StockDataset] = {}
    for sym, ds in cache.items():
        sect = sym_to_sector.get(sym)
        if sect is None:
            continue
        peers = [s for s in sector_to_syms.get(sect, []) if s in all_returns]

        new_ds = StockDataset.__new__(StockDataset)
        new_ds.__dict__.update(ds.__dict__)
        new_ds.X_tech = ds.X_tech
        new_ds.X_sent = ds.X_sent

        new_y = np.full(len(ds.y), -1, dtype=np.int64)
        for i, anchor_date in enumerate(ds.dates):
            anchor   = pd.Timestamp(anchor_date).normalize()
            ret_stock = all_returns.get(sym, {}).get(anchor)
            if ret_stock is None:
                continue
            peer_rets = [
                all_returns[p][anchor]
                for p in peers
                if anchor in all_returns.get(p, {})
            ]
            if len(peer_rets) < 2:
                continue
            new_y[i] = 1 if ret_stock > float(np.mean(peer_rets)) else 0

        valid = new_y >= 0
        new_ds.y     = new_y[valid]
        new_ds.dates = ds.dates[valid]
        new_cache[sym] = new_ds

    return new_cache


def exp2_sector_relative(
    cache: dict[str, StockDataset],
    full_split: Split,
) -> dict:
    """Train with sector-relative outperformance as the prediction target."""
    logger.info("=" * 60)
    logger.info("E2: Sector-relative target (cross-sectional)")
    logger.info("=" * 60)

    sectors_raw  = yaml.safe_load(SECTORS_PATH.open())
    sym_to_sector = {sym: sec for sec, syms in sectors_raw.items() for sym in syms}

    logger.info("Computing sector-relative targets …")
    sector_cache = _compute_sector_relative(cache, sym_to_sector, horizon=3)
    logger.info("Sector-relative cache: %d symbols", len(sector_cache))

    all_y = np.concatenate([ds.y for ds in sector_cache.values()])
    pos_pct = float(100 * all_y.mean())
    logger.info("Class balance: %.1f %% positive (outperformers)", pos_pct)

    symbols   = list(sector_cache.keys())
    sub_split = make_subset_split(symbols, full_split)

    on_gpu = torch.cuda.is_available()
    training, compute = _make_training_config(on_gpu)

    builder = DataLoaderBuilder(sector_cache, sub_split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    held_loader = (
        builder.build_held_out_loader() if sub_split.held_out_symbols else None
    )

    logger.info(
        "Splits — train: %d, val: %d, test: %d, held-out: %d",
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset),
        len(held_loader.dataset) if held_loader else 0,
    )

    model   = _make_model()
    trainer = Trainer(model, training, compute)
    t0      = time.monotonic()
    tr      = trainer.fit(train_loader, val_loader)
    r_test  = trainer.bootstrap_evaluate(test_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    r_ho    = (
        trainer.bootstrap_evaluate(held_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        if held_loader else None
    )
    duration = time.monotonic() - t0

    logger.info(
        "E2 done in %.1fs: HO AUC=%.4f Brier=%.4f ECE=%.4f epoch=%d",
        duration,
        r_ho.auc_mean if r_ho else float("nan"),
        r_ho.brier_mean if r_ho else float("nan"),
        r_ho.ece_mean if r_ho else float("nan"),
        tr.best_epoch,
    )

    return {
        "key": "E2_sector_relative",
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "class_balance_pct_positive": pos_pct,
        "n_symbols": len(symbols),
        "duration_s": duration,
        "test":     _eval_dict(r_test),
        "held_out": _eval_dict(r_ho) if r_ho else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# E3: Full 992-symbol H1+H4 scale validation
# ---------------------------------------------------------------------------

def exp3_full_scale() -> dict:
    """H1+H4 combo on the complete 992-symbol universe."""
    logger.info("=" * 60)
    logger.info("E3: Full 992-symbol H1+H4 scale validation")
    logger.info("=" * 60)

    if not FULL_SPLIT_PATH.exists():
        logger.error("splits.yml not found at %s — skipping E3", FULL_SPLIT_PATH)
        return {"key": "E3_full_scale", "error": "splits.yml not found"}

    full_split = Split.load(FULL_SPLIT_PATH)
    logger.info(
        "Full split: %d train, %d held-out symbols",
        len(full_split.train_symbols), len(full_split.held_out_symbols),
    )

    prices    = PriceRepository()
    sentiment = SentimentRepository()

    datasets: dict[str, StockDataset] = {}
    skipped = 0
    t0_build = time.monotonic()

    for symbol in full_split.all_symbols:
        try:
            price_df = prices.load_years(symbol, PRICE_YEARS)
        except FileNotFoundError:
            skipped += 1
            continue
        sent_df = sentiment.load(symbol) if sentiment.exists(symbol) else None
        try:
            datasets[symbol] = StockDataset(
                symbol=symbol,
                price_df=price_df,
                sentiment_df=sent_df,
                window=20,
                horizon=3,
                target_threshold=BASE_THRESHOLD,
                has_news_feature=False,
            )
        except RuntimeError as exc:
            logger.warning("skip %s: %s", symbol, exc)
            skipped += 1

    logger.info(
        "Built %d datasets, skipped %d  (%.1fs)",
        len(datasets), skipped, time.monotonic() - t0_build,
    )

    on_gpu = torch.cuda.is_available()
    training, compute = _make_training_config(on_gpu)

    builder = DataLoaderBuilder(datasets, full_split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    held_loader = builder.build_held_out_loader()

    logger.info(
        "Splits — train: %d, val: %d, test: %d, held-out: %d",
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset),  len(held_loader.dataset),
    )

    model   = _make_model()
    trainer = Trainer(model, training, compute)
    t0      = time.monotonic()
    tr      = trainer.fit(train_loader, val_loader)
    r_test  = trainer.bootstrap_evaluate(test_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    r_ho    = trainer.bootstrap_evaluate(held_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    duration = time.monotonic() - t0

    logger.info(
        "E3 done in %.1fs: HO AUC=%.4f Brier=%.4f ECE=%.4f epoch=%d",
        duration, r_ho.auc_mean, r_ho.brier_mean, r_ho.ece_mean, tr.best_epoch,
    )

    return {
        "key": "E3_full_scale",
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "n_datasets": len(datasets),
        "n_skipped": skipped,
        "n_train_symbols": len(full_split.train_symbols),
        "n_held_out_symbols": len(full_split.held_out_symbols),
        "duration_s": duration,
        "test":     _eval_dict(r_test),
        "held_out": _eval_dict(r_ho),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("CUDA: %s", torch.cuda.is_available())

    logger.info("Building 50-symbol dataset cache (shared by E1 and E2) …")
    cache      = build_dataset_cache()
    fast_split = Split.load(SPLIT_PATH)

    results: dict[str, dict] = {}
    results["E1"] = exp1_news_day_conditional(cache, fast_split)
    results["E2"] = exp2_sector_relative(cache, fast_split)
    results["E3"] = exp3_full_scale()

    out_path = RESULTS_DIR / "next_level_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", out_path)

    print()
    print("=" * 82)
    print("NEXT LEVEL EXPERIMENTS SUMMARY")
    print("=" * 82)
    print(
        f"{'Experiment':<30} {'Epoch':>6} {'HO AUC':>8} "
        f"{'HO Brier':>10} {'HO ECE':>8} {'HO Rec':>8} {'HO Prec':>8}"
    )
    print("-" * 82)
    print(
        f"{'[H1+H4 winner, 50-sym]':<30} {'13':>6} {'0.5270':>8} "
        f"{'0.2473':>10} {'0.011':>8} {'0.153':>8} {'0.493':>8}"
    )
    print("-" * 82)

    for key, res in results.items():
        if "error" in res:
            print(f"{res.get('key', key):<30}  ERROR: {res['error']}")
            continue
        ho = res.get("held_out")
        if ho is None:
            print(f"{res.get('key', key):<30}  no held-out data")
            continue
        print(
            f"{res.get('key', key):<30} {res['best_epoch']:>6} "
            f"{ho['auc_mean']:>8.4f} {ho['brier_mean']:>10.4f} "
            f"{ho['ece_mean']:>8.4f} {ho['recall_mean']:>8.3f} "
            f"{ho['precision_mean']:>8.3f}"
        )


if __name__ == "__main__":
    main()
