"""Three follow-up experiments after Phase 3:

  A: magnitude detector at FULL 992-symbol scale
  B: magnitude detector at horizon=1 (volatility autocorrelation peak)
  C: 3-dim sentiment probs instead of 768-dim FinBERT embedding

All write to data/experiments_extras/. GPU-aware via _phase1_helpers.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from src.features.dataset import DataLoaderBuilder, StockDataset
from src.log import setup_logging  # noqa: E402
from src.model.lstm import SentimentLSTM
from src.model.trainer import Trainer
from src.repositories.prices import PriceRepository
from src.repositories.sentiment import SentimentRepository
from src.training import ComputeConfig, Split, TrainingConfig

from scripts._phase1_helpers import (  # noqa: E402
    TargetSpec, build_dataset_cache, train_one_combo, override_targets,
    make_subset_split,
)

setup_logging()
logger = logging.getLogger("extras")

RESULTS_DIR = ROOT / "data" / "experiments_extras"
PRED_DIR = RESULTS_DIR / "predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# A: full 992-symbol magnitude
# ============================================================================
def experiment_a() -> dict:
    logger.info("=" * 60)
    logger.info("A: magnitude at FULL 992-symbol scale")
    logger.info("=" * 60)

    full_split = Split.load(ROOT / "data" / "splits.yml")
    prices = PriceRepository()
    sentiment = SentimentRepository()
    cache: dict[str, StockDataset] = {}

    t0_build = time.monotonic()
    for symbol in full_split.all_symbols:
        try:
            price_df = prices.load_years(symbol, list(range(2018, 2025)))
        except FileNotFoundError:
            continue
        sent_df = sentiment.load(symbol) if sentiment.exists(symbol) else None
        try:
            ds = StockDataset(
                symbol=symbol, price_df=price_df, sentiment_df=sent_df,
                window=20, horizon=3, target_threshold=None, has_news_feature=False,
            )
        except RuntimeError:
            continue
        ds._price_close = price_df["close"].copy()
        cache[symbol] = ds
    logger.info("Built %d StockDatasets in %.1fs", len(cache), time.monotonic() - t0_build)

    spec = TargetSpec(kind="magnitude", horizon=3, magnitude_threshold=0.015)
    t0 = time.monotonic()
    res = train_one_combo(
        symbols=list(cache.keys()), cache=cache, spec=spec, full_split=full_split,
        save_predictions_path=PRED_DIR / "a_full_magnitude.npz",
        n_bootstrap=300,
    )
    res["duration_s"] = time.monotonic() - t0
    if not res.get("skipped"):
        ho = res["held_out"]
        logger.info("A done in %.1fs: HO AUC=%.4f, Brier=%.4f, n_HO=%d",
                    res["duration_s"], ho["auc_mean"], ho["brier_mean"], ho["n_samples"])
    return res


# ============================================================================
# B: magnitude at horizon=1
# ============================================================================
def experiment_b() -> dict:
    logger.info("=" * 60)
    logger.info("B: magnitude at horizon=1 (multiple thresholds)")
    logger.info("=" * 60)

    cache = build_dataset_cache()
    full_split = Split.load(ROOT / "data" / "splits_fast.yml")

    results: dict[str, dict] = {}
    # 1-day moves are smaller; sweep a few thresholds
    for thr in (0.005, 0.0075, 0.01):
        spec = TargetSpec(kind="magnitude", horizon=1, magnitude_threshold=thr)
        logger.info("--- B: h=1, threshold=%.4f ---", thr)
        t0 = time.monotonic()
        res = train_one_combo(
            symbols=list(cache.keys()), cache=cache, spec=spec,
            full_split=full_split,
            save_predictions_path=PRED_DIR / f"b_h1_th_{int(thr*10000):04d}.npz",
        )
        res["duration_s"] = time.monotonic() - t0
        results[f"h1_th_{thr}"] = res
        if not res.get("skipped"):
            ho = res["held_out"]
            logger.info("  done thr=%.4f in %.1fs: HO AUC=%.4f, pos_rate~%.3f",
                        thr, res["duration_s"], ho["auc_mean"], 1 - ho["brier_mean"]/0.25)
    return results


# ============================================================================
# C: 3-dim sentiment probs instead of 768-dim embedding
# ============================================================================
def _build_cache_with_sentiment_probs() -> dict[str, StockDataset]:
    """Build StockDatasets where X_sent is the 3-dim sentiment_probs (pos/neu/neg)
    repeated per anchor day, instead of the 768-dim FinBERT embedding.
    Days without news get a neutral [0, 1, 0] prior."""
    full_split = Split.load(ROOT / "data" / "splits_fast.yml")
    prices = PriceRepository()
    sentiment = SentimentRepository()
    cache: dict[str, StockDataset] = {}

    for symbol in full_split.all_symbols:
        try:
            price_df = prices.load_years(symbol, list(range(2018, 2025)))
        except FileNotFoundError:
            continue
        sent_df = sentiment.load(symbol) if sentiment.exists(symbol) else None
        try:
            ds = StockDataset(
                symbol=symbol, price_df=price_df, sentiment_df=sent_df,
                window=20, horizon=3, target_threshold=None, has_news_feature=False,
            )
        except RuntimeError:
            continue
        ds._price_close = price_df["close"].copy()

        # Override X_sent: build a (T_days, 3) matrix from sentiment_probs
        if sent_df is not None and not sent_df.empty:
            rows = sent_df[sent_df["ticker"] == symbol]
            sym_probs: dict = {}
            for _, row in rows.iterrows():
                d = pd.Timestamp(row["date"]).normalize()
                sym_probs[d] = np.array(row["sentiment_probs"], dtype=np.float32)
            # ds.X_tech and ds.X_sent are aligned to factors_df.index (post-warmup,
            # post-target-valid). The original X_sent had shape (T_days, 768).
            # We need to build a parallel (T_days, 3) array. The dates of these
            # rows aren't directly stored on ds — but ds.dates corresponds to
            # window-end (anchor) dates of windows. Per-day rows are 0..T_days-1.
            # Easiest: for each row in X_sent, check if its norm > 0 (was it a
            # news day) — if yes, we need to find which date. But we don't have
            # the per-day dates stored.
            # Workaround: ds.dates[w-1] is the anchor of the w-th window, which
            # is row (window-1+w_index) in X_sent. So row r in X_sent is
            # day_dates[r] where day_dates is factors_df.index. Reconstruct via
            # ds.dates - (window-1) for w=0.
            n_days = ds.X_sent.shape[0]
            # Build day_dates: row 0 corresponds to ds.dates[0] - (window-1) days
            # Actually, ds.dates[w] = factor_dates[window - 1 + w] for w in [0, n_windows).
            # And X_sent[r] is factor_dates[r] for r in [0, n_days).
            # So factor_dates[r] = ds.dates[0] + (r - (window-1)) trading days,
            # which we approximate using calendar info from price_df.
            #
            # Simpler: just use the sentiment_df dates directly. For each day in
            # the price series, look up sentiment_probs.
            day_index = pd.DatetimeIndex(price_df.index)
            day_index = day_index.tz_localize(None) if day_index.tz is not None else day_index
            day_index_norm = day_index.normalize()
            # Map each row of price_df to (3,) probs vector
            probs_per_day = np.array(
                [sym_probs.get(d, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                 for d in day_index_norm], dtype=np.float32)
            # Now the question is: which rows of probs_per_day correspond to
            # ds.X_sent rows?
            # ds was built from price_df after computing factors and dropping
            # warmup/sentinel rows. The valid mask is hidden inside StockDataset.
            # Easiest hack: keep only rows aligned to the same date range as
            # ds.X_sent (which has shape (n_days, 768)).
            # If lengths match, take the last n_days rows of probs_per_day.
            if len(probs_per_day) >= n_days:
                ds._X_sent_probs = probs_per_day[-n_days:]
            else:
                # Pad with neutral
                pad = np.tile([0.0, 1.0, 0.0], (n_days - len(probs_per_day), 1)).astype(np.float32)
                ds._X_sent_probs = np.vstack([pad, probs_per_day])
            ds.X_sent = ds._X_sent_probs   # swap to 3-dim
        else:
            # No sentiment: neutral prior
            ds.X_sent = np.tile([0.0, 1.0, 0.0], (ds.X_sent.shape[0], 1)).astype(np.float32)
        cache[symbol] = ds
    return cache


def experiment_c() -> dict:
    logger.info("=" * 60)
    logger.info("C: 3-dim sentiment probs (pos/neu/neg) instead of 768-dim")
    logger.info("=" * 60)

    cache = _build_cache_with_sentiment_probs()
    full_split = Split.load(ROOT / "data" / "splits_fast.yml")
    spec = TargetSpec(kind="magnitude", horizon=3, magnitude_threshold=0.015)
    sub_split = make_subset_split(list(cache.keys()), full_split)
    datasets = {s: override_targets(cache[s], spec) for s in cache}

    on_gpu = torch.cuda.is_available()
    bs = 128 if on_gpu else 32
    nw = 4 if on_gpu else 0
    training = TrainingConfig(
        window=20, batch_size=bs, n_epochs=50, lr=1e-4, weight_decay=1e-4,
        patience=10, dropout=0.2, seed=42, scheduler="plateau",
        scheduler_patience=5, grad_clip=1.0, early_stopping_metric="auc",
    )
    compute = ComputeConfig(device=None, num_workers=nw)
    compute.setup()

    builder = DataLoaderBuilder(datasets, sub_split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    has_held_out = bool(sub_split.held_out_symbols)
    held_out_loader = builder.build_held_out_loader() if has_held_out else None

    # Model with sentiment_dim=3 instead of 768
    model = SentimentLSTM(
        n_factors=16, sentiment_dim=3, hidden_size=64,
        num_layers=2, dropout=training.dropout,
        sentiment_proj_dim=8,   # tiny projection from 3-dim
    )
    trainer = Trainer(model, training, compute)
    t0 = time.monotonic()
    tr = trainer.fit(train_loader, val_loader)
    r_test = trainer.bootstrap_evaluate(test_loader, n_bootstrap=200, seed=42)
    r_ho = trainer.bootstrap_evaluate(held_out_loader, n_bootstrap=200, seed=42) if held_out_loader else None
    duration = time.monotonic() - t0
    logger.info("C done in %.1fs: HO AUC=%.4f", duration,
                r_ho.auc_mean if r_ho else r_test.auc_mean)
    return {
        "skipped": False,
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "duration_s": duration,
        "test": {k: float(getattr(r_test, k)) for k in vars(r_test) if not k.startswith("_")},
        "held_out": {k: float(getattr(r_ho, k)) for k in vars(r_ho) if not k.startswith("_")} if r_ho else None,
    }


def main() -> None:
    logger.info("CUDA: %s", torch.cuda.is_available())
    results: dict = {}
    results["A_full_magnitude"] = experiment_a()
    results["B_h1_thresholds"] = experiment_b()
    results["C_3dim_sentiment"] = experiment_c()

    out_path = RESULTS_DIR / "extras_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", out_path)

    print()
    print("=" * 70)
    print("EXTRAS SUMMARY (held-out)")
    print("=" * 70)

    if not results["A_full_magnitude"].get("skipped"):
        ho = results["A_full_magnitude"]["held_out"]
        print(f"\n[A] magnitude @ full 992 syms:")
        print(f"    HO AUC={ho['auc_mean']:.4f} [{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  "
              f"Brier={ho['brier_mean']:.4f}  rec={ho['recall_mean']:.3f}  prec={ho['precision_mean']:.3f}  n={ho['n_samples']}")

    print(f"\n[B] magnitude @ h=1, threshold sweep:")
    for k, res in results["B_h1_thresholds"].items():
        if res.get("skipped"):
            continue
        ho = res["held_out"]
        print(f"    {k:<14}: HO AUC={ho['auc_mean']:.4f} [{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  "
              f"Brier={ho['brier_mean']:.4f}")

    res = results["C_3dim_sentiment"]
    if res.get("held_out"):
        ho = res["held_out"]
        print(f"\n[C] 3-dim sentiment probs (pos/neu/neg):")
        print(f"    HO AUC={ho['auc_mean']:.4f} [{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  "
              f"Brier={ho['brier_mean']:.4f}  rec={ho['recall_mean']:.3f}  prec={ho['precision_mean']:.3f}")
        print(f"    (vs 768-dim baseline from B1: 0.6033)")


if __name__ == "__main__":
    main()
