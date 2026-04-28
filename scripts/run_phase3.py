"""Phase 3 — sentiment ablation, asymmetric tails, sector-relative.

Experiments (all on the 50-symbol fast subset):
  B1: magnitude detector — both / tech_only / sentiment_only ablation
  A1: big_up + big_down (two binary models — together = 3-class)
  A2: tail risk (return < -2% in next 3 days)
  C1: sector-relative outperformance (return > sector_mean_return)

All use the adopted combo (target_threshold=0.005 where relevant,
sentiment_proj_dim=64, lr=1e-4, dr=0.2). GPU-aware via train_one_combo.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._phase1_helpers import (  # noqa: E402
    TargetSpec, build_dataset_cache, train_one_combo,
)
from src.log import setup_logging  # noqa: E402
from src.training import Split  # noqa: E402

setup_logging()
logger = logging.getLogger("phase3")

RESULTS_DIR = ROOT / "data" / "experiments_phase3"
PRED_DIR = RESULTS_DIR / "predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SECTORS_PATH = ROOT / "data" / "sectors_50.yml"
SPLIT_PATH = ROOT / "data" / "splits_fast.yml"

MAG_SPEC = TargetSpec(kind="magnitude", horizon=3, magnitude_threshold=0.015)
BIG_UP_SPEC = TargetSpec(kind="big_up", horizon=3, magnitude_threshold=0.015)
BIG_DOWN_SPEC = TargetSpec(kind="big_down", horizon=3, magnitude_threshold=0.015)
TAIL_RISK_SPEC = TargetSpec(kind="big_down", horizon=3, magnitude_threshold=0.020)


def apply_sector_relative_target(cache: dict, sectors: dict[str, list[str]], horizon: int = 3) -> dict:
    """Replace each ds.y with 1 if stock return > sector mean return that day, else 0.
    Returns a new cache dict (does not mutate input)."""
    new_cache = {}
    sym_to_sector = {s: sec for sec, syms in sectors.items() for s in syms}

    # Compute per-sector daily mean returns
    sector_returns: dict[str, dict[pd.Timestamp, float]] = {sec: {} for sec in sectors}
    per_symbol_returns: dict[str, dict[pd.Timestamp, float]] = {}

    for sym, ds in cache.items():
        close = ds._price_close.copy()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None) if close.index.tz is not None else pd.DatetimeIndex(close.index)
        close_sorted = close.sort_index()
        # h-day return per anchor day
        ret = (close_sorted.shift(-horizon) - close_sorted) / close_sorted
        per_symbol_returns[sym] = {ts.normalize(): float(v) for ts, v in ret.items() if pd.notna(v)}

    # For each (sector, date), compute mean across constituents
    for sec, syms in sectors.items():
        all_dates: set = set()
        for sym in syms:
            if sym in per_symbol_returns:
                all_dates.update(per_symbol_returns[sym].keys())
        for d in all_dates:
            vals = [per_symbol_returns[sym].get(d) for sym in syms if sym in per_symbol_returns and d in per_symbol_returns[sym]]
            if vals:
                sector_returns[sec][d] = float(np.mean(vals))

    # Now relabel each dataset
    for sym, ds in cache.items():
        sec = sym_to_sector.get(sym)
        if sec is None:
            continue
        new_ds = type(ds).__new__(type(ds))
        new_ds.__dict__.update(ds.__dict__)
        new_ds.X_tech = ds.X_tech
        new_ds.X_sent = ds.X_sent
        new_ds._price_close = ds._price_close
        new_y = np.full(len(ds.y), -1, dtype=np.int64)
        sym_rets = per_symbol_returns.get(sym, {})
        sec_rets = sector_returns.get(sec, {})
        for i, d in enumerate(ds.dates):
            anchor = pd.Timestamp(d).tz_localize(None).normalize() if pd.Timestamp(d).tz is not None else pd.Timestamp(d).normalize()
            if anchor not in sym_rets or anchor not in sec_rets:
                continue
            new_y[i] = 1 if sym_rets[anchor] > sec_rets[anchor] else 0
        valid = new_y >= 0
        new_ds.y = new_y[valid]
        new_ds.dates = ds.dates[valid]
        new_cache[sym] = new_ds
    return new_cache


def main() -> None:
    sectors: dict[str, list[str]] = yaml.safe_load(open(SECTORS_PATH))
    full_split = Split.load(SPLIT_PATH)
    cache = build_dataset_cache()
    all_symbols = list(cache.keys())
    logger.info("Cache built: %d symbols", len(all_symbols))
    logger.info("CUDA available: %s | device count: %d",
                torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)

    results: dict[str, dict] = {}

    # ============ B1: ablation on magnitude ============
    logger.info("=" * 60)
    logger.info("B1: magnitude ablation (both / tech_only / sentiment_only)")
    logger.info("=" * 60)
    b1: dict[str, dict] = {}
    for ablation in ("both", "tech_only", "sentiment_only"):
        logger.info("--- B1: %s ---", ablation)
        t0 = time.monotonic()
        res = train_one_combo(
            symbols=all_symbols, cache=cache, spec=MAG_SPEC,
            full_split=full_split, ablation=ablation,
            save_predictions_path=PRED_DIR / f"b1_magnitude_{ablation}.npz",
        )
        res["duration_s"] = time.monotonic() - t0
        b1[ablation] = res
        if not res.get("skipped"):
            ho = res["held_out"]
            logger.info("  done %s in %.1fs: HO AUC=%.4f",
                        ablation, res["duration_s"], ho["auc_mean"])
    results["b1_ablation_magnitude"] = b1

    # ============ A1: big_up + big_down (split 3-class) ============
    logger.info("=" * 60)
    logger.info("A1: big_up + big_down (combined = 3-class signal)")
    logger.info("=" * 60)
    a1: dict[str, dict] = {}
    for name, spec in [("big_up", BIG_UP_SPEC), ("big_down", BIG_DOWN_SPEC)]:
        logger.info("--- A1: %s ---", name)
        t0 = time.monotonic()
        res = train_one_combo(
            symbols=all_symbols, cache=cache, spec=spec,
            full_split=full_split,
            save_predictions_path=PRED_DIR / f"a1_{name}.npz",
        )
        res["duration_s"] = time.monotonic() - t0
        a1[name] = res
        if not res.get("skipped"):
            logger.info("  done %s in %.1fs: HO AUC=%.4f",
                        name, res["duration_s"], res["held_out"]["auc_mean"])
    results["a1_big_up_down"] = a1

    # ============ A2: tail risk ============
    logger.info("=" * 60)
    logger.info("A2: tail risk (return < -2%% in next 3 days)")
    logger.info("=" * 60)
    t0 = time.monotonic()
    res = train_one_combo(
        symbols=all_symbols, cache=cache, spec=TAIL_RISK_SPEC,
        full_split=full_split,
        save_predictions_path=PRED_DIR / "a2_tail_risk.npz",
    )
    res["duration_s"] = time.monotonic() - t0
    results["a2_tail_risk"] = res
    if not res.get("skipped"):
        logger.info("A2 done in %.1fs: HO AUC=%.4f",
                    res["duration_s"], res["held_out"]["auc_mean"])

    # ============ C1: sector-relative outperformance ============
    logger.info("=" * 60)
    logger.info("C1: sector-relative outperformance (pooled-on-50, sector-relative target)")
    logger.info("=" * 60)
    rel_cache = apply_sector_relative_target(cache, sectors, horizon=3)
    logger.info("Sector-relative cache built: %d symbols", len(rel_cache))
    t0 = time.monotonic()
    # We use the direction-style spec but the targets are already overwritten;
    # train_one_combo will call override_targets which would clobber them.
    # Instead, give it a TargetSpec it accepts then bypass via a stub: easiest
    # is to use direction kind, then re-override targets in the cache.
    # Workaround: build datasets manually here.
    from src.features.dataset import DataLoaderBuilder
    from src.model.lstm import SentimentLSTM
    from src.model.trainer import Trainer
    from src.training import ComputeConfig, TrainingConfig
    from scripts._phase1_helpers import make_subset_split

    sub_split = make_subset_split(list(rel_cache.keys()), full_split)
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
    builder = DataLoaderBuilder(rel_cache, sub_split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    held_out_loader = builder.build_held_out_loader() if sub_split.held_out_symbols else None
    model = SentimentLSTM(
        n_factors=16, sentiment_dim=768, hidden_size=64,
        num_layers=2, dropout=training.dropout, sentiment_proj_dim=64,
    )
    trainer = Trainer(model, training, compute)
    tr = trainer.fit(train_loader, val_loader)
    r_test = trainer.bootstrap_evaluate(test_loader, n_bootstrap=200, seed=42)
    r_ho = trainer.bootstrap_evaluate(held_out_loader, n_bootstrap=200, seed=42) if held_out_loader else None
    duration = time.monotonic() - t0
    results["c1_sector_relative"] = {
        "skipped": False,
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "duration_s": duration,
        "test": {k: float(getattr(r_test, k)) for k in vars(r_test) if not k.startswith("_")},
        "held_out": {k: float(getattr(r_ho, k)) for k in vars(r_ho) if not k.startswith("_")} if r_ho else None,
    }
    if r_ho:
        logger.info("C1 done in %.1fs: HO AUC=%.4f", duration, r_ho.auc_mean)

    # ============ Save and summarise ============
    out_path = RESULTS_DIR / "phase3_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", out_path)

    print()
    print("=" * 70)
    print("PHASE 3 SUMMARY (held-out)")
    print("=" * 70)

    # B1
    print(f"\n[B1] magnitude ablation:")
    for abl, res in results["b1_ablation_magnitude"].items():
        if res.get("skipped"):
            continue
        ho = res["held_out"]
        print(f"  {abl:<18}: HO AUC={ho['auc_mean']:.4f} "
              f"[{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  "
              f"Brier={ho['brier_mean']:.4f}  rec={ho['recall_mean']:.3f}  prec={ho['precision_mean']:.3f}")

    # A1
    print(f"\n[A1] split 3-class:")
    for name, res in results["a1_big_up_down"].items():
        if res.get("skipped"):
            continue
        ho = res["held_out"]
        print(f"  {name:<10}: HO AUC={ho['auc_mean']:.4f} "
              f"[{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  "
              f"pos_rate~{ho['n_samples'] and 'see brier'}, "
              f"Brier={ho['brier_mean']:.4f}  rec={ho['recall_mean']:.3f}  prec={ho['precision_mean']:.3f}")

    # A2
    res = results["a2_tail_risk"]
    if not res.get("skipped"):
        ho = res["held_out"]
        print(f"\n[A2] tail risk (return < -2%):    HO AUC={ho['auc_mean']:.4f} "
              f"[{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  "
              f"Brier={ho['brier_mean']:.4f}  rec={ho['recall_mean']:.3f}  prec={ho['precision_mean']:.3f}")

    # C1
    res = results["c1_sector_relative"]
    if res.get("held_out"):
        ho = res["held_out"]
        print(f"\n[C1] sector-relative outperformance:  HO AUC={ho['auc_mean']:.4f} "
              f"[{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  "
              f"Brier={ho['brier_mean']:.4f}")


if __name__ == "__main__":
    main()
