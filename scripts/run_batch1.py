"""Batch 1 follow-ups: F1 (3-class), F2 (sector ensemble), F3 (horizon ladder), F6 (calibration).

All on the 50-symbol fast subset, all on GPU when available, all magnitude-related.
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
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from src.features.dataset import DataLoaderBuilder
from src.log import setup_logging  # noqa: E402
from src.model.lstm import SentimentLSTM
from src.model.trainer import Trainer
from src.training import ComputeConfig, Split, TrainingConfig

from scripts._phase1_helpers import (  # noqa: E402
    TargetSpec, build_dataset_cache, train_one_combo, override_targets, make_subset_split,
)

setup_logging()
logger = logging.getLogger("batch1")

RESULTS_DIR = ROOT / "data" / "experiments_batch1"
PRED_DIR = RESULTS_DIR / "predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SECTORS_PATH = ROOT / "data" / "sectors_50.yml"
SPLIT_PATH = ROOT / "data" / "splits_fast.yml"


# ============================================================================
# F1: 3-class direct (no_move / big_up / big_down)
# ============================================================================
def f1_three_class(cache: dict, full_split: Split, threshold: float = 0.015, horizon: int = 3) -> dict:
    """Compute 3-class targets per dataset, train an LSTM with 3-output head."""
    logger.info("=" * 60)
    logger.info("F1: 3-class direct (threshold=%.3f, horizon=%d)", threshold, horizon)
    logger.info("=" * 60)

    # Build 3-class targets per symbol
    new_cache = {}
    for sym, ds in cache.items():
        new_ds = type(ds).__new__(type(ds))
        new_ds.__dict__.update(ds.__dict__)
        new_ds.X_tech = ds.X_tech
        new_ds.X_sent = ds.X_sent
        new_ds._price_close = ds._price_close

        close = ds._price_close.copy()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None) if close.index.tz is not None else pd.DatetimeIndex(close.index)
        close_dict = {ts.normalize(): float(v) for ts, v in close.items()}
        sorted_dates = sorted(close_dict.keys())
        date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

        new_y = np.full(len(ds.y), -1, dtype=np.int64)
        for i, anchor_date in enumerate(ds.dates):
            anchor = pd.Timestamp(anchor_date).tz_localize(None).normalize() if pd.Timestamp(anchor_date).tz is not None else pd.Timestamp(anchor_date).normalize()
            if anchor not in date_to_idx:
                continue
            idx = date_to_idx[anchor]
            if idx + horizon >= len(sorted_dates):
                continue
            c_now = close_dict[sorted_dates[idx]]
            c_fut = close_dict[sorted_dates[idx + horizon]]
            ret = (c_fut - c_now) / c_now
            if ret > threshold:
                new_y[i] = 1   # big_up
            elif ret < -threshold:
                new_y[i] = 2   # big_down
            else:
                new_y[i] = 0   # no_move
        valid = new_y >= 0
        new_ds.y = new_y[valid]
        new_ds.dates = ds.dates[valid]
        new_cache[sym] = new_ds

    sub_split = make_subset_split(list(new_cache.keys()), full_split)
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

    builder = DataLoaderBuilder(new_cache, sub_split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    held_out_loader = builder.build_held_out_loader() if sub_split.held_out_symbols else None

    # Model: 3-output head
    model = SentimentLSTM(
        n_factors=16, sentiment_dim=768, hidden_size=64,
        num_layers=2, dropout=training.dropout, sentiment_proj_dim=64,
        n_classes=3,
    )
    trainer = Trainer(model, training, compute)
    t0 = time.monotonic()
    tr = trainer.fit(train_loader, val_loader)
    duration = time.monotonic() - t0

    # Custom 3-class evaluation: collect probs+targets, compute per-class AUC + accuracy
    probs_ho, targets_ho, _ = trainer._collect_predictions(held_out_loader) if held_out_loader else (None, None, None)
    probs_test, targets_test, _ = trainer._collect_predictions(test_loader)

    def evaluate_3class(probs, targets):
        if probs is None:
            return None
        preds = probs.argmax(axis=1)
        accuracy = float((preds == targets).mean())
        # Per-class one-vs-rest AUC
        aucs = {}
        for c, name in [(0, "no_move"), (1, "big_up"), (2, "big_down")]:
            try:
                aucs[name] = float(roc_auc_score((targets == c).astype(int), probs[:, c]))
            except ValueError:
                aucs[name] = float("nan")
        # Macro
        aucs["macro"] = float(np.nanmean(list(aucs.values())))
        # Class distribution
        class_dist = {f"class_{c}_rate": float((targets == c).mean()) for c in (0, 1, 2)}
        # Confusion-style: of predictions where model says class c, what fraction are right?
        per_class_prec = {}
        per_class_rec = {}
        for c, name in [(0, "no_move"), (1, "big_up"), (2, "big_down")]:
            mask = preds == c
            per_class_prec[name] = float((targets[mask] == c).mean()) if mask.sum() > 0 else 0.0
            class_mask = targets == c
            per_class_rec[name] = float((preds[class_mask] == c).mean()) if class_mask.sum() > 0 else 0.0
        return {
            "accuracy": accuracy, "auc": aucs,
            "precision": per_class_prec, "recall": per_class_rec,
            "class_rate": class_dist, "n": int(len(targets)),
        }

    res_ho = evaluate_3class(probs_ho, targets_ho)
    res_test = evaluate_3class(probs_test, targets_test)

    # Save
    np.savez(PRED_DIR / "f1_3class_ho.npz", probs=probs_ho, targets=targets_ho) if probs_ho is not None else None
    logger.info("F1 done in %.1fs: macro AUC=%.4f, acc=%.4f",
                duration, res_ho["auc"]["macro"] if res_ho else 0, res_ho["accuracy"] if res_ho else 0)
    return {
        "best_epoch": tr.best_epoch, "best_val_auc": tr.best_val_auc, "duration_s": duration,
        "test": res_test, "held_out": res_ho,
    }


# ============================================================================
# F2: per-sector magnitude ensemble (uses saved predictions from Phase 1+2)
# ============================================================================
def f2_sector_ensemble() -> dict:
    """Pool the per-sector magnitude predictions; compare aggregate AUC to pooled magnitude."""
    logger.info("=" * 60)
    logger.info("F2: per-sector magnitude ensemble (offline stacking)")
    logger.info("=" * 60)

    sectors = yaml.safe_load(open(SECTORS_PATH))
    pred_root = ROOT / "data" / "experiments_phase1" / "predictions"

    pooled_path = pred_root / "e4_pooled_magnitude.npz"
    if not pooled_path.exists():
        return {"error": f"missing {pooled_path}"}
    pooled_data = np.load(pooled_path)
    pooled_p_big = pooled_data["probs"][:, 1]
    pooled_targets = pooled_data["targets"]
    pooled_auc = float(roc_auc_score(pooled_targets, pooled_p_big))

    sector_probs = []
    sector_targets = []
    sector_aucs = {}
    for sec in sectors:
        path = pred_root / f"e4_sector_{sec}_magnitude.npz"
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        d = np.load(path)
        p_big = d["probs"][:, 1]
        targ = d["targets"]
        try:
            sector_aucs[sec] = float(roc_auc_score(targ, p_big))
        except ValueError:
            sector_aucs[sec] = float("nan")
        sector_probs.append(p_big)
        sector_targets.append(targ)

    # Concatenate per-sector specialist outputs into a single (window, prob, target) bag
    if not sector_probs:
        return {"error": "no sector predictions"}
    routed_probs = np.concatenate(sector_probs)
    routed_targets = np.concatenate(sector_targets)
    routed_auc = float(roc_auc_score(routed_targets, routed_probs))

    # Bootstrap CIs
    def bootstrap_ci(t, p, n_boot=500, seed=42):
        rng = np.random.default_rng(seed)
        N = len(t)
        aucs = []
        for _ in range(n_boot):
            idx = rng.choice(N, size=N, replace=True)
            try:
                aucs.append(roc_auc_score(t[idx], p[idx]))
            except ValueError:
                pass
        arr = np.array(aucs)
        return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    pooled_mean, pooled_lo, pooled_hi = bootstrap_ci(pooled_targets, pooled_p_big)
    routed_mean, routed_lo, routed_hi = bootstrap_ci(routed_targets, routed_probs)

    logger.info("pooled AUC=%.4f [%.4f,%.4f]", pooled_mean, pooled_lo, pooled_hi)
    logger.info("routed AUC=%.4f [%.4f,%.4f]", routed_mean, routed_lo, routed_hi)
    return {
        "pooled_auc": pooled_mean, "pooled_ci": [pooled_lo, pooled_hi], "pooled_n": int(len(pooled_targets)),
        "routed_auc": routed_mean, "routed_ci": [routed_lo, routed_hi], "routed_n": int(len(routed_targets)),
        "delta_routed_minus_pooled": routed_mean - pooled_mean,
        "per_sector_aucs": sector_aucs,
    }


# ============================================================================
# F3: horizon ladder (h=3, 5, 10) — h=1 already done
# ============================================================================
def f3_horizon_ladder(cache: dict, full_split: Split) -> dict:
    logger.info("=" * 60)
    logger.info("F3: magnitude horizon ladder (threshold=1.5%%)")
    logger.info("=" * 60)
    out = {}
    for h in (3, 5, 10):
        spec = TargetSpec(kind="magnitude", horizon=h, magnitude_threshold=0.015)
        logger.info("--- F3: h=%d ---", h)
        t0 = time.monotonic()
        res = train_one_combo(
            symbols=list(cache.keys()), cache=cache, spec=spec, full_split=full_split,
            save_predictions_path=PRED_DIR / f"f3_h{h:02d}_magnitude.npz",
        )
        res["duration_s"] = time.monotonic() - t0
        out[f"h{h:02d}"] = res
        if not res.get("skipped"):
            logger.info("  done h=%d in %.1fs: HO AUC=%.4f",
                        h, res["duration_s"], res["held_out"]["auc_mean"])
    return out


# ============================================================================
# F6: calibration on h=1 winner
# ============================================================================
def f6_calibration() -> dict:
    """Bin P(big) deciles and report empirical big-move rate per bin."""
    logger.info("=" * 60)
    logger.info("F6: calibration on h=1 thr=1%% magnitude detector")
    logger.info("=" * 60)
    pred_path = ROOT / "data" / "experiments_extras" / "predictions" / "b_h1_th_0100.npz"
    if not pred_path.exists():
        return {"error": f"missing {pred_path}"}
    d = np.load(pred_path)
    p_big = d["probs"][:, 1]
    targets = d["targets"]
    n = len(targets)
    base_rate = float(targets.mean())

    # Decile bins
    bin_edges = np.percentile(p_big, np.arange(0, 101, 10))
    bins = []
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < 9:
            mask = (p_big >= lo) & (p_big < hi)
        else:
            mask = (p_big >= lo) & (p_big <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        emp_rate = float(targets[mask].mean())
        mean_pred = float(p_big[mask].mean())
        # Lift over base rate
        lift = emp_rate / base_rate if base_rate > 0 else 0.0
        bins.append({
            "decile": i + 1,
            "p_big_range": [float(lo), float(hi)],
            "n": n_bin,
            "mean_predicted": mean_pred,
            "empirical_big_rate": emp_rate,
            "lift_vs_base_rate": lift,
        })

    # Threshold-strategy table: at each tau, what's the precision and coverage?
    threshold_table = []
    for tau in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        mask = p_big > tau
        n_act = int(mask.sum())
        if n_act < 10:
            threshold_table.append({"tau": tau, "n": n_act, "skipped": True})
            continue
        precision = float(targets[mask].mean())
        coverage = n_act / n
        threshold_table.append({
            "tau": tau, "n": n_act, "coverage": coverage,
            "precision": precision, "lift": precision / base_rate,
        })

    logger.info("Base rate: %.3f, n=%d", base_rate, n)
    return {
        "n_samples": n,
        "base_rate": base_rate,
        "decile_bins": bins,
        "threshold_strategy": threshold_table,
    }


def main():
    cache = build_dataset_cache()
    full_split = Split.load(SPLIT_PATH)
    logger.info("CUDA: %s", torch.cuda.is_available())

    results = {}
    results["F1_3class"] = f1_three_class(cache, full_split)
    results["F2_sector_ensemble"] = f2_sector_ensemble()
    results["F3_horizon_ladder"] = f3_horizon_ladder(cache, full_split)
    results["F6_calibration"] = f6_calibration()

    out = RESULTS_DIR / "batch1_results.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", out)

    # ============ Summary ============
    print()
    print("=" * 70)
    print("BATCH 1 SUMMARY")
    print("=" * 70)

    # F1
    r = results["F1_3class"]
    if r.get("held_out"):
        ho = r["held_out"]
        print(f"\n[F1] 3-class direct (no_move/big_up/big_down):")
        print(f"  HO macro AUC={ho['auc']['macro']:.4f}  accuracy={ho['accuracy']:.4f}  n={ho['n']}")
        print(f"  per-class AUC: no_move={ho['auc']['no_move']:.4f}  big_up={ho['auc']['big_up']:.4f}  big_down={ho['auc']['big_down']:.4f}")
        print(f"  per-class precision: {r['held_out']['precision']}")
        print(f"  per-class recall:    {r['held_out']['recall']}")
        print(f"  class distribution:  {r['held_out']['class_rate']}")

    # F2
    r = results["F2_sector_ensemble"]
    if "error" not in r:
        print(f"\n[F2] per-sector magnitude ensemble:")
        print(f"  pooled AUC={r['pooled_auc']:.4f} {r['pooled_ci']}  (n={r['pooled_n']})")
        print(f"  routed AUC={r['routed_auc']:.4f} {r['routed_ci']}  (n={r['routed_n']})")
        print(f"  delta (routed - pooled): {r['delta_routed_minus_pooled']:+.4f}")
        print(f"  per-sector specialist AUCs:")
        for sec, auc in r["per_sector_aucs"].items():
            print(f"    {sec:<22}: {auc:.4f}")

    # F3
    r = results["F3_horizon_ladder"]
    print(f"\n[F3] magnitude horizon ladder (50-sym, threshold=1.5%):")
    for h_key, hr in r.items():
        if hr.get("skipped"):
            continue
        ho = hr["held_out"]
        print(f"  {h_key:<6}: HO AUC={ho['auc_mean']:.4f} [{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  Brier={ho['brier_mean']:.4f}")
    # add h=1 as reference from extras
    extras = ROOT / "data" / "experiments_extras" / "extras_results.json"
    if extras.exists():
        ed = json.loads(extras.read_text())
        if "B_h1_thresholds" in ed and "h1_th_0.01" in ed["B_h1_thresholds"]:
            ho = ed["B_h1_thresholds"]["h1_th_0.01"]["held_out"]
            print(f"  h=01* : HO AUC={ho['auc_mean']:.4f} [{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  (from extras, threshold=1%)")

    # F6
    r = results["F6_calibration"]
    if "error" not in r:
        print(f"\n[F6] calibration on h=1 thr=1% magnitude (n={r['n_samples']}, base_rate={r['base_rate']:.3f}):")
        print(f"  Decile bins (mean_pred -> empirical big-rate):")
        for b in r["decile_bins"]:
            bar = "#" * int(b["empirical_big_rate"] * 30)
            print(f"    decile {b['decile']:>2}  pred={b['mean_predicted']:.3f}  emp={b['empirical_big_rate']:.3f}  "
                  f"lift={b['lift_vs_base_rate']:.2f}x  n={b['n']:<5} {bar}")
        print(f"  Threshold strategy (tau -> precision):")
        for t in r["threshold_strategy"]:
            if t.get("skipped"):
                continue
            print(f"    tau={t['tau']:.2f}  n={t['n']:>5}  cov={t['coverage']:.1%}  "
                  f"precision={t['precision']:.3f}  lift={t['lift']:.2f}x")


if __name__ == "__main__":
    main()
