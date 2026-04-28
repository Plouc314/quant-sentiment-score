"""10 new experiments building on the H1+H4 winning combo.

Groups:
  A — threshold sweep:  0.002 / 0.003 / 0.007 / 0.010 / 0.015
  B — horizon combos:   H1+H4+h1 / H1+H4+h5 / H1+H4+h10
  C — architecture:     hidden=128 / num_layers=3

All run on the 50-symbol fast subset; results saved to data/experiments_new/.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from src.features.dataset import DataLoaderBuilder, StockDataset
from src.log import setup_logging
from src.model.lstm import SentimentLSTM
from src.model.trainer import Trainer
from src.repositories.prices import PriceRepository
from src.repositories.sentiment import SentimentRepository
from src.training import ComputeConfig, Split, TrainingConfig

from scripts._phase1_helpers import (
    TargetSpec, build_dataset_cache, train_one_combo, override_targets,
    make_subset_split,
)

setup_logging()
logger = logging.getLogger("new_experiments")

SPLIT_PATH = ROOT / "data" / "splits_fast.yml"
RESULTS_DIR = ROOT / "data" / "experiments_new"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 200
SEED = 42

# Winning combo defaults
BASE_THRESHOLD = 0.005
BASE_PROJ_DIM = 64
BASE_HIDDEN = 64
BASE_LAYERS = 2
BASE_HORIZON = 3


# ---------------------------------------------------------------------------
# Architecture runner (needed when train_one_combo's hardcoded arch won't do)
# ---------------------------------------------------------------------------

def run_arch_experiment(
    key: str,
    cache: dict[str, StockDataset],
    full_split: Split,
    threshold: float = BASE_THRESHOLD,
    horizon: int = BASE_HORIZON,
    hidden_size: int = BASE_HIDDEN,
    num_layers: int = BASE_LAYERS,
    proj_dim: int = BASE_PROJ_DIM,
) -> dict:
    """Train with custom architecture; mirrors train_one_combo's training settings."""
    spec = TargetSpec(kind="direction", horizon=horizon, return_threshold=threshold)
    symbols = list(cache.keys())
    sub_split = make_subset_split(symbols, full_split)
    datasets = {s: override_targets(cache[s], spec) for s in symbols if s in cache}

    on_gpu = torch.cuda.is_available()
    bs = 128 if on_gpu else 32
    nw = 4 if on_gpu else 0

    training = TrainingConfig(
        window=20, batch_size=bs, n_epochs=50, lr=1e-4, weight_decay=1e-4,
        patience=10, dropout=0.2, seed=SEED, scheduler="plateau",
        scheduler_patience=5, grad_clip=1.0, early_stopping_metric="auc",
    )
    compute = ComputeConfig(device=None, num_workers=nw)
    compute.setup()

    builder = DataLoaderBuilder(datasets, sub_split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    held_out_loader = builder.build_held_out_loader() if sub_split.held_out_symbols else None

    model = SentimentLSTM(
        n_factors=16, sentiment_dim=768,
        hidden_size=hidden_size, num_layers=num_layers,
        dropout=training.dropout, sentiment_proj_dim=proj_dim,
    )
    trainer = Trainer(model, training, compute)

    t0 = time.monotonic()
    tr = trainer.fit(train_loader, val_loader)
    r_test = trainer.bootstrap_evaluate(test_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    r_ho = (
        trainer.bootstrap_evaluate(held_out_loader, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        if held_out_loader is not None else None
    )
    duration = time.monotonic() - t0

    logger.info(
        "%s done in %.1fs: HO AUC=%.4f Brier=%.4f epoch=%d",
        key, duration,
        r_ho.auc_mean if r_ho else float("nan"),
        r_ho.brier_mean if r_ho else float("nan"),
        tr.best_epoch,
    )
    return {
        "key": key,
        "skipped": False,
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "duration_s": duration,
        "params": {
            "threshold": threshold, "horizon": horizon,
            "hidden_size": hidden_size, "num_layers": num_layers,
            "proj_dim": proj_dim,
        },
        "test": {k: float(getattr(r_test, k)) for k in vars(r_test) if not k.startswith("_")},
        "held_out": (
            {k: float(getattr(r_ho, k)) for k in vars(r_ho) if not k.startswith("_")}
            if r_ho is not None else None
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    full_split = Split.load(SPLIT_PATH)
    logger.info("CUDA: %s", torch.cuda.is_available())
    logger.info("Building dataset cache …")
    cache = build_dataset_cache()
    symbols = list(cache.keys())

    results: dict[str, dict] = {}

    # =========================================================================
    # Group A: threshold sweep (proj_dim=64, horizon=3)
    # =========================================================================
    for thresh, key in [
        (0.002, "A1_thresh_0020"),
        (0.003, "A2_thresh_0030"),
        (0.007, "A3_thresh_0070"),
        (0.010, "A4_thresh_0100"),
        (0.015, "A5_thresh_0150"),
    ]:
        logger.info("=" * 60)
        logger.info("Running %s (threshold=%.3f)", key, thresh)
        logger.info("=" * 60)
        spec = TargetSpec(kind="direction", horizon=BASE_HORIZON, return_threshold=thresh)
        t0 = time.monotonic()
        res = train_one_combo(symbols, cache, spec, full_split, seed=SEED, n_bootstrap=N_BOOTSTRAP)
        res["key"] = key
        res["params"] = {"threshold": thresh, "horizon": BASE_HORIZON, "proj_dim": BASE_PROJ_DIM}
        res["duration_s"] = time.monotonic() - t0
        res["timestamp"] = datetime.now(timezone.utc).isoformat()
        results[key] = res
        if not res.get("skipped") and res.get("held_out"):
            logger.info(
                "  %s: HO AUC=%.4f Brier=%.4f epoch=%d",
                key, res["held_out"]["auc_mean"], res["held_out"]["brier_mean"], res["best_epoch"],
            )

    # =========================================================================
    # Group B: horizon combos with H1+H4
    # =========================================================================
    for horizon, key in [
        (1,  "B1_combo_h1"),
        (5,  "B2_combo_h5"),
        (10, "B3_combo_h10"),
    ]:
        logger.info("=" * 60)
        logger.info("Running %s (threshold=%.3f, horizon=%d)", key, BASE_THRESHOLD, horizon)
        logger.info("=" * 60)
        spec = TargetSpec(kind="direction", horizon=horizon, return_threshold=BASE_THRESHOLD)
        t0 = time.monotonic()
        res = train_one_combo(symbols, cache, spec, full_split, seed=SEED, n_bootstrap=N_BOOTSTRAP)
        res["key"] = key
        res["params"] = {"threshold": BASE_THRESHOLD, "horizon": horizon, "proj_dim": BASE_PROJ_DIM}
        res["duration_s"] = time.monotonic() - t0
        res["timestamp"] = datetime.now(timezone.utc).isoformat()
        results[key] = res
        if not res.get("skipped") and res.get("held_out"):
            logger.info(
                "  %s: HO AUC=%.4f Brier=%.4f epoch=%d",
                key, res["held_out"]["auc_mean"], res["held_out"]["brier_mean"], res["best_epoch"],
            )

    # =========================================================================
    # Group C: architecture variants (threshold=0.005, horizon=3)
    # =========================================================================
    for params, key in [
        ({"hidden_size": 128, "num_layers": BASE_LAYERS}, "C1_hidden128"),
        ({"hidden_size": BASE_HIDDEN, "num_layers": 3},   "C2_layers3"),
    ]:
        logger.info("=" * 60)
        logger.info("Running %s %s", key, params)
        logger.info("=" * 60)
        res = run_arch_experiment(key, cache, full_split, **params)
        results[key] = res

    # =========================================================================
    # Save all results
    # =========================================================================
    out_path = RESULTS_DIR / "new_experiments_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", out_path)

    # =========================================================================
    # Summary table
    # =========================================================================
    print()
    print("=" * 80)
    print("NEW EXPERIMENTS SUMMARY (held-out set, 50-symbol fast subset)")
    print("=" * 80)
    print(f"{'Key':<22} {'Epoch':>6} {'HO AUC':>8} {'HO Brier':>10} {'HO ECE':>8} {'HO Rec':>8} {'HO Prec':>8}")
    print("-" * 80)

    # Reference row
    print(f"{'[H1+H4 winner]':<22} {'13':>6} {'0.5270':>8} {'0.2473':>10} {'0.011':>8} {'0.153':>8} {'0.493':>8}")
    print("-" * 80)

    for key, res in results.items():
        if res.get("skipped"):
            print(f"{key:<22} SKIPPED")
            continue
        ho = res.get("held_out")
        if ho is None:
            print(f"{key:<22} no held-out data")
            continue
        print(
            f"{key:<22} {res['best_epoch']:>6} "
            f"{ho['auc_mean']:>8.4f} {ho['brier_mean']:>10.4f} "
            f"{ho['ece_mean']:>8.4f} {ho['recall_mean']:>8.3f} {ho['precision_mean']:>8.3f}"
        )

    print()
    print("Best by HO AUC:")
    ranked = [
        (k, r) for k, r in results.items()
        if not r.get("skipped") and r.get("held_out")
    ]
    ranked.sort(key=lambda x: x[1]["held_out"]["auc_mean"], reverse=True)
    for k, r in ranked[:5]:
        ho = r["held_out"]
        print(f"  {k:<22} AUC={ho['auc_mean']:.4f}  Brier={ho['brier_mean']:.4f}")


if __name__ == "__main__":
    main()
