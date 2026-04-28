"""Phase 1 — topology experiments (E1 + E2 + E3).

E1: pooled on all 50 symbols
E2: per-supersector (5 sectors, ~6-15 symbols each)
E3: per-stock on all 50 symbols, aggregate

All use the same combo: target_threshold=0.005, sentiment_proj_dim=64,
lr=1e-4, dropout=0.2, val-AUC early stopping.

Saves held-out predictions to data/experiments_phase1/predictions/ for
downstream two-stage stacking.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._phase1_helpers import (  # noqa: E402
    TargetSpec, build_dataset_cache, train_one_combo,
)
from src.log import setup_logging  # noqa: E402
from src.training import Split  # noqa: E402

setup_logging()
logger = logging.getLogger("topology")

RESULTS_DIR = ROOT / "data" / "experiments_phase1"
PRED_DIR = RESULTS_DIR / "predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SECTORS_PATH = ROOT / "data" / "sectors_50.yml"
SPLIT_PATH = ROOT / "data" / "splits_fast.yml"

DIRECTION_SPEC = TargetSpec(
    kind="direction", horizon=3, return_threshold=0.005,
)


def main() -> None:
    sectors: dict[str, list[str]] = yaml.safe_load(open(SECTORS_PATH))
    full_split = Split.load(SPLIT_PATH)
    cache = build_dataset_cache()
    all_symbols = list(cache.keys())
    logger.info("Cache built: %d symbols", len(all_symbols))

    results: dict[str, dict] = {}

    # ============ E1: pooled on all 50 ============
    logger.info("=" * 60)
    logger.info("E1: pooled on all 50 symbols")
    logger.info("=" * 60)
    t0 = time.monotonic()
    res_e1 = train_one_combo(
        symbols=all_symbols,
        cache=cache,
        spec=DIRECTION_SPEC,
        full_split=full_split,
        save_predictions_path=PRED_DIR / "e1_pooled_direction.npz",
    )
    res_e1["duration_s"] = time.monotonic() - t0
    results["e1_pooled"] = res_e1
    logger.info("E1 done in %.1fs: HO AUC=%.4f, Brier=%.4f, n_HO=%d",
                res_e1["duration_s"], res_e1["held_out"]["auc_mean"],
                res_e1["held_out"]["brier_mean"], res_e1["held_out"]["n_samples"])

    # ============ E2: per-supersector ============
    logger.info("=" * 60)
    logger.info("E2: per-supersector (%d sectors)", len(sectors))
    logger.info("=" * 60)
    sector_results: dict[str, dict] = {}
    for sector_name, sector_symbols in sectors.items():
        logger.info("--- sector: %s (%d symbols) ---", sector_name, len(sector_symbols))
        t0 = time.monotonic()
        res = train_one_combo(
            symbols=sector_symbols,
            cache=cache,
            spec=DIRECTION_SPEC,
            full_split=full_split,
            save_predictions_path=PRED_DIR / f"e2_sector_{sector_name}_direction.npz",
        )
        res["duration_s"] = time.monotonic() - t0
        sector_results[sector_name] = res
        if res.get("skipped"):
            logger.warning("  skipped: %s", res.get("reason"))
        else:
            logger.info("  done in %.1fs: HO AUC=%.4f, n_HO=%d",
                        res["duration_s"], res["held_out"]["auc_mean"],
                        res["held_out"]["n_samples"])
    results["e2_sectors"] = sector_results

    # ============ E3: per-stock on all 50 ============
    logger.info("=" * 60)
    logger.info("E3: per-stock on all 50")
    logger.info("=" * 60)
    per_stock_results: dict[str, dict] = {}
    for symbol in all_symbols:
        logger.info("--- stock: %s ---", symbol)
        t0 = time.monotonic()
        res = train_one_combo(
            symbols=[symbol],
            cache=cache,
            spec=DIRECTION_SPEC,
            full_split=full_split,
            save_predictions_path=None,   # too many files
            n_bootstrap=100,              # cheaper per-stock
        )
        res["duration_s"] = time.monotonic() - t0
        per_stock_results[symbol] = res
        if res.get("skipped"):
            logger.info("  skipped %s: %s", symbol, res.get("reason"))
        else:
            ho = res["held_out"]
            tst = res["test"]
            metric = ho if res["n_held_out_symbols"] > 0 else tst
            label = "HO" if res["n_held_out_symbols"] > 0 else "TST"
            logger.info("  %s done in %.1fs: %s AUC=%.4f, n=%d",
                        symbol, res["duration_s"], label,
                        metric["auc_mean"], metric["n_samples"])
    results["e3_per_stock"] = per_stock_results

    # ============ Save and summarise ============
    out_path = RESULTS_DIR / "topology_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", out_path)

    print()
    print("=" * 70)
    print("PHASE 1 SUMMARY (held-out unless noted)")
    print("=" * 70)

    # E1
    ho = results["e1_pooled"]["held_out"]
    print(f"\n[E1] pooled-on-50:                HO AUC={ho['auc_mean']:.4f} "
          f"[{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  Brier={ho['brier_mean']:.4f}")

    # E2
    print(f"\n[E2] per-sector:")
    sector_aucs = []
    for name, res in results["e2_sectors"].items():
        if res.get("skipped"):
            print(f"  {name:<22}: skipped ({res.get('reason')})")
            continue
        ho = res["held_out"]
        if res["n_held_out_symbols"] == 0:
            tst = res["test"]
            print(f"  {name:<22}: (no HO)  TST AUC={tst['auc_mean']:.4f}  n={tst['n_samples']}")
        else:
            sector_aucs.append(ho["auc_mean"])
            print(f"  {name:<22}: HO AUC={ho['auc_mean']:.4f} "
                  f"[{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  n={ho['n_samples']}")
    if sector_aucs:
        print(f"  {'macro-mean':<22}: HO AUC={sum(sector_aucs)/len(sector_aucs):.4f}")

    # E3
    per_stock_aucs: list[tuple[str, float, str]] = []
    for sym, res in results["e3_per_stock"].items():
        if res.get("skipped"):
            continue
        if res["n_held_out_symbols"] > 0:
            per_stock_aucs.append((sym, res["held_out"]["auc_mean"], "HO"))
        else:
            per_stock_aucs.append((sym, res["test"]["auc_mean"], "TST"))
    per_stock_aucs.sort(key=lambda r: -r[1])
    aucs_only = [a for _, a, _ in per_stock_aucs]
    if aucs_only:
        import statistics
        print(f"\n[E3] per-stock (n={len(aucs_only)} models):")
        print(f"  mean   AUC: {statistics.mean(aucs_only):.4f}")
        print(f"  median AUC: {statistics.median(aucs_only):.4f}")
        print(f"  top-10 AUC: {sum(aucs_only[:10])/10:.4f}")
        print(f"  top-25 AUC: {sum(aucs_only[:25])/25:.4f}")
        print(f"  worst   AUC: {min(aucs_only):.4f}  best AUC: {max(aucs_only):.4f}")
        print(f"  top-10 stocks: {[(s, round(a, 3)) for s, a, _ in per_stock_aucs[:10]]}")


if __name__ == "__main__":
    main()
