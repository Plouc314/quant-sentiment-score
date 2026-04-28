"""Phase 2 — big-move detector (E4).

Same architecture and topology variants as topology.py, but the target is
|return at horizon| > 1.0% (volatility / "is something happening" task).

Saves held-out predictions for E5 stacking.
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
logger = logging.getLogger("bigmove")

RESULTS_DIR = ROOT / "data" / "experiments_phase1"
PRED_DIR = RESULTS_DIR / "predictions"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SECTORS_PATH = ROOT / "data" / "sectors_50.yml"
SPLIT_PATH = ROOT / "data" / "splits_fast.yml"

MAGNITUDE_SPEC = TargetSpec(
    kind="magnitude", horizon=3, magnitude_threshold=0.015,
)


def main() -> None:
    sectors: dict[str, list[str]] = yaml.safe_load(open(SECTORS_PATH))
    full_split = Split.load(SPLIT_PATH)
    cache = build_dataset_cache()
    all_symbols = list(cache.keys())
    logger.info("Cache built: %d symbols", len(all_symbols))

    results: dict[str, dict] = {}

    # ============ E4a: pooled big-move detector ============
    logger.info("=" * 60)
    logger.info("E4a: pooled big-move detector (|ret(h=3)| > 1%%)")
    logger.info("=" * 60)
    t0 = time.monotonic()
    res = train_one_combo(
        symbols=all_symbols,
        cache=cache,
        spec=MAGNITUDE_SPEC,
        full_split=full_split,
        save_predictions_path=PRED_DIR / "e4_pooled_magnitude.npz",
    )
    res["duration_s"] = time.monotonic() - t0
    results["e4a_pooled_magnitude"] = res
    logger.info("E4a done in %.1fs: HO AUC=%.4f, Brier=%.4f, n_HO=%d",
                res["duration_s"], res["held_out"]["auc_mean"],
                res["held_out"]["brier_mean"], res["held_out"]["n_samples"])

    # ============ E4b: per-sector big-move detector ============
    logger.info("=" * 60)
    logger.info("E4b: per-sector big-move detector")
    logger.info("=" * 60)
    sector_results: dict[str, dict] = {}
    for sector_name, sector_symbols in sectors.items():
        logger.info("--- sector: %s (%d symbols) ---", sector_name, len(sector_symbols))
        t0 = time.monotonic()
        s_res = train_one_combo(
            symbols=sector_symbols,
            cache=cache,
            spec=MAGNITUDE_SPEC,
            full_split=full_split,
            save_predictions_path=PRED_DIR / f"e4_sector_{sector_name}_magnitude.npz",
        )
        s_res["duration_s"] = time.monotonic() - t0
        sector_results[sector_name] = s_res
        if s_res.get("skipped"):
            logger.warning("  skipped: %s", s_res.get("reason"))
        else:
            logger.info("  done in %.1fs: HO AUC=%.4f, n_HO=%d",
                        s_res["duration_s"], s_res["held_out"]["auc_mean"],
                        s_res["held_out"]["n_samples"])
    results["e4b_sectors_magnitude"] = sector_results

    # ============ Save and summarise ============
    out_path = RESULTS_DIR / "bigmove_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Saved %s", out_path)

    print()
    print("=" * 70)
    print("PHASE 2 SUMMARY — BIG-MOVE DETECTOR (held-out)")
    print("=" * 70)
    ho = results["e4a_pooled_magnitude"]["held_out"]
    print(f"\n[E4a] pooled magnitude:           HO AUC={ho['auc_mean']:.4f} "
          f"[{ho['auc_ci_low']:.4f},{ho['auc_ci_high']:.4f}]  Brier={ho['brier_mean']:.4f}  "
          f"rec={ho['recall_mean']:.3f} prec={ho['precision_mean']:.3f}")

    print(f"\n[E4b] per-sector magnitude:")
    sector_aucs = []
    for name, res in results["e4b_sectors_magnitude"].items():
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


if __name__ == "__main__":
    main()
