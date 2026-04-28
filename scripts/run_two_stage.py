"""Phase 3 — two-stage stacking (E5).

Loads held-out predictions saved by run_topology.py and run_bigmove.py and
combines them: signed_score = (2*P(up) - 1) * P(big_move). Reports AUC of
this stacked score against the original direction labels, plus the rank
correlation against actual signed returns.

This is a no-training step — pure stacking of already-computed predictions.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("two_stage")

PRED_DIR = ROOT / "data" / "experiments_phase1" / "predictions"
RESULTS_DIR = ROOT / "data" / "experiments_phase1"


def load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["probs"], data["targets"]


def main() -> None:
    direction_path = PRED_DIR / "e1_pooled_direction.npz"
    magnitude_path = PRED_DIR / "e4_pooled_magnitude.npz"

    if not direction_path.exists() or not magnitude_path.exists():
        logger.error("Need both %s and %s. Run run_topology.py and run_bigmove.py first.",
                     direction_path, magnitude_path)
        sys.exit(1)

    dir_probs, dir_targets = load_pred(direction_path)
    mag_probs, mag_targets = load_pred(magnitude_path)

    # Sanity: predictions are computed on the same DataLoader (held-out) so
    # length/order should match for pooled-on-50 + pooled-magnitude. Verify.
    if len(dir_probs) != len(mag_probs):
        logger.error("Length mismatch: direction=%d, magnitude=%d", len(dir_probs), len(mag_probs))
        sys.exit(1)

    p_up = dir_probs[:, 1]
    p_big = mag_probs[:, 1]
    direction_label = dir_targets               # 0/1, original direction labels
    magnitude_label = mag_targets               # 0/1, big-move labels

    print()
    print("=" * 70)
    print("PHASE 3 — TWO-STAGE STACKING (held-out, n=%d)" % len(p_up))
    print("=" * 70)

    # Stage 1 alone (direction)
    auc_dir = roc_auc_score(direction_label, p_up)
    print(f"\n[Stage 1 alone — direction]:    HO AUC = {auc_dir:.4f}")

    # Stage 2 alone (magnitude)
    auc_mag = roc_auc_score(magnitude_label, p_big)
    print(f"[Stage 2 alone — magnitude]:    HO AUC = {auc_mag:.4f}")

    # Two-stage signed score: positive when model predicts up AND big move
    signed_score = (2 * p_up - 1) * p_big       # in [-1, 1], magnitude-weighted

    # Evaluate against direction (does it improve direction prediction?)
    # Map signed_score into [0, 1] for AUC vs direction label
    score_normed = (signed_score + 1) / 2
    auc_two_stage_vs_dir = roc_auc_score(direction_label, score_normed)
    print(f"\n[Two-stage product vs direction]:   HO AUC = {auc_two_stage_vs_dir:.4f}  "
          f"(delta vs Stage 1: {auc_two_stage_vs_dir - auc_dir:+.4f})")

    # Top-K trades: rank by |signed_score| (confidence of magnitude * direction)
    # and check direction accuracy on top-K.
    print(f"\n[Top-K confidence — direction accuracy on top-K most confident]:")
    abs_score = np.abs(signed_score)
    pred_direction = (signed_score > 0).astype(int)
    correct = (pred_direction == direction_label).astype(int)
    order = np.argsort(-abs_score)
    for k in (50, 200, 1000, 5000):
        if k > len(order):
            continue
        top = order[:k]
        acc = correct[top].mean()
        # Also AUC restricted to top-K
        try:
            auc_top = roc_auc_score(direction_label[top], score_normed[top])
        except ValueError:
            auc_top = float("nan")
        print(f"  top-{k:>5}:  accuracy={acc:.4f}  AUC={auc_top:.4f}")

    # Filter strategy: only act when P(big move) > some threshold
    print(f"\n[Filter strategy — direction AUC restricted to P(big_move) > tau]:")
    for tau in (0.3, 0.4, 0.5, 0.6):
        mask = p_big > tau
        n = mask.sum()
        if n < 100:
            print(f"  tau={tau}:  n={n} (too few, skip)")
            continue
        try:
            auc_filt = roc_auc_score(direction_label[mask], p_up[mask])
        except ValueError:
            auc_filt = float("nan")
        print(f"  tau={tau}:  n={n}  direction AUC={auc_filt:.4f}  "
              f"(delta vs all: {auc_filt - auc_dir:+.4f})")

    out = {
        "n_samples": int(len(p_up)),
        "stage1_direction_auc": float(auc_dir),
        "stage2_magnitude_auc": float(auc_mag),
        "two_stage_vs_direction_auc": float(auc_two_stage_vs_dir),
    }
    (RESULTS_DIR / "two_stage_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
