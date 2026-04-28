"""F3 — conditional direction given big move (offline analysis).

Loads saved predictions:
  - data/experiments_phase1/predictions/e1_pooled_direction.npz   (P(up), direction labels)
  - data/experiments_phase3/predictions/b1_magnitude_both.npz     (P(big), magnitude labels)

Both are aligned (same pooled-on-50 held-out loader, same window order).

Tests:
  1. Conditional direction AUC = AUC of P(up) restricted to windows where
     magnitude_label == 1 (actual big moves happened).
  2. Confidence-filtered direction AUC = AUC of P(up) restricted to windows
     where P(big) > tau (model thinks big move is likely).
  3. Combined ranking score = P(up) * P(big) — tested as a "directional bet
     weighted by magnitude confidence" signal.

If conditional > unconditional with non-overlapping CI, the two-stage signal
is operationally meaningful.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("conditional")

DIR_PATH = ROOT / "data" / "experiments_phase1" / "predictions" / "e1_pooled_direction.npz"
MAG_PATH = ROOT / "data" / "experiments_phase3" / "predictions" / "b1_magnitude_both.npz"


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n: int = 500, seed: int = 42):
    """Bootstrap CI on AUC."""
    rng = np.random.default_rng(seed)
    aucs = []
    N = len(y_true)
    for _ in range(n):
        idx = rng.choice(N, size=N, replace=True)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            pass
    arr = np.array(aucs)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)


def main() -> None:
    if not DIR_PATH.exists() or not MAG_PATH.exists():
        logger.error("missing predictions: %s, %s", DIR_PATH, MAG_PATH)
        sys.exit(1)

    dir_data = np.load(DIR_PATH)
    mag_data = np.load(MAG_PATH)

    p_up = dir_data["probs"][:, 1]
    direction_label = dir_data["targets"]
    p_big = mag_data["probs"][:, 1]
    magnitude_label = mag_data["targets"]

    assert len(p_up) == len(p_big), f"length mismatch: {len(p_up)} vs {len(p_big)}"
    n = len(p_up)
    print()
    print("=" * 70)
    print(f"CONDITIONAL DIRECTION (held-out, n={n})")
    print("=" * 70)
    print(f"\nLabel base rates:  direction(up)={direction_label.mean():.3f}  "
          f"magnitude(big)={magnitude_label.mean():.3f}")

    # Unconditional reference
    auc, lo, hi = bootstrap_auc(direction_label, p_up)
    print(f"\n[Unconditional direction]: AUC={auc:.4f} [{lo:.4f}, {hi:.4f}]   (reference)")

    # Conditional on actual big move
    print(f"\n[Conditional on actual big move (magnitude_label==1)]:")
    for label_val, label_name in [(1, "actual big move"), (0, "actual small move")]:
        mask = magnitude_label == label_val
        n_sub = int(mask.sum())
        auc, lo, hi = bootstrap_auc(direction_label[mask], p_up[mask])
        delta = auc - 0.5249  # vs unconditional reference
        print(f"  subset={label_name:<22}  n={n_sub:>6}  "
              f"AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  delta_vs_unconditional={delta:+.4f}")

    # Conditional on model's magnitude prediction
    print(f"\n[Conditional on P(big_move) > tau]:")
    for tau in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7):
        mask = p_big > tau
        n_sub = int(mask.sum())
        if n_sub < 200:
            print(f"  tau={tau}:  n={n_sub:<6} (too few, skip)")
            continue
        auc, lo, hi = bootstrap_auc(direction_label[mask], p_up[mask])
        delta = auc - 0.5249
        frac = n_sub / n
        print(f"  tau={tau:<5}  n={n_sub:>6}  ({frac:.0%} of HO)  "
              f"AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  delta={delta:+.4f}")

    # Top-K most confident magnitude predictions, then check direction AUC
    print(f"\n[Top-K most confident magnitude predictions, direction AUC on those]:")
    order = np.argsort(-p_big)  # most confident big-move first
    for k in (50, 100, 500, 1000, 5000):
        if k > n:
            continue
        idx = order[:k]
        try:
            auc, lo, hi = bootstrap_auc(direction_label[idx], p_up[idx])
        except ValueError:
            continue
        # Also: precision (fraction that actually were big moves)
        actual_big = magnitude_label[idx].mean()
        delta = auc - 0.5249
        print(f"  top-{k:<5}  AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  "
              f"actual_big={actual_big:.3f}  delta={delta:+.4f}")

    # Combined score: P(up) * P(big) for direction, P(down) * P(big) for short
    print(f"\n[Combined score P(up)*P(big) — does magnitude weighting help direction?]:")
    score = p_up * p_big
    auc, lo, hi = bootstrap_auc(direction_label, score)
    print(f"  combined (full HO):  AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  delta={auc-0.5249:+.4f}")

    # Combined on big-move subset
    mask = magnitude_label == 1
    auc, lo, hi = bootstrap_auc(direction_label[mask], score[mask])
    print(f"  combined on actual big moves only:  AUC={auc:.4f} [{lo:.4f},{hi:.4f}]")


if __name__ == "__main__":
    main()
