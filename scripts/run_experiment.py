"""Run a single ML experiment from a YAML config file.

Usage::

    python scripts/run_experiment.py configs/experiment_01.yml
    python scripts/run_experiment.py configs/experiment_01.yml --results-dir data/experiments
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: E402  (triggers load_dotenv)
from src.experiment import ExperimentConfig, run_experiment  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.repositories.experiments import ExperimentRepository  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ML experiment from a YAML config.")
    parser.add_argument("config", type=Path, help="Path to the experiment YAML config file.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory to save results JSON (default: data/experiments/).",
    )
    args = parser.parse_args()

    setup_logging()

    config = ExperimentConfig.from_yaml(args.config)
    result = run_experiment(config)

    repo = ExperimentRepository(data_dir=args.results_dir)
    path = repo.save(result)

    # Print summary
    r = result
    t, h = r.temporal_test, r.held_out
    print(f"\n{'=' * 60}")
    print(f"Experiment : {r.config.name}")
    print(f"Model      : {r.config.model_type}")
    print(f"Duration   : {r.duration_seconds:.1f}s")
    print(f"Best epoch : {r.training.best_epoch}")
    print(f"Symbols    : {r.n_symbols_used} used, {r.n_symbols_skipped} skipped")
    print(f"Windows    : {r.n_train_windows} train, {r.n_val_windows} val, {r.n_test_windows} test, {r.n_held_out_windows} held-out")
    print(f"\n{'Metric':<12} {'Temporal test':>16} {'Held-out':>16}")
    print(f"{'-' * 12} {'-' * 16} {'-' * 16}")
    for metric in ["auc", "accuracy", "precision", "recall", "brier", "ece", "pr_auc"]:
        tv = getattr(t, f"{metric}_mean")
        hv = getattr(h, f"{metric}_mean")
        print(f"{metric:<12} {tv:>16.4f} {hv:>16.4f}")
    print(f"{'=' * 60}")
    print(f"Results saved: {path}")


if __name__ == "__main__":
    main()
