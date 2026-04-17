"""Run all experiment configs in a directory sequentially.

Usage::

    python scripts/run_sweep.py
    python scripts/run_sweep.py --configs-dir configs
    python scripts/run_sweep.py --configs-dir configs --results-dir data/experiments
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: E402  (triggers load_dotenv)
from src.experiment import ExperimentConfig, run_experiment  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.repositories.experiments import (  # noqa: E402
    ExperimentRepository,
    compare_experiments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments in a config directory.")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML config files (default: configs/).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: data/experiments/).",
    )
    args = parser.parse_args()

    setup_logging()
    repo = ExperimentRepository(data_dir=args.results_dir)

    configs = sorted(args.configs_dir.glob("*.yml"))
    if not configs:
        print(f"No YAML configs found in {args.configs_dir}")
        return

    print(f"Found {len(configs)} configs:")
    for p in configs:
        print(f"  {p.name}")
    print()

    results = []
    for i, config_path in enumerate(configs, 1):
        config = ExperimentConfig.from_yaml(config_path)
        print(f"[{i}/{len(configs)}] Running: {config.name} ({config_path.name})")

        try:
            result = run_experiment(config)
            repo.save(result)
            results.append(result)
            print(
                f"  Done in {result.duration_seconds:.1f}s — "
                f"test AUC={result.temporal_test.auc_mean:.3f}, "
                f"test Brier={result.temporal_test.brier_mean:.3f}"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")

        print()

    if len(results) >= 2:
        print("=" * 70)
        print("COMPARISON (sorted by test Brier score, lower is better)")
        print("=" * 70)
        df = compare_experiments(results, sort_by="test_brier")
        print(df.to_string(index=False))
    elif results:
        r = results[0]
        print(f"Only one experiment completed: {r.config.name}")
        print(f"  test AUC={r.temporal_test.auc_mean:.3f}, test Brier={r.temporal_test.brier_mean:.3f}")


if __name__ == "__main__":
    main()
