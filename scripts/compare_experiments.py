"""Compare all saved experiment results in a ranked table.

Usage::

    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --sort-by test_auc
    python scripts/compare_experiments.py --model-type lstm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: E402  (triggers load_dotenv)
from src.repositories.experiments import ExperimentRepository, compare_experiments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved experiment results.")
    parser.add_argument(
        "--sort-by",
        default="test_brier",
        help="Column to sort by (default: test_brier).",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Filter to a specific model type (lstm or transformer).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing results JSON files (default: data/experiments/).",
    )
    args = parser.parse_args()

    repo = ExperimentRepository(data_dir=args.results_dir)
    results = repo.load_all()

    if not results:
        print("No experiment results found.")
        return

    if args.model_type:
        results = [r for r in results if r.config.model_type == args.model_type]

    df = compare_experiments(results, sort_by=args.sort_by)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
