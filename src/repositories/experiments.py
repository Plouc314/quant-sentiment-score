from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentRepository:
    """Saves and loads experiment results as JSON files.

    Layout::

        <data_dir>/
            baseline_lstm.json
            experiment_01.json
            ...
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "experiments"
        self._data_dir = data_dir

    def save(self, result: object) -> Path:
        """Save an ``ExperimentResult`` as ``<data_dir>/<name>.json``.

        The *result* is imported late to avoid circular imports (the
        ``experiment`` module imports from ``repositories``).
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)
        name = result.config.name  # type: ignore[attr-defined]
        path = self._data_dir / f"{name}.json"
        data = _serialise(result)
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved experiment result: %s", path)
        return path

    def load(self, name: str) -> object:
        """Load an experiment result by name.

        Returns an ``ExperimentResult`` (imported lazily).
        Raises ``FileNotFoundError`` if absent.
        """
        path = self._data_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No experiment result: {path}")
        data = json.loads(path.read_text())
        return _deserialise(data)

    def list(self) -> list[str]:
        """Return names of all saved experiment results."""
        if not self._data_dir.exists():
            return []
        return [p.stem for p in sorted(self._data_dir.glob("*.json"))]

    def load_all(self) -> list[object]:
        """Load every saved experiment result."""
        return [self.load(name) for name in self.list()]


def compare_experiments(
    results: list,
    sort_by: str = "test_brier",
) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of ``ExperimentResult`` objects.

    Each row is one experiment.  Columns include key config params and all
    metric means for both temporal-test and held-out evaluation sets.

    Parameters
    ----------
    sort_by:
        Column name to sort by.  Brier/ECE sort ascending (lower is better);
        all others sort descending.
    """
    rows: list[dict] = []
    for r in results:
        cfg = r.config
        row: dict = {
            "name": cfg.name,
            "model_type": cfg.model_type,
            "lr": cfg.training.lr,
            "dropout": cfg.training.dropout,
            "window": cfg.training.window,
            "batch_size": cfg.training.batch_size,
            "best_epoch": r.training.best_epoch,
            "duration_s": round(r.duration_seconds, 1),
        }
        for prefix, er in [("test", r.temporal_test), ("ho", r.held_out)]:
            row[f"{prefix}_auc"] = er.auc_mean
            row[f"{prefix}_accuracy"] = er.accuracy_mean
            row[f"{prefix}_precision"] = er.precision_mean
            row[f"{prefix}_recall"] = er.recall_mean
            row[f"{prefix}_brier"] = er.brier_mean
            row[f"{prefix}_ece"] = er.ece_mean
            row[f"{prefix}_pr_auc"] = er.pr_auc_mean
        rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        ascending = "brier" in sort_by or "ece" in sort_by
        df = df.sort_values(sort_by, ascending=ascending, ignore_index=True)
    return df


# ------------------------------------------------------------------
# Private: serialisation helpers
# ------------------------------------------------------------------


def _serialise(result: object) -> dict:
    """Convert an ExperimentResult to a JSON-safe dict."""
    return asdict(result)  # type: ignore[arg-type]


def _deserialise(data: dict) -> object:
    """Reconstruct an ExperimentResult from a JSON dict."""
    from ..experiment import ExperimentConfig, ExperimentResult
    from ..model.trainer import EvalResult, TrainingResult
    from ..training import ComputeConfig, TrainingConfig

    def _build(cls: type, d: dict) -> object:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    training_cfg = _build(TrainingConfig, data["config"]["training"])
    compute_cfg = _build(ComputeConfig, data["config"]["compute"])
    config = ExperimentConfig(
        **{
            k: v
            for k, v in data["config"].items()
            if k not in ("training", "compute")
        },
        training=training_cfg,
        compute=compute_cfg,
    )

    training_result = _build(TrainingResult, data["training"])
    temporal_test = _build(EvalResult, data["temporal_test"])
    held_out = _build(EvalResult, data["held_out"])

    baseline_tt = (
        _build(EvalResult, data["baseline_temporal_test"])
        if data.get("baseline_temporal_test")
        else None
    )
    baseline_ho = (
        _build(EvalResult, data["baseline_held_out"])
        if data.get("baseline_held_out")
        else None
    )

    return ExperimentResult(
        config=config,
        training=training_result,
        temporal_test=temporal_test,
        held_out=held_out,
        baseline_temporal_test=baseline_tt,
        baseline_held_out=baseline_ho,
        timestamp=data["timestamp"],
        duration_seconds=data["duration_seconds"],
        n_train_windows=data["n_train_windows"],
        n_val_windows=data["n_val_windows"],
        n_test_windows=data["n_test_windows"],
        n_held_out_windows=data["n_held_out_windows"],
        n_symbols_used=data["n_symbols_used"],
        n_symbols_skipped=data["n_symbols_skipped"],
    )
