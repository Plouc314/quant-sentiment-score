from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .features.dataset import DataLoaderBuilder, StockDataset
from .model.lstm import SentimentLSTM
from .model.trainer import EvalResult, Trainer, TrainingResult
from .model.transformer import SentimentTransformer
from .repositories.models import ModelRepository
from .repositories.prices import PriceRepository
from .repositories.sentiment import SentimentRepository
from .training import ComputeConfig, Split, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Everything needed to run one training experiment."""

    name: str
    model_type: str  # "lstm" or "transformer"
    split_path: str
    price_years: list[int]
    training: TrainingConfig
    compute: ComputeConfig
    # Model architecture
    hidden_size: int = 64
    num_layers: int = 2
    d_model: int = 64
    nhead: int = 4
    n_encoder_layers: int = 2
    dim_feedforward: int = 128
    # Evaluation
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42
    # Hypothesis parameters
    horizon: int = 3
    target_threshold: float | None = None
    pos_weight: float | None = None
    sentiment_proj_dim: int | None = None
    has_news_feature: bool = False
    # Optional: compare against this checkpoint
    baseline_checkpoint: str | None = None
    description: str = ""

    def to_yaml(self, path: Path) -> None:
        """Serialise to a YAML file."""
        data = asdict(self)
        data["training"] = asdict(self.training)
        data["compute"] = {
            "device": self.compute.device,
            "num_workers": self.compute.num_workers,
            "n_threads": self.compute.n_threads,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        training_fields = {fi.name for fi in fields(TrainingConfig)}
        training = TrainingConfig(
            **{k: v for k, v in data.pop("training").items() if k in training_fields}
        )
        compute_fields = {fi.name for fi in fields(ComputeConfig)}
        compute = ComputeConfig(
            **{k: v for k, v in data.pop("compute").items() if k in compute_fields}
        )
        own_fields = {fi.name for fi in fields(cls)} - {"training", "compute"}
        filtered = {k: v for k, v in data.items() if k in own_fields}
        return cls(training=training, compute=compute, **filtered)


@dataclass
class ExperimentResult:
    """Complete output of a single experiment run."""

    config: ExperimentConfig
    training: TrainingResult
    temporal_test: EvalResult
    held_out: EvalResult
    baseline_temporal_test: EvalResult | None
    baseline_held_out: EvalResult | None
    timestamp: str
    duration_seconds: float
    n_train_windows: int
    n_val_windows: int
    n_test_windows: int
    n_held_out_windows: int
    n_symbols_used: int
    n_symbols_skipped: int


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run one full training experiment.

    Encapsulates the ``train.ipynb`` flow: load data, build datasets,
    train, evaluate (bootstrap), save checkpoint, and optionally compare
    against a baseline.
    """
    if config.model_type not in ("lstm", "transformer"):
        raise ValueError(f"Unknown model_type: {config.model_type!r}")

    split_path = Path(config.split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    config.compute.setup()
    t0 = time.monotonic()

    # --- Load split ---
    split = Split.load(split_path)
    logger.info(
        "Split: %d train, %d held-out symbols",
        len(split.train_symbols),
        len(split.held_out_symbols),
    )

    # --- Load price data ---
    prices = PriceRepository()
    price_dfs: dict = {}
    for symbol in split.all_symbols:
        try:
            price_dfs[symbol] = prices.load_years(symbol, config.price_years)
        except FileNotFoundError:
            price_dfs[symbol] = None

    # --- Load sentiment data ---
    sent_repo = SentimentRepository()
    sentiment_dfs: dict = {}
    for symbol in split.all_symbols:
        if sent_repo.exists(symbol):
            sentiment_dfs[symbol] = sent_repo.load(symbol)
        else:
            sentiment_dfs[symbol] = None

    # --- Build datasets ---
    datasets: dict[str, StockDataset] = {}
    skipped: list[str] = []
    for symbol in split.all_symbols:
        df = price_dfs.get(symbol)
        if df is None:
            skipped.append(symbol)
            continue
        try:
            datasets[symbol] = StockDataset(
                symbol=symbol,
                price_df=df,
                sentiment_df=sentiment_dfs.get(symbol),
                window=config.training.window,
                horizon=config.horizon,
                target_threshold=config.target_threshold,
                has_news_feature=config.has_news_feature,
            )
        except RuntimeError as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
            skipped.append(symbol)

    logger.info("Datasets: %d built, %d skipped", len(datasets), len(skipped))

    # --- Build loaders ---
    builder = DataLoaderBuilder(datasets, split, config.training, config.compute)
    train_loader, val_loader, test_loader = builder.build()
    held_out_loader = builder.build_held_out_loader()

    # --- Create model ---
    n_factors = 16 + (1 if config.has_news_feature else 0)

    if config.model_type == "lstm":
        model = SentimentLSTM(
            n_factors=n_factors,
            sentiment_dim=768,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.training.dropout,
            sentiment_proj_dim=config.sentiment_proj_dim,
        )
    else:
        model = SentimentTransformer(
            n_factors=n_factors,
            sentiment_dim=768,
            d_model=config.d_model,
            nhead=config.nhead,
            n_layers=config.n_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.training.dropout,
            sentiment_proj_dim=config.sentiment_proj_dim,
        )

    # --- Train ---
    class_weights = None
    if config.pos_weight is not None:
        class_weights = torch.tensor([1.0, config.pos_weight])

    trainer = Trainer(model, config.training, config.compute, class_weights=class_weights)
    training_result = trainer.fit(train_loader, val_loader)
    logger.info(
        "Training done: best_epoch=%d, best_val_auc=%.4f",
        training_result.best_epoch,
        training_result.best_val_auc,
    )

    # --- Evaluate ---
    r_test = trainer.bootstrap_evaluate(
        test_loader, n_bootstrap=config.n_bootstrap, seed=config.bootstrap_seed,
    )
    r_ho = trainer.bootstrap_evaluate(
        held_out_loader, n_bootstrap=config.n_bootstrap, seed=config.bootstrap_seed,
    )
    logger.info(
        "Temporal test: AUC=%.3f, Brier=%.3f | Held-out: AUC=%.3f, Brier=%.3f",
        r_test.auc_mean, r_test.brier_mean, r_ho.auc_mean, r_ho.brier_mean,
    )

    # --- Save checkpoint ---
    model_repo = ModelRepository()
    model_repo.save(
        config.name,
        model,
        {
            "model_type": config.model_type,
            "window": config.training.window,
            "n_classes": 2,
            "training_history": training_result.history,
            "best_epoch": training_result.best_epoch,
            "best_val_loss": training_result.best_val_loss,
            "best_val_auc": training_result.best_val_auc,
            "temporal_test_auc": r_test.auc_mean,
            "held_out_auc": r_ho.auc_mean,
        },
    )

    # --- Baseline comparison (optional) ---
    baseline_tt: EvalResult | None = None
    baseline_ho: EvalResult | None = None
    if config.baseline_checkpoint:
        baseline_tt, baseline_ho = _evaluate_baseline(
            config, model_repo, test_loader, held_out_loader,
        )

    duration = time.monotonic() - t0
    return ExperimentResult(
        config=config,
        training=training_result,
        temporal_test=r_test,
        held_out=r_ho,
        baseline_temporal_test=baseline_tt,
        baseline_held_out=baseline_ho,
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_seconds=round(duration, 2),
        n_train_windows=len(train_loader.dataset),
        n_val_windows=len(val_loader.dataset),
        n_test_windows=len(test_loader.dataset),
        n_held_out_windows=len(held_out_loader.dataset),
        n_symbols_used=len(datasets),
        n_symbols_skipped=len(skipped),
    )


def _evaluate_baseline(
    config: ExperimentConfig,
    model_repo: ModelRepository,
    test_loader,
    held_out_loader,
) -> tuple[EvalResult, EvalResult]:
    """Load a baseline checkpoint and evaluate on the same loaders."""
    ckpt = model_repo.load(config.baseline_checkpoint)
    cfg = ckpt.config

    if cfg["model_type"] == "lstm":
        baseline_model = SentimentLSTM(
            n_factors=16,
            sentiment_dim=768,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
        )
    else:
        baseline_model = SentimentTransformer(
            n_factors=16,
            sentiment_dim=768,
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            n_layers=cfg.get("n_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 128),
        )

    baseline_model.load_state_dict(ckpt.state_dict)
    baseline_trainer = Trainer(baseline_model, config.training, config.compute)

    bl_test = baseline_trainer.bootstrap_evaluate(
        test_loader, n_bootstrap=config.n_bootstrap, seed=config.bootstrap_seed,
    )
    bl_ho = baseline_trainer.bootstrap_evaluate(
        held_out_loader, n_bootstrap=config.n_bootstrap, seed=config.bootstrap_seed,
    )
    logger.info(
        "Baseline: test AUC=%.3f, Brier=%.3f | held-out AUC=%.3f, Brier=%.3f",
        bl_test.auc_mean, bl_test.brier_mean, bl_ho.auc_mean, bl_ho.brier_mean,
    )
    return bl_test, bl_ho
