"""Feature ablation sweep: tech-only, sentiment-only, both.

Bypasses the YAML config loop — directly builds datasets, zeros one
modality, then runs train+eval. All other settings match the adopted
combo (target_threshold=0.005, sentiment_proj_dim=64, lr=1e-4, dr=0.2).
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

import src  # noqa: E402  (load_dotenv)
from src.features.dataset import DataLoaderBuilder, StockDataset  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.model.lstm import SentimentLSTM  # noqa: E402
from src.model.trainer import Trainer  # noqa: E402
from src.repositories.prices import PriceRepository  # noqa: E402
from src.repositories.sentiment import SentimentRepository  # noqa: E402
from src.training import ComputeConfig, Split, TrainingConfig  # noqa: E402

setup_logging()
logger = logging.getLogger("ablation")

SPLIT_PATH = ROOT / "data" / "splits_fast.yml"
PRICE_YEARS = list(range(2018, 2025))
RESULTS_DIR = ROOT / "data" / "experiments_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ABLATIONS = ("both", "tech_only", "sentiment_only")


def build_datasets(split: Split) -> dict[str, StockDataset]:
    prices = PriceRepository()
    sentiment = SentimentRepository()
    datasets: dict[str, StockDataset] = {}
    for symbol in split.all_symbols:
        try:
            price_df = prices.load_years(symbol, PRICE_YEARS)
        except FileNotFoundError:
            continue
        sent_df = sentiment.load(symbol) if sentiment.exists(symbol) else None
        try:
            datasets[symbol] = StockDataset(
                symbol=symbol,
                price_df=price_df,
                sentiment_df=sent_df,
                window=20,
                horizon=3,
                target_threshold=0.005,
                has_news_feature=False,
            )
        except RuntimeError as exc:
            logger.warning("skip %s: %s", symbol, exc)
    return datasets


def apply_ablation(datasets: dict[str, StockDataset], mode: str) -> None:
    for ds in datasets.values():
        if mode == "tech_only":
            ds.X_sent = np.zeros_like(ds.X_sent)
        elif mode == "sentiment_only":
            ds.X_tech = np.zeros_like(ds.X_tech)
        elif mode == "both":
            pass
        else:
            raise ValueError(f"Unknown ablation mode: {mode!r}")


def run_one(mode: str, split: Split, datasets_template: dict[str, StockDataset]) -> dict:
    logger.info("=" * 60)
    logger.info("ABLATION: %s", mode)
    logger.info("=" * 60)

    # Deep-copy arrays so the next iteration starts from clean data.
    datasets: dict[str, StockDataset] = {}
    for s, src_ds in datasets_template.items():
        ds = StockDataset.__new__(StockDataset)
        ds.__dict__.update(src_ds.__dict__)
        ds.X_tech = src_ds.X_tech.copy()
        ds.X_sent = src_ds.X_sent.copy()
        datasets[s] = ds
    apply_ablation(datasets, mode)

    training = TrainingConfig(
        window=20, batch_size=32, n_epochs=50, lr=1e-4, weight_decay=1e-4,
        patience=10, dropout=0.2, seed=42, scheduler="plateau",
        scheduler_patience=5, grad_clip=1.0, early_stopping_metric="auc",
    )
    compute = ComputeConfig(device=None, num_workers=0)
    compute.setup()

    builder = DataLoaderBuilder(datasets, split, training, compute)
    train_loader, val_loader, test_loader = builder.build()
    held_out_loader = builder.build_held_out_loader()

    model = SentimentLSTM(
        n_factors=16, sentiment_dim=768, hidden_size=64,
        num_layers=2, dropout=training.dropout, sentiment_proj_dim=64,
    )
    trainer = Trainer(model, training, compute)

    t0 = time.monotonic()
    tr = trainer.fit(train_loader, val_loader)
    r_test = trainer.bootstrap_evaluate(test_loader, n_bootstrap=200, seed=42)
    r_ho = trainer.bootstrap_evaluate(held_out_loader, n_bootstrap=200, seed=42)
    duration = time.monotonic() - t0

    logger.info("Done %s in %.1fs: HO AUC=%.4f, Brier=%.4f",
                mode, duration, r_ho.auc_mean, r_ho.brier_mean)

    result = {
        "mode": mode,
        "best_epoch": tr.best_epoch,
        "best_val_auc": tr.best_val_auc,
        "duration_s": duration,
        "test": {k: getattr(r_test, k) for k in vars(r_test) if not k.startswith("_")},
        "held_out": {k: getattr(r_ho, k) for k in vars(r_ho) if not k.startswith("_")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = RESULTS_DIR / f"ablation_{mode}.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    split = Split.load(SPLIT_PATH)
    logger.info("Loaded split: %d train, %d held-out symbols",
                len(split.train_symbols), len(split.held_out_symbols))
    datasets_template = build_datasets(split)
    logger.info("Built %d datasets", len(datasets_template))

    rows = []
    for mode in ABLATIONS:
        rows.append(run_one(mode, split, datasets_template))

    print()
    print("=" * 70)
    print("ABLATION RESULTS (held-out)")
    print("=" * 70)
    print(f"{'mode':<18} {'epoch':>6} {'AUC':>8} {'Brier':>8} {'ECE':>8} {'Rec':>8} {'Prec':>8}")
    for r in rows:
        ho = r["held_out"]
        print(f"{r['mode']:<18} {r['best_epoch']:>6} {ho['auc_mean']:>8.4f} "
              f"{ho['brier_mean']:>8.4f} {ho['ece_mean']:>8.4f} "
              f"{ho['recall_mean']:>8.3f} {ho['precision_mean']:>8.3f}")


if __name__ == "__main__":
    main()
