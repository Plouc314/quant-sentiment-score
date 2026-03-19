"""Training loop and evaluation for stock movement prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train with early stopping on validation AUC.

    Parameters
    ----------
    model:
        A model whose ``forward(tech, sentiment)`` returns logits ``(batch, 1)``.
    train_loader, val_loader:
        DataLoaders yielding ``(tech, sentiment, target)`` batches.
    n_epochs:
        Maximum training epochs.
    lr:
        Learning rate for Adam.
    patience:
        Stop after this many epochs without validation AUC improvement.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    Dict with training history (``train_loss``, ``val_loss``, ``val_auc``,
    ``val_accuracy`` lists) and ``best_epoch``, ``best_val_auc``.
    The model weights are restored to the best checkpoint.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_epoch = 0
    best_state: dict | None = None
    epochs_no_improve = 0

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_accuracy": [],
    }

    for epoch in range(1, n_epochs + 1):
        train_loss = _train_epoch(model, optimizer, criterion, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        scheduler.step(val_metrics["auc"])

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | val_acc=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["accuracy"],
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        **history,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model on a DataLoader.

    Returns
    -------
    Dict with keys: ``loss``, ``accuracy``, ``auc``, ``precision``, ``recall``.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0

    with torch.no_grad():
        for tech, sentiment, targets in loader:
            tech = tech.to(device)
            sentiment = sentiment.to(device)
            targets = targets.to(device)

            logits = model(tech, sentiment)
            loss = criterion(logits, targets.unsqueeze(1))
            total_loss += loss.item() * len(targets)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    logits_arr = np.concatenate(all_logits).squeeze()
    targets_arr = np.concatenate(all_targets)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "loss": total_loss / len(targets_arr),
        "accuracy": float(accuracy_score(targets_arr, preds)),
        "precision": float(precision_score(targets_arr, preds, zero_division=0)),
        "recall": float(recall_score(targets_arr, preds, zero_division=0)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(targets_arr, probs))
    except ValueError:
        metrics["auc"] = 0.5  # single class in batch

    return metrics


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for tech, sentiment, targets in loader:
        tech = tech.to(device)
        sentiment = sentiment.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(tech, sentiment)
        loss = criterion(logits, targets.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n_samples += len(targets)

    return total_loss / n_samples
