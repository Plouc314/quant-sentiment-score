"""Training loop and evaluation for stock movement prediction."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BootstrapResult(TypedDict):
    """Confidence intervals for evaluation metrics from :func:`bootstrap_evaluate`."""

    auc_mean: float
    auc_ci_low: float
    auc_ci_high: float
    accuracy_mean: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    precision_mean: float
    precision_ci_low: float
    precision_ci_high: float
    recall_mean: float
    recall_ci_low: float
    recall_ci_high: float
    n_bootstrap: int
    """Actual number of valid resamples used (may be < requested if single-class draws occur)."""
    n_samples: int
    """Total number of test samples."""


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
        A model whose ``forward(tech, sentiment, fundamentals)`` returns logits
        ``(batch, 1)``.  ``fundamentals`` may be an empty tensor when the model
        has ``n_fundamentals=0``.
    train_loader, val_loader:
        DataLoaders yielding ``(tech, sentiment, fundamentals, target)`` batches.
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
    criterion = nn.BCEWithLogitsLoss()
    probs, targets_arr, total_loss = _collect_predictions(model, loader, device, criterion)
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


def bootstrap_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap confidence intervals for AUC, accuracy, precision, and recall.

    Runs a single forward pass to collect all predictions, then resamples with
    replacement ``n_bootstrap`` times entirely in numpy — no repeated GPU passes.

    Parameters
    ----------
    model:
        Trained model.
    loader:
        DataLoader yielding ``(tech, sentiment, fundamentals, target)`` batches.
    device:
        ``"cpu"`` or ``"cuda"``.
    n_bootstrap:
        Number of bootstrap resamples.
    ci:
        Confidence level, e.g. ``0.95`` for a 95% CI.
    seed:
        Optional random seed.  Uses ``numpy.random.default_rng`` — does not
        affect global numpy state.

    Returns
    -------
    :class:`BootstrapResult` with mean and CI bounds for each metric.
    """
    probs, targets, _ = _collect_predictions(model, loader, device)
    preds = (probs >= 0.5).astype(int)
    n_samples = len(targets)

    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci
    lo, hi = alpha / 2 * 100, (1.0 - alpha / 2) * 100

    aucs: list[float] = []
    accs: list[float] = []
    precs: list[float] = []
    recs: list[float] = []

    n_auc_skipped = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        t, p, pr = targets[idx], probs[idx], preds[idx]

        try:
            aucs.append(float(roc_auc_score(t, p)))
        except ValueError:
            # Single-class resample — only AUC is undefined; other metrics are fine
            n_auc_skipped += 1

        accs.append(float(accuracy_score(t, pr)))
        precs.append(float(precision_score(t, pr, zero_division=0)))
        recs.append(float(recall_score(t, pr, zero_division=0)))

    if n_auc_skipped:
        logger.debug(
            "bootstrap_evaluate: %d/%d resamples were single-class (AUC skipped)",
            n_auc_skipped,
            n_bootstrap,
        )

    def _ci(samples: list[float]) -> tuple[float, float, float]:
        arr = np.array(samples)
        return float(arr.mean()), float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    auc_mean, auc_lo, auc_hi = _ci(aucs)
    acc_mean, acc_lo, acc_hi = _ci(accs)
    pre_mean, pre_lo, pre_hi = _ci(precs)
    rec_mean, rec_lo, rec_hi = _ci(recs)

    return {
        "auc_mean": auc_mean,
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "accuracy_mean": acc_mean,
        "accuracy_ci_low": acc_lo,
        "accuracy_ci_high": acc_hi,
        "precision_mean": pre_mean,
        "precision_ci_low": pre_lo,
        "precision_ci_high": pre_hi,
        "recall_mean": rec_mean,
        "recall_ci_low": rec_lo,
        "recall_ci_high": rec_hi,
        "n_bootstrap": n_bootstrap - n_auc_skipped,
        "n_samples": n_samples,
    }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a single forward pass and return (probs, targets, total_loss).

    ``total_loss`` is the sum of per-sample losses (divide by n_samples for
    mean loss).  When *criterion* is ``None``, total_loss is always 0.0.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0

    with torch.no_grad():
        for tech, sentiment, fundamentals, targets in loader:
            tech = tech.to(device)
            sentiment = sentiment.to(device)
            fundamentals = fundamentals.to(device)
            targets = targets.to(device)

            logits = model(tech, sentiment, fundamentals)

            if criterion is not None:
                total_loss += criterion(logits, targets.unsqueeze(1)).item() * len(targets)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    logits_arr = np.concatenate(all_logits).squeeze()
    targets_arr = np.concatenate(all_targets)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))  # sigmoid
    return probs, targets_arr, total_loss


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

    for tech, sentiment, fundamentals, targets in loader:
        tech = tech.to(device)
        sentiment = sentiment.to(device)
        fundamentals = fundamentals.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(tech, sentiment, fundamentals)
        loss = criterion(logits, targets.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n_samples += len(targets)

    return total_loss / n_samples
