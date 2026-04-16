from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from ..training import ComputeConfig, TrainingConfig

logger = logging.getLogger(__name__)

# Label semantics: 0 = down, 1 = up
N_CLASSES = 2


@dataclass
class TrainingResult:
    best_epoch:    int
    best_val_loss: float
    best_val_auc:  float
    history: dict[str, list[float]]
    """Per-epoch lists: train_loss, val_loss, val_auc, val_accuracy."""


@dataclass
class EvalResult:
    auc_mean:      float
    auc_ci_low:    float
    auc_ci_high:   float
    accuracy_mean:    float
    accuracy_ci_low:  float
    accuracy_ci_high: float
    precision_mean:    float
    precision_ci_low:  float
    precision_ci_high: float
    recall_mean:    float
    recall_ci_low:  float
    recall_ci_high: float
    n_bootstrap: int
    n_samples:   int


class Trainer:
    """Trains and evaluates a binary stock movement prediction model.

    Classes: 0 = down, 1 = up.

    Parameters
    ----------
    model:
        A model whose ``forward(tech, sentiment)``
        returns logits of shape ``(batch, 2)``.
    config:
        Training hyperparameters (lr, patience, n_epochs, …).
    compute_config:
        Hardware configuration (device, …).
    class_weights:
        Optional weight tensor of shape ``(2,)`` for ``CrossEntropyLoss``.
        Use to counteract class imbalance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        compute_config: ComputeConfig,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self._model  = model.to(compute_config.device)
        self._config = config
        self._compute = compute_config
        self._class_weights = class_weights

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        """Train with early stopping on validation AUC.

        Restores the model to its best checkpoint before returning.
        """
        config  = self._config
        compute = self._compute
        model   = self._model

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=config.scheduler_patience
            )
        elif config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.step_size, gamma=config.gamma
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler!r}")
        weight = (
            self._class_weights.to(compute.device)
            if self._class_weights is not None else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        best_loss      = float("inf")
        best_auc       = 0.0
        best_epoch     = 0
        best_state: dict | None = None
        no_improve     = 0
        history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_auc": [], "val_accuracy": []}

        for epoch in range(1, config.n_epochs + 1):
            train_loss  = self._train_epoch(optimizer, criterion, train_loader)
            val_metrics = self._evaluate_epoch(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_auc"].append(val_metrics["auc"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            if config.scheduler == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | val_acc=%.4f",
                epoch, train_loss, val_metrics["loss"], val_metrics["auc"], val_metrics["accuracy"],
            )

            if val_metrics["loss"] < best_loss:
                best_loss  = val_metrics["loss"]
                best_auc   = val_metrics["auc"]
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= config.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, config.patience)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return TrainingResult(best_epoch=best_epoch, best_val_loss=best_loss, best_val_auc=best_auc, history=history)

    def bootstrap_evaluate(
        self,
        loader: DataLoader,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int | None = None,
    ) -> EvalResult:
        """Bootstrap confidence intervals for binary metrics.

        Computes AUC (on P(up)), accuracy, precision, and recall for the
        positive class with percentile bootstrap CIs.
        """
        probs, targets, _ = self._collect_predictions(loader)
        preds     = probs.argmax(axis=1)
        p_up      = probs[:, 1]
        n         = len(targets)
        rng       = np.random.default_rng(seed)
        alpha     = 1.0 - ci
        lo, hi    = alpha / 2 * 100, (1.0 - alpha / 2) * 100

        aucs:  list[float] = []
        accs:  list[float] = []
        precs: list[float] = []
        recs:  list[float] = []
        n_skipped = 0

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            t, p_prob, pr = targets[idx], p_up[idx], preds[idx]
            try:
                aucs.append(float(roc_auc_score(t, p_prob)))
            except ValueError:
                n_skipped += 1
            accs.append(float(accuracy_score(t, pr)))
            precs.append(float(precision_score(t, pr, zero_division=0)))
            recs.append(float(recall_score(t, pr, zero_division=0)))

        def _ci(samples: list[float]) -> tuple[float, float, float]:
            arr = np.array(samples)
            return float(arr.mean()), float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

        am, al, ah = _ci(aucs)
        cm, cl, ch = _ci(accs)
        pm, pl, ph = _ci(precs)
        rm, rl, rh = _ci(recs)

        return EvalResult(
            auc_mean=am,      auc_ci_low=al,      auc_ci_high=ah,
            accuracy_mean=cm,    accuracy_ci_low=cl,    accuracy_ci_high=ch,
            precision_mean=pm,   precision_ci_low=pl,   precision_ci_high=ph,
            recall_mean=rm,   recall_ci_low=rl,   recall_ci_high=rh,
            n_bootstrap=n_bootstrap - n_skipped,
            n_samples=n,
        )

    def calibration_curve(
        self,
        loader: DataLoader,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the calibration (reliability) curve for the positive class.

        Returns
        -------
        ``(fraction_of_positives, mean_predicted_value)`` for P(up).
        """
        from sklearn.calibration import calibration_curve as _sklearn_calibration_curve

        probs, targets, _ = self._collect_predictions(loader)
        return _sklearn_calibration_curve(
            targets, probs[:, 1], n_bins=n_bins, strategy="uniform",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        loader: DataLoader,
    ) -> float:
        model  = self._model
        device = self._compute.device
        model.train()
        total_loss = 0.0
        n_samples  = 0

        for tech, sentiment, targets in loader:
            tech      = tech.to(device)
            sentiment = sentiment.to(device)
            targets   = targets.to(device)

            optimizer.zero_grad()
            logits = model(tech, sentiment)
            loss   = criterion(logits, targets)
            loss.backward()
            if self._config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self._config.grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(targets)
            n_samples  += len(targets)

        return total_loss / n_samples

    def _evaluate_epoch(self, loader: DataLoader) -> dict[str, float]:
        criterion = nn.CrossEntropyLoss()
        probs, targets, total_loss = self._collect_predictions(loader, criterion)
        preds = probs.argmax(axis=1)
        try:
            auc = float(roc_auc_score(targets, probs[:, 1]))
        except ValueError:
            auc = 0.5
        return {
            "loss":      total_loss / len(targets),
            "auc":       auc,
            "accuracy":  float(accuracy_score(targets, preds)),
        }

    def _collect_predictions(
        self,
        loader: DataLoader,
        criterion: nn.Module | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Collect softmax probabilities and targets from the full dataset.

        Returns
        -------
        ``(probs, targets, total_loss)`` where probs has shape ``(N, 2)``.
        """
        model  = self._model
        device = self._compute.device
        model.eval()
        all_probs:   list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        total_loss = 0.0

        with torch.no_grad():
            for tech, sentiment, targets in loader:
                tech      = tech.to(device)
                sentiment = sentiment.to(device)
                targets   = targets.to(device)

                logits = model(tech, sentiment)
                if criterion is not None:
                    total_loss += criterion(logits, targets).item() * len(targets)

                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        probs_arr   = np.concatenate(all_probs)
        targets_arr = np.concatenate(all_targets)

        return probs_arr, targets_arr, total_loss
