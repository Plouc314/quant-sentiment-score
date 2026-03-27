from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from IPython import display
import matplotlib.pyplot as plt


def train_epoch(model: nn.Module, optimizer, criterion, train_loader, device: str) -> float:
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model: nn.Module, criterion, test_loader, device: str) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def train_cycle(
    model: nn.Module,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    n_epochs: int,
    device: str,
    scheduler=None,
) -> tuple[list[float], list[float]]:
    train_loss_log: list[float] = []
    test_loss_log: list[float] = []
    start_time = time.time()

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        test_loss = evaluate(model, criterion, test_loader, device)

        if scheduler:
            scheduler.step()

        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)

        display.clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_log, label="Train MSE", color="royalblue", lw=2)
        plt.plot(test_loss_log, label="Test MSE", color="darkorange", lw=2)
        plt.title(f"Training Progress [Epoch {epoch + 1}/{n_epochs}]")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        print(f"Running Time: {elapsed:.1f}s")

    print("\nLearning finished!")
    return train_loss_log, test_loss_log


def directional_accuracy(model: nn.Module, test_loader, device: str = "cpu") -> float:
    """Fraction of predictions with the correct sign (up vs down).

    For trading, direction matters more than magnitude. A model that
    correctly predicts *whether* the next bar is positive or negative
    is actionable even if the magnitude is off.

    Returns
    -------
    Float in [0, 1] — 0.5 is chance level for balanced returns.
    """
    model.eval()
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            preds.append(output.cpu().numpy())
            actuals.append(batch_y.numpy())

    preds_arr = np.vstack(preds).flatten()
    actuals_arr = np.vstack(actuals).flatten()
    return float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))


def plot_prediction(model: nn.Module, test_loader, scaler=None, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Plot predicted vs actual values from the test set.

    When *scaler* is ``None`` (default), the target is assumed to be
    log-returns — predicted and actual log-returns are plotted directly.

    When *scaler* is provided (legacy mode), inverse-transforms predictions
    back to raw price using the original ``MinMaxScaler``.

    Returns
    -------
    ``(preds, actuals)`` — raw arrays before any inverse transform.
    """
    model.eval()
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            preds.append(output.cpu().numpy())
            actuals.append(batch_y.numpy())

    preds_arr = np.vstack(preds).flatten()
    actuals_arr = np.vstack(actuals).flatten()

    if scaler is not None:
        # Legacy: inverse-transform scaled close price
        def _denormalize(data: np.ndarray) -> np.ndarray:
            dummy = np.zeros((len(data), 7))
            dummy[:, 3] = data
            return scaler.inverse_transform(dummy)[:, 3]

        plot_preds = _denormalize(preds_arr)
        plot_actuals = _denormalize(actuals_arr)
        ylabel = "Price"
        title = "Stock Price Prediction"
    else:
        plot_preds = preds_arr
        plot_actuals = actuals_arr
        ylabel = "Log Return"
        title = "Predicted vs Actual Log Return"

    plt.figure(figsize=(12, 6))
    plt.plot(plot_actuals, label="Actual", color="royalblue", alpha=0.7)
    plt.plot(plot_preds, label="Predicted", color="crimson", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return preds_arr, actuals_arr
