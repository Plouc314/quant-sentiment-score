from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelCheckpoint:
    state_dict: dict
    tech_scaler: StandardScaler
    config: dict
    """Architecture + training config needed to reconstruct the model."""


class ModelRepository:
    """Saves and loads trained model checkpoints.

    Each checkpoint bundles the model state dict, fitted scalers, and
    the architecture config required to reconstruct the model for inference.

    Layout::

        <data_dir>/
            universal_lstm_v1.pt
            universal_transformer_v1.pt
            ...
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parents[2] / "data" / "models"
        self._data_dir = data_dir

    def save(
        self,
        name: str,
        model: nn.Module,
        tech_scaler: StandardScaler,
        config: dict,
    ) -> None:
        """Save a checkpoint as ``<data_dir>/<name>.pt``."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "tech_scaler": tech_scaler,
                "config": config,
            },
            self._data_dir / f"{name}.pt",
        )

    def load(self, name: str) -> ModelCheckpoint:
        """Load a checkpoint by name.  Raises ``FileNotFoundError`` if absent."""
        path = self._data_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No model checkpoint: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        return ModelCheckpoint(
            state_dict=data["state_dict"],
            tech_scaler=data["tech_scaler"],
            config=data["config"],
        )

    def list(self) -> list[str]:
        """Return names of all saved checkpoints."""
        if not self._data_dir.exists():
            return []
        return [p.stem for p in sorted(self._data_dir.glob("*.pt"))]
