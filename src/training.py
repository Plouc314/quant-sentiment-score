from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml


@dataclass
class ComputeConfig:
    """Hardware and process configuration, separate from model hyperparameters."""

    device: str | None = None  # None → auto-detect: "cuda" if available, else "cpu"
    num_workers: int = 0
    n_threads: int | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self) -> None:
        """Configure the process for training.

        Sets the multiprocessing start method to ``spawn`` (required for CUDA
        DataLoader workers) and optionally caps the number of PyTorch CPU threads.
        Call this once at the top of a notebook or script before any DataLoaders
        are created.
        """
        mp.set_start_method("spawn", force=True)
        if self.n_threads is not None:
            torch.set_num_threads(self.n_threads)


@dataclass
class TrainingConfig:
    window: int = 64
    batch_size: int = 32
    n_epochs: int = 100
    lr: float = 1e-3
    patience: int = 15
    dropout: float = 0.2
    include_momentum_slope: bool = True


@dataclass
class Split:
    """Train/held-out assignment and per-symbol cutoff dates.

    Cutoffs are used to split each training symbol's windows into
    train (before ``cutoff - val_months``), val, and test (after cutoff).
    Held-out symbols are evaluated on all their windows.
    """

    train_symbols: list[str]
    held_out_symbols: list[str]
    cutoffs: dict[str, pd.Timestamp]
    val_months: int

    @property
    def all_symbols(self) -> list[str]:
        return self.train_symbols + self.held_out_symbols

    @classmethod
    def create(
        cls,
        symbols: list[str],
        held_out_frac: float = 0.2,
        nominal_cutoff: str = "2019-10-01",
        stagger_days: int = 45,
        val_months: int = 2,
        seed: int = 42,
    ) -> Split:
        """Randomly assign symbols to train/held-out and stagger per-symbol cutoffs.

        Parameters
        ----------
        symbols:
            Full screened universe.
        held_out_frac:
            Fraction of symbols reserved as held-out (never seen during training).
        nominal_cutoff:
            Central train/test boundary date (YYYY-MM-DD).
        stagger_days:
            Each symbol's cutoff is offset by a uniform random draw in
            ``[-stagger_days, +stagger_days]`` to avoid a single sharp boundary.
        val_months:
            Calendar months immediately before each cutoff reserved for validation.
        seed:
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        shuffled = list(symbols)
        rng.shuffle(shuffled)

        n_held_out = int(len(shuffled) * held_out_frac)
        held_out = shuffled[:n_held_out]
        train = shuffled[n_held_out:]

        nominal = pd.Timestamp(nominal_cutoff)
        cutoffs: dict[str, pd.Timestamp] = {}
        for symbol in train + held_out:
            offset = int(rng.integers(-stagger_days, stagger_days + 1))
            cutoffs[symbol] = nominal + pd.Timedelta(days=offset)

        return cls(
            train_symbols=train,
            held_out_symbols=held_out,
            cutoffs=cutoffs,
            val_months=val_months,
        )

    def save(self, path: Path) -> None:
        """Write to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "train_symbols": self.train_symbols,
            "held_out_symbols": self.held_out_symbols,
            "cutoffs": {s: str(ts.date()) for s, ts in self.cutoffs.items()},
            "val_months": self.val_months,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Path) -> Split:
        """Load from a YAML file written by :meth:`save`."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Support legacy splits.yml format (cutoffs nested under "cutoffs" key,
        # val_months under "config")
        val_months = data.get("val_months") or data.get("config", {}).get(
            "val_months", 2
        )
        cutoffs = {s: pd.Timestamp(d) for s, d in data["cutoffs"].items()}

        return cls(
            train_symbols=data["train_symbols"],
            held_out_symbols=data["held_out_symbols"],
            cutoffs=cutoffs,
            val_months=val_months,
        )
