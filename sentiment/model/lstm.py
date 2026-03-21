"""LSTM model for binary stock movement prediction with sentiment fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM for binary stock movement prediction with sentiment + fundamental fusion.

    Architecture::

        sentiment_proj : Linear(sentiment_dim, n_factors)
        lstm           : LSTM(n_factors * 2, hidden_size, num_layers)
        classifier     : Linear(hidden_size + n_fundamentals, 1)

    Time-series features (technical indicators, projected sentiment embeddings)
    flow through the LSTM to capture temporal dynamics over the rolling window.
    Fundamental factors (P/E, ROE, etc.) are slow-moving snapshots — they are
    injected at the classifier stage alongside the LSTM's final hidden state,
    conditioning the prediction without polluting the recurrent cells with
    repeated identical values across timesteps.
    """

    def __init__(
        self,
        n_factors: int = 16,
        sentiment_dim: int = 768,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_fundamentals: int = 0,
    ) -> None:
        super().__init__()
        self.n_fundamentals = n_fundamentals
        self.sentiment_proj = nn.Linear(sentiment_dim, n_factors)
        self.lstm = nn.LSTM(
            input_size=n_factors * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_size + n_fundamentals, 1)

    def forward(
        self,
        tech: torch.Tensor,
        sentiment: torch.Tensor,
        fundamentals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        tech:
            Technical factors of shape ``(batch, window, n_factors)``.
        sentiment:
            Sentiment embeddings of shape ``(batch, window, sentiment_dim)``.
        fundamentals:
            Fundamental snapshot of shape ``(batch, n_fundamentals)``, or
            ``None`` when the model was built with ``n_fundamentals=0``.

        Returns
        -------
        Logits of shape ``(batch, 1)``.  Apply sigmoid for probabilities.
        """
        projected = self.sentiment_proj(sentiment)        # (batch, window, n_factors)
        fused = torch.cat([tech, projected], dim=-1)      # (batch, window, n_factors*2)
        out, _ = self.lstm(fused)                         # (batch, window, hidden_size)
        last = out[:, -1, :]                              # (batch, hidden_size)

        if self.n_fundamentals > 0:
            if fundamentals is None:
                raise RuntimeError(
                    f"model expects n_fundamentals={self.n_fundamentals} but fundamentals=None"
                )
            last = torch.cat([last, fundamentals], dim=-1)  # (batch, hidden_size + n_fund)

        return self.classifier(last)                      # (batch, 1)
