"""LSTM model for binary stock movement prediction with sentiment fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM for binary stock movement prediction with sentiment + fundamental fusion.

    Architecture::

        sentiment_proj : Linear(sentiment_dim, n_factors)
        lstm           : LSTM(n_factors * 2, hidden_size, num_layers)
        classifier     : Linear(hidden_size + n_fundamentals + n_sentiment_probs, hidden_size)
                         → ReLU → Dropout → BatchNorm1d
                         → Linear(hidden_size, 1)

    Time-series features (technical indicators, projected sentiment embeddings)
    flow through the LSTM to capture temporal dynamics over the rolling window.

    Fundamental factors (P/E, ROE, etc.) and FinBERT class probabilities are
    slow-moving / daily snapshots — they are injected at the classifier stage
    alongside the LSTM's final hidden state, conditioning the prediction without
    polluting the recurrent cells with repeated identical values across timesteps.

    ``n_sentiment_probs=3`` enables the 3-class FinBERT probability injection
    (``[P(pos), P(neg), P(neutral)]``).  Default ``0`` disables it for
    backward compatibility when running without a sentiment pipeline.
    """

    def __init__(
        self,
        n_factors: int = 16,
        sentiment_dim: int = 768,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_fundamentals: int = 0,
        n_sentiment_probs: int = 0,
    ) -> None:
        super().__init__()
        self.n_fundamentals = n_fundamentals
        self.n_sentiment_probs = n_sentiment_probs
        self.sentiment_proj = nn.Linear(sentiment_dim, n_factors)
        self.lstm = nn.LSTM(
            input_size=n_factors * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        classifier_in = hidden_size + n_fundamentals + n_sentiment_probs
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        tech: torch.Tensor,
        sentiment: torch.Tensor,
        fundamentals: torch.Tensor | None = None,
        sentiment_probs: torch.Tensor | None = None,
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
            ``None`` / empty tensor when the model was built with ``n_fundamentals=0``.
        sentiment_probs:
            Daily FinBERT class probabilities of shape ``(batch, n_sentiment_probs)``,
            or ``None`` / empty tensor when the model was built with ``n_sentiment_probs=0``.

        Returns
        -------
        Logits of shape ``(batch, 1)``.  Apply sigmoid for probabilities.
        """
        projected = self.sentiment_proj(sentiment)        # (batch, window, n_factors)
        fused = torch.cat([tech, projected], dim=-1)      # (batch, window, n_factors*2)
        out, _ = self.lstm(fused)                         # (batch, window, hidden_size)
        last = out[:, -1, :]                              # (batch, hidden_size)

        if self.n_fundamentals > 0:
            if fundamentals is None or fundamentals.shape[-1] == 0:
                raise RuntimeError(
                    f"model expects n_fundamentals={self.n_fundamentals} but got empty tensor"
                )
            last = torch.cat([last, fundamentals], dim=-1)

        if self.n_sentiment_probs > 0:
            if sentiment_probs is None or sentiment_probs.shape[-1] == 0:
                raise RuntimeError(
                    f"model expects n_sentiment_probs={self.n_sentiment_probs} but got empty tensor"
                )
            last = torch.cat([last, sentiment_probs], dim=-1)

        return self.classifier(last)                      # (batch, 1)
