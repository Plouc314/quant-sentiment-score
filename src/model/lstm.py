from __future__ import annotations

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM for binary stock movement prediction with sentiment fusion.

    Architecture::

        sentiment_proj : Linear(sentiment_dim → n_factors)
        lstm           : LSTM(n_factors*2 + n_sentiment_probs, hidden_size, num_layers)
        classifier     : Linear(hidden_size, hidden_size)
                         → ReLU → Dropout → BatchNorm1d → Linear(1)

    Technical indicators, projected sentiment embeddings, and FinBERT class
    probabilities flow through the LSTM to capture temporal dynamics over the
    window.
    """

    def __init__(
        self,
        n_factors: int = 16,
        sentiment_dim: int = 768,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_sentiment_probs: int = 0,
    ) -> None:
        super().__init__()
        self.n_sentiment_probs = n_sentiment_probs

        self.sentiment_proj = nn.Linear(sentiment_dim, n_factors)
        self.lstm = nn.LSTM(
            input_size=n_factors * 2 + n_sentiment_probs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        tech: torch.Tensor,
        sentiment: torch.Tensor,
        sentiment_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tech:            ``(batch, window, n_factors)``
        sentiment:       ``(batch, window, sentiment_dim)``
        sentiment_probs: ``(batch, window, n_sentiment_probs)`` or ``None``

        Returns
        -------
        Logits of shape ``(batch, 1)``.
        """
        projected = self.sentiment_proj(sentiment)
        parts = [tech, projected]
        if self.n_sentiment_probs > 0:
            if sentiment_probs is None or sentiment_probs.shape[-1] == 0:
                raise RuntimeError(
                    f"model expects n_sentiment_probs={self.n_sentiment_probs} but received empty tensor"
                )
            parts.append(sentiment_probs)

        out, _ = self.lstm(torch.cat(parts, dim=-1))
        last = out[:, -1, :]
        return self.classifier(last)
