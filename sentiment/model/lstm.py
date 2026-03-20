"""LSTM model for binary stock movement prediction with sentiment fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM for binary stock movement prediction with sentiment fusion.

    Architecture::

        sentiment_proj : Linear(sentiment_dim, n_factors)
        lstm           : LSTM(n_factors * 2, hidden_size, num_layers)
        classifier     : Linear(hidden_size, 1)

    The projection layer is trained end-to-end, learning which embedding
    dimensions are predictive of price movement.  By projecting to the
    same dimensionality as the technical factors, neither feature type
    dominates the other in the concatenated input.
    """

    def __init__(
        self,
        n_factors: int = 16,
        sentiment_dim: int = 768,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sentiment_proj = nn.Linear(sentiment_dim, n_factors)
        self.lstm = nn.LSTM(
            input_size=n_factors * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        tech: torch.Tensor,
        sentiment: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        tech:
            Technical factors of shape ``(batch, window, n_factors)``.
        sentiment:
            Sentiment embeddings of shape ``(batch, window, sentiment_dim)``.

        Returns
        -------
        Logits of shape ``(batch, 1)``.  Apply sigmoid for probabilities.
        """
        projected = self.sentiment_proj(sentiment)        # (batch, window, n_factors)
        fused = torch.cat([tech, projected], dim=-1)     # (batch, window, n_factors*2)
        out, _ = self.lstm(fused)                        # (batch, window, hidden_size)
        return self.classifier(out[:, -1, :])            # (batch, 1)
