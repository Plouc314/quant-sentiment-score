from __future__ import annotations

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM for binary stock movement prediction with sentiment fusion.

    Classes: 0 = down, 1 = up.

    Two operating modes depending on ``use_sentiment_proj``:

    **With projection** (default, original architecture)::

        sentiment_proj : Linear(sentiment_dim → n_factors)
        lstm           : LSTM(n_factors * 2, hidden_size, num_layers)
                         ↑ tech (n_factors) + projected sentiment (n_factors)
        classifier     : Linear(hidden_size, hidden_size)
                         → ReLU → Dropout → BatchNorm1d → Linear(n_classes)

    **Without projection** (Plan A / B — scalar score baked into tech features)::

        lstm           : LSTM(n_factors, hidden_size, num_layers)
                         ↑ tech only (n_factors already includes sentiment scalar)
        classifier     : same as above

    When ``use_sentiment_proj=False`` the ``sentiment`` argument to ``forward``
    is accepted but ignored, so the DataLoader contract is unchanged.
    """

    def __init__(
        self,
        n_factors: int = 16,
        sentiment_dim: int = 768,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 2,
        use_sentiment_proj: bool = True,
    ) -> None:
        super().__init__()
        self.n_classes          = n_classes
        self.use_sentiment_proj = use_sentiment_proj

        if use_sentiment_proj:
            self.sentiment_proj = nn.Linear(sentiment_dim, n_factors)
            lstm_input = n_factors * 2
        else:
            self.sentiment_proj = None
            lstm_input = n_factors

        self.lstm = nn.LSTM(
            input_size=lstm_input,
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
            nn.Linear(hidden_size, n_classes),
        )
        self._init_weights()

    def forward(
        self,
        tech: torch.Tensor,
        sentiment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tech:      ``(batch, window, n_factors)``
        sentiment: ``(batch, window, sentiment_dim)`` — ignored when
                   ``use_sentiment_proj=False``

        Returns
        -------
        Logits of shape ``(batch, n_classes)``.
        """
        if self.use_sentiment_proj:
            projected = self.sentiment_proj(sentiment)
            lstm_in   = torch.cat([tech, projected], dim=-1)
        else:
            lstm_in = tech
        out, _ = self.lstm(lstm_in)
        last = out[:, -1, :]
        return self.classifier(last)

    def _init_weights(self) -> None:
        for module in [self.sentiment_proj, *self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
