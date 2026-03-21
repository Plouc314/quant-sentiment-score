"""Transformer encoder model for binary stock movement prediction with sentiment fusion."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SentimentTransformer(nn.Module):
    """Transformer encoder for binary stock movement prediction with sentiment + fundamental fusion.

    Architecture::

        sentiment_proj : Linear(sentiment_dim, n_factors)
        input_proj     : Linear(n_factors * 2, d_model)
        pos_embedding  : Embedding(max_seq_len, d_model)   [learned]
        encoder        : TransformerEncoder(d_model, nhead, n_layers, dim_feedforward, dropout)
        dropout        : Dropout(dropout)
        classifier     : Linear(d_model + n_fundamentals, 1)

    Time-series features (technical indicators + projected sentiment) flow through
    the Transformer encoder.  Mean pooling over the sequence dimension replaces
    the LSTM's final hidden state — the last token has no recurrent privilege in
    a Transformer, so mean pooling is more stable for short classification sequences.

    Fundamental factors are injected at the classifier stage alongside the pooled
    representation, identical to :class:`~sentiment.model.lstm.SentimentLSTM`.

    Notes
    -----
    ``dim_feedforward`` defaults to 128 (not PyTorch's default of 2048).  With
    test sets of ~50–80 windows the default feedforward dimension causes severe
    overfitting.  Increase it only if validation loss has stabilised and you have
    more data.
    """

    def __init__(
        self,
        n_factors: int = 16,
        sentiment_dim: int = 768,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        n_fundamentals: int = 0,
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()
        self.n_fundamentals = n_fundamentals

        self.sentiment_proj = nn.Linear(sentiment_dim, n_factors)
        self.input_proj = nn.Linear(n_factors * 2, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model + n_fundamentals, 1)

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
        batch, window, _ = tech.shape

        # Project sentiment to same dim as tech, then fuse
        projected = self.sentiment_proj(sentiment)           # (batch, window, n_factors)
        fused = torch.cat([tech, projected], dim=-1)         # (batch, window, n_factors*2)

        # Project to d_model and add learned positional encoding
        x = self.input_proj(fused)                           # (batch, window, d_model)
        positions = torch.arange(window, device=tech.device).unsqueeze(0)  # (1, window)
        x = x + self.pos_embedding(positions)               # (batch, window, d_model)

        # Transformer encoder
        x = self.encoder(x)                                  # (batch, window, d_model)

        # Mean pool over sequence dimension
        pooled = self.dropout(x.mean(dim=1))                 # (batch, d_model)

        # Inject fundamentals at classifier stage
        if self.n_fundamentals > 0:
            if fundamentals is None:
                raise RuntimeError(
                    f"model expects n_fundamentals={self.n_fundamentals} but fundamentals=None"
                )
            pooled = torch.cat([pooled, fundamentals], dim=-1)  # (batch, d_model + n_fund)

        return self.classifier(pooled)                       # (batch, 1)
