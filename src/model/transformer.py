from __future__ import annotations

import torch
import torch.nn as nn


class SentimentTransformer(nn.Module):
    """Transformer encoder for binary stock movement prediction with sentiment fusion.

    Architecture::

        sentiment_proj : Linear(sentiment_dim → n_factors)
        input_proj     : Linear(n_factors*2 + n_sentiment_probs → d_model)
        pos_embedding  : Embedding(max_seq_len, d_model)   [learned]
        encoder        : TransformerEncoder(d_model, nhead, n_layers, dim_feedforward)
        classifier     : Linear(d_model → 1)

    Mean pooling over the sequence replaces the LSTM's final hidden state —
    the last token has no recurrent privilege in a Transformer.

    Notes
    -----
    ``dim_feedforward`` defaults to 128 (not PyTorch's default 2048) to
    prevent overfitting on small per-stock datasets.
    """

    def __init__(
        self,
        n_factors: int = 16,
        sentiment_dim: int = 768,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 6,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        n_sentiment_probs: int = 0,
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()
        self.n_sentiment_probs = n_sentiment_probs

        self.sentiment_proj = nn.Linear(sentiment_dim, n_factors)
        self.input_proj     = nn.Linear(n_factors * 2 + n_sentiment_probs, d_model)
        self.pos_embedding  = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

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
        _, window, _ = tech.shape
        if window > self.pos_embedding.num_embeddings:
            raise RuntimeError(
                f"window ({window}) > max_seq_len ({self.pos_embedding.num_embeddings})"
            )

        projected = self.sentiment_proj(sentiment)
        parts = [tech, projected]
        if self.n_sentiment_probs > 0:
            if sentiment_probs is None or sentiment_probs.shape[-1] == 0:
                raise RuntimeError(
                    f"model expects n_sentiment_probs={self.n_sentiment_probs} but received empty tensor"
                )
            parts.append(sentiment_probs)

        x = self.input_proj(torch.cat(parts, dim=-1))
        x = x + self.pos_embedding(torch.arange(window, device=tech.device).unsqueeze(0))
        pooled = self.dropout(self.encoder(x).mean(dim=1))
        return self.classifier(pooled)
