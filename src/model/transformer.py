from __future__ import annotations

import torch
import torch.nn as nn


class SentimentTransformer(nn.Module):
    """Transformer encoder for binary stock movement prediction with sentiment fusion.

    Classes: 0 = down, 1 = up.

    Architecture (matches reference paper)::

        sentiment_proj : Linear(sentiment_dim → n_factors)
        cat_relu       : ReLU applied after [tech ‖ projected_sentiment]
        input_proj     : Linear(n_factors * 2 → d_model)
        pos_embedding  : Embedding(max_seq_len, d_model)   [learned]
        encoder        : TransformerEncoder(d_model, nhead, n_layers, dim_feedforward)
        classifier     : Linear(d_model → n_classes)

    Mean pooling over the sequence replaces the LSTM's final hidden state.

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
        max_seq_len: int = 100,
        n_classes: int = 2,
        sentiment_proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        proj_dim = sentiment_proj_dim if sentiment_proj_dim is not None else n_factors
        self.sentiment_proj = nn.Linear(sentiment_dim, proj_dim)
        self.cat_relu       = nn.ReLU()
        self.input_proj     = nn.Linear(n_factors + proj_dim, d_model)
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
        self.classifier = nn.Linear(d_model, n_classes)
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
        sentiment: ``(batch, window, sentiment_dim)``

        Returns
        -------
        Logits of shape ``(batch, n_classes)``.
        """
        _, window, _ = tech.shape
        if window > self.pos_embedding.num_embeddings:
            raise RuntimeError(
                f"window ({window}) > max_seq_len ({self.pos_embedding.num_embeddings})"
            )

        projected = self.sentiment_proj(sentiment)
        x = self.cat_relu(torch.cat([tech, projected], dim=-1))
        x = self.input_proj(x)
        x = x + self.pos_embedding(torch.arange(window, device=tech.device).unsqueeze(0))
        pooled = self.dropout(self.encoder(x).mean(dim=1))
        return self.classifier(pooled)

    def _init_weights(self) -> None:
        for module in [self.sentiment_proj, self.input_proj, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
