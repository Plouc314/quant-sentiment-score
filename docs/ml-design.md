# ML Design Notes

## How the Models Work

### Data & Target

**Target variable:** Binary classification — 1 if `close[t+2] > close[t-1]`, 0 otherwise. This is a 3-day executable signal (buy today, sell two days later, benchmarked against yesterday's close).

**Features:**

| Input | Shape | Details |
|-------|-------|---------|
| Technical | `(batch, 64, 16)` | 16 scale-invariant OHLCV indicators (RSI, MACD, Bollinger bands, ATR, OBV slope, etc.) |
| Sentiment embeddings | `(batch, 64, 768)` | FinBERT CLS vectors, projected down to 16 dims inside the model |
| Sentiment probs | `(batch, 64, 3)` | FinBERT [neg, neutral, pos] class probabilities |
| Fundamentals | `(batch, n_fund)` | Quarterly snapshots forward-filled to daily, injected at the classifier stage only |

Sliding window: **64 trading days** (~3 months).

---

### Train / Val / Test Split

Temporal per-symbol split with a cross-symbol held-out set:

```
Each training symbol (80%):
  ├── Train:  dates < (cutoff − 2 months)
  ├── Val:    (cutoff − 2 months) ≤ dates < cutoff
  └── Test:   dates ≥ cutoff

Held-out symbols (20%):  all windows, never trained on
```

Nominal cutoff is **2019-10-01** with a ±45-day per-symbol stagger.

---

### Training Loop

- **Loss:** `BCEWithLogitsLoss` with optional `pos_weight` for class imbalance
- **Optimizer:** Adam lr=1e-3
- **Scheduler:** `ReduceLROnPlateau` on val AUC (factor=0.5, patience=5)
- **Early stopping:** patience=15 epochs on best val AUC, restores best checkpoint
- **Grad clipping:** max_norm=1.0

---

### Model Architectures

**LSTM** (`hidden_size=32, num_layers=2`):
1. Project sentiment 768 → 16 dims
2. Concat [tech(16×2), projected_sentiment(16), sent_probs(3)] per timestep
3. 2-layer LSTM → last hidden state
4. Concat fundamentals → Linear(32+n_fund, 32) → ReLU → Dropout → BN → Linear(32,1)

**Transformer** (`d_model=64, nhead=4, n_layers=6`):
1. Same input construction, then project → 64 dims + learned positional embedding
2. 6× TransformerEncoderLayer (dim_feedforward=128)
3. Mean pool over sequence → Dropout → Linear(64+n_fund, 1)

---

### Evaluation

`bootstrap_evaluate()` with n=1000 resamples:
- **Metrics:** AUC, accuracy, precision, recall — all with 95% percentile CI
- **Two evaluation sets:** temporal test (same symbols, post-cutoff) and held-out (unseen symbols)

---

## Assessment

### What's well-designed

**The staggered per-symbol cutoff** mirrors real deployment (you don't train/deploy on all stocks simultaneously) and prevents data leakage at the boundary.

**The cross-symbol held-out set** is the most important evaluation. If the model only generalizes temporally on the same symbols it trained on, it has likely memorized stock-specific patterns. The held-out set tests whether it learned anything general.

**Injecting fundamentals at classifier stage, not into recurrent cells** is architecturally sound — quarterly-frequency signals would pollute the temporal dynamics if fed as a sequence.

**16 scale-invariant technical features** — computing ratios (close/SMA, ATR/close) rather than raw prices makes the features cross-stock comparable.

**StandardScaler fitted only on train windows** and reused on val/test/held-out is correct — no leakage there.

---

### What probably won't work

**The target is very noisy.** `close[t+2] > close[t-1]` is close to a coin flip for most stocks on most days. Even a "good" model realistically maxes out at AUC ~0.55. The bootstrap CI will be wide, and distinguishing a real signal from luck requires far more test windows than 3 years of data provides.

**The Transformer is likely too large for this data.** 6 encoder layers with d_model=64 and 4 heads is a substantial model for per-stock datasets that might have 400–500 training windows. The LSTM (32 hidden, 2 layers) is much better-sized. The `dim_feedforward=128` reduction shows awareness of this, but 6 layers is still heavy.

**FinBERT embeddings (768 dims) projected to 16 dims** then concatenated with 16-dim technical features effectively discards almost all the semantic content. A linear 768→16 bottleneck loses enormous amounts of information. Either keep the projection larger (64–128) or use only the 3-class probabilities and skip the full embedding in the sequence encoder.

**Concatenating [tech×2, sentiment×1, probs×1] before the LSTM** gives technical features double the weight of sentiment by construction (`n_factors*2` vs `n_factors`). This looks unintentional.

**The 3-year price window (2018–2020)** is very short and includes the COVID crash as the only major regime shift. Generalization to different volatility regimes is untested.

---

### What could be improved

**Target construction:** `close[t+2] > close[t-1]` ignores transaction costs. A more actionable target would be `(close[t+2] - close[t]) / close[t] > threshold` where threshold exceeds the round-trip spread. This filters out noise around zero.

**Sentiment alignment:** Forward-filling zero vectors when sentiment data is missing treats "no news" the same as "no signal available." A binary `has_news` feature or a learned null embedding would be cleaner.

**The scaler is fitted across all symbols' training windows pooled together.** Per-stock z-scoring before the pooled fit might reduce cross-stock distribution differences.

**No calibration metrics.** For a classifier feeding a downstream trading strategy, calibration matters more than raw AUC. Brier score and reliability diagrams should be tracked alongside AUC.

**No walk-forward / expanding window retraining.** The single temporal cutoff tests stale generalization. Real use would require periodic retraining — it's worth testing whether the model degrades gracefully over time post-cutoff.
