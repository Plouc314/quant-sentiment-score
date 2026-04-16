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

Fundamentals are deliberately **not** a model input — see "Why fundamentals are not features" below.

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
2. Concat [tech(16), projected_sentiment(16), sent_probs(3)] per timestep
3. 2-layer LSTM → last hidden state
4. Linear(32, 32) → ReLU → Dropout → BN → Linear(32, 1)

**Transformer** (`d_model=64, nhead=4, n_layers=6`):
1. Same input construction, then project → 64 dims + learned positional embedding
2. 6× TransformerEncoderLayer (dim_feedforward=128)
3. Mean pool over sequence → Dropout → Linear(64, 1)

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

**16 scale-invariant technical features** — computing ratios (close/SMA, ATR/close) rather than raw prices makes the features cross-stock comparable.

**StandardScaler fitted only on train windows** and reused on val/test/held-out is correct — no leakage there.

---

### What probably won't work

**The target is very noisy.** `close[t+2] > close[t-1]` is close to a coin flip for most stocks on most days. Even a "good" model realistically maxes out at AUC ~0.55. The bootstrap CI will be wide, and distinguishing a real signal from luck requires far more test windows than 3 years of data provides.

**The Transformer is likely too large for this data.** 6 encoder layers with d_model=64 and 4 heads is a substantial model for per-stock datasets that might have 400–500 training windows. The LSTM (32 hidden, 2 layers) is much better-sized. The `dim_feedforward=128` reduction shows awareness of this, but 6 layers is still heavy.

**FinBERT embeddings (768 dims) projected to 16 dims** then concatenated with 16-dim technical features effectively discards almost all the semantic content. A linear 768→16 bottleneck loses enormous amounts of information. Either keep the projection larger (64–128) or use only the 3-class probabilities and skip the full embedding in the sequence encoder.

**The 3-year price window (2018–2020)** is very short and includes the COVID crash as the only major regime shift. Generalization to different volatility regimes is untested.

---

### What could be improved

**Target construction:** `close[t+2] > close[t-1]` ignores transaction costs. A more actionable target would be `(close[t+2] - close[t]) / close[t] > threshold` where threshold exceeds the round-trip spread. This filters out noise around zero.

**Sentiment alignment:** Forward-filling zero vectors when sentiment data is missing treats "no news" the same as "no signal available." A binary `has_news` feature or a learned null embedding would be cleaner.

**The scaler is fitted across all symbols' training windows pooled together.** Per-stock z-scoring before the pooled fit might reduce cross-stock distribution differences.

**No calibration metrics.** For a classifier feeding a downstream trading strategy, calibration matters more than raw AUC. Brier score and reliability diagrams should be tracked alongside AUC.

**No walk-forward / expanding window retraining.** The single temporal cutoff tests stale generalization. Real use would require periodic retraining — it's worth testing whether the model degrades gracefully over time post-cutoff.

---

### Why fundamentals are not features

An earlier version of this pipeline concatenated a 10-dim fundamentals vector (PE, PB, PS, ROE, op_margin, profit_margin, DE, beta, …) into the classifier head of both models. This was removed. Rationale:

1. **The reference paper (electronics-12-03960) doesn't use fundamentals as features either.** Section 2.2 uses them only to compute a weighted stock score and screen the universe (threshold 0.65, 4565 → 4129 stocks). The deep hybrid model in Sec 2.4 only consumes technical factors + news embeddings.
2. **We can't implement the paper's screen honestly.** The paper uses RESSET/CSMAR, which provide historical point-in-time fundamentals. `yfinance` only exposes "latest" snapshots — forward-filling those across training windows leaks future information.
3. **The momentum slope that used to ride along in `X_fund` is not a fundamental.** It lives in `features/screening.py::apply_momentum_gate` as a post-inference filter, matching the paper's Sec 2.5 momentum rotation trading strategy.

---

## Experimental Findings (Empirical Log)

This section records what was actually run and what was learned. Updated as experiments complete.

---

### Experiment 1 — Per-Stock LSTM (reference hyperparameters)

**Setup:** One LSTM per stock, 49 US large-cap tickers, `window=20`, `horizon=3` (close[t+3] > close[t]), StepLR(step=10, γ=0.1), 150 epochs, batch=16. Sentiment-gated windows only (anchor day must have a non-zero embedding).

**Results:** Mean test AUC = **0.491** (below random). 23/49 stocks above 0.50. Best individual results: T (0.634), GOOGL (0.586), GS (0.580).

**Key observation:** `best_epoch` was 1–5 for most stocks, meaning the randomly initialized model generalized better than any trained version. The model peaked before meaningful learning occurred.

**Root causes identified:**

| # | Cause | Detail |
|---|---|---|
| 1 | LR schedule too aggressive | StepLR(10, 0.1) over 150 epochs → lr=1e-6 by epoch 30. Designed for 550+ Chinese stocks with thousands of windows; destroys learning on small US per-stock datasets |
| 2 | Temporal regime mismatch | Train: 2018–2023 (pre-AI rally). Test: June 2023+ (NVDA +200%, AI mania, Fed pause). NVDA achieved AUC=0.411 — *worse* than random |
| 3 | US market efficiency | US large-cap news is processed by HFT algos within milliseconds. The Chinese reference used East Money retail forum posts that causally drive price (readers trade on them). No equivalent mechanism for Bloomberg/Reuters articles |
| 4 | Small training sets | LIN (136 windows), DUK (162), AMT (179) — far too few to train 25k parameters. Models fitting to noise |
| 5 | Sentiment signal quality | FinBERT on broad financial news captures macro/sector noise, not stock-specific signal. The 768→16 projection cannot learn what matters from 500 examples |

---

### Experiment 2 — Sector-Level LSTM, Absolute Targets

**Setup:** Architecture B — one model per sector (8 sectors × 4 horizons = 32 models). Sector price index = equal-weight mean of constituent OHLCV. Sector embedding = mean of constituent FinBERT embeddings per day. Target: `close[t+horizon] > close[t]`. Horizons: T+5, T+10, T+21, T+42. Switched to `ReduceLROnPlateau` (patience=10, patience_stop=20).

**Results:** Mean AUC across all 32 models ≈ 0.50–0.54. Top results: ConsumerStaples T+42 (0.694), Industrials T+42 (0.680), Energy T+42 (0.668).

**Critical finding — class imbalance from positive market drift:** The top results were artefacts. `best_epoch=1` combined with `accuracy=0.871` for ConsumerStaples T+42 revealed the model simply predicted "up" for everything — correct 87% of the time because over a 42-day window in a bull market, the index almost always rises. The AUC was random initialization luck, not learned signal.

**What the scheduler fix achieved:** `best_epoch` moved from 1–5 (StepLR) to 1–20 (plateau) for most models. An improvement, but many were still at epoch 1–3, confirming the target was the core problem.

---

### Experiment 3 — Sector-Level LSTM, Relative Targets

**Setup:** Same as Experiment 2 but with cross-sector relative labels. For each day `t` and horizon `N`, a sector is labelled 1 if its `N`-day forward return is strictly above the median of all 8 sectors on the same day, 0 otherwise. This gives ~50/50 balance by construction.

**Sanity check finding:** The positive rates were *not* exactly 0.50 — they ranged from 0.39 (Energy) to 0.60 (Financials). This is not a bug: it reflects the real 2018–2024 sector dynamics (Tech and ConsumerStaples systematically outperformed; Energy and Industrials underperformed). Residual imbalance is much smaller than the absolute-target case but non-zero.

**Results:** Mean test AUC across 32 models ≈ 0.47–0.48. Most credible results (best_epoch ≥ 5):

| Sector | Horizon | best_epoch | Test AUC | CI |
|---|---|---|---|---|
| Energy | T+42 | 9 | 0.602 | [0.545, 0.662] |
| Energy | T+10 | 14 | 0.477 | [0.421, 0.532] |
| UtilTelecom | T+21 | 14 | 0.455 | [0.394, 0.519] |
| Industrials | T+21 | 20 | 0.433 | [0.377, 0.489] |
| Technology | T+21 | 12 | 0.505 | [0.443, 0.563] |

Only Energy T+42 has a CI that does not straddle 0.5. One credible result out of 32.

---

### Experiment 4 — Ablation: Technical Features Only

**Setup:** Identical to Experiment 3 but `ds.X_sent = np.zeros_like(ds.X_sent)` before DataLoader construction. The LSTM architecture is unchanged — the `sentiment_proj` layer still exists but always receives a zero input.

**Results:**

```
Mean delta (full − techonly): +0.009
Cases where full > techonly:   18 / 32  (coin flip)
Cases where full > techonly by >0.02:  12 / 32
```

**Verdict: FinBERT embeddings are contributing nothing.** The sentiment projection layer is net-negative in 14/32 cases, sometimes severely:

| Sector | Horizon | Full AUC | Tech-only AUC | Delta |
|---|---|---|---|---|
| Healthcare | T+21 | 0.464 | **0.574** | −0.110 |
| Energy | T+5 | 0.442 | **0.528** | −0.086 |
| UtilTelecom | T+21 | 0.455 | **0.537** | −0.082 |

**Most credible result overall:** Healthcare T+21, tech-only, `best_epoch=18`, AUC=0.574. A model that actually trained for 18 epochs with no sentiment and reached above-random performance. This is the strongest signal found so far.

**Why sentiment hurts:** The 768→16 linear projection cannot identify which of 768 FinBERT dimensions are relevant for sector prediction from ~1000 training windows. The projected vectors are effectively noisy. They occupy 16 of the 32 LSTM input dimensions, diluting the 16 meaningful technical features by 50%.

---

### Key Structural Insight: Why the Chinese Paper Works and Ours Doesn't

| Dimension | Reference (Chinese) | Ours (US) |
|---|---|---|
| Market efficiency | Low — 70% retail volume, sentiment causally drives price | High — HFT processes news in milliseconds |
| Sentiment source | East Money forum — readers trade on what they read | Bloomberg/Reuters — priced in before publication |
| Embedding model | Fine-tuned ALBERT on financial Chinese (384-dim) | Pre-trained FinBERT on English financial text (768-dim) |
| Universe | 550+ stocks, including mid/small-cap | 49 US mega-caps (most efficient stocks in the world) |
| Test period | June–Dec 2022 | June 2023+ (AI regime shift) |
| Reported AUC | ~0.56–0.60 | 0.49 (with sentiment), 0.50–0.57 (without) |

The paper works because there is a causal mechanism: Chinese retail investors read East Money and then buy the stock. Our English news pipeline has no equivalent mechanism for large-cap US names.

---

## Solution Plans: Fixing the Sentiment Signal

The ablation confirms the architecture and training setup are not the bottleneck. The bottleneck is **what the sentiment input represents**. Three concrete approaches, ranked by implementation cost and expected impact.

---

### Plan A — Replace Embedding with Scalar Probability Score (Low cost, high priority)

**Rationale:** FinBERT produces `(p_negative, p_neutral, p_positive)` alongside the 768-dim embedding. The scalar `p_positive − p_negative` is FinBERT's supervised answer to "is this text positive or negative" — a direct compression of 768 dims into the only dimension that matters for directional prediction. Rather than asking a linear layer to rediscover this compression from 1000 examples, use the pre-computed answer directly.

**Implementation:** Add `p_positive − p_negative` (aggregated daily across sector constituents) as a 17th technical feature column. No changes to the LSTM architecture — it disappears into the technical feature vector alongside RSI and MACD. The `sentiment_proj` layer and its 768-dim input are removed entirely.

**Requires:** Verifying that `sentiment_probs` is still stored in the parquet files (it was removed from model input but the pipeline may still compute and store it). If not, re-running the embedding step.

**Expected impact:** Moderate. The scalar signal is clean and task-aligned, but a single number loses all semantic nuance. Worth trying first because of low cost.

---

### Plan B — Sentiment Surprise (Rolling Delta)

**Rationale:** Absolute sentiment level is efficiently priced. What may not be priced is a sudden *change* in sentiment — a shift in how the market talks about a sector. The delta `sentiment_score[t] − mean(sentiment_score[t−20:t])` captures whether today's news is unusually positive or negative relative to recent history.

**Implementation:** Post-process the daily `p_positive − p_negative` scalar with a 20-day rolling mean subtraction before adding to the feature vector. Pure data transformation, no model changes.

**Expected impact:** Potentially higher than Plan A for capturing narrative shifts (e.g., Energy sector narrative turning negative before oil price drops). Complements Plan A rather than replacing it.

---

### Plan C — Sector-Filtered Embeddings (Medium cost)

**Rationale:** Currently we average FinBERT embeddings across all articles mentioning any constituent stock. An article saying "the Federal Reserve raised rates today" gets embedded and averaged into every sector's representation, even though it affects all sectors equally. That shared macro signal is already priced across all sectors simultaneously and contributes nothing to *relative* sector prediction.

**Implementation:** Before embedding aggregation, filter to sentences that explicitly name the sector's constituent companies (not just mentions in passing). Then compute the sector embedding. The idiosyncratic, company-specific signal is what survives after the cross-sector comparison removes the shared macro component.

**This connects to Root Cause 5** (sentiment signal quality): filtering to company-specific sentences gives the 768→16 projection something meaningful to compress rather than macro noise.

---

### Plan D — Two-Stream Architecture (High cost, exploratory)

**Rationale:** The current architecture concatenates sentiment and technical features at the input level, forcing the LSTM to jointly process two very different signal types at every timestep. A two-stream design processes them separately and combines at the output:

```
tech_stream:      LSTM(16) → hidden_t
sentiment_stream: MLP(1)   → sentiment_t     (scalar p_pos - p_neg)
combined:         Linear([hidden_t, sentiment_t]) → logit
```

This lets the LSTM specialise on temporal price patterns while sentiment enters as a modulating scalar at the classifier level — much closer to how a human analyst would use news ("the technical setup is bullish, and sentiment just turned positive, so buy").

**Expected impact:** Uncertain. Higher complexity with small datasets risks overfitting. Only worth trying after Plans A and B are evaluated.
