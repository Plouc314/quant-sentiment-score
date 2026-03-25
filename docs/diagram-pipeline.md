# Pipeline Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              QUANT-SENTIMENT-SCORE — FULL PIPELINE                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

 DATA SOURCES                 FEATURE ENGINEERING                   WINDOWING              MODEL INPUT
 ────────────                 ───────────────────                   ─────────              ───────────

 ┌─────────────┐   OHLCV+VWAP  ┌────────────────────────────────┐
 │  Alpaca API │ ────────────► │        TechnicalFactors         │  sliding window
 │  (daily     │               │  Trend   : SMA5/20/60, MACD    │  size=64, stride=1 ──► X_tech
 │   bars)     │               │  Momentum: MACD sig, RSI,       │                        (N, 64, 16)
 └─────────────┘               │           Stoch K/D, ROC-10    │
         │                     │  Volatility: ATR, BB%B, BBwidth │
         │                     │  Volume  : vol/SMA20, OBV slope │
         │  (close prices)     │           VWAP/close            │
         │                     │  Returns : log_return           │
         │                     │  ──────────────────────────     │
         │                     │  per-day vector: (16,)          │
         │                     │  StandardScaler (train only)    │
         │                     └────────────────────────────────┘
         │
         │  (last 20 closes                                                               X_fundamental
         │   per window)  ──── momentum slope ──────────────────────────────────────────► appended
         │                     (lin. reg. slope)                                          as feature 10
         │
 ┌─────────────┐  Stories   ┌─────────────────────┐   raw text   ┌──────────────────┐
 │  MediaCloud │ ─────────► │   ArticleExtractor   │ ──────────► │  BART Summarizer │
 │  API        │            │   (trafilatura)      │             │  (facebook/       │
 └─────────────┘            └─────────────────────┘             │  bart-large-cnn) │
                                                                 └──────────────────┘
                                                                          │ summary + title
                                                                          ▼
                                                                 ┌──────────────────┐
                                                                 │  FinBERT Encoder │
                                                                 │  (ProsusAI/      │
                                                                 │   finbert)       │
                                                                 └──────────────────┘
                                                                  │               │
                                                          embedding (768,)    probs (3,)
                                                          per article         [P(pos),P(neg),P(neu)]
                                                                  │               │
                                                          daily aggregation   daily aggregation
                                                          (mean over all      (mean over all
                                                           articles/day)       articles/day)
                                                                  │               │
                                                          align to price      align to price
                                                          date index          date index
                                                          zero if no news     zero if no news
                                                                  │               │
                                                          sliding window      snapshot at
                                                          size=64, stride=1   window end date ──► X_sentiment_probs
                                                                  │                                (N, 3)
                                                                  ▼
                                                              X_sentiment
                                                              (N, 64, 768)

 ┌─────────────┐  quarterly  ┌─────────────────────────────────────────┐
 │  yfinance   │ ──────────► │  FundamentalCache  (fundamentals.csv)   │
 │  snapshots  │             │  PE, fwdPE, PB, PS, ROE,                │
 └─────────────┘             │  op_margin, profit_margin, D/E, beta    │
                             │  ──────────────────────────────────      │
                             │  forward-fill quarterly → daily          │
                             │  snapshot at window end date             │
                             │  StandardScaler (train only)             │  ──────────────────────► X_fundamental
                             └─────────────────────────────────────────┘                           (N, 9) or (N,10)


══════════════════════════════════════════════════════════════════════════════════════════════════════
                                          MODEL
══════════════════════════════════════════════════════════════════════════════════════════════════════

                    ┌── MODEL A: LSTM ────────────────────────────────────────────────────┐
                    │                                                                      │
  X_tech            │  ┌─────────────────────────────────────────────────────────────┐   │
  (N, 64, 16) ──────┼─►│                                                             │   │
                    │  │  sentiment_proj: Linear(768 → 16)                           │   │
  X_sentiment       │  │  X_sentiment (N,64,768) ──────────────────► (N, 64, 16)    │   │
  (N, 64, 768) ─────┼─►│                                                    │        │   │
                    │  │                                            concat on feat   │   │
                    │  │  X_tech (N, 64, 16) ───────────────────────────── │        │   │
                    │  │                                                    ▼        │   │
                    │  │                                           (N, 64, 32)       │   │
                    │  │                                                    │        │   │
                    │  │                                  LSTM(input=32, hidden=32)  │   │
                    │  │                                  2 layers, dropout=0.2      │   │
                    │  │                                  reads days 1→64 in order   │   │
                    │  │                                                    │        │   │
                    │  │                                       take LAST hidden state│   │
                    │  │                                            (N, 32)          │   │
                    │  └─────────────────────────────────────────────────────────────┘   │
                    │                                               │                     │
  X_fundamental     │                                          concat                     │
  (N, 10) ──────────┼──────────────────────────────────────────── │                     │
                    │                                              ▼                      │
  X_sentiment_probs │                                        (N, 32+10+3 = 45)           │
  (N, 3) ───────────┼──────────────────────────────────────────── │                     │
                    │                                              ▼                      │
                    │                          classifier: Linear(45→32) → ReLU          │
                    │                                      → Dropout(0.2)                 │
                    │                                      → BatchNorm1d                  │
                    │                                      → Linear(32→1)                 │
                    │                                              │                      │
                    └──────────────────────────────────────────────┼──────────────────────┘
                                                                   │
                    ┌── MODEL B: TRANSFORMER ─────────────────────────────────────────────┐
                    │                                                                      │
  X_tech            │  ┌─────────────────────────────────────────────────────────────┐   │
  (N, 64, 16) ──────┼─►│                                                             │   │
                    │  │  sentiment_proj: Linear(768 → 16)                           │   │
  X_sentiment       │  │  X_sentiment (N,64,768) ──────────────────► (N, 64, 16)    │   │
  (N, 64, 768) ─────┼─►│                                                    │        │   │
                    │  │                                            concat on feat   │   │
                    │  │  X_tech (N, 64, 16) ───────────────────────────── │        │   │
                    │  │                                                    ▼        │   │
                    │  │                                           (N, 64, 32)       │   │
                    │  │                                                    │        │   │
                    │  │                           input_proj: Linear(32 → 64)       │   │
                    │  │                                           (N, 64, 64)       │   │
                    │  │                                                    │        │   │
                    │  │                        + pos_embedding [0..63]              │   │
                    │  │                          (learned position tags)            │   │
                    │  │                                           (N, 64, 64)       │   │
                    │  │                                                    │        │   │
                    │  │              TransformerEncoder: 6 layers                   │   │
                    │  │              each layer:                                    │   │
                    │  │                Multi-head self-attention (4 heads)          │   │
                    │  │                → all 64 days attend to each other           │   │
                    │  │                Feedforward(64 → 128 → 64)                  │   │
                    │  │                                           (N, 64, 64)       │   │
                    │  │                                                    │        │   │
                    │  │                              MEAN POOL over 64 days         │   │
                    │  │                                      → (N, 64)              │   │
                    │  └─────────────────────────────────────────────────────────────┘   │
                    │                                               │                     │
  X_fundamental     │                                          concat                     │
  (N, 10) ──────────┼──────────────────────────────────────────── │                     │
                    │                                              ▼                      │
  X_sentiment_probs │                                       (N, 64+10+3 = 77)            │
  (N, 3) ───────────┼──────────────────────────────────────────── │                     │
                    │                                              ▼                      │
                    │                            classifier: Linear(77 → 1)              │
                    │                                                                     │
                    └──────────────────────────────────────────────┼──────────────────────┘
                                                                   │
                                                                   ▼
                                                             sigmoid(logit)
                                                                   │
                                                     ┌─────────────┴─────────────┐
                                                     │      P(price goes up)     │
                                                     │        threshold 0.5      │
                                                     └─────────────┬─────────────┘
                                                                   │
                                                        ┌──────────┴──────────┐
                                                        │  Momentum Gate      │
                                                        │  (optional, option A)│
                                                        │  discard if 20-day  │
                                                        │  slope ≤ 0          │
                                                        └──────────┬──────────┘
                                                                   │
                                                            BUY signal  (y=1)
                                                            target: close[t+2] > close[t-1]
```

## Notes

- **N** = number of windows = (trading days − 60 warmup − window − 2), where 60 is the SMA-60 lookback and 2 is for the target computation
- The **LSTM summary is 32 numbers**, the **Transformer summary is 64** (`d_model`), so their classifier inputs differ (45 vs 77) but the final output shape is identical
- `X_sentiment_probs` is a **snapshot** (last day of window only), while `X_sentiment` is the full **64-day sequence** — both come from FinBERT but serve different roles in the model
- The **momentum gate** is an optional post-inference filter (option A from `design.md`) — it runs after the model, not inside it
