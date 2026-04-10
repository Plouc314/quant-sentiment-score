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
                               │  Volatility: ATR, BB%B, BBwidth │
                               │  Volume  : vol/SMA20, OBV slope │
                               │           VWAP/close            │
                               │  Returns : log_return           │
                               │  ──────────────────────────     │
                               │  per-day vector: (16,)          │
                               │  StandardScaler (train only)    │
                               └────────────────────────────────┘

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
                                                          sliding window      sliding window
                                                          size=64, stride=1   size=64, stride=1 ──► X_sentiment_probs
                                                                  │                                  (N, 64, 3)
                                                                  ▼
                                                              X_sentiment
                                                              (N, 64, 768)


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
  X_sentiment_probs │  │  X_tech (N, 64, 16) ───────────────────────────── │        │   │
  (N, 64, 3) ───────┼─►│  X_sentiment_probs (N, 64, 3) ─────────────────── │        │   │
                    │  │                                                    ▼        │   │
                    │  │                                           (N, 64, 35)       │   │
                    │  │                                                    │        │   │
                    │  │                                  LSTM(input=35, hidden=32)  │   │
                    │  │                                  2 layers, dropout=0.2      │   │
                    │  │                                  reads days 1→64 in order   │   │
                    │  │                                                    │        │   │
                    │  │                                       take LAST hidden state│   │
                    │  │                                            (N, 32)          │   │
                    │  └─────────────────────────────────────────────────────────────┘   │
                    │                                              │                      │
                    │                                              ▼                      │
                    │                          classifier: Linear(32→32) → ReLU          │
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
  X_sentiment_probs │  │  X_tech (N, 64, 16) ───────────────────────────── │        │   │
  (N, 64, 3) ───────┼─►│  X_sentiment_probs (N, 64, 3) ─────────────────── │        │   │
                    │  │                                                    ▼        │   │
                    │  │                                           (N, 64, 35)       │   │
                    │  │                                                    │        │   │
                    │  │                           input_proj: Linear(35 → 64)       │   │
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
                    │                                              │                      │
                    │                                              ▼                      │
                    │                            classifier: Linear(64 → 1)              │
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

- **N** = number of windows = (trading days − 60 warmup − window − 2 + 1), where 60 is the SMA-60 lookback, 2 is for the target computation, and +1 accounts for the inclusive sliding window count
- The **LSTM summary is 32 numbers**, the **Transformer summary is 64** (`d_model`) — these are the classifier inputs and the final output shape is identical
- `X_sentiment_probs` is a **64-day sequence** (same window as `X_sentiment`), concatenated with tech and projected sentiment before the temporal model — both come from FinBERT and capture temporal sentiment dynamics
- The **momentum gate** is an optional post-inference filter (option A from `design.md`) — it runs after the model, not inside it. This matches the paper (Sec 2.5), which uses the 20-day slope for momentum rotation trading *after* prediction, not as a model feature.
- **Fundamentals are intentionally absent.** The reference paper (Sec 2.2) uses them only as a pre-screen over the stock universe, not as model features. We don't implement that screen because yfinance does not provide historical point-in-time fundamentals (the paper uses RESSET/CSMAR); forward-filling a current snapshot across training windows would leak future information into the past.
