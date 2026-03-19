# Plan: Hybrid Deep Neural Model for Stock Prediction & Trading

## Decisions Made

### 1. Classification, not Regression
- Switch from MSE regression (predict next close price) to **binary classification** (rise/fall)
- Use LSTM with `BCEWithLogitsLoss`
- Prediction target: will the stock rise on day t+2 vs t-1 (accounts for trading delay, matches paper)
- Output: single logit, sigmoid at inference for probability

### 2. Feature Fusion: Project + Concatenate
- Compute N technical factors from OHLCV
- Project 768-dim FinBERT embedding to N-dim via a learned `nn.Linear(768, N)` (trained end-to-end)
- Concatenate: [tech_factors | projected_embedding] = 2N-dim input per timestep
- Feed time series of 2N-dim vectors into LSTM

### 3. Replace Raw OHLCV with Technical Factors
- Drop raw OHLCV columns (open/high/low/close/volume/trade_count/vwap)
- Compute derived technical factors
- Technical factors encode meaningful market patterns; raw prices have high redundancy

---

## Technical Factor Comparison: Paper vs. Ours

### Paper's 16 Factors (from their GitHub: SallyLi0606/Quant)
All computed with TA-Lib, MinMaxScaler per-window normalization.

| # | Factor | Category |
|---|--------|----------|
| 1 | MA(5) | Trend |
| 2 | MA(30) | Trend |
| 3 | MA(60) | Trend |
| 4 | EMA(5) | Trend |
| 5 | EMA(30) | Trend |
| 6 | EMA(60) | Trend |
| 7 | MACD histogram (6/15/6) | Trend |
| 8 | MACD histogram (12/26/9) | Trend |
| 9 | MACD histogram (30/60/30) | Trend |
| 10 | RSI(14) | Momentum |
| 11 | Williams %R(14) | Momentum |
| 12 | MOM(14) | Momentum |
| 13 | CMO(14) | Momentum |
| 14 | Ultimate Oscillator(7/14/28) | Momentum |
| 15 | OBV | Volume |
| 16 | Chaikin A/D Oscillator(3/10) | Volume |

**Observations on paper's set:**
- Heavily trend-biased: 9/16 are trend indicators (6 raw MAs + 3 MACDs)
- Raw MAs are not normalized to close — relies on MinMaxScaler to fix scale differences across stocks
- MA(5) and EMA(5) carry largely overlapping information (same for 30 and 60 pairs)
- Zero volatility indicators — no ATR, no Bollinger Bands
- Only 2 volume indicators

### Our 16 Factors (implemented in `sentiment/features/technical.py`)
More balanced across categories, ratio-based normalization, includes volatility.

**Trend (4):**
| # | Factor | Formula | Justification |
|---|--------|---------|---------------|
| 1 | Close / SMA(5) | close / 5-day SMA | Short-term trend position. >1 = above average. Ratio is naturally normalized across stocks. |
| 2 | Close / SMA(20) | close / 20-day SMA | Medium-term trend. 20-day = standard monthly benchmark. |
| 3 | Close / SMA(60) | close / 60-day SMA | Long-term trend. Captures sustained up/downtrends over ~3 months. |
| 4 | MACD | (EMA12 - EMA26) / close | Trend strength & direction. Normalized by close for cross-stock comparability. |

**Momentum (5):**
| # | Factor | Formula | Justification |
|---|--------|---------|---------------|
| 5 | MACD Signal | EMA(9) of MACD | Crossovers between MACD and Signal are classic entry/exit triggers. |
| 6 | RSI(14) | 100 - 100/(1 + avg_gain/avg_loss) | Standard momentum oscillator [0-100]. >70 overbought, <30 oversold. |
| 7 | Stochastic %K(14) | (close - low_14) / (high_14 - low_14) | Where close sits in 14-day range. Complementary to RSI. |
| 8 | Stochastic %D | 3-day SMA of %K | Smoothed stochastic. %K/%D crossovers are signals. |
| 9 | ROC(10) | (close - close_10d) / close_10d | Pure momentum. Unbounded, captures extreme moves. |

**Volatility (3):**
| # | Factor | Formula | Justification |
|---|--------|---------|---------------|
| 10 | ATR(14) / Close | 14-day avg true range / close | Normalized volatility. High = big swings = uncertain predictions. |
| 11 | BB %B(20) | (close - BB_lower) / (BB_upper - BB_lower) | Position within Bollinger Bands. Near 0 = potential bounce, near 1 = potential pullback. |
| 12 | BB Width(20) | (BB_upper - BB_lower) / SMA(20) | Volatility expansion/contraction. Narrow bands often precede breakouts. |

**Volume (3):**
| # | Factor | Formula | Justification |
|---|--------|---------|---------------|
| 13 | Volume / SMA(20, vol) | volume / 20-day avg volume | Relative volume. >1 = unusual activity. Confirms price moves. |
| 14 | OBV slope(10) | 10-day linreg slope of OBV | Volume trend. Rising OBV = buyers accumulating, even if price is flat. |
| 15 | VWAP / Close | vwap / close | Above/below volume-weighted fair value. Free from Alpaca data. |

**Returns (1):**
| # | Factor | Formula | Justification |
|---|--------|---------|---------------|
| 16 | Log return | ln(close_t / close_{t-1}) | Daily price change. Additive, approximately normal. Most direct signal. |

**Advantages over paper's set:**
- More balanced: 4 trend / 5 momentum / 3 volatility / 3 volume / 1 returns (vs 9/5/0/2/0)
- Ratio-based normalization: factors are naturally scale-invariant without relying on per-window MinMaxScaler
- Volatility coverage: ATR + Bollinger Bands capture regime changes the paper ignores
- Less redundancy: no MA/EMA duplication at same timeframes

---

## Implemented Modules

### `sentiment/features/technical.py` — 16 Technical Indicators
- `FACTOR_COLUMNS`: constant list of 16 factor names
- `TechnicalFactors.compute(df)`: OHLCV+VWAP DataFrame → 16-column factor DataFrame
- Pure pandas/numpy, no TA-Lib dependency

### `sentiment/features/dataloader.py` — Dataset Building & Fusion
- `compute_targets(close)`: binary target (t+2 vs t-1)
- `align_sentiment(index, sentiment_df, ticker)`: align 768-dim embeddings with market dates
- `build_dataset(df, tech, sentiment_df, ticker, window)`: full pipeline → sliding windows
- `make_loaders(dataset, test_frac, val_frac, batch_size)`: chronological split + StandardScaler + DataLoaders
- `FusedStockDataset`: PyTorch Dataset for (tech, sentiment, target) triplets

### `sentiment/model/lstm.py` — Classification Model
- `SentimentLSTM(n_factors, sentiment_dim, hidden_size, num_layers, dropout)`
- Architecture: Linear(768→16) projection → concat → LSTM(32, 2 layers) → Linear(32→1) classifier
- Input: (batch, window, 16) tech + (batch, window, 768) sentiment → (batch, 1) logit

### `sentiment/model/train.py` — Training & Evaluation
- `train_model(model, train_loader, val_loader, ...)`: Adam + BCEWithLogitsLoss + early stopping on val AUC
- `evaluate(model, loader, device)`: accuracy, AUC, precision, recall

---

## Analysis & Observations (What Still Needs to Be Done)

### A. Sentiment Pipeline Gaps

1. **Fine-tuning gap**: The paper fine-tunes ALBERT on ChnSentiCorp before applying to stock news. We use FinBERT off-the-shelf with no task-specific fine-tuning. FinBERT's financial pretraining partially compensates, but fine-tuning on a labeled financial sentiment dataset could improve quality.

2. **Three-class collapse**: FinBERT outputs 3 classes (positive/negative/neutral). We collapse to binary (positive=1, else=0), treating neutral as negative. Consider whether neutral articles should be excluded or handled differently.

3. **Summarizer domain specificity**: We use general-purpose BART-CNN; the paper uses a domain-adapted Chinese summarizer. A finance-tuned summarizer could improve summary quality for financial jargon.

4. **Embedding extraction method**: We use mean pooling (robust for our no-fine-tuning setup). Paper likely uses [CLS] token (standard when model is fine-tuned with [CLS] as sentence representation). Current choice is reasonable.

### B. Prediction Model Gaps

5. ~~**Feature fusion not connected**~~ → DONE: `sentiment/features/dataloader.py` + `sentiment/model/lstm.py`

6. ~~**No technical factors**~~ → DONE: `sentiment/features/technical.py`

7. ~~**Classification vs regression**~~ → DONE: `BCEWithLogitsLoss` in `sentiment/model/train.py`

8. ~~**Prediction horizon**~~ → DONE: `compute_targets()` uses t+2 vs t-1

9. **Transformer model**: Paper compares LSTM vs Transformer (N=6, h=8, d=64). Our codebase has placeholder comment only. Implement as a secondary model for comparison.

### C. Trading Strategy Gaps

10. **No stock screening**: Paper narrows 4565 -> 4129 -> 577 stocks through fundamental + sentiment screening. We have no screening — we manually specify symbols.

11. **No momentum rotation**: Paper uses 20-day linear regression slope on close prices, keeps top 60% by momentum.

12. **No trading signals**: No buy/sell logic based on model predictions.

13. **No risk management**: No profit-taking (+22%) or stop-loss (-8%) implementation.

14. **No backtesting framework**: No portfolio simulation, no ARR/MDR metrics, no benchmark comparison. This is needed to evaluate the strategy.

### D. Infrastructure Gaps

15. ~~**No train/val/test split strategy**~~ → DONE: chronological split in `make_loaders()`

16. **No per-stock model evaluation**: Paper selects top 50 stocks by validation accuracy. We train/evaluate on a single stock.

17. **No multi-stock training**: Paper trains across 577 stocks. We train on one stock at a time.
