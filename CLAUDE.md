# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

Copy `.env.example` to `.env` and fill in the Alpaca API credentials.

## Architecture

Quantitative trading sentiment analysis based on https://doi.org/10.3390/electronics12183960.

The `src/` package is organised into five layers:

**Providers** — thin wrappers around external APIs (Alpaca, Yahoo Finance, Kaggle). Each provider fetches data and returns it as a DataFrame or TypedDict; no persistence logic.

**Repositories** — read/write data to disk. Each repository owns one data type and one directory layout. No business logic — just load and store.

**Embeddings** — two-step NLP pipeline: long articles are summarised before being encoded by FinBERT into 768-dim vectors and 3-class sentiment probabilities. `aggregate_daily()` collapses per-article encodings into one row per ticker per day.

**Features** — builds PyTorch-ready datasets. `TechnicalFactors` computes 16 scale-invariant OHLCV indicators. `StockDataset` aligns prices, embeddings, and fundamentals into flat arrays with lazy windowing. `DataLoaderBuilder` applies temporal splits, fits scalers on training data, and returns DataLoaders.

**Model** — `SentimentLSTM` and `SentimentTransformer` share the same input contract. `Trainer` handles the training loop, early stopping, and bootstrap evaluation.

**Top-level:**
- `models.py` — `Article`, `Fundamentals`, `ArticleEncoding` TypedDicts
- `training.py` — `TrainingConfig`, `ComputeConfig`, `Split`
- `log.py` — `setup_logging()`; call this at the top of each notebook/script

`src/__init__.py` calls `load_dotenv()` — no other file should call it.

Notebooks live in `notebooks/` and are the primary entry point:
- `fetch_data.ipynb` — populate all repositories (prices, fundamentals, news, sentiment)
- `train.ipynb` — train a model and save a checkpoint
- `evaluate.ipynb` — load a checkpoint and run bootstrap evaluation

## Coding Conventions

See `docs/conventions.md` for full details. Key points:

- **Thin, single-responsibility classes** — each wraps one external concern (API or file store), no inheritance
- **Fail loudly** — raise `RuntimeError`/`FileNotFoundError` for bad state; only explicit boundaries return `None`
- **No global mutable state** — credentials passed via `__init__`, resolved with `api_key = api_key or os.environ["KEY"]`
- **Types** — full annotations, `X | None` syntax, `TypedDict` for records, concrete types over abstract
- **Data dir defaults** — `Path(__file__).parents[N] / "data" / "<name>"` (relative to source file, not cwd)
- **Logging** — `logger = logging.getLogger(__name__)` at module level; library code never configures handlers
- **Pagination** — `while True: ...; if not token: break` pattern
- Private attributes prefixed with `_`; private helpers at bottom of class
