# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

Copy `.env.example` to `.env` and fill in API credentials (Alpaca, MediaCloud).

## Architecture

Quantitative trading sentiment analysis based on https://doi.org/10.3390/electronics12183960.

The `sentiment/` package has two independent pipelines:

**Market data pipeline:**
- `sources/alpaca.py` — `AlpacaSource` fetches OHLCV bars from Alpaca API with retry/backoff
- `sources/cache.py` — `MarketDataCache` persists bars as CSV under `data/historical-prices/<SYMBOL>/<YEAR>.csv`

**News pipeline:**
- `sources/news/search.py` — `NewsSearch` wraps MediaCloud API, returns `Story` objects
- `sources/news/extractor.py` — `ArticleExtractor` uses trafilatura to fetch and extract article text; parallel by source with per-source rate limiting
- `sources/news/blacklist.py` — `SourceBlacklist` tracks per-source fetch failures and auto-blacklists sources that exceed the failure threshold
- `sources/news/repository.py` — `ArticleRepository` stores articles in `data/news/` (CSV index + per-article `.txt` files)
- `sources/news/models.py` — `Story` and `Article` TypedDicts

`sentiment/__init__.py` calls `load_dotenv()` — no other file should call it.

Notebooks live in `notebooks/` and are the primary entry point. Call `setup_logging()` from `sentiment/log.py` at the top of each notebook/script.

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
