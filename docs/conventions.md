# Coding Conventions

## General principles

- **Explicit over clever** — no magic, no metaclasses, no decorators beyond stdlib. Code reads top to bottom.
- **Thin classes, no inheritance** — each class wraps one external concern (an API, a file store). No base classes or mixins.
- **Fail loudly** — bad state raises `RuntimeError` or `FileNotFoundError`; nothing is silently swallowed except at explicit boundaries (e.g. `extract()` returning `None`).
- **No global mutable state** — all config/credentials go through `__init__`, never module-level globals (except constants like `BASE_URL`).

## Typing

- Full type annotations on all public methods (parameters and return type).
- `X | None` union syntax (Python 3.10+), not `Optional[X]`.
- Prefer concrete types (`list[str]`, `dict[str, list[dict]]`) over `Sequence`, `Mapping`, etc.
- `TypedDict` for structured data records

## Credentials / configuration

- API keys default to `None` in `__init__`, then resolved via `os.environ["KEY"]`.
- Pattern: `api_key = api_key or os.environ["ALPACA_API_KEY"]` — explicit fallback, hard error if missing.
- `.env` loaded once in `sentiment/__init__.py` via `python-dotenv`; no other file calls `load_dotenv()`.

## Class structure

- Private attributes prefixed with `_` (`self._api`, `self._df`).
- Public data attributes (no leading `_`) only when intentionally part of the API.
- Private helpers go at the bottom, after all public methods.
- No `@property` unless genuinely needed — direct attribute access is preferred.

## Data layer

- Storage classes expose `store()` / `load()` as the primary interface.
- `data_dir` defaults to `Path(__file__).parents[N] / "data" / "<name>"` — resolved relative to the source file, not cwd.
- DataFrames for tabular data; `TypedDict` for individual records.
- CSV is the persistence format.

## Docstrings

- Class docstring describes the concept and any important layout/structure.
- Method docstrings: one-line summary, then elaboration if needed. Args/Returns sections only when the signature isn't self-evident.

## Logging

- Each module that needs logging declares `logger = logging.getLogger(__name__)` at module level (lowercase).
- Library code only logs, never configures handlers.
- `setup_logging()` in `sentiment/log.py` is called once at entry points (scripts, notebooks).
