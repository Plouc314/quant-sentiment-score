"""Coverage-based stock screening.

Filters a ticker universe by average monthly news article count, allowing
downstream sentiment encoding to run only on stocks with sufficient signal.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import pandas as pd

from ..sources.news.repository import ArticleRepository

logger = logging.getLogger(__name__)


class CoverageStats(TypedDict):
    ticker: str
    months_tracked: int
    avg_articles_per_month: float
    passes: bool


def momentum_slope(closes: pd.Series, window: int = 20) -> float:
    """Linear regression slope of the last `window` closing prices.

    Positive slope → uptrend (pass); negative → downtrend (filter out).
    Uses normalised x-axis (0..window-1) so the slope is in price-units-per-day.
    """
    prices = closes.iloc[-window:].values
    slope: float = np.polyfit(np.arange(window), prices, 1)[0]
    return slope


def apply_momentum_gate(
    probs: np.ndarray,
    targets: np.ndarray,
    closes: pd.Series,
    win_dates: np.ndarray,
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply momentum pre-filter at inference: retain only uptrend windows.

    Computes the linear-regression slope of the last *window* closes for each
    window end date and returns arrays masked to uptrend windows (slope > 0).

    Parameters
    ----------
    probs:
        Model output probabilities, shape ``(N,)``.
    targets:
        Ground-truth labels, shape ``(N,)``.
    closes:
        Full close price series with DatetimeIndex (must cover all *win_dates*).
    win_dates:
        Array of window end dates, length N — typically
        ``FusedDataset["dates"][test_start:]``.
    window:
        Number of trailing closes used to fit the slope (default 20 trading
        days, ~1 month).

    Returns
    -------
    ``(filtered_probs, filtered_targets, uptrend_mask)`` where *uptrend_mask*
    is a boolean array of length N.  Use it to index back into the original
    arrays or to report coverage.

    Notes
    -----
    This is a hard inference-time gate: predictions for downtrend windows are
    discarded without the model having seen the slope signal during training.
    See ``docs/design.md`` for alternative approaches.
    """
    slopes = np.array(
        [_slope_at(closes, pd.Timestamp(d), window) for d in win_dates],
        dtype=np.float32,
    )
    mask = slopes > 0
    return probs[mask], targets[mask], mask


def screen_by_coverage(
    repository: ArticleRepository,
    tickers: list[str],
    start: tuple[int, int],
    end: tuple[int, int],
    min_avg_articles: float = 5.0,
) -> pd.DataFrame:
    """Filter *tickers* by average monthly article coverage.

    Reads only index metadata (no article text) from *repository* for each
    calendar month in [*start*, *end*].  Returns a DataFrame with one row per
    ticker and columns: ticker, months_tracked, avg_articles_per_month, passes.

    Parameters
    ----------
    repository:
        ArticleRepository instance.
    tickers:
        Universe of ticker symbols to evaluate.
    start:
        ``(year, month)`` of the first calendar month to include, inclusive.
    end:
        ``(year, month)`` of the last calendar month to include, inclusive.
    min_avg_articles:
        Minimum average articles per calendar month for a ticker to pass.
        Months with zero articles for a ticker count toward the denominator —
        a stock that attracts news only occasionally should not pass the filter.
    """
    ticker_set = set(tickers)
    months = _iter_months(start, end)
    counts: dict[str, int] = {t: 0 for t in tickers}

    for year, month in months:
        df = repository.read_month_index(year, month)
        if df.empty:
            continue
        # explode list-valued tickers column to get one row per (article, ticker)
        exploded = df[["id", "tickers"]].explode("tickers")
        exploded = exploded[exploded["tickers"].isin(ticker_set)]
        for ticker, group in exploded.groupby("tickers"):
            counts[str(ticker)] += len(group)

    n_months = len(months)
    rows: list[CoverageStats] = []
    for ticker in tickers:
        total = counts[ticker]
        avg = total / n_months if n_months > 0 else 0.0
        passes = avg >= min_avg_articles
        rows.append(
            {
                "ticker": ticker,
                "months_tracked": n_months,
                "avg_articles_per_month": round(avg, 2),
                "passes": passes,
            }
        )

    result = pd.DataFrame(rows)
    n_pass = result["passes"].sum()
    logger.info(
        "Coverage screen: %d/%d tickers pass (min_avg=%.1f over %d months)",
        n_pass,
        len(tickers),
        min_avg_articles,
        n_months,
    )
    return result


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _slope_at(closes: pd.Series, end_ts: pd.Timestamp, window: int) -> float:
    """Return momentum slope at *end_ts* using the last *window* closes."""
    loc = closes.index.get_indexer([end_ts], method="pad")[0]
    if loc < window - 1:
        return 0.0
    return momentum_slope(closes.iloc[loc - window + 1 : loc + 1], window=window)


def _iter_months(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples from *start* to *end* inclusive."""
    months: list[tuple[int, int]] = []
    year, month = start
    end_year, end_month = end
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months
