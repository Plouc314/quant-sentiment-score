"""Coverage-based stock screening.

Filters a ticker universe by average monthly news article count, allowing
downstream sentiment encoding to run only on stocks with sufficient signal.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import pandas as pd

from ..sources.news.repository import ArticleRepository

logger = logging.getLogger(__name__)


class CoverageStats(TypedDict):
    ticker: str
    months_tracked: int
    avg_articles_per_month: float
    passes: bool


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
