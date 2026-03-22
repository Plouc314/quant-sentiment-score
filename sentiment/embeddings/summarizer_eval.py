"""Evaluation utilities for comparing summarizer quality on financial articles."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from rouge_score import rouge_scorer

from ..sources.news.models import Article
from .encoder import SentimentEncoder
from .summarizer import Summarizer

if TYPE_CHECKING:
    from ..features.technical import TechnicalFactors

logger = logging.getLogger(__name__)

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def evaluate_rouge(
    summarizer: Summarizer,
    articles: list[Article],
    gold_fn: Callable[[Article], str],
) -> dict:
    """Compute ROUGE scores for *summarizer* on *articles*.

    Parameters
    ----------
    summarizer:
        A :class:`~sentiment.embeddings.summarizer.Summarizer` instance.
    articles:
        List of articles to evaluate (text must be populated).
    gold_fn:
        Callable that extracts the gold reference summary from an article.
        Common choices::

            # First paragraph as lead-paragraph proxy
            lambda a: (a["text"] or "").split("\\n\\n")[0].strip()

            # Title only
            lambda a: a["title"] or ""

    Returns
    -------
    Dict with keys:

    - ``rouge1_f``, ``rouge2_f``, ``rougeL_f`` — mean F1 scores (0–1)
    - ``rouge1_p``, ``rouge2_p``, ``rougeL_p`` — mean precision scores
    - ``rouge1_r``, ``rouge2_r``, ``rougeL_r`` — mean recall scores
    - ``bypass_rate`` — fraction of articles skipped by the short-content bypass
    - ``mean_seconds_per_article`` — wall-clock time per article (bypass included)
    - ``n_articles`` — number of articles evaluated
    """
    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)

    scores: dict[str, list[float]] = {
        f"{rt}_{m}": [] for rt in _ROUGE_TYPES for m in ("f", "p", "r")
    }
    n_bypass = 0
    total_time = 0.0

    for article in articles:
        content = (article.get("text") or "").strip()
        gold = gold_fn(article).strip()
        if not content or not gold:
            continue

        t0 = time.perf_counter()
        summary = summarizer.summarize(content)
        total_time += time.perf_counter() - t0

        if summary == content:
            n_bypass += 1

        result = scorer.score(gold, summary)
        for rt in _ROUGE_TYPES:
            scores[f"{rt}_f"].append(result[rt].fmeasure)
            scores[f"{rt}_p"].append(result[rt].precision)
            scores[f"{rt}_r"].append(result[rt].recall)

    n = len(scores["rouge1_f"])
    means = {k: float(sum(v) / len(v)) if v else 0.0 for k, v in scores.items()}
    means["bypass_rate"] = n_bypass / n if n else 0.0
    means["mean_seconds_per_article"] = total_time / n if n else 0.0
    means["n_articles"] = n
    return means


def evaluate_downstream_auc(
    summarizer_model: str | None,
    ticker_articles: dict[str, list[Article]],
    df: pd.DataFrame,
    technical: TechnicalFactors,
    ticker: str,
    device: str = "cpu",
    n_epochs: int = 50,
    seed: int | None = None,
    fundamental_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for a given summarizer and return downstream model AUC.

    Runs summarize → FinBERT encode → build dataset → train LSTM → bootstrap
    evaluate on the test split.  This is the only honest comparison between
    summarizer candidates: ROUGE measures faithfulness to source, not trading
    signal quality.

    Parameters
    ----------
    summarizer_model:
        HuggingFace seq2seq model name, or ``None`` to skip summarization
        (raw article text is passed directly to FinBERT, which truncates at
        512 tokens).  Useful as a no-summarization baseline.
    ticker_articles:
        ``{ticker: [Article, ...]}`` — same format as
        ``SentimentPipeline.process_ticker_articles``.
    df:
        OHLCV DataFrame with DatetimeIndex for *ticker*.
    technical:
        ``TechnicalFactors`` instance.
    ticker:
        Ticker symbol.
    device:
        ``"cpu"`` or ``"cuda"``.
    n_epochs:
        Maximum training epochs (early stopping applies, so most runs stop
        earlier).
    seed:
        Random seed for reproducibility.
    fundamental_df:
        Optional ``FundamentalCache.load_df(ticker)`` output.

    Returns
    -------
    Dict with keys:

    - ``summarizer_model`` — the model name passed in (or ``None``)
    - ``auc_mean``, ``auc_ci_low``, ``auc_ci_high`` — bootstrap AUC (95% CI)
    - ``best_epoch`` — epoch at which the best val AUC was reached
    - ``best_val_auc`` — best validation AUC during training
    - ``n_test_samples`` — number of test windows evaluated
    """
    # Lazy imports avoid circular dependencies and keep module load fast
    from ..features.dataloader import build_dataset, make_loaders
    from ..model import SentimentLSTM, bootstrap_evaluate, train_model
    from .pipeline import SentimentPipeline

    logger.info(
        "evaluate_downstream_auc: summarizer_model=%s  ticker=%s",
        summarizer_model,
        ticker,
    )

    pipeline = SentimentPipeline(device=device, summarizer_model=summarizer_model)
    sentiment_df = pipeline.process_ticker_articles(ticker_articles)

    dataset = build_dataset(
        df,
        technical,
        sentiment_df=sentiment_df if not sentiment_df.empty else None,
        ticker=ticker,
        fundamental_df=fundamental_df,
    )

    n_fund = dataset["X_fundamental"].shape[1]
    n_sprob = dataset["X_sentiment_probs"].shape[1]

    train_loader, val_loader, test_loader, _, _ = make_loaders(dataset)
    model = SentimentLSTM(n_fundamentals=n_fund, n_sentiment_probs=n_sprob)
    history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        device=device,
        seed=seed,
    )

    result = bootstrap_evaluate(model, test_loader, device=device, seed=seed)
    return {
        "summarizer_model": summarizer_model,
        "auc_mean": result["auc_mean"],
        "auc_ci_low": result["auc_ci_low"],
        "auc_ci_high": result["auc_ci_high"],
        "best_epoch": history["best_epoch"],
        "best_val_auc": history["best_val_auc"],
        "n_test_samples": result["n_samples"],
    }


def label_agreement_rate(
    encoder: SentimentEncoder,
    summarizer: Summarizer,
    articles: list[Article],
) -> float:
    """Fraction of articles where summarization preserves the FinBERT sentiment label.

    For each article, encodes the full text (truncated to FinBERT's 512-token limit)
    and encodes the summary, then checks whether the argmax class agrees.  A high
    agreement rate means the summarizer is not distorting the sentiment signal.

    Articles with no text or where the short-content bypass fires (summary == text)
    are excluded from the denominator.

    Returns
    -------
    Agreement rate in ``[0.0, 1.0]``, or ``float("nan")`` when no articles
    qualify (all skipped by bypass or empty text).  Callers must check
    ``math.isnan(result)`` before using the return value in comparisons.
    """
    agreed = 0
    total = 0

    for article in articles:
        content = (article.get("text") or "").strip()
        if not content:
            continue

        summary = summarizer.summarize(content)
        if summary == content:
            continue  # bypass: nothing to compare

        label_full, _, _ = encoder.encode(content)
        label_sum, _, _ = encoder.encode(summary)

        total += 1
        if label_full == label_sum:
            agreed += 1

    if total == 0:
        logger.warning("label_agreement_rate: no non-bypass articles found")
        return float("nan")

    return agreed / total
