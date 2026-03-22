"""Evaluation utilities for comparing summarizer quality on financial articles."""

from __future__ import annotations

import logging
import time
from typing import Callable

from rouge_score import rouge_scorer

from ..sources.news.models import Article
from .encoder import SentimentEncoder
from .summarizer import Summarizer

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
