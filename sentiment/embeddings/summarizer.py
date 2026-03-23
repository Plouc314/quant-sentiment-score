from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# BART-CNN generation defaults (from summarization.ipynb POC)
_BART_MAX_INPUT = 1024
_BART_MAX_OUTPUT = 128
_BART_MIN_OUTPUT = 30
_BART_NUM_BEAMS = 4
_BART_LENGTH_PENALTY = 2.0

# FinBERT tokenisation limit (used for short-content bypass)
_FINBERT_MAX_LENGTH = 512


class Summarizer:
    """Seq2seq summarizer — compresses long article text for downstream encoding.

    Defaults to ``facebook/bart-large-cnn``; pass any HuggingFace seq2seq model
    name (e.g. ``human-centered-summarization/financial-summarization-pegasus``)
    to swap the backbone without changing any other code.

    Pass ``model_name=None`` for a no-op summarizer that returns text unchanged.
    In this mode no model is loaded and :meth:`summarize` simply returns its
    input — useful for downstream AUC evaluation where you want to measure the
    baseline of feeding raw (FinBERT-truncated) text directly to the encoder.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str | None = "facebook/bart-large-cnn",
    ) -> None:
        self.device = torch.device(device)
        self._noop = model_name is None
        if not self._noop:
            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._model.eval().to(self.device)

    def summarize(self, content: str) -> str:
        """Compress article content to a summary suitable for FinBERT encoding.

        If the content is already short enough for FinBERT (≤ 512 tokens),
        summarization is skipped and the content is returned as-is.

        When the summarizer was constructed with ``model_name=None`` the input
        is always returned unchanged (FinBERT will truncate at 512 tokens).
        """
        if not content or not content.strip():
            return ""

        if self._noop:
            return content

        # Short-content bypass: use BART tokenizer as a proxy for FinBERT length.
        # BART and FinBERT tokenizers produce similar token counts for financial
        # prose, so this avoids loading a second model just for the length check.
        bart_tokens = self._tok(content, truncation=False)
        if len(bart_tokens["input_ids"]) <= _FINBERT_MAX_LENGTH:
            return content

        inputs = self._tok(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=_BART_MAX_INPUT,
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self._model.generate(
                inputs["input_ids"],
                max_length=_BART_MAX_OUTPUT,
                min_length=_BART_MIN_OUTPUT,
                num_beams=_BART_NUM_BEAMS,
                length_penalty=_BART_LENGTH_PENALTY,
            )

        return self._tok.decode(summary_ids[0], skip_special_tokens=True)
