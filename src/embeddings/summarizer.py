from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

from .encoder import FINBERT_MAX_LENGTH

logger = logging.getLogger(__name__)

_FINBERT_MODEL = "ProsusAI/finbert"

# BART-CNN generation constants (input cap and minimum output are not tuneable
# per-instance — they are architectural limits of the BART-CNN checkpoint).
_BART_MAX_INPUT = 1024
_BART_MIN_OUTPUT = 30


class Summarizer:
    """Seq2seq summarizer — compresses long article text for downstream encoding.

    Defaults to ``facebook/bart-large-cnn``; pass any HuggingFace seq2seq model
    name to swap the backbone without changing any other code.

    Pass ``model_name=None`` for a no-op summarizer that returns text unchanged.
    In this mode no model is loaded and :meth:`summarize` simply returns its
    input — useful for evaluating the baseline of feeding raw (FinBERT-truncated)
    text directly to the encoder.

    Parameters
    ----------
    num_beams:
        Beam-search width.  ``1`` selects greedy decoding — ~4× faster than the
        original default of 4 with negligible quality loss for a 3-class
        sentiment classifier downstream.
    length_penalty:
        Exponent applied to sequence length when scoring beams
        (``score = log_prob / length ** length_penalty``).  Has no effect when
        ``num_beams=1``.  The original default of 2.0 aggressively biased the
        decoder toward longer outputs; 1.0 is length-neutral.
    max_output:
        Maximum number of tokens to generate.  64 is sufficient when the output
        is prepended with the article title before FinBERT encodes it.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str | None = "facebook/bart-large-cnn",
        finbert_tokenizer: PreTrainedTokenizerBase | None = None,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        max_output: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self._noop = model_name is None
        self._num_beams = num_beams
        self._length_penalty = length_penalty
        self._max_output = max_output
        if not self._noop:
            self._tok = AutoTokenizer.from_pretrained(model_name)
            # fp16 halves memory bandwidth on CUDA/MPS; avoid on CPU where
            # native fp16 ALUs are absent and it would be slower.
            dtype = (
                torch.float16
                if self.device.type in ("cuda", "mps")
                else torch.float32
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=dtype
            )
            self._model.eval().to(self.device)
            self._finbert_tok: PreTrainedTokenizerBase = (
                finbert_tokenizer
                if finbert_tokenizer is not None
                else AutoTokenizer.from_pretrained(_FINBERT_MODEL)
            )

    @property
    def noop(self) -> bool:
        """If the summarizer is deactivated"""
        return self._noop

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

        # Short-content bypass: use FinBERT's own tokenizer for the length check.
        # BART (BPE) and FinBERT (WordPiece) tokenize financial jargon very
        # differently, so the BART count is an unreliable proxy.
        finbert_tokens = self._finbert_tok(content, truncation=False)
        if len(finbert_tokens["input_ids"]) <= FINBERT_MAX_LENGTH:
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
                max_length=self._max_output,
                min_length=_BART_MIN_OUTPUT,
                num_beams=self._num_beams,
                length_penalty=self._length_penalty,
            )

        return self._tok.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_batch(self, contents: list[str], batch_size: int = 16) -> list[str]:
        """Summarize a list of article bodies, returning one string per input.

        Articles whose body is empty or already fits within FinBERT's 512-token
        limit bypass BART entirely and are returned as-is.  Only genuinely long
        articles are sent through the seq2seq model, in padded batches of
        ``batch_size`` to amortise encoder and decoder overhead on GPU/MPS.

        When the summarizer was constructed with ``model_name=None`` all inputs
        are returned unchanged.
        """
        if self._noop:
            return list(contents)

        results: list[str] = [""] * len(contents)
        long_indices: list[int] = []
        long_contents: list[str] = []

        # Pass 1 — short-content bypass (tokenisation only, no model inference).
        for i, content in enumerate(contents):
            if not content or not content.strip():
                results[i] = ""
                continue
            n_tokens = len(self._finbert_tok(content, truncation=False)["input_ids"])
            if n_tokens <= FINBERT_MAX_LENGTH:
                results[i] = content
            else:
                long_indices.append(i)
                long_contents.append(content)

        if not long_contents:
            return results

        n_long = len(long_contents)
        logger.info(
            "Summarising %d / %d articles with BART (batch_size=%d)",
            n_long,
            len(contents),
            batch_size,
        )

        # Pass 2 — batched BART inference on long articles only.
        for batch_start in range(0, n_long, batch_size):
            batch = long_contents[batch_start : batch_start + batch_size]

            inputs = self._tok(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=_BART_MAX_INPUT,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                summary_ids = self._model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self._max_output,
                    min_length=_BART_MIN_OUTPUT,
                    num_beams=self._num_beams,
                    length_penalty=self._length_penalty,
                )

            for j, sid in enumerate(summary_ids):
                results[long_indices[batch_start + j]] = self._tok.decode(
                    sid, skip_special_tokens=True
                )

            done = min(batch_start + batch_size, n_long)
            logger.info("Summarised %d / %d articles", done, n_long)

        return results
