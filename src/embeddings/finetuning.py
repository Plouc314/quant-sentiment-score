"""Fine-tune FinBERT on the Financial PhraseBank dataset.

Usage::

    from src.embeddings.finetuning import fine_tune_finbert
    output_dir = fine_tune_finbert("models/finbert-fpb")

The saved model is a drop-in replacement for the base FinBERT weights in
:class:`~src.embeddings.encoder.SentimentEncoder`::

    enc = SentimentEncoder(model_name_or_path="models/finbert-fpb")
"""

from __future__ import annotations

import logging
from pathlib import Path

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from .encoder import FINBERT_MAX_LENGTH

logger = logging.getLogger(__name__)

# FinBERT id2label: {0: "positive", 1: "negative", 2: "neutral"}
# Financial PhraseBank ClassLabel: {0: "negative", 1: "neutral", 2: "positive"}
# Mapping: FPB index → FinBERT index
_FPB_TO_FINBERT: list[int] = [1, 2, 0]


def fine_tune_finbert(
    output_dir: str | Path,
    *,
    base_model: str = "ProsusAI/finbert",
    fpb_config: str = "sentences_allagree",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """Fine-tune FinBERT on Financial PhraseBank and save to *output_dir*.

    The fine-tuned model preserves FinBERT's label convention
    ``{0: positive, 1: negative, 2: neutral}`` so it is a drop-in replacement
    in :class:`~src.embeddings.encoder.SentimentEncoder` without any changes
    to downstream label logic.

    Parameters
    ----------
    output_dir:
        Directory where the fine-tuned model and tokenizer are saved.
        Created if it does not exist.
    base_model:
        HuggingFace model ID or local path to start from.
    fpb_config:
        Financial PhraseBank config name.  ``sentences_allagree`` (default)
        restricts to sentences where all annotators agreed — the cleanest
        label signal (~2 264 sentences).  Alternatives: ``sentences_75agree``,
        ``sentences_66agree``.
    num_train_epochs:
        Number of full passes over the training set.
    learning_rate:
        Peak learning rate for AdamW.
    per_device_train_batch_size:
        Batch size per device during training.
    per_device_eval_batch_size:
        Batch size per device during evaluation.
    weight_decay:
        L2 regularisation coefficient (applied to all params except biases and
        LayerNorm weights).
    warmup_ratio:
        Fraction of total training steps used for linear LR warmup.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Resolved path to the saved model directory.
    """
    output_dir = Path(output_dir)

    logger.info("loading tokenizer and model from %s", base_model)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)

    logger.info("loading Financial PhraseBank (%s)", fpb_config)
    dataset = _load_and_prepare(fpb_config, tok, seed)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=_compute_metrics,
    )

    logger.info(
        "starting fine-tuning: epochs=%d, lr=%s, batch=%d",
        num_train_epochs,
        learning_rate,
        per_device_train_batch_size,
    )
    trainer.train()
    logger.info("fine-tuning complete — saving to %s", output_dir)

    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))

    return output_dir.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_and_prepare(fpb_config: str, tok: AutoTokenizer, seed: int) -> DatasetDict:
    """Load FPB, remap labels to FinBERT convention, tokenize, split 90/10."""
    raw = load_dataset("takala/financial_phrasebank", fpb_config)
    split = raw["train"].train_test_split(
        test_size=0.1, seed=seed, stratify_by_column="label"
    )
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    dataset = dataset.map(_remap_labels)
    dataset = dataset.map(
        lambda batch: tok(
            batch["sentence"],
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            padding="max_length",
        ),
        batched=True,
    )
    dataset = dataset.remove_columns(["sentence"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset


def _remap_labels(example: dict) -> dict:
    """Translate FPB label integer to FinBERT label integer.

    FPB: {0: negative, 1: neutral, 2: positive}
    FinBERT: {0: positive, 1: negative, 2: neutral}
    """
    return {"label": _FPB_TO_FINBERT[example["label"]]}


def _compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        **accuracy_metric.compute(predictions=preds, references=pred.label_ids),
        **f1_metric.compute(predictions=preds, references=pred.label_ids, average="macro"),
    }
