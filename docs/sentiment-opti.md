# Sentiment Pipeline — Optimizations

Summary of all performance optimizations applied to the summarization + FinBERT
encoding pipeline (`src/embeddings/summarizer.py`, `src/embeddings/pipeline.py`,
`src/embeddings/filter.py`).

### `ArticleFilter` — reducing the article set before any model inference

`ArticleFilter.filter()` is applied before the pipeline and reduces the number
of articles that reach BART and FinBERT through three successive steps:

**HTML stripping.** Raw article bodies often contain HTML markup. Tags are
stripped and entities decoded before any tokenisation occurs. This matters for
correctness as well as speed: markup tokens inflate the FinBERT token count,
causing articles to be incorrectly routed through BART when their actual text
content would fit within the 512-token limit.

**Minimum body length filter (`min_body_chars`).** Articles whose body is
shorter than the configured character threshold are dropped entirely. Very short
bodies add negligible sentiment signal but still incur tokenisation and
(potentially) BART overhead.

**Per-day sampling (`sample_above`, `sample_ratio`, `max_per_day`).** On days
with many articles, a random subset is drawn rather than encoding every article.
This applies two levers: a ratio-based sample (e.g. keep 50% of articles on
days above a threshold) and a hard per-day ceiling. Because BART and FinBERT
costs scale linearly with article count, reducing the article set directly
reduces total compute.

### Short-content bypass (`Summarizer.summarize`)
BART is skipped entirely for articles whose body already fits within FinBERT's
512-token limit. The check uses the FinBERT tokenizer (WordPiece), not BART's
BPE tokenizer — the two diverge significantly on financial jargon, so using
BART's count would be an unreliable proxy.

### FinBERT batched encoding (`SentimentEncoder.encode_batch`)
FinBERT inference runs in padded batches across all articles rather than one at
a time, amortising the forward-pass overhead on MPS/CUDA.

### Greedy decoding — `num_beams=1` (was 4)
Beam search with B beams performs B parallel decoder forward passes at every
generation step. With `num_beams=1` (greedy decoding), each long article
triggers at most `max_output` decoder passes instead of `num_beams × max_output`.
This is the largest single speedup: **~4× faster** on the generation step with
negligible quality loss for a 3-class downstream classifier.

### Length penalty neutralised — `length_penalty=1.0` (was 2.0)
The original value of 2.0 quadratically penalised shorter beam candidates,
biasing the decoder to generate close to `max_output` tokens even when the
content did not warrant it. At 1.0 (neutral), the decoder stops as soon as the
best sequence ends naturally. Has no effect when `num_beams=1`, but is the
correct default for any future experiment with `num_beams > 1`.

### Batched BART inference (`Summarizer.summarize_batch`)
Previously `encode_articles` called `summarize()` in a Python for-loop, one
article at a time. The new `summarize_batch` method separates articles into two
groups in a first pass (tokenisation only, no model inference):

- **Short articles** (≤ 512 FinBERT tokens) — returned as-is, BART not called.
- **Long articles** (> 512 tokens) — collected and processed in padded batches.

The BART encoder runs once per batch across all inputs simultaneously; the
decoder generates all sequences in parallel at each step. `attention_mask` is
passed explicitly to handle padding correctly. Speedup scales with batch size
and device parallelism (significant on MPS/CUDA, moderate on CPU).

### fp16 weights on CUDA/MPS
Model weights are loaded directly in `torch.float16` when the device is CUDA or
MPS, via `torch_dtype` in `from_pretrained`. This halves memory bandwidth at
inference time. Loading in fp16 directly avoids the peak memory spike that
`model.half()` would cause (which allocates fp32 weights first, then converts).
Falls back to fp32 on CPU where native fp16 ALUs are absent and fp16 would be
slower.

### Configurable summarization model
`SUMMARIZER_MODEL` is a top-level variable in the notebook config section.
Swapping the backbone requires no code changes. Available options:

| Model | Notes |
|---|---|
| `None` | No summarization — fastest, FinBERT truncates at 512 tokens |
| `sshleifer/distilbart-cnn-6-6` | Distilled from BART-large-cnn, news-aligned, ~2× faster |
| `sshleifer/distilbart-cnn-12-6` | Distilled, higher quality, ~1.5× faster |
| `facebook/bart-large-cnn` | Full model, best quality, slowest |
