# News Pipeline

## Current pipeline

### Steps

```
[NewsSearch] → list[Story] → [ArticleRepository.filter_new] → [ArticleExtractor] → [ArticleRepository.store]
```

1. **Search** — `NewsSearch.search(query, start_date, end_date)` calls the MediaCloud API and paginates through all results. Returns a flat `list[Story]` (id, url, title, publish_date, source_name, language). No article text yet.

2. **Deduplication** — `ArticleRepository.filter_new(stories)` drops any story whose URL is already in the index. Prevents re-fetching articles across runs.

3. **Extraction** — `ArticleExtractor.extract_many(stories)` fetches full HTML for each URL and extracts article text using trafilatura. Stories are grouped by `source_name` and each source bucket is processed sequentially (with a configurable inter-request delay) to be polite to publishers. Different sources run in parallel up to a thread pool limit.

   Sources that accumulate too many fetch failures are automatically skipped for the rest of the run via `SourceBlacklist` (sliding window of last 10 attempts; blacklisted at 5 failures).

4. **Storage** — `ArticleRepository.store(article)` writes the text to `data/news/articles/<id>.txt` and appends a row to an in-memory CSV index. `flush()` persists the index.

### Components

| Class | File | Responsibility |
|---|---|---|
| `NewsSearch` | `sources/news/search.py` | MediaCloud query → paginated `list[Story]` |
| `ArticleExtractor` | `sources/news/extractor.py` | URL → `Article` (HTML fetch + trafilatura extraction), parallel by source |
| `SourceBlacklist` | `sources/news/blacklist.py` | Per-source failure tracking, persistent blacklist |
| `ArticleRepository` | `sources/news/repository.py` | Disk store: `index.csv` + per-article `.txt` files |
| `Story`, `Article` | `sources/news/models.py` | TypedDicts for the two pipeline stages |

### Storage layout

```
data/news/
    index.csv                  # id, url, title, publish_date, source_name, language
    articles/
        <id>.txt               # raw extracted article text
    blacklist.csv              # permanently blacklisted source names
    source_attempts.csv        # rolling window of fetch outcomes per source
```

---

## Differences from the paper

### How the paper collects news

The paper uses a single Chinese financial portal (East Money) that organises its content by stock ticker. Articles arrive **pre-associated with a company** — no search query is needed. All 3.9 M articles across 4 500+ stocks come from this one source, so there is no publisher diversity and no source-quality problem to solve.

### Our situation: key divergences

**1. Company-to-article matching must be explicit**

MediaCloud is a general search engine over thousands of publishers. Nothing links a story to a ticker automatically. We must construct a query per symbol (e.g. `"Apple" OR "AAPL"`) and tag articles at collection time. Currently `Story` and `Article` carry no `symbol` field — that association needs to be made at the notebook/orchestration level and either stored in the index or embedded in a separate data structure.

**2. General / macro articles**

The paper is silent on macro news (elections, conflicts, interest rate decisions, etc.) because East Money only surfaces stock-specific content. With MediaCloud, a broad query will inevitably return both company-specific and macro articles. Two reasonable positions:

- **Ignore macro articles** (stick close to the paper): keep queries tight enough (company name + ticker, financial context) that general articles don't appear.
- **Include macro articles as a separate signal**: treat them as a market-wide sentiment factor. This goes beyond what the paper validates and would require additional modelling work.

**3. Source quality and diversity**

Because the paper uses one curated source, it applies no source weighting or filtering. With MediaCloud we pull from blogs, wire services, local newspapers, and specialist financial outlets simultaneously. `SourceBlacklist` already handles sources that are unreachable; beyond that, downstream sentiment scoring may need to decide whether to weight `source_name` (e.g. Reuters vs. a random blog) or simply treat all articles equally as the paper does.

### Recommended approach (faithful to the paper)

- One query per ticker at collection time; store the `symbol` alongside each article in the index.
- Keep queries narrow (company name, ticker symbol) so that results are company-specific rather than macro.
- Do not apply source weighting for now; treat all successfully extracted articles equally.
- Revisit macro news and source quality only if downstream sentiment results are unsatisfying.
