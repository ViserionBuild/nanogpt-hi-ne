# Hindi IndicCorp Data Pipeline & Tokenizer

Phase 1 data pipeline for **Model H (Hindi)**: a monolingual decoder-only
Transformer language model trained from scratch.

## Overview

This pipeline downloads, preprocesses, and tokenizes a representative
subset of the [AI4Bharat IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
Hindi corpus to produce an approximately **400M content-token** final corpus,
combined with **≥20% manually collected data** (OCR/scraping), tokenized
with a SentencePiece BPE tokenizer trained from scratch.

> **Important terminology**: The initial download target is **500 million
> whitespace-separated words**, NOT 500M BPE tokens.  The actual BPE
> token count is measured after tokenizer training (script 10) and
> verified again in script 14, always in **content tokens** — BOS/EOS
> IDs added during tokenization are counted separately from the 400M
> content-token target.

---

## Dataset Sources

| Field | Value |
|---|---|
| **Public** | AI4Bharat IndicCorpV2 — Hindi (hi-1.txt through hi-5.txt) |
| **Manual** | Hindi Wikipedia (dump) + Hindi news (Dainik Jagran, Hindustan, ABP News enabled by default; 7 more sources configured but disabled — see `hindi_data.yaml`'s `manual.news` block and `hindi/scripts/manual/03_collect_news.py`) |
| License | CC BY-NC 4.0 (IndicCorp) |
| HuggingFace | `ai4bharat/IndicCorpV2` |

---

## Deduplication (CL3410 Lecture 3)

Script 06 runs two passes over the normalized + Hindi-filtered +
quality-filtered corpus (the appropriate cleaned representation — not raw
text; Devanagari/ZWNJ/ZWJ/punctuation/whitespace pass through unchanged):

1. **Exact duplicate detection** — unchanged from before: SHA-256 line
   hash, SQLite- or memory-backed, O(1) memory.
2. **Near-duplicate / sub-document duplicate detection** (new) — on the
   exact-deduplicated output:
   - represent each line as its set of overlapping **3-word shingles**
   - summarize each shingle set with a **MinHash** signature
   - use **LSH banding for candidate generation** — an incoming line is
     only ever compared against previously-kept lines sharing a band,
     never the whole corpus (no O(N²) document-to-document comparison)
   - **confirm** each candidate with an actual similarity estimate before
     removing anything: Jaccard ≥ `similarity_threshold` → **near
     duplicate**; overlap/containment ≥ `sub_document_overlap_threshold`
     (for a pair whose lengths differ substantially) → **sub-document
     duplicate** (e.g. a repeated boilerplate sentence embedded in
     otherwise-different longer text)

This two-step "generate candidates cheaply, then confirm by similarity"
design is what keeps a line that merely shares a common short phrase with
another from being removed — a band collision alone never causes a
removal. Reusable building blocks (`MinHasher`, LSH banding, Jaccard/
overlap estimators, `MinHashDedupIndex`) live in `shared/near_dup.py` and
are shared with the decontamination stage (09) — see there for its
character-shingle configuration.

Configuration: `deduplication:` block in `hindi_data.yaml` (shingle size/
type, MinHash permutations, LSH bands/rows, similarity + overlap
thresholds, minimum document word count). Metadata:
`hindi/metadata/deduplication_report.json` (input lines, exact/near/
sub-document duplicates removed, final lines, estimated words, removal
percentage).

---

## Decontamination (Leakage-Safe Pipeline Ordering)

Per CL3410 Lecture 3 ("Decontamination"): hold out the test set first,
then search the candidate training corpus for **exact and near-duplicate**
matches of the held-out validation/test data and remove them from
training — before any tokenizer, vocabulary, or corpus-selection statistic
is computed. `validation.txt` and `test.txt` are produced once (script 08)
and are never touched again by any later stage.

```
08_split_dataset.py   → splits the DEDUPLICATED public+manual corpora into
                         train-candidate / validation / test. validation.txt
                         and test.txt are written straight to `data/final/`
                         and FROZEN from this point on.
        ↓
09_decontaminate.py   → removes EXACT (SHA-256) and NEAR-DUPLICATE
                         (MinHash + LSH banding, see shared/near_dup.py)
                         matches of validation.txt/test.txt from the
                         train-candidate corpus. Near-duplicate matching is
                         required — exact-match alone is not sufficient,
                         since paraphrased test items can also leak.
```

Both scripts are streaming/SQLite-backed and memory-bounded: only the
(small) held-out validation+test signatures are held in an index; the
training corpus is scanned once, line by line — never loaded into RAM,
even at hundreds of millions of lines.

Configuration lives under `decontamination:` in `hindi_data.yaml`
(`exact_match`, `near_duplicate`, `validation`, `test` toggles, plus
`near_duplicate_config` for shingle size / MinHash permutations / LSH
bands & rows).

Script 15 independently re-verifies this at the end of the pipeline (it
does **not** trust script 09's own report) and **fails loudly** if:
  * any exact or near-duplicate of validation/test is found in
    `final/train.txt`,
  * the final tokenizer was trained on anything other than `final/train.txt`.

---

## Token Counting & Reporting (CL3410 Lecture 3)

The professor's slide: *"Do deduplicate what you have collected, then
count your tokens again. Report both numbers."* Whitespace word count is
**not** the final BPE token count — the pipeline is careful to keep these
distinct at every stage:

| Stage | Script | Reports |
|---|---|---|
| RAW / NORMALIZED / QUALITY FILTERED / DEDUPLICATED | `07_statistics.py` | lines, estimated words (+ characters for raw) — `corpus_statistics.json` |
| After the ONE final tokenizer is trained | `10_train_tokenizer.py` | actual SentencePiece BPE content tokens, per-source + total, vs. targets — `token_count_check.json` |
| Actual selected/tokenized corpus | `14_tokenize_corpus.py` | content/BOS/EOS/stored tokens for `final/train.txt` — `tokenization_metadata.json` |
| Independent re-verification | `15_verify_final_corpus.py` | re-tokenizes `final/train.txt` from scratch (does not trust 14's self-report) — `verification_report.json` |

All four log the same four lines wherever they compare a measured count
against the target:
```
FINAL CONTENT BPE TOKENS: <n>
TARGET CONTENT BPE TOKENS: 400,000,000
DIFFERENCE: <+/-n>
PERCENTAGE FROM TARGET: <+/-n%>
```

**The 400M target means 400,000,000 BPE CONTENT tokens of `final/train.txt`
specifically** — not whitespace words, not characters, not BOS/EOS-inclusive
stored tokens, and not train+validation+test summed (validation/test are
sized independently as 5%/5% of the pre-selection pool, not a slice of the
400M target). This is enforced two ways:
  * `selection.target_unit: bpe_content_tokens` in `hindi_data.yaml` is a
    machine-checkable declaration — scripts 11/14/15 assert it and refuse
    to run if it's missing or wrong.
  * `selection.indiccorp_target_tokens` (not `target_tokens`, a
    previously-latent config-key mismatch that silently fell back to a
    hardcoded default) is now what's actually read.

**No silent duplication**: if the deduplicated+decontaminated corpus has
*fewer* BPE content tokens than the target, script 10 fails loudly with a
structured shortfall report (available vs. required vs. shortfall
percentage) — it never repeats/duplicates lines to reach 400M. Selection
only ever subsamples (`fraction <= 1.0`, asserted). If repetition/epochs
are ever introduced for training, that must be reported separately from
the unique corpus token count (`data_reuse: {duplicated_to_reach_target,
epochs}` field reserved for this in the JSON reports).

Run `python hindi/scripts/15_verify_final_corpus.py --config hindi/configs/hindi_data.yaml`
to see the consolidated report — it pulls together 07's RAW→DEDUPLICATED
line/word counts, 10's post-dedup BPE availability check
(`token_count_check.json`), and its own independently re-measured final
BPE count into one JSON
(`hindi/metadata/verification_report.json`, `full_progression` +
`target_check` fields) with a `pipeline_flow` diagram:
```
RAW CORPUS -> NORMALIZED -> QUALITY FILTERED -> DEDUPLICATED
  -> ACTUAL BPE TOKEN COUNT -> TARGET SELECTION
```

---

## Single-Tokenizer Architecture

There is exactly ONE tokenizer in this pipeline — a SentencePiece BPE
model (`model_type="bpe"`; BPE is never hand-implemented). Corpus
selection (script 11) does not need token counts to run, so there is no
dependency cycle to break and no need for a temporary tokenizer at all:

```
FINAL tokenizer (script 10, the only tokenizer)
  └── Trained on `final/train.txt` (the combined, decontaminated
      IndicCorp + Wikipedia/manual corpus produced by script 11).
      NEVER sees validation/test text.
  └── After training, script 10 reloads this SAME model and runs a
      streaming token-count pass — separately — over
      `indiccorp_train.txt`, `wikipedia_train.txt`, and `train.txt`,
      reporting per-source/total BPE token counts, vocabulary size, a
      sum-consistency check, and any shortfall against the 400M
      IndicCorp / 100M Wikipedia targets (reported only — never
      triggers a second training run).
  └── Used by scripts 13, 14, and model pretraining/finetuning
  └── Saved to: hindi/tokenizer/model/hindi_bpe_32000.model
```

---

## Pipeline Architecture

```
01_download_indiccorp.py   — HTTP Range partial download (~500M words)
        │
        │  ALSO (parallel, independent collection): scripts/manual/
        │  01_download_wikipedia.py + 02_extract_wikipedia.py produce ONE
        │  wikipedia_hi_raw.txt directly (no merge step needed); scripts/
        │  manual/03_collect_news.py collects each enabled news outlet, then
        │  manual/05_aggregate_news_corpus.py merges those outlet shards into
        │  ONE news_raw.txt (04_convert_news_jsonl_to_raw.py is an optional,
        │  non-required debugging aid, not part of this flow). From this
        │  point on, indiccorp / wikipedia / news are THREE separate real
        │  files/sources flowing through 03-09 independently.
        ↓
03_normalize.py            — Unicode NFC + whitespace normalization
04_filter_hindi.py         — Devanagari script ratio filtering
05_quality_filter.py       — content quality checks
06_deduplicate.py          — exact (SHA-256) + near-duplicate + sub-document
                              dedup (3-word shingles, MinHash, LSH candidate
                              generation, Jaccard/overlap confirmation) —
                              indiccorp, wikipedia, and news deduplicated
                              TOGETHER in one cross-source pass, with
                              precedence wikipedia > news > indiccorp
        ↓
07_statistics.py           — per-stage corpus statistics
        ↓
08_split_dataset.py        — split every source's deduplicated corpus
                              (IndicCorp, Wikipedia, News) into
                              train-candidate / validation / test (90/5/5).
                              validation.txt + test.txt → data/final/, FROZEN.
        ↓
09_decontaminate.py        — remove exact + near-duplicate (MinHash/LSH)
                              matches of validation/test from every
                              source's train-candidate
        ↓
11_select_corpus.py        — combine decontaminated train-candidate
                              IndicCorp + Wikipedia + News sources
                              (no subsampling/trimming; no tokenizer used) →
                              writes {source}_train.txt for each of the
                              three sources, then concatenates
                              wikipedia_train.txt + news_train.txt into
                              manual_train.txt, then indiccorp_train.txt +
                              manual_train.txt into train.txt, all in
                              data/final/
        ↓
10_train_tokenizer.py      — train the ONE final tokenizer on
                              final/train.txt, then reload it and stream
                              per-source (IndicCorp/Wikipedia/News, plus
                              derived Manual = Wikipedia + News) BPE token
                              counts, vocab size, sum check, and shortfall
                              report against the 400M/100M targets
        ↓
13_evaluate_tokenizer.py   — fertility, UNK rate, comparison (read-only
                              eval against validation.txt — never trained on)
        ↓
14_tokenize_corpus.py      — streaming binary uint32 token IDs + verify
        ↓
15_verify_final_corpus.py  — comprehensive verification, INCLUDING an
                              independent leakage re-check (exact + near-dup,
                              train vs validation/test) that fails loudly
                              on any residual contamination
```

### Key design decisions

1. **Separate provenance**: Public and manual data are tracked separately
   throughout the pipeline. Script 11 selects from each source independently
   and enforces the ≥20% manual content-token requirement.

2. **Streaming / memory-bounded**: Every stage processes the corpus
   line-by-line. No `rng.permutation`, no per-line NumPy arrays. This
   includes decontamination — held-out signatures are the only thing
   held in memory/SQLite; the training corpus is scanned once.

3. **Content tokens only**: The 400M target is measured in actual BPE
   content tokens (BOS/EOS excluded).

4. **No sentence cutting**: Selection and splitting never cut a sentence.

5. **Test set held out before contamination is possible**: validation/test
   are split off the deduplicated corpus (a deterministic, config-threshold
   cleaning stage with no data-dependent tuning) BEFORE corpus selection or
   the final tokenizer — the stages that actually make data-driven
   decisions about what enters training — ever run. See "Decontamination"
   above.

---

## Pipeline Commands

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1: Collect manual data — Wikipedia + News

The Hindi **Manual** corpus has exactly two subcategories, collected by
`hindi/scripts/manual/collect_*.py` (same architecture as the Nepali
pipeline's `nepali/scripts/manual/` package):

```text
manual/
├── wikipedia   (hindi/scripts/manual/01_download_wikipedia.py + 02_extract_wikipedia.py)
└── news        (hindi/scripts/manual/03_collect_news.py)
```

IndicCorp remains a separate, existing top-level source — it is never
classified as Manual.

**Step 1a — Wikipedia** (two plain, standalone scripts — run one, then the
other; no orchestrating wrapper):

```bash
# 1. Download the dump (resumable, idempotent — skips if already complete)
python hindi/scripts/manual/01_download_wikipedia.py \
    --config hindi/configs/hindi_data.yaml

# 2. Extract plain text from the downloaded dump
python hindi/scripts/manual/02_extract_wikipedia.py \
    --config hindi/configs/hindi_data.yaml
```

`01_download_wikipedia.py` downloads the latest **completed**
`hiwiki-YYYYMMDD-pages-articles.xml.bz2` dump (current-revision article
text only — never `pages-meta-history`), resolved dynamically via
`manual.wikipedia.dump_url: "auto"`, resumable via HTTP Range, validated
by size/bz2-magic/SHA-1. Re-running with no flags never redoes a
completed download; `--overwrite` forces a clean re-download.

`02_extract_wikipedia.py` then streams that dump into plain
paragraph-per-line text at
`hindi/data/raw/manual/wikipedia/wikipedia_hi_raw.txt`, with an
extraction report at
`hindi/data/metadata/wikipedia_extraction_report.json`. It exits loudly
if the output already exists — pass `--overwrite` to re-extract.

**Step 1b — News** (sitemap discovery + generic article extraction,
individually enable/disable-able per source in `hindi_data.yaml`'s
`manual.news` block):

```bash
python hindi/scripts/manual/03_collect_news.py \
    --config hindi/configs/hindi_data.yaml

# One source only:
python hindi/scripts/manual/03_collect_news.py \
    --config hindi/configs/hindi_data.yaml --source dainik_jagran

# Fast, network-free check:
python hindi/scripts/manual/03_collect_news.py \
    --config hindi/configs/hindi_data.yaml --dry-run

# Connectivity check only, no articles collected:
python hindi/scripts/manual/03_collect_news.py \
    --config hindi/configs/hindi_data.yaml --smoke-test
```

Of the 10 named Hindi news sources evaluated for this project, **3 are
enabled by default** (verified permissive robots.txt + working sitemap
discovery + real Devanagari text recoverable via a plain HTTP GET):

| Source | Status |
|---|---|
| `dainik_jagran` | enabled |
| `hindustan` | enabled |
| `abp` | enabled |
| `amar_ujala` | disabled — robots.txt disallows ClaudeBot/Claude-Web/anthropic-ai |
| `dainik_bhaskar` | disabled — robots.txt disallows anthropic-ai/CCBot/GPTBot |
| `jansatta` | disabled — robots.txt disallows ClaudeBot/Claude-Web/anthropic-ai |
| `navbharat_times` | disabled — article body is JS-rendered, not in static HTML |
| `aaj_tak` | disabled — no working article sitemap found |
| `ndtv` | disabled — robots.txt fetch blocked at the edge (Akamai) |
| `prabhat_khabar` | disabled — fully client-side-rendered (Next.js), zero static text |

See `hindi/scripts/manual/03_collect_news.py`'s module docstring for the
full verification evidence, and each source's `disabled_reason` in
`hindi_data.yaml`. Re-verify and flip `enabled: true` if a disabled
source's situation changes — do not enable a source whose robots.txt
disallows AI crawlers by name.

Each source's sitemap is a **rolling window** (recent articles, not a
full archive) — reaching the ~120-150M raw/quality-token collection
target (for a 100M+ final-token contribution after cleaning/dedup)
requires running this collector **repeatedly over time** (e.g. daily);
resumability means every run only adds new articles.

Output is sharded JSONL under `hindi/data/raw/manual/news/`, one shard
sequence per source (`dainik_jagran_00000.jsonl`, `hindustan_00000.jsonl`,
`abp_00000.jsonl`, ...), preserving per-article title/URL/source
name/publication timestamp (when available)/collection timestamp.

**Step 1c — Convert news JSONL to raw text (OPTIONAL, NOT required):**

`manual/05_aggregate_news_corpus.py` (next step) reads `03_collect_news.py`'s
sharded JSONL **directly** — this conversion step is skipped by the actual
pipeline. `04_convert_news_jsonl_to_raw.py` is kept only as an optional
standalone debugging aid (a human-readable flattened text file per news
source); nothing downstream consumes its output, so you do not need to run
it before Step 1d.

```bash
# Optional — for manual inspection only, not part of the required flow:
python hindi/scripts/manual/04_convert_news_jsonl_to_raw.py \
    --config hindi/configs/hindi_data.yaml --source all
```

**Step 1d — Aggregate the enabled news outlets into one provenance-tracked
news corpus:**

```bash
python hindi/scripts/manual/05_aggregate_news_corpus.py \
    --config hindi/configs/hindi_data.yaml
```

This script aggregates **news only** — Wikipedia is out of scope for it
entirely, since `01_download_wikipedia.py` + `02_extract_wikipedia.py`
already produce a single `wikipedia_hi_raw.txt` with no merge step
needed. Combines every enabled news outlet's shards into
`hindi/data/intermediate/news/news_raw.txt` (one line per document —
feeds directly into `03_normalize.py --input ... --output ...`) plus
`news_provenance.jsonl` (traces every line back to its
`source_name`/`source_url`) and
`hindi/metadata/news_aggregation_report.json` (documents/words per news
source, and TOTAL NEWS — nothing is ever silently dropped; every
excluded document is counted under a named reason). Source provenance is
preserved throughout:

```text
news/dainik_jagran
news/hindustan
news/abp
...
wikipedia
indiccorp
```

Indiccorp, Wikipedia, and News are three separate real files/sources from
this point on — each flows through the same common pipeline (steps 4-7
below) independently, using `--source indiccorp` / `--source wikipedia` /
`--source news` respectively, so `06_deduplicate.py`'s cross-source pass
can find each source's quality-filtered output side by side. Wikipedia's
raw file (`hindi/data/raw/manual/wikipedia/wikipedia_hi_raw.txt`) and
News's aggregated file (`hindi/data/intermediate/news/news_raw.txt`) are
both fed directly into `03_normalize.py`, e.g.:

```bash
python hindi/scripts/03_normalize.py --config hindi/configs/hindi_data.yaml \
    --source news
```

Wikipedia and News only merge into a single combined `manual` stream much
later, at `11_select_corpus.py` (see Step 11 below), which concatenates
their independently-decontaminated `*_train.txt` outputs into
`manual_train.txt` — never before then.

### Step 2: Download IndicCorp candidate

```bash
python hindi/scripts/01_download_indiccorp.py \
    --config hindi/configs/hindi_data.yaml
```

### Step 3: Normalize

```bash
python hindi/scripts/03_normalize.py \
    --config hindi/configs/hindi_data.yaml
```

### Step 4: Hindi filtering

```bash
python hindi/scripts/04_filter_hindi.py \
    --config hindi/configs/hindi_data.yaml
```

### Step 5: Quality filtering

```bash
python hindi/scripts/05_quality_filter.py \
    --config hindi/configs/hindi_data.yaml
```

### Step 6: Deduplicate

```bash
python hindi/scripts/06_deduplicate.py \
    --config hindi/configs/hindi_data.yaml
```

Cross-source exact + MinHash/LSH near-duplicate deduplication across every
source that has quality-filtered output present (IndicCorp, Wikipedia,
and any collected/enabled news source) — reads script 05's per-source
outputs from `hindi/data/intermediate/quality/` directly, no `--input`
override needed. Source precedence on a cross-source duplicate (which
copy is kept): Wikipedia > news sources > IndicCorp.

### Step 7: Statistics

```bash
python hindi/scripts/07_statistics.py \
    --config hindi/configs/hindi_data.yaml
```

### Step 8: Manual data collection recap

Steps 1a-1d above (`hindi/scripts/manual/01_download_wikipedia.py` +
`02_extract_wikipedia.py`, `03_collect_news.py`, `manual/05_aggregate_news_corpus.py`
— Step 1c, `04_convert_news_jsonl_to_raw.py`, is optional/not required, see
Step 1c above) collect Wikipedia (already one file) and aggregate the enabled
news outlets into one file; run each of Wikipedia and News through
`03_normalize.py` → `04_filter_hindi.py` →
`05_quality_filter.py` → `06_deduplicate.py` (Step 3-6 above) with
`--source wikipedia` and `--source news` respectively the same as
IndicCorp, before continuing to Step 9.

`hindi/scripts/collect_manual.py` (a generic PDF-OCR / ad-hoc-URL-scrape
stub, distinct from the Wikipedia+news collectors — never filled in with
real PDFs/URLs) still exists for that different, orthogonal use case, but
`hindi/scripts/clean_normalize.py` and `hindi/scripts/make_splits.py`
(the older combine/stats/split scripts it fed into) have been removed —
they read from a flat `data/raw/`/`data/manual/` layout that the current
per-source pipeline no longer produces, and their outputs were not
consumed by anything after `08_split_dataset.py`/`11_select_corpus.py`
were generalized to the per-source model above. If `collect_manual.py`
is ever used for a real ad-hoc source, its output now needs to be fed
into `03_normalize.py` onward directly (same as any other source),
rather than through the removed combine step.
above.

### Step 9: Split into train-candidate / validation / test (HOLD OUT TEST FIRST)

```bash
python hindi/scripts/08_split_dataset.py \
    --config hindi/configs/hindi_data.yaml
```

Splits every source that has deduplicated output present — IndicCorp,
Wikipedia, and any collected news source (`dainik_jagran`, `hindustan`,
`abp`, ...). A source that is ENABLED in `hindi_data.yaml` but whose
deduplicated file is missing is a hard failure (never a silent skip —
the manual corpus can't silently shrink); a source that is simply
disabled contributes an empty split, logged as such. Writes
`hindi/data/final/validation.txt` and `hindi/data/final/test.txt`
(FROZEN from here on) plus
`hindi/data/intermediate/split/{source}_train_candidate.txt` for every
source in `PUBLIC_SOURCE`/`MANUAL_SOURCES`.

### Step 10: Decontaminate the training corpus

```bash
python hindi/scripts/09_decontaminate.py \
    --config hindi/configs/hindi_data.yaml
```

Removes exact and near-duplicate (MinHash/LSH) matches of validation/test
from every source's train-candidate corpus. Writes
`hindi/data/intermediate/decontaminated/{source}_train_decontam.txt` (one
per `PUBLIC_SOURCE`/`MANUAL_SOURCES` entry) and
`hindi/metadata/decontamination_report.json`.

### Step 11: Combine decontaminated sources into final/train.txt

```bash
python hindi/scripts/11_select_corpus.py \
    --config hindi/configs/hindi_data.yaml
```

Streams and concatenates every decontaminated source — no
subsampling/trimming, no tokenizer involved. Writes
`hindi/data/final/indiccorp_train.txt`, `hindi/data/final/wikipedia_train.txt`,
and `hindi/data/final/news_train.txt` (News is already the single
pre-merged aggregate of every enabled outlet, produced by
`manual/05_aggregate_news_corpus.py` upstream — there is no per-outlet
`final/{outlet}_train.txt`), then concatenates
`wikipedia_train.txt` + `news_train.txt` into
`hindi/data/final/manual_train.txt`, and finally `indiccorp_train.txt` +
`manual_train.txt` into `hindi/data/final/train.txt`. If Wikipedia or
News has no decontaminated file (e.g. disabled), it is skipped with a
warning, not a crash.

### Step 12: Train the ONE final tokenizer

```bash
python hindi/scripts/10_train_tokenizer.py \
    --config hindi/configs/hindi_data.yaml \
    --vocab-sizes 16000 32000 48000
```

After training, reloads the same tokenizer and streams it separately over
each of `final/indiccorp_train.txt`, `final/wikipedia_train.txt`, and
`final/news_train.txt` to report actual BPE content-token counts
**natively per source** (News is measured directly from its own real
`news_train.txt` — already the pre-merged aggregate of every enabled
outlet — not summed from per-outlet files), plus:

```text
Wikipedia tokens:       XX M
News tokens:            XX M
Manual total:           XX M   (= Wikipedia + News)
IndicCorp tokens:       XX M
Final corpus:           XX M
```

reported before/after dedup where applicable (see `07_statistics.py` for
the raw→deduplicated progression), plus a sum-consistency check
(`indiccorp + manual == total`, where `manual = wikipedia + news`) and
named shortfall warnings against the `indiccorp_target_tokens` (400M) /
`manual_target_tokens` (100M, covering Wikipedia+News combined) /
`total_target_tokens` (500M) config targets. No second tokenizer is ever
trained to compensate for a shortfall; quality thresholds are never
lowered to reach a token target.

### Step 13: Evaluate tokenizer

```bash
python hindi/scripts/13_evaluate_tokenizer.py \
    --config hindi/configs/hindi_data.yaml
```

### Step 14: Final tokenization (binary)

```bash
python hindi/scripts/14_tokenize_corpus.py \
    --config hindi/configs/hindi_data.yaml \
    --tokenizer-model hindi/tokenizer/model/hindi_bpe_32000.model
```

### Step 15: Final verification (includes leakage checks)

```bash
python hindi/scripts/15_verify_final_corpus.py \
    --config hindi/configs/hindi_data.yaml
```

Fails loudly (non-zero exit) if validation/test text is found — exact or
near-duplicate — anywhere in `final/train.txt`, or if any tokenizer was
trained on a held-out file.

This step also assembles `hindi/metadata/provenance.json` — see
[Provenance Manifest](#provenance-manifest-cl3410-lecture-3) below.

---

## Provenance Manifest (CL3410 Lecture 3)

CL3410 asks for provenance: source, licence, collection date, and which
filters were applied to the final corpus. Rather than leaving this
scattered across each stage's own report JSON, `15_verify_final_corpus.py`
assembles one consolidated, read-only summary — `hindi/metadata/provenance.json`
— written every time verification runs (checked as a pass/fail requirement:
"provenance.json written"). It does not replace or restructure any
existing per-stage report; it only reads them and copies forward what it
finds:

```json
{
  "source": {"source_urls": [...], "source_name": "...", "source_paper": "..."},
  "licence": "CC BY-NC 4.0",
  "date": {
    "verification_run_at_utc": "<timestamp of this verification run>",
    "download_completed_at_utc": "<from indiccorp_selection_state.json, or null>"
  },
  "filters_applied": {
    "normalization": {"ran": true},
    "hindi_script_filtering": {"ran": true},
    "quality_filtering": {"ran": true},
    "deduplication": {
      "report_exists": true,
      "exact_duplicates_removed": 12345,
      "near_duplicates_removed": 678,
      "sub_document_duplicates_removed": 90
    },
    "decontamination": {"report_exists": true},
    "safety_blocklist_filtering": {"configured": true, "blocklist_file_exists": true}
  }
}
```

`source`/`licence` are read straight from `hindi_data.yaml`'s `indiccorp:`
block (previously declared there but never propagated anywhere — now
they are). `download_completed_at_utc` is recovered from
`indiccorp_selection_state.json`'s `status`/`timestamp_utc` fields
(`01_download_indiccorp.py` records a UTC timestamp on every state save;
only a `status: completed` save counts as the download's completion
time). Each `filters_applied` entry reflects whether that stage's real
output/report actually exists on disk at verification time — it is not a
hardcoded "yes" list, so a pipeline run that skipped a stage shows that
honestly as `false`.

---

## Small Test Mode

For testing the pipeline end-to-end with small data:

```bash
# Download only 1M words
python hindi/scripts/01_download_indiccorp.py \
    --config hindi/configs/hindi_data.yaml \
    --target-words 1000000 --chunk-size-bytes 10485760

# ... run scripts 03-07, manual/01_download_wikipedia.py + 02_extract_wikipedia.py, manual/03_collect_news.py,
# manual/05_aggregate_news_corpus.py (news only — reads 03_collect_news.py's JSONL
# shards directly; 04_convert_news_jsonl_to_raw.py is optional, not required;
# Wikipedia needs no aggregation step, it's already one file), then run
# indiccorp/wikipedia/news each through 03-06 with --source <name>,
# 08 (split), 09 (decontaminate), then:

# Combine decontaminated sources into final/train.txt (no target/trim flags —
# selection carries all decontaminated data forward; shortfalls against the
# targets are reported by script 10 after the tokenizer is trained)
python hindi/scripts/11_select_corpus.py \
    --config hindi/configs/hindi_data.yaml
```

---

## Directory Structure

```
hindi/
├── configs/
│   ├── hindi_data.yaml          # Central pipeline configuration
│   ├── pretrain_config.yaml     # Model pretraining configuration
│   └── finetune_config.yaml     # Finetuning configuration
├── scripts/
│   └── manual/                  # Manual-corpus collectors (Wikipedia + news)
│       ├── manual_common.py     # Shared schema/CLI/HTTP helpers (library)
│       ├── 01_download_wikipedia.py    # Wikipedia dump download
│       ├── 02_extract_wikipedia.py     # Wikipedia dump text extraction
│       ├── 03_collect_news.py      # Sitemap-based news collector, 10 configurable sources
│       ├── 05_aggregate_news_corpus.py # Merges enabled news outlets into one file
│       └── 04_convert_news_jsonl_to_raw.py  # DEPRECATED/optional — not required (05_aggregate_news_corpus.py reads JSONL directly)
├── data/
│   ├── raw/
│   │   ├── indiccorp/           # Downloaded candidate corpus + state
│   │   └── manual/
│   │       ├── wikipedia/       # Dump + extracted text + collector metadata
│   │       └── news/            # Sharded JSONL, one shard sequence per
│   │                             source (dainik_jagran_*.jsonl, ...)
│   ├── manual/                  # Legacy OCR/scrape (collect_manual.py) data
│   ├── intermediate/
│   │   ├── news/                 # manual/05_aggregate_news_corpus.py output (news ONLY —
│   │   │   ├── news_raw.txt             # Wikipedia is never touched by this script;
│   │   │   └── news_provenance.jsonl    # its raw file lives at raw/manual/wikipedia/
│   │   │                                # above and needs no aggregation step)
│   │   ├── normalized/
│   │   ├── filtered/
│   │   ├── quality/              # Script 05 output, one file per source
│   │   ├── deduplicated/         # Script 06 output, one file per source
│   │   │                          (indiccorp_hi_deduplicated.txt,
│   │   │                          wikipedia_hi_deduplicated.txt,
│   │   │                          news_hi_deduplicated.txt)
│   │   ├── split/                 # Script 08 output — train-candidate
│   │   │   └── {source}_train_candidate.txt   (still needs decontamination)
│   │   ├── decontaminated/        # Script 09 output — train-candidate
│   │   │   └── {source}_train_decontam.txt    AFTER removing val/test overlap
│   │   └── selected/              # Script 11 intermediate (per-source selected)
│   ├── final/                   # {source}_train.txt + manual_train.txt +
│   │                             train.txt (11) / validation.txt + test.txt
│   │                             (08, FROZEN) / 90:5:5 by line count
│   └── tokenized/               # Binary token-ID files
├── metadata/
│   ├── deduplication_report.json     # Script 06 exact/near/sub-doc counts
│   ├── corpus_selection_report.json  # Public/manual provenance
│   ├── decontamination_report.json   # Exact/near-dup removal counts
│   ├── verification_report.json      # Final requirement + leakage checks
│   ├── provenance.json               # Script 15: consolidated source/licence/
│   │                                 # date/filters-applied manifest
│   ├── token_count_check.json
│   ├── split_statistics.json
│   └── tokenizer_statistics.json
├── scripts/                     # numbered pipeline scripts (01-15) + helpers
├── tokenizer/
│   ├── model/                   # The ONE final tokenizer (hindi_bpe_32000.model)
│   ├── vocab/
│   └── config/
└── README.md                    # This file
```

---

## Manual Data Requirement

The project requires:

```
manual final CONTENT TOKENS / total final CONTENT TOKENS >= 0.20
```

This means at least 80M content tokens (for a 400M target) must come from
manually collected data — Hindi Wikipedia (`hindi/scripts/manual/01_download_wikipedia.py` + `02_extract_wikipedia.py`,
already one file, no aggregation needed) plus Hindi news
(`hindi/scripts/manual/03_collect_news.py`, aggregated by
`manual/05_aggregate_news_corpus.py` into one file) (never `hindi/data/manual/`,
which holds only the older, out-of-scope OCR/scrape workflow). Wikipedia
and news remain two separate real files/sources all the way through
scripts 03-09; "manual" is the accounting/target grouping of the two,
not an early-pipeline merge — they only combine into one
`manual_train.txt` file at `11_select_corpus.py`.

`manual/05_aggregate_news_corpus.py` / `07_statistics.py` report early
documents/words/lines counts for visibility, but do not themselves
enforce the 20% floor — that happens at:

**Enforcement points:**
1. `10_train_tokenizer.py` — actual BPE token enforcement using the one final tokenizer, reported per source (Wikipedia / News / IndicCorp, plus derived Manual = Wikipedia + News)
2. `15_verify_final_corpus.py` — final verification using the final tokenizer

If manual data is insufficient, the pipeline **FAILS** with a clear error:
```
Manual data requirement failed:
  manual content tokens = X
  total content tokens = Y
  manual fraction = Z%
  required minimum = 20%
```

---

## Reproducibility

- All random operations use **seed=42** (configurable).
- Deterministic hash `blake2b("seed:line_index") → [0,1)` for selection/splitting.
- Pipeline is deterministic given the same input and configuration.

---

## Academic Constraints

- ✅ No pretrained language model
- ✅ No pretrained tokenizer (trained from scratch on Hindi training data)
- ✅ No HuggingFace Transformer model classes
- ✅ No shared tokenizer/vocabulary with the lower-resource language
- ✅ ≥20% manual data enforced via actual BPE content tokens
- ✅ ~400M content-token target verified end-to-end
