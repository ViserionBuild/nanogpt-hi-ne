# Hindi (Model H) — Phase 1 Workflow

Complete workflow and architecture for the **Model H (Hindi)** Phase 1 data
pipeline: corpus collection, cleaning, splitting, and tokenizer construction.

Every path below is relative to the repository root. The single source of truth
for all thresholds, paths, and targets is
[`hindi/configs/hindi_data.yaml`](configs/hindi_data.yaml) — no script hardcodes
a value that the config declares.

---

## 1. What this pipeline produces

| Output | Path | Tracked in git |
|---|---|---|
| Final training corpus | `hindi/data/final/train.txt` | no → Drive |
| Held-out splits | `hindi/data/final/{validation,test}.txt` | no → Drive |
| Per-source final corpora | `hindi/data/final/{indiccorp,wikipedia,news,manual}_train.txt` | no → Drive |
| Tokenizer models | `hindi/tokenizer/model/hindi_bpe_{8000…16000}.model` | **yes** |
| Vocabulary files | `hindi/tokenizer/vocab/hindi_bpe_*.vocab` | **yes** |
| Binary token IDs | `hindi/data/tokenized/{train,validation,test}.bin` | no → Drive |
| Statistics / reports | `hindi/metadata/*.json` | **yes** |
| Per-stage run summaries | `hindi/data/run_stats/*.json` | **yes** |

**Achieved:** 632,352,544 BPE content tokens in `train.txt` (7,830,015 documents)
— 26.5% above the ~500M target. Selected tokenizer: BPE, vocab 12,000.

---

## 2. Sources

| Lane | Source | Type | Licence |
|---|---|---|---|
| Public | AI4Bharat IndicCorpV2 Hindi (`hi-1/2/3.txt` shards) | downloaded | CC BY-NC 4.0 |
| Manual | Hindi Wikipedia `pages-articles` dump | collected | CC BY-SA |
| Manual | Hindi news — ABP, Live Hindustan, Dainik Jagran | collected | per-outlet |

IndicCorp is never classified as manual. Wikipedia and News stay two separate
real files all the way through stages 02–08; **"manual" is an accounting
grouping**, not an early merge — they only combine into one `manual_train.txt`
at stage 10.

Seven further news outlets were evaluated and excluded (four denied access, three
had no working article sitemap). Each carries a `disabled_reason` in the config.

---

## 3. Pipeline architecture

Twelve stages. Every stage is **streaming** — the corpus is scanned line by line
and never loaded into RAM, at any size.

```
01_download_indiccorp.py          manual/01_download_wikipedia.py
  HTTP Range, resumable,             + 02_extract_wikipedia.py
  SHA-256 verified                 manual/03_collect_news.py
        │                            + 05_aggregate_news_corpus.py
        │                                      │
        └──────────────┬───────────────────────┘
                       ↓
        THREE independent sources from here on:
        indiccorp · wikipedia · news
                       ↓
02_normalize.py            NFC, whitespace/CRLF collapse, control-char strip
                           (ZWNJ/ZWJ preserved — semantically meaningful)
                       ↓
03_filter_hindi.py         Devanagari ratio ≥ 0.60 keep · 0.40–0.60 → review
                       ↓
04_quality_filter.py       length, URL, HTML, repeated-char, punctuation,
                           digit-ratio heuristics + safety blocklist
                       ↓
05_deduplicate.py          CROSS-SOURCE, all three sources in ONE pass:
  (or 05_deduplicate_mac)    exact SHA-256, then MinHash/LSH near-dup
                             precedence: wikipedia > news > indiccorp
                       ↓
06_statistics.py           descriptive stats before any irreversible step
                       ↓
07_split_dataset.py        deterministic 90/5/5 hash split, per source
                           validation.txt + test.txt → final/, FROZEN
                       ↓
08_decontaminate.py        remove exact + near-dup matches of the FROZEN
  (or 08_decontaminate_ram)  val/test from every train candidate
                       ↓
10_concatenate_final_corpus.py   ← RUNS BEFORE 09, see note below
                           {source}_train.txt, then
                           manual_train.txt = wikipedia + news, then
                           train.txt = indiccorp + manual
                       ↓
09_train_tokenizer.py      ONE SentencePiece BPE per vocab size, trained on
                           final/train.txt ONLY; then reloads and streams
                           per-source token counts vs. targets
                       ↓
11_evaluate_tokenizer.py   fertility, chars/token, UNK, Zipf curves, examples
                           — all measured on held-out validation.txt
                       ↓
12_tokenize_corpus.py      streaming uint32 .bin with self-verifying re-read
```

> **Run order caveat.** Script `10_concatenate_final_corpus.py` must run
> **before** `09_train_tokenizer.py`, despite the numbering. The tokenizer trains
> on `final/train.txt`, which script 10 is what produces. The numbers are
> filenames, not execution order, at this one point in the pipeline.

### Disk-saving cascade

Stages 03, 04, and 05 each delete their own predecessor's output for a source
immediately after successfully writing their own. Only `raw/`, `deduplicated/`,
`split/`, `decontaminated/`, and `final/` survive a full run. Per-stage counts
survive regardless, in `data/run_stats/`.

---

## 4. Two design decisions that matter

### Split before decontamination (leakage safety)

Stage 07 runs **before** the tokenizer is trained. `validation.txt` and
`test.txt` are written once and frozen; stage 08 then removes their matches from
the train side only.

Consequence: **the tokenizer never sees a single held-out line**, so Phase 2
perplexity is measured on genuinely unseen text. Exact-match removal alone would
not be enough — paraphrased held-out items leak too — so decontamination uses
MinHash/LSH near-duplicate matching with a deliberately *looser* index than
deduplication (16 permutations, 5-word shingles vs. 32/3). Recall matters more
than precision when the cost of a miss is train/test leakage.

### Cross-source dedup with manual precedence

All three sources are deduplicated **together** in one pass, not separately. When
the same text exists in both a manual source and IndicCorp, the **manual copy is
kept** and the IndicCorp copy dropped (precedence: wikipedia > news > indiccorp).

This protects the manual lane at the public lane's expense — the correct
direction, given the ≥20% manual requirement. 43,681 of the 242,381 exact
duplicates removed were cross-source removals of this kind.

The near-dup stage is two-step by design: LSH banding generates candidates
cheaply, then each candidate is **confirmed** by an actual Jaccard/overlap
estimate before anything is removed. A band collision alone never causes a
removal, so a line sharing one common phrase with another survives.

---

## 5. Measured results

### Per-stage document counts

| Stage | IndicCorp | Wikipedia | News | Total |
|---|---|---|---|---|
| normalized (02) | 8,787,343 | 936,593 | 340,892 | 10,064,828 |
| script filter (03) | 8,779,002 | 884,851 | 283,287 | 9,947,140 |
| quality filter (04) | 8,742,935 | 820,086 | 277,305 | 9,840,326 |
| dedup (05) | 8,520,630 | 672,411 | 147,487 | 9,340,528 |
| train candidate (07) | 7,668,144 | 605,562 | 132,808 | 8,406,514 |
| decontaminated (08) | 7,169,358 | 575,485 | 85,172 | **7,830,015** |

Held out at stage 07: **validation 467,228** / **test 466,786** documents.

**Dedup:** 5.08% removed — 242,381 exact (43,681 cross-source) + 257,417
near-dup (205,718 whole-document, 51,699 sub-document). Peak RSS 12.6 GB,
~7,854 lines/s.

**Decontamination:** 576,499 near-dup matches of val/test removed; 0 exact
remained. News lost 36% — expected, since outlets syndicate and re-run stories.

### Final token counts (BPE content tokens, BOS/EOS excluded)

| Source | Lane | Tokens | Share | Fertility |
|---|---|---|---|---|
| IndicCorpV2 | public | 517,934,945 | 0.8191 | 1.2883 |
| Wikipedia | manual | 59,965,651 | 0.0948 | 1.5153 |
| News | manual | 54,451,948 | 0.0861 | 1.3026 |
| **Manual combined** | manual | **114,417,599** | **0.1809** | 1.4061 |
| **Total (train)** | | **632,352,544** | 1.0000 | 1.3082 |

**Manual fraction 0.1809 is below the 0.20 requirement.** Both sub-targets pass
(IndicCorp 517.9M vs. 400M; manual 114.4M vs. 100M, +14.4%) — the public lane
overshot by ~120M and diluted the ratio. Trimming IndicCorp by 60,264,549 tokens
would reach exactly 0.20 while leaving 572M training tokens. Not applied; see
`report/phase1_report.md` §7.

---

## 6. Tokenizer architecture

**Exactly one tokenizer**, SentencePiece BPE (`model_type="bpe"` — BPE is never
hand-implemented), trained on `final/train.txt` and nothing else. It never sees
validation or test text.

| Setting | Value |
|---|---|
| Algorithm | BPE (SentencePiece) |
| Character coverage | 0.9995 |
| `input_sentence_size` | 5,000,000 (shuffled sample) |
| `max_sentence_length` | 16,384 |
| Byte fallback | enabled |
| Special tokens | `<pad>`=0 `<unk>`=1 `<bos>`=2 `<eos>`=3 |
| Vocab sizes swept | 8,000 / 10,000 / 12,000 / 14,000 / 16,000 |
| **Selected** | **12,000** |

### Vocabulary-size selection (held-out validation)

| Vocab | Fertility | Chars/token | Used vocab |
|---|---|---|---|
| 8,000 | 1.3823 | 3.7143 | 7,869 |
| 10,000 | 1.3361 | 3.8427 | 9,868 |
| **12,000** | **1.3042** | **3.9366** | **11,868** |
| 14,000 | 1.2804 | 4.0098 | 13,867 |
| 16,000 | 1.2619 | 4.0686 | 15,866 |

Fertility falls monotonically, so no size is optimal on compression alone.
Marginal return decides it: 10K→12K buys 0.032 fertility, 14K→16K only 0.019.
Past 12K each 2,000 embedding rows costs transformer body capacity at a fixed
~25M parameter budget. 11,868/12,000 entries are actually used on held-out text,
so the size is not wasteful.

**UNK rate is 0.0 at every vocab size.** Byte fallback puts all 256 byte pieces
`<0x00>`–`<0xFF>` in the vocabulary, so uncovered characters decompose into byte
tokens rather than emitting `<unk>` — making `<unk>` structurally unreachable and
UNK rate useless for selection. Hence fertility as the criterion.

### Token counting discipline

Whitespace words and BPE tokens are **different quantities** and the pipeline
never conflates them:

- The IndicCorp *download* target is 500M **whitespace words** — a crude measure
  used only to fetch enough candidate data.
- The corpus target is 500M **BPE content tokens**, BOS/EOS excluded, measured
  after the tokenizer exists.
- `selection.target_unit: bpe_content_tokens` in the config is a
  machine-checkable declaration that downstream scripts assert against.

**No silent duplication.** If the corpus falls short of a target, script 09
reports the shortfall — it never repeats lines to reach a number, never lowers
quality thresholds, and never trains a second tokenizer to compensate.

---

## 7. Commands, in run order

```bash
pip install -r requirements.txt
```

### Collect (stage 01)

```bash
# Public lane
python hindi/scripts/01_download_indiccorp.py --config hindi/configs/hindi_data.yaml

# Manual lane — Wikipedia
python hindi/scripts/manual/01_download_wikipedia.py --config hindi/configs/hindi_data.yaml
python hindi/scripts/manual/02_extract_wikipedia.py  --config hindi/configs/hindi_data.yaml

# Manual lane — News (resumable; sitemaps are a rolling window, so run repeatedly
# over days to accumulate volume — each run only adds new articles)
python hindi/scripts/manual/03_collect_news.py --config hindi/configs/hindi_data.yaml
python hindi/scripts/manual/05_aggregate_news_corpus.py --config hindi/configs/hindi_data.yaml
```

`04_convert_news_jsonl_to_raw.py` is an **optional debugging aid** — stage 05
aggregation reads the JSONL shards directly. Nothing downstream consumes its
output.

### Clean — run once per source (stages 02–04)

```bash
for SRC in indiccorp wikipedia news; do
  python hindi/scripts/02_normalize.py      --config hindi/configs/hindi_data.yaml --source $SRC
  python hindi/scripts/03_filter_hindi.py   --config hindi/configs/hindi_data.yaml --source $SRC
  python hindi/scripts/04_quality_filter.py --config hindi/configs/hindi_data.yaml --source $SRC
done
```

### Dedup, stats, split, decontaminate (stages 05–08)

```bash
# All three sources together in one cross-source pass — no --source flag
python hindi/scripts/05_deduplicate.py   --config hindi/configs/hindi_data.yaml
python hindi/scripts/06_statistics.py    --config hindi/configs/hindi_data.yaml
python hindi/scripts/07_split_dataset.py --config hindi/configs/hindi_data.yaml
python hindi/scripts/08_decontaminate.py --config hindi/configs/hindi_data.yaml
```

Use `05_deduplicate_mac.py` / `08_decontaminate_ram.py` for the in-memory
variants — identical output, RAM instead of SQLite.

### Concatenate, then train tokenizer (stages 10, 09 — in that order)

```bash
python hindi/scripts/10_concatenate_final_corpus.py --config hindi/configs/hindi_data.yaml
python hindi/scripts/09_train_tokenizer.py --config hindi/configs/hindi_data.yaml
```

### Evaluate and tokenize (stages 11–12)

```bash
python hindi/scripts/11_evaluate_tokenizer.py --config hindi/configs/hindi_data.yaml
python hindi/scripts/12_tokenize_corpus.py \
    --config hindi/configs/hindi_data.yaml \
    --tokenizer-model hindi/tokenizer/model/hindi_bpe_12000.model
```

---

## 8. Where to look for what

| Question | File |
|---|---|
| Did stage N run, and how much survived? | `hindi/data/run_stats/` (stages 02–04, one JSON per run) |
| Final token counts vs. targets? | `hindi/metadata/token_count_check.json` — authoritative |
| Dedup removal breakdown? | `hindi/metadata/deduplication_report_mac.json` |
| Leakage removal counts? | `hindi/metadata/decontamination_report.json` |
| Split sizes and method? | `hindi/metadata/split_statistics.json` |
| Corpus health before splitting? | `hindi/metadata/corpus_statistics.json` |
| Tokenizer sweep results? | `hindi/metadata/tokenizer_statistics.json` |
| Per-article news provenance? | `hindi/data/intermediate/news/news_provenance.jsonl` |

Everything under `hindi/metadata/` is a **report**, never corpus data — safe to
delete and regenerate by re-running its producing script.

---

## 9. Reproducibility

- Seed **42** throughout; deterministic hash `blake2b("seed:line_index") → [0,1)`
  for splitting.
- Every stage is deterministic given the same input and config.
- Downloads are SHA-256 verified and resumable; news collection is resumable per
  source via a page cursor.
- One caveat: `09_train_tokenizer.py` passes `seed_sentencepiece_size=42`, which
  is SentencePiece's *seed-piece count*, not a random seed. Combined with
  `input_sentence_size=5000000`, the 5M-line training sample is therefore not
  seeded — re-running the trainer resamples and can produce a slightly different
  tokenizer. Setting `input_sentence_size: 0` (use all data) removes the sampling
  and the issue together.

---

## 10. Constraints honoured

| Constraint | Status |
|---|---|
| No pretrained language model | **PASS** |
| No pretrained tokenizer — BPE trained from scratch on Hindi train.txt | **PASS** |
| No HuggingFace Transformer classes | **PASS** |
| No tokenizer/vocabulary shared with Nepali | **PASS** |
| Tokenizer never trained on validation/test | **PASS** |
| Train decontaminated against frozen val/test | **PASS** |
| ~500M content-token target | **PASS** — 632.4M |
