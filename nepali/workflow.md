# Nepali (Model L) — Phase 1 Workflow

Complete workflow and architecture for the **Model L (Nepali)** Phase 1 data
pipeline: corpus collection, cleaning, splitting, and tokenizer construction.

Every path below is relative to the repository root. The single source of truth
for all thresholds, paths, and targets is
[`nepali/configs/nepali_data.yaml`](configs/nepali_data.yaml) — no script
hardcodes a value that the config declares.

---

## 1. What this pipeline produces

| Output | Path | Tracked in git |
|---|---|---|
| Final training corpus | `nepali/data/final/train.txt` | no, on Drive |
| Held-out splits | `nepali/data/final/{validation,test}.txt` | no, on Drive |
| Per-source final corpora | `nepali/data/final/{indiccorp,onlinekhabar,khabarhub,manual}_train.txt` | no, on Drive |
| Tokenizer models | `nepali/tokenizer/model/nepali_bpe_{6000...16000}.model` | **yes** |
| Vocabulary files | `nepali/tokenizer/vocab/nepali_bpe_*.vocab` | **yes** |
| Binary token IDs | `nepali/data/tokenized/{train,validation,test}.bin` | no, on Drive |
| Statistics and reports | `nepali/metadata/*.json` | **yes** |
| Per-stage run summaries | `nepali/data/run_stats/*.json` | **yes** |

**Achieved:** 637,535,988 BPE content tokens in `train.txt` (10,981,726
documents) — 27.5% above the ~500M target. Selected tokenizer: BPE, vocab 12,000.

Nepali exceeded the target by a wider margin than Hindi did, so **no shortfall
justification is required for Model L** despite it being the lower-resource
language.

---

## 2. Sources

| Lane | Source | Type | Licence |
|---|---|---|---|
| Public | AI4Bharat IndicCorpV2 Nepali (single `ne.txt`, ~14.8 GB) | downloaded | CC BY-NC 4.0 |
| Manual | Onlinekhabar (WordPress REST API) | collected | per-outlet |
| Manual | Khabarhub (WordPress REST API) | collected | per-outlet |

IndicCorp is never classified as manual. Onlinekhabar and Khabarhub stay two
separate real files all the way through stages 02–08; **"manual" is an accounting
grouping**, not an early merge — they only combine into one `manual_train.txt`
at stage 09.

### Why the manual lane is news-only

Nepali has **no queryable Wikisource wiki** (`ne.wikisource.org` does not exist
as an independent wiki), and no Wikipedia sub-lane was built. The manual lane is
therefore news-only, unlike Hindi's Wikipedia-plus-news composition. This is a
realistic lower-resource constraint, not an oversight.

Both sources are open WordPress REST endpoints (`/wp-json/wp/v2/posts`),
collected with a resumable per-source page cursor and URL-level deduplication,
sharded into JSONL at 20,000 documents per shard. Khabarhub's robots.txt was
verified to disallow only `/wp-admin/`, leaving article collection permitted.

---

## 3. Pipeline architecture

Twelve stages. Every stage is **streaming** — the corpus is scanned line by line
and never loaded into RAM, at any size.

```
01_download_indiccorp.py            manual/collect_news.py
  HTTP Range, resumable,              Onlinekhabar + Khabarhub REST APIs
  SHA-256 verified                    + convert_news_jsonl_to_raw.py
        |                                       |
        +-----------------+---------------------+
                          |
        THREE independent sources from here on:
        indiccorp . onlinekhabar . khabarhub
                          |
02_normalize.py             NFC, whitespace/CRLF collapse, control-char strip
                            (ZWNJ/ZWJ preserved — semantically meaningful)
                          |
03_filter_devanagari_script.py
                            Devanagari ratio >= 0.70 keep, 0.40-0.70 to review
                          |
04_quality_filter.py        length, URL, HTML, repeated-char, punctuation,
                            digit-ratio heuristics + safety blocklist
                          |
05_deduplicate.py           CROSS-SOURCE, all three sources in ONE pass:
  (or 05_deduplicate_mac)     exact SHA-256, then MinHash/LSH near-dup
                              precedence: onlinekhabar > khabarhub > indiccorp
                          |
06_corpus_statistics.py     descriptive stats before any irreversible step
                          |
07_split_train_val_test.py  deterministic 90/5/5 hash split, per source
                            validation.txt + test.txt to final/, FROZEN
                          |
08_decontaminate.py         remove exact + near-dup matches of the FROZEN
  (or 08_decontaminate_ram)   val/test from every train candidate
                          |
09_concatenate_final_corpus.py
                            {source}_train.txt, then
                            manual_train.txt = onlinekhabar + khabarhub, then
                            train.txt = indiccorp + manual
                          |
10_train_tokenizer.py       ONE SentencePiece BPE per vocab size, trained on
                            final/train.txt ONLY; then reloads and streams
                            per-source token counts vs. targets
                          |
11_evaluate_tokenizer.py    fertility, chars/token, UNK, Zipf curves, examples
                            — all measured on held-out validation.txt
                          |
12_tokenize_corpus.py       streaming uint32 .bin with self-verifying re-read
```

Unlike the Hindi pipeline, Nepali's script numbering **matches execution order
exactly** — run 01 through 12 in sequence. (Hindi swaps 09 and 10.)

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
kept** and the IndicCorp copy dropped (precedence: onlinekhabar > khabarhub >
indiccorp). This protects the manual lane at the public lane's expense — the
correct direction, given the >=20% manual requirement.

The near-dup stage is two-step by design: LSH banding generates candidates
cheaply, then each candidate is **confirmed** by an actual Jaccard/overlap
estimate before anything is removed. A band collision alone never causes a
removal.

---

## 5. Measured results

### Per-stage document counts

| Stage | IndicCorp | Onlinekhabar | Khabarhub | Total |
|---|---|---|---|---|
| normalized (02) | 14,946,717 | 297,891 | 257,041 | 15,501,649 |
| script filter (03) | 14,855,758 | 297,649 | 256,792 | 15,410,199 |
| quality filter (04) | 14,821,096 | 288,229 | 254,103 | 15,363,428 |
| dedup (05) | 13,329,271 | 282,629 | 233,910 | 13,845,810 |
| train candidate (07) | 11,996,006 | 254,527 | 210,628 | 12,461,161 |
| decontaminated (08) | 10,656,979 | 175,998 | 148,749 | **10,981,726** |

Held out at stage 07: **validation 692,772** / **test 691,877** documents.

**Dedup:** 9.88% removed — 323,984 exact (61 cross-source) + 1,193,634 near-dup
(895,595 whole-document, 298,039 sub-document). Peak RSS 8.7 GB, ~5,637 lines/s
over 2,725 s.

The near-duplicate rate is roughly **double Hindi's** (9.88% vs. 5.08%). That is
the expected signature of a smaller-language crawl: less unique text exists on
the open web, so more of it gets scraped repeatedly.

**Decontamination:** 1,479,435 near-dup matches of val/test removed; 0 exact
remained. Both news sources lost about 30% of their train candidates — the
syndication effect, same as Hindi.

### Final token counts (BPE content tokens, BOS/EOS excluded)

| Source | Lane | Tokens | Share | Fertility |
|---|---|---|---|---|
| IndicCorpV2 | public | 520,693,966 | 0.8167 | 1.4656 |
| Onlinekhabar | manual | 70,435,290 | 0.1105 | 1.4073 |
| Khabarhub | manual | 46,406,732 | 0.0728 | 1.4059 |
| **Manual combined** | manual | **116,842,022** | **0.1833** | — |
| **Total (train)** | | **637,535,988** | 1.0000 | 1.4544 |

| Split | Documents | Content tokens | Stored (incl. BOS/EOS) |
|---|---|---|---|
| train | 10,981,726 | 637,535,988 | 659,499,440 |
| validation | 692,772 | 42,326,843 | 43,712,387 |
| test | 691,877 | 42,396,121 | 43,779,875 |

**Manual fraction 0.1833 is below the 0.20 requirement.** Both sub-targets pass
(IndicCorp 520.7M vs. 400M; manual 116.8M vs. 100M, +16.8%) — the public lane
overshot by ~120M and diluted the ratio. Trimming IndicCorp by 53,325,878 tokens
would reach exactly 0.20 while leaving 584M training tokens. Not applied; see
`report/phase1_report.md` section 7.

---

## 6. Tokenizer architecture

**Exactly one tokenizer**, SentencePiece BPE (`model_type="bpe"` — BPE is never
hand-implemented), trained on `final/train.txt` and nothing else. It never sees
validation or test text, and shares nothing with the Hindi tokenizer.

| Setting | Value |
|---|---|
| Algorithm | BPE (SentencePiece) |
| Character coverage | 0.9995 |
| `input_sentence_size` | 5,000,000 (shuffled sample) |
| `max_sentence_length` | 16,384 |
| Byte fallback | enabled |
| Special tokens | `<pad>`=0 `<unk>`=1 `<bos>`=2 `<eos>`=3 |
| Vocab sizes swept | 6,000 / 8,000 / 10,000 / 12,000 / 14,000 / 16,000 |
| **Selected** | **12,000** |

### Vocabulary-size selection (held-out validation)

| Vocab | Fertility | Chars/token | Used vocab |
|---|---|---|---|
| 6,000 | 1.6444 | 3.9116 | 5,894 |
| 8,000 | 1.5484 | 4.1539 | 7,894 |
| 10,000 | 1.4852 | 4.3309 | 9,894 |
| **12,000** | **1.4394** | **4.4686** | **11,894** |
| 14,000 | 1.4043 | 4.5802 | 13,894 |
| 16,000 | 1.3766 | 4.6724 | 15,894 |

Fertility falls monotonically, so no size is optimal on compression alone.
Marginal return decides it: 10K to 12K buys 0.046 fertility, 14K to 16K only
0.028. Past 12K each 2,000 embedding rows costs transformer body capacity at a
fixed ~25M parameter budget. 11,894/12,000 entries are actually used on held-out
text, so the size is not wasteful. Using the same 12,000 for both languages also
keeps Model H and Model L comparable in Phase 2.

**UNK rate is 0.0 at every vocab size.** Byte fallback puts all 256 byte pieces
`<0x00>` through `<0xFF>` in the vocabulary, so uncovered characters decompose
into byte tokens rather than emitting `<unk>` — making `<unk>` structurally
unreachable and UNK rate useless for selection. Hence fertility as the criterion.

### Nepali versus Hindi

Nepali's fertility is consistently higher (1.4394 vs. 1.3042 at vocab 12,000).
This is a **morphology effect**: Nepali's agglutinative case marking generates far
more distinct surface forms per lemma, which the same corpus also reports as more
unique words (6.80M vs. Hindi's 3.96M) despite Nepali being the lower-resource
language.

The learned vocabulary shows the same typological difference directly. Nepali's
top tokens include bare **agglutinative suffixes** with no word-boundary marker
(`को`, `मा`, `का`, `ले`, `लाई`), while Hindi's are dominated by free-standing
**postpositions** carrying the boundary marker (`▁के`, `▁में`, `▁की`). BPE
recovered this from raw text alone.

Chars/token is **not** comparable across the two languages — Nepali reads higher
(4.4686 vs. 3.9366) because of longer orthographic words and the news-heavy
manual lane, not because its tokenizer is better. Fertility is the fair
comparison.

### Token counting discipline

Whitespace words and BPE tokens are **different quantities** and the pipeline
never conflates them:

- The IndicCorp *download* target is 470M **whitespace words** — a crude measure
  used only to fetch enough candidate data.
- The corpus target is 500M **BPE content tokens**, BOS/EOS excluded, measured
  after the tokenizer exists.
- `selection.target_unit: bpe_content_tokens` in the config is a
  machine-checkable declaration that downstream scripts assert against.

**No silent duplication.** If the corpus falls short of a target, script 10
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
python nepali/scripts/01_download_indiccorp.py --config nepali/configs/nepali_data.yaml

# Manual lane — both news sources (resumable per-source page cursor;
# run repeatedly over days to accumulate volume)
python nepali/scripts/manual/collect_news.py --config nepali/configs/nepali_data.yaml

# Flatten the JSONL shards into one-document-per-line raw text
python nepali/scripts/manual/convert_news_jsonl_to_raw.py --config nepali/configs/nepali_data.yaml
```

Unlike Hindi, `convert_news_jsonl_to_raw.py` is **required** here — there is no
separate aggregation step, and stage 02 reads its output directly.

### Clean — run once per source (stages 02-04)

```bash
for SRC in indiccorp onlinekhabar khabarhub; do
  python nepali/scripts/02_normalize.py                --config nepali/configs/nepali_data.yaml --source $SRC
  python nepali/scripts/03_filter_devanagari_script.py --config nepali/configs/nepali_data.yaml --source $SRC
  python nepali/scripts/04_quality_filter.py           --config nepali/configs/nepali_data.yaml --source $SRC
done
```

### Dedup, stats, split, decontaminate (stages 05-08)

```bash
# All three sources together in one cross-source pass — no --source flag
python nepali/scripts/05_deduplicate.py          --config nepali/configs/nepali_data.yaml
python nepali/scripts/06_corpus_statistics.py    --config nepali/configs/nepali_data.yaml
python nepali/scripts/07_split_train_val_test.py --config nepali/configs/nepali_data.yaml
python nepali/scripts/08_decontaminate.py        --config nepali/configs/nepali_data.yaml
```

Use `05_deduplicate_mac.py` / `08_decontaminate_ram.py` for the in-memory
variants — identical output, RAM instead of SQLite. Note that
`08_decontaminate_ram.py` writes to the **same** `decontamination_report.json`
filename as `08_decontaminate.py`, so a second run overwrites the first report.

### Concatenate, then train tokenizer (stages 09-10)

```bash
python nepali/scripts/09_concatenate_final_corpus.py --config nepali/configs/nepali_data.yaml
python nepali/scripts/10_train_tokenizer.py          --config nepali/configs/nepali_data.yaml
```

### Evaluate and tokenize (stages 11-12)

```bash
python nepali/scripts/11_evaluate_tokenizer.py --config nepali/configs/nepali_data.yaml
python nepali/scripts/12_tokenize_corpus.py \
    --config nepali/configs/nepali_data.yaml \
    --tokenizer-model nepali/tokenizer/model/nepali_bpe_12000.model
```

---

## 8. Where to look for what

| Question | File |
|---|---|
| Did stage N run, and how much survived? | `nepali/data/run_stats/` (stages 02-04, one JSON per run) |
| Final token counts vs. targets? | `nepali/metadata/token_count_check.json` — authoritative |
| Dedup removal breakdown? | `nepali/metadata/deduplication_report_mac.json` |
| Leakage removal counts? | `nepali/metadata/decontamination_report.json` |
| Split sizes and method? | `nepali/metadata/split_statistics.json` |
| Corpus health before splitting? | `nepali/metadata/corpus_statistics.json` |
| Tokenizer sweep results? | `nepali/metadata/tokenizer_statistics.json` |
| Per-source train token counts? | `nepali/data/tokenized/train_content_token_counts.json` |
| Per-script input/output reference? | `nepali/nepali_pipeline_io_reference.md` |

Everything under `nepali/metadata/` is a **report**, never corpus data — safe to
delete and regenerate by re-running its producing script.

---

## 9. Reproducibility

- Seed **42** throughout; deterministic hash `blake2b("seed:line_index")` mapped
  to [0,1) for splitting.
- Every stage is deterministic given the same input and config.
- Downloads are SHA-256 verified and resumable; news collection is resumable per
  source via a page cursor.
- One caveat: `10_train_tokenizer.py` passes `seed_sentencepiece_size=42`, which
  is SentencePiece's *seed-piece count*, not a random seed. Combined with
  `input_sentence_size=5000000`, the 5M-line training sample is therefore not
  seeded — re-running the trainer resamples and can produce a slightly different
  tokenizer. Setting `input_sentence_size: 0` (use all data) removes the sampling
  and the issue together. Note that 5M lines is 45.5% of Nepali's 10.98M-line
  `train.txt`, a lower sampling fraction than Hindi's 63.9%.

---

## 10. Constraints honoured

| Constraint | Status |
|---|---|
| Model L drawn from the allowed lower-resource list | **PASS** — Nepali |
| No pretrained language model | **PASS** |
| No pretrained tokenizer — BPE trained from scratch on Nepali train.txt | **PASS** |
| No HuggingFace Transformer classes | **PASS** |
| No tokenizer/vocabulary shared with Hindi | **PASS** |
| Tokenizer never trained on validation/test | **PASS** |
| Train decontaminated against frozen val/test | **PASS** |
| ~500M content-token target | **PASS** — 637.5M, no shortfall justification needed |
