# Hindi Corpus Pipeline — Run Instructions

## Pipeline order

```text
1a. 01_download_indiccorp.py
1b. manual/01_download_wikipedia.py + manual/02_extract_wikipedia.py   (independent of 1a)
1c. manual/03_collect_news.py
1d. manual/05_aggregate_news_corpus.py   (news only — Wikipedia needs no merge)
2.  02_normalize.py
3.  03_filter_hindi.py
4.  04_quality_filter.py
5.  05_deduplicate.py  (or 05_deduplicate_mac.py on Mac)
6.  06_statistics.py
7.  07_split_dataset.py
8.  08_decontaminate.py  (or 08_decontaminate_ram.py on Mac)
9.  10_concatenate_final_corpus.py
10. 09_train_tokenizer.py
11. 11_evaluate_tokenizer.py
12. 12_tokenize_corpus.py
```

THREE SOURCES: `indiccorp` (public, AI4Bharat IndicCorpV2 Hindi), and two
manual sources — `wikipedia` (Hindi Wikipedia dump) and `news` (up to 10
configured Hindi news outlets, merged into one file). "Manual" is not a
real fourth source/file until step 9 — it's an accounting group
(Wikipedia + News) used only for the ≥20% manual-fraction target.

Note steps 9/10 run out of filename order: `10_concatenate_final_corpus.py`
runs BEFORE `09_train_tokenizer.py`, because corpus selection needs no
token counts (no subsampling happens), so there's no dependency cycle.
Filenames keep their original numeric prefixes from history; the list
above is the authoritative execution order.

Run all commands from the repository root.

## Targets

```yaml
selection:
  target_words: 500000000          # rough IndicCorp download target (whitespace words)
  indiccorp_target_tokens: 400000000
  manual_target_tokens: 100000000
  total_target_tokens: 500000000
  target_unit: bpe_content_tokens

manual:
  min_fraction: 0.20
  enforce: true
```

All targets except `target_words` are measured in actual SentencePiece
BPE content tokens (never whitespace words, never BOS/EOS), and are only
ever measured once the final tokenizer is trained (`09_train_tokenizer.py`).

## 1a. Download IndicCorp

```bash
python hindi/scripts/01_download_indiccorp.py \
    --config hindi/configs/hindi_data.yaml \
    --target-words 1000000 --chunk-size-bytes 16777216   # small test first
```

```bash
python hindi/scripts/01_download_indiccorp.py \
    --config hindi/configs/hindi_data.yaml --resume
```

## 1b. Wikipedia (independent of 1a)

```bash
python hindi/scripts/manual/01_download_wikipedia.py \
    --config hindi/configs/hindi_data.yaml

python hindi/scripts/manual/02_extract_wikipedia.py \
    --config hindi/configs/hindi_data.yaml
```

Writes `hindi/data/raw/manual/wikipedia/wikipedia_hi_raw.txt`. Resumable/
idempotent; `--overwrite` forces a redo.

## 1c. News (independent of 1a/1b)

```bash
python hindi/scripts/manual/03_collect_news.py \
    --config hindi/configs/hindi_data.yaml
```

Collects every source enabled in `hindi_data.yaml`'s `manual.news` block
(3 enabled by default: `dainik_jagran`, `hindustan`, `abp` — 7 others are
disabled with a documented `disabled_reason`, see the script's module
docstring). One source only:

```bash
python hindi/scripts/manual/03_collect_news.py \
    --config hindi/configs/hindi_data.yaml --source dainik_jagran --max-articles 5
```

`--dry-run` (network-free) / `--smoke-test` (connectivity only) are also
available. Discovery capability is per-source (`sitemap_rolling` vs.
`sitemap_index_dated`/`sitemap_index_static` — see each source's
`discovery` block); a rolling-window source may need repeated runs over
time to accumulate volume.

## 1d. Aggregate news into one file

```bash
python hindi/scripts/manual/05_aggregate_news_corpus.py \
    --config hindi/configs/hindi_data.yaml
```

Combines every enabled news source's collected shards into
`hindi/data/intermediate/news/news_raw.txt` plus `news_provenance.jsonl`
and an aggregation report (written to
`hindi/data/raw/manual/news/news_aggregation_report.json`). Wikipedia is
out of scope for this script — it already produces one file and needs no
merge step. Re-running requires `--overwrite` to regenerate.

`hindi/scripts/manual/04_convert_news_jsonl_to_raw.py` is an optional
standalone debugging aid only — not required, since step 1d reads the
collector's JSONL directly.

### Optional — convert news JSONL to flattened raw text (debugging aid only)

```bash
python hindi/scripts/manual/04_convert_news_jsonl_to_raw.py \
    --config hindi/configs/hindi_data.yaml --source all --overwrite
```

Flattens `03_collect_news.py`'s sharded JSONL into one
`{source}_hi_raw.txt` per news source, for manual inspection only.
Nothing downstream reads this output — `manual/05_aggregate_news_corpus.py`
(step 1d) reads the JSONL shards directly, so this step can be skipped
entirely in the normal pipeline flow.

## 2. Normalize (run per source)

```bash
python hindi/scripts/02_normalize.py --config hindi/configs/hindi_data.yaml --source indiccorp
python hindi/scripts/02_normalize.py --config hindi/configs/hindi_data.yaml --source wikipedia
python hindi/scripts/02_normalize.py --config hindi/configs/hindi_data.yaml --source news
```

## 3. Hindi-language filter (run per source)

```bash
python hindi/scripts/03_filter_hindi.py --config hindi/configs/hindi_data.yaml --source indiccorp
python hindi/scripts/03_filter_hindi.py --config hindi/configs/hindi_data.yaml --source wikipedia
python hindi/scripts/03_filter_hindi.py --config hindi/configs/hindi_data.yaml --source news
```

## 4. Quality filter (run per source)

```bash
python hindi/scripts/04_quality_filter.py --config hindi/configs/hindi_data.yaml --source indiccorp
python hindi/scripts/04_quality_filter.py --config hindi/configs/hindi_data.yaml --source wikipedia
python hindi/scripts/04_quality_filter.py --config hindi/configs/hindi_data.yaml --source news
```

Steps 2-4 each auto-delete their immediate predecessor's output file once
their own output is written successfully (see `shared/run_stats.py`), and
each run writes a stats JSON to `hindi/data/run_stats/`.

## 5. Deduplicate (cross-source, one shared pass)

```bash
python hindi/scripts/05_deduplicate.py --config hindi/configs/hindi_data.yaml
```

Mac-optimized version:

```bash
python hindi/scripts/05_deduplicate_mac.py \
    --config hindi/configs/hindi_data.yaml --in-memory
```

Do not run both. Reads all three sources' quality-filtered output
together (exact + MinHash/LSH near-dup, cross-source). Source precedence
on a cross-source duplicate: **Wikipedia > News > IndicCorp**. Writes
`indiccorp_hi_deduplicated.txt`, `wikipedia_hi_deduplicated.txt`,
`news_hi_deduplicated.txt`, then deletes each source's quality-stage
input file.

## 6. Statistics

```bash
python hindi/scripts/06_statistics.py --config hindi/configs/hindi_data.yaml
```

Reports on the `deduplicated` stage only (per source + manual summary +
combined total) — the earlier raw/normalized/filtered/quality stages are
already gone by this point (auto-deleted by steps 2-4), and each of those
steps already recorded its own counts in `run_stats/`.

## 7. Split train/validation/test

```bash
python hindi/scripts/07_split_dataset.py --config hindi/configs/hindi_data.yaml
```

Splits each of the three deduplicated sources deterministically (~90%
train / 5% validation / 5% test). Writes FROZEN `final/validation.txt` +
`final/test.txt` (combined across all three sources) and per-source
`split/{source}_train_candidate.txt`. `indiccorp` is always required;
`wikipedia`/`news` are each required independently based on their own
`enabled` flag.

## 8. Decontaminate

```bash
python hindi/scripts/08_decontaminate.py --config hindi/configs/hindi_data.yaml
```

Mac version:

```bash
python hindi/scripts/08_decontaminate_ram.py --config hindi/configs/hindi_data.yaml
```

Removes any train-candidate document that duplicates the frozen
validation/test sets, per source. Writes
`decontaminated/{indiccorp,wikipedia,news}_train_decontam.txt`.

## 9. Concatenate final corpus

```bash
python hindi/scripts/10_concatenate_final_corpus.py \
    --config hindi/configs/hindi_data.yaml
```

Must run BEFORE tokenizer training — the tokenizer trains on this
script's output. Combines the three decontaminated sources (no
subsampling/trimming) into:

```text
final/indiccorp_train.txt
final/wikipedia_train.txt
final/news_train.txt
final/manual_train.txt   (= wikipedia_train.txt + news_train.txt)
final/train.txt          (= indiccorp_train.txt + manual_train.txt)
```

with a strict line-count reconciliation check (wikipedia + news ==
manual; indiccorp + manual == train.txt).

## 10. Train the final tokenizer

There is **one final/production tokenizer**, but this stage can train
multiple **candidate vocabulary-size variants** for comparison:

```bash
python hindi/scripts/09_train_tokenizer.py \
    --config hindi/configs/hindi_data.yaml \
    --vocab-sizes 16000 32000 48000
```

`--config` alone trains every size listed under `tokenizer.vocab_sizes`
in the config. `--overwrite` is only needed to retrain an existing model.
These are candidate tokenizers, not multiple production tokenizers —
step 11 evaluates them so you can pick one.

Immediately after training, the script reloads the (default-vocab-size)
tokenizer and streams it separately over `final/indiccorp_train.txt`,
`final/wikipedia_train.txt`, `final/news_train.txt` to report:

```text
IndicCorp / Wikipedia / News / Manual (=Wikipedia+News) / Total BPE content tokens
Manual fraction (required >= 20%)
Shortfall against the IndicCorp/Manual/Total targets, if any
```

No second tokenizer is ever trained to close a shortfall, and quality
thresholds are never lowered to reach a token target. Report written to
`hindi/metadata/token_count_check.json`.

## 11. Evaluate tokenizer

```bash
python hindi/scripts/11_evaluate_tokenizer.py \
    --config hindi/configs/hindi_data.yaml
```

Evaluates every trained `hindi_bpe_*.model` variant against
`final/validation.txt`: fertility (tokens/word), UNK rate, chars/token,
top-token frequencies, vocabulary utilization, worked tokenization
examples. Generates `token_frequency_distribution.png` (Zipf plot,
always) and `tokenizer_comparison.png` (fertility/chars-per-token/UNK-rate
bars, only once 2+ variants exist) under `hindi/reports/figures/`, plus
`tokenizer_statistics.json` under `hindi/metadata/`. Once a vocab size is
selected, step 12 must use that model for the actual final tokenization.

## 12. Tokenize corpus

```bash
python hindi/scripts/12_tokenize_corpus.py \
    --config hindi/configs/hindi_data.yaml \
    --tokenizer-model hindi/tokenizer/model/hindi_bpe_32000.model
```

Tokenizes `final/train.txt` / `validation.txt` / `test.txt` into `uint32`
binary token-id files under `hindi/data/tokenized/`. When tokenizing
`train.txt`, also streams `indiccorp_train.txt`/`wikipedia_train.txt`/
`news_train.txt` with the same tokenizer to print/save a source
breakdown (`"source_breakdown"` key in `tokenization_metadata.json`).

## Complete command sequence

```bash
python hindi/scripts/01_download_indiccorp.py --config hindi/configs/hindi_data.yaml --resume

python hindi/scripts/manual/01_download_wikipedia.py --config hindi/configs/hindi_data.yaml
python hindi/scripts/manual/02_extract_wikipedia.py --config hindi/configs/hindi_data.yaml

python hindi/scripts/manual/03_collect_news.py --config hindi/configs/hindi_data.yaml
python hindi/scripts/manual/05_aggregate_news_corpus.py --config hindi/configs/hindi_data.yaml

python hindi/scripts/02_normalize.py --config hindi/configs/hindi_data.yaml --source indiccorp
python hindi/scripts/02_normalize.py --config hindi/configs/hindi_data.yaml --source wikipedia
python hindi/scripts/02_normalize.py --config hindi/configs/hindi_data.yaml --source news

python hindi/scripts/03_filter_hindi.py --config hindi/configs/hindi_data.yaml --source indiccorp
python hindi/scripts/03_filter_hindi.py --config hindi/configs/hindi_data.yaml --source wikipedia
python hindi/scripts/03_filter_hindi.py --config hindi/configs/hindi_data.yaml --source news

python hindi/scripts/04_quality_filter.py --config hindi/configs/hindi_data.yaml --source indiccorp
python hindi/scripts/04_quality_filter.py --config hindi/configs/hindi_data.yaml --source wikipedia
python hindi/scripts/04_quality_filter.py --config hindi/configs/hindi_data.yaml --source news

python hindi/scripts/05_deduplicate.py --config hindi/configs/hindi_data.yaml
# Mac: python hindi/scripts/05_deduplicate_mac.py --config hindi/configs/hindi_data.yaml --in-memory

python hindi/scripts/06_statistics.py --config hindi/configs/hindi_data.yaml

python hindi/scripts/07_split_dataset.py --config hindi/configs/hindi_data.yaml

python hindi/scripts/08_decontaminate.py --config hindi/configs/hindi_data.yaml
# Mac: python hindi/scripts/08_decontaminate_ram.py --config hindi/configs/hindi_data.yaml

python hindi/scripts/10_concatenate_final_corpus.py --config hindi/configs/hindi_data.yaml

# Train candidate tokenizer variants (16K / 32K / 48K)
python hindi/scripts/09_train_tokenizer.py --config hindi/configs/hindi_data.yaml --vocab-sizes 16000 32000 48000
# Evaluate the candidate variants and select ONE final tokenizer
python hindi/scripts/11_evaluate_tokenizer.py --config hindi/configs/hindi_data.yaml

# Then tokenize using ONLY the selected tokenizer (example: 32K)
python hindi/scripts/12_tokenize_corpus.py --config hindi/configs/hindi_data.yaml --tokenizer-model hindi/tokenizer/model/hindi_bpe_32000.model
```

## Token accounting notes

- **Content tokens vs. BOS/EOS**: every target (400M/100M/500M) means
  SentencePiece content tokens only. BOS/EOS are never counted toward a
  target.
- **Train vs. validation/test**: only `train.txt`'s content-token count is
  compared against the targets. `validation.txt`/`test.txt` are reported
  separately, never added into the target comparison.
- **Manual fraction**: `manual tokens / total tokens >= 0.20`, measured
  from actual BPE content tokens, never whitespace words.

## Pre-run checklist

- [ ] `hindi_data.yaml`'s `indiccorp.source_urls` points only to
      IndicCorpV2 Hindi sources (`hi-*.txt`).
- [ ] `manual.min_fraction` is `0.20`, `manual.enforce` is `true`.
- [ ] `manual.wikipedia.enabled` is `true` and the dump has been collected.
- [ ] `manual.news` sources are configured as intended (`enabled: true/false`
      per source).
- [ ] Enough manual material has plausibly been collected toward the
      100M-manual-token target — check with step 6's stats before running
      the expensive later stages, since some news sources are
      rolling-window only and may need repeated collection runs.
- [ ] `deduplicated/indiccorp_hi_deduplicated.txt` exists, and
      `wikipedia_hi_deduplicated.txt`/`news_hi_deduplicated.txt` each
      exist iff that source is enabled.

## Notes

- `hindi/scripts/collect_manual.py` and `hindi/scripts/collect_public.py`
  have been removed — they were unused stubs, not part of this workflow.
- There is no independent post-hoc leakage-reverification stage (the
  Hindi equivalent of Nepali's now-removed verification step was also
  removed) — `08_decontaminate.py`'s own pass against frozen
  validation/test is the one authoritative check.
