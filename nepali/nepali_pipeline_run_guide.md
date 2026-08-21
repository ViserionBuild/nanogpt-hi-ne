# Nepali Corpus Pipeline — Mac M4 + 24 GB RAM

## Pipeline order

```text
1a. 01_download_indiccorp.py
1b. collect_news.py     (Onlinekhabar + Khabarhub — independent of 1a)
1c. convert_news_jsonl_to_raw.py
2. 02_normalize.py
3. 03_filter_devanagari_script.py
4. 04_quality_filter.py
5. 05_deduplicate_mac.py   <-- Mac-specific
6. 06_corpus_statistics.py
7. 07_split_train_val_test.py
8. 08_decontaminate.py
9. 09_concatenate_final_corpus.py
10. 10_train_tokenizer.py
11. 11_evaluate_tokenizer.py
12. 12_tokenize_corpus.py
```

TWO SOURCES ONLY: `indiccorp` (public, AI4Bharat IndicCorpV2) and manual
news (`onlinekhabar` + `khabarhub`, each collected independently via the
same open WordPress REST API mechanism). Wikipedia and Setopati were both
removed from the active pipeline (see `collect_news.py`'s module
docstring for the full rationale).

Run all commands from the repository root.

## 1a. Download IndicCorp

```bash
python nepali/scripts/01_download_indiccorp.py \
    --config nepali/configs/nepali_data.yaml \
    --resume
```

## 1b. Collect news

Independent of 1a — can run before, after, or in parallel with it.

```bash
python nepali/scripts/manual/collect_news.py \
    --config nepali/configs/nepali_data.yaml
```

Optional per-source caps:

```bash
python nepali/scripts/manual/collect_news.py \
    --config nepali/configs/nepali_data.yaml \
    --max_onlinekhabar 20000 \
    --max_khabarhub 20000
```

## 2. Convert JSONL to raw text

```bash
python nepali/scripts/manual/convert_news_jsonl_to_raw.py \
    --config nepali/configs/nepali_data.yaml
```

Expected news raw files:

```text
onlinekhabar_ne_raw.txt
khabarhub_ne_raw.txt
```

## 3. Normalize each source

```bash
python nepali/scripts/02_normalize.py \
    --config nepali/configs/nepali_data.yaml \
    --source indiccorp
```

```bash
python nepali/scripts/02_normalize.py \
    --config nepali/configs/nepali_data.yaml \
    --source onlinekhabar
```

```bash
python nepali/scripts/02_normalize.py \
    --config nepali/configs/nepali_data.yaml \
    --source khabarhub
```

## 4. Filter

```bash
python nepali/scripts/03_filter_devanagari_script.py \
    --config nepali/configs/nepali_data.yaml \
    --source indiccorp
```

```bash
python nepali/scripts/03_filter_devanagari_script.py \
    --config nepali/configs/nepali_data.yaml \
    --source onlinekhabar
```

```bash
python nepali/scripts/03_filter_devanagari_script.py \
    --config nepali/configs/nepali_data.yaml \
    --source khabarhub
```

## 5. Quality filter

```bash
python nepali/scripts/04_quality_filter.py \
    --config nepali/configs/nepali_data.yaml \
    --source indiccorp
```

```bash
python nepali/scripts/04_quality_filter.py \
    --config nepali/configs/nepali_data.yaml \
    --source onlinekhabar
```

```bash
python nepali/scripts/04_quality_filter.py \
    --config nepali/configs/nepali_data.yaml \
    --source khabarhub
```

## 6. Mac-optimized deduplication

**Use the Mac version on the M4 + 24 GB machine:**

```bash
python nepali/scripts/05_deduplicate_mac.py \
    --config nepali/configs/nepali_data.yaml \
    --in-memory
```

Do not run the normal `05_deduplicate.py` as well.

Before the full run, benchmark the Mac version on a sample using
`--sample-lines` (e.g. `--sample-lines 500000`).

Monitor:

- lines/sec
- RAM usage
- candidates/document
- comparisons
- near duplicates removed
- sub-document duplicates removed
- elapsed time

## 7. Statistics

Reports on the `deduplicated` stage only (per source + combined) — the
earlier raw/normalized/filtered/quality stages are already gone by this
point (auto-deleted by 02/03/05 as each finishes), and normalize
(script 02) already records its own raw-stage line counts in
`run_stats/`.

```bash
python nepali/scripts/06_corpus_statistics.py \
    --config nepali/configs/nepali_data.yaml
```

## 8. Split dataset

```bash
python nepali/scripts/07_split_train_val_test.py \
    --config nepali/configs/nepali_data.yaml
```

## 9. Decontaminate

For MAC:

```bash
python nepali/scripts/08_decontaminate_ram.py \
    --config nepali/configs/nepali_data.yaml
```

Otherwise:

```bash
python nepali/scripts/08_decontaminate.py \
    --config nepali/configs/nepali_data.yaml
```

## 10. Concatenate final corpus

```bash
python nepali/scripts/09_concatenate_final_corpus.py \
    --config nepali/configs/nepali_data.yaml
```

Combines the decontaminated per-source files (indiccorp, onlinekhabar,
khabarhub) into `final/train.txt` — must run BEFORE tokenizer training
(step 11), since the tokenizer trains on this script's output.

## 11. Train tokenizer

There is **one final/production tokenizer**, but this stage can train multiple **candidate vocabulary-size variants** for comparison.

For example, to train 6k, 8K, 12K, and 16K candidates:

```bash
python nepali/scripts/10_train_tokenizer.py \
    --config nepali/configs/nepali_data.yaml \
    --vocab-sizes 6000 8000 10000 12000 14000 16000
```

This command trains **4 tokenizer variants** from the same `final/train.txt`:

```text
final/train.txt
       |
       +--> SentencePiece BPE 6K
       |
       +--> SentencePiece BPE 8K
       |
       +--> SentencePiece BPE 12K
       |
       +--> SentencePiece BPE 16K
```

These are **candidate tokenizers**, not three production tokenizers. The next step (script 11 in this guide's numbering, `11_evaluate_tokenizer.py`) evaluates the variants using tokenizer metrics such as fertility, UNK rate, characters-per-token, and token-frequency statistics. After evaluation, select **one** vocabulary size as the final tokenizer.

If you already know the desired vocabulary size, you can train only one candidate, for example:

```bash
python nepali/scripts/10_train_tokenizer.py \
    --config nepali/configs/nepali_data.yaml \
    --vocab-sizes 12000
```

Once one tokenizer is selected, script 12 (`12_tokenize_corpus.py`) must use that selected model for the actual final corpus tokenization. Do **not** tokenize the production corpus separately with all three candidates.

## 12. Evaluate tokenizer

```bash
python nepali/scripts/11_evaluate_tokenizer.py \
    --config nepali/configs/nepali_data.yaml
```

## 13. Tokenize corpus

```bash
python nepali/scripts/12_tokenize_corpus.py \
    --config nepali/configs/nepali_data.yaml
```
OR
```bash
python nepali/scripts/12_tokenize_corpus.py \
    --config nepali/configs/nepali_data.yaml \
    --tokenizer-model nepali/tokenizer/model/nepali_bpe_12000.model
```

Step 13 (`12_tokenize_corpus.py`) is the final step in the pipeline.
There is no separate post-hoc analysis/verification stage —
`10_train_tokenizer.py`'s own token-count report is the one authoritative
check.

## Pre-production checks

Before the full Mac run:

```bash
python -m py_compile \
    nepali/scripts/05_deduplicate_mac.py \
    shared/near_dup_mac.py
```

Then:

```bash
python nepali/scripts/05_deduplicate_mac.py --help
```

Validate the Mac deduplication implementation on a small deterministic corpus against the existing implementation before processing the full corpus.

## Complete command sequence

```bash
python nepali/scripts/01_download_indiccorp.py --config nepali/configs/nepali_data.yaml --resume

python nepali/scripts/manual/collect_news.py --config nepali/configs/nepali_data.yaml

python nepali/scripts/manual/convert_news_jsonl_to_raw.py --config nepali/configs/nepali_data.yaml

python nepali/scripts/02_normalize.py --config nepali/configs/nepali_data.yaml --source indiccorp
python nepali/scripts/02_normalize.py --config nepali/configs/nepali_data.yaml --source onlinekhabar
python nepali/scripts/02_normalize.py --config nepali/configs/nepali_data.yaml --source khabarhub

python nepali/scripts/03_filter_devanagari_script.py --config nepali/configs/nepali_data.yaml --source indiccorp
python nepali/scripts/03_filter_devanagari_script.py --config nepali/configs/nepali_data.yaml --source onlinekhabar
python nepali/scripts/03_filter_devanagari_script.py --config nepali/configs/nepali_data.yaml --source khabarhub

python nepali/scripts/04_quality_filter.py --config nepali/configs/nepali_data.yaml --source indiccorp
python nepali/scripts/04_quality_filter.py --config nepali/configs/nepali_data.yaml --source onlinekhabar
python nepali/scripts/04_quality_filter.py --config nepali/configs/nepali_data.yaml --source khabarhub

python nepali/scripts/05_deduplicate_mac.py --config nepali/configs/nepali_data.yaml --in-memory

python nepali/scripts/06_corpus_statistics.py --config nepali/configs/nepali_data.yaml

python nepali/scripts/07_split_train_val_test.py --config nepali/configs/nepali_data.yaml

python nepali/scripts/08_decontaminate_ram.py --config nepali/configs/nepali_data.yaml

python nepali/scripts/09_concatenate_final_corpus.py --config nepali/configs/nepali_data.yaml

# Train candidate tokenizer variants (16K / 32K / 48K)
python nepali/scripts/10_train_tokenizer.py --config nepali/configs/nepali_data.yaml --vocab-sizes 6000 8000 10000 12000 14000 16000
# Evaluate the candidate variants and select ONE final tokenizer
python nepali/scripts/11_evaluate_tokenizer.py --config nepali/configs/nepali_data.yaml

# Then tokenize using ONLY the selected tokenizer (example: 32K)
python nepali/scripts/12_tokenize_corpus.py --config nepali/configs/nepali_data.yaml --tokenizer-model nepali/tokenizer/model/nepali_bpe_32000.model
```

**Important:** `05_deduplicate_mac.py` and `shared/near_dup_mac.py` are the only Mac-specific components (the exact-duplicate logic is shared verbatim with `05_deduplicate.py`; only the near-duplicate index storage differs — in-memory vs. SQLite). The rest of the pipeline is unchanged between platforms.
