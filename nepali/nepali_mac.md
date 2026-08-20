
## 6. Mac-optimized deduplication

**Use the Mac version on the M4 + 24 GB machine:**

```bash
python nepali/scripts/05_deduplicate_mac.py \
    --config nepali/configs/nepali_data.yaml \
    --in-memory
```

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

If you already know the desired vocabulary size, you can train only one candidate, for example:

## 12. Evaluate tokenizer

```bash
python nepali/scripts/11_evaluate_tokenizer.py \
    --config nepali/configs/nepali_data.yaml
```

## 13. Tokenize corpus

```bash
python nepali/scripts/12_tokenize_corpus.py \
    --config nepali/configs/nepali_data.yaml \
    --tokenizer-model nepali/tokenizer/model/nepali_bpe_12000.model
```

## 14. Analyze tokenizer + corpus

```bash
python nepali/scripts/13_analyze_tokenizer_and_corpus.py \
    --config nepali/configs/nepali_data.yaml
```

Final, read-only analysis/report — reads the outputs of every prior
stage and writes `tokenizer_corpus_analysis.json` / `.txt` (+ optional
figures) to `nepali/metadata/`. Does not modify anything.
