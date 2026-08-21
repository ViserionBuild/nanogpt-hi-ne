
## 5. Mac-optimized deduplication

**Use the Mac version on the M4 + 24 GB machine:**

```bash
python hindi/scripts/05_deduplicate_mac.py \
    --config hindi/configs/hindi_data.yaml \
    --in-memory
```

## 6. Statistics

Reports on the `deduplicated` stage only (per source + manual summary +
combined total) — the earlier raw/normalized/filtered/quality stages are
already gone by this point (auto-deleted by 02/03/04 as each finishes).

```bash
python hindi/scripts/06_statistics.py \
    --config hindi/configs/hindi_data.yaml
```

## 7. Split dataset

```bash
python hindi/scripts/07_split_dataset.py \
    --config hindi/configs/hindi_data.yaml
```

## 8. Decontaminate

For MAC:

```bash
python hindi/scripts/08_decontaminate_ram.py \
    --config hindi/configs/hindi_data.yaml
```

## 9. Concatenate final corpus

```bash
python hindi/scripts/10_concatenate_final_corpus.py \
    --config hindi/configs/hindi_data.yaml
```

Combines the decontaminated per-source files (indiccorp, wikipedia, news)
into `final/train.txt` — must run BEFORE tokenizer training (step 10),
since the tokenizer trains on this script's output.

## 10. Train tokenizer

There is **one final/production tokenizer**, but this stage can train multiple **candidate vocabulary-size variants** for comparison.

For example, to train 16K, 32K, and 48K candidates:

```bash
python hindi/scripts/09_train_tokenizer.py \
    --config hindi/configs/hindi_data.yaml \
    --vocab-sizes 8000 10000 12000 14000 16000
```

If you already know the desired vocabulary size, you can train only one candidate, for example:

## 11. Evaluate tokenizer

```bash
python hindi/scripts/11_evaluate_tokenizer.py \
    --config hindi/configs/hindi_data.yaml
```

## 12. Tokenize corpus

```bash
python hindi/scripts/12_tokenize_corpus.py \
    --config hindi/configs/hindi_data.yaml \
    --tokenizer-model hindi/tokenizer/model/hindi_bpe_12000.model
```

This is the final step — `12_tokenize_corpus.py` is the last stage in the
pipeline. There is no separate post-hoc analysis/verification stage;
`09_train_tokenizer.py`'s own token-count report is the one authoritative
check.
