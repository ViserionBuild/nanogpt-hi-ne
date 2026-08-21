# Nepali Pipeline — Inputs / Outputs / Summary per Script

Reference table for what each script reads, what it writes (corpus files,
JSON reports, plots), and what it actually does. Run order top to bottom.
All paths relative to repo root.

Two shared conventions used throughout:

- **run_stats** — scripts `02`/`03`/`04` each write one JSON file to
  `nepali/data/run_stats/` per run, named `{stage}_{source}_{timestamp}.json`
  (e.g. `03_indiccorp_20260821T120000Z.json`), never overwritten.
- **disk-saving cascade** — scripts `03`, `04`, `05`/`05_mac` each delete
  their own immediate predecessor stage's file for a source right after
  successfully writing their own output (only when using the default
  config-resolved path, not a manual `--input`/`--output` override).

| # | Script | Input | Output (files / JSON / plots) | Summary |
|---|---|---|---|---|
| 1a | `01_download_indiccorp.py` | none (network: IndicCorpV2 `ne.txt`) | `raw/indiccorp/indiccorp_ne_500m_words_candidate.txt`, `..._selection_state.json` (resume state), `..._selection_report.json` | Downloads IndicCorpV2 Nepali via HTTP range chunks until ~470M whitespace words reached; resumable, verified via SHA-256. |
| 1b | `manual/collect_news.py` | none (network: Onlinekhabar + Khabarhub WordPress REST API) | `raw/manual/news/{onlinekhabar,khabarhub}_NNNNN.jsonl` shards, `collection_state.json`, `news_collection_summary.json` | Collects Nepali news articles from both open REST APIs, dedupes by URL, writes sharded JSONL; resumable per-source page cursor. |
| 1c | `manual/convert_news_jsonl_to_raw.py` | `raw/manual/news/{source}_*.jsonl` | `raw/manual/news/{onlinekhabar,khabarhub}_ne_raw.txt` | Flattens JSONL shards into one-document-per-line raw text for `02_normalize.py`. |
| 2 | `02_normalize.py` | `raw/.../{source}` raw text | `intermediate/normalized/{source}_ne_normalized.txt` + run_stats `02_{source}_*.json` | NFC-normalizes text, collapses whitespace/CRLF, strips unsafe control chars (keeps ZWNJ/ZWJ). |
| 3 | `03_filter_devanagari_script.py` | `normalized/{source}_ne_normalized.txt` | `filtered/{source}_ne_filtered.txt`, `{source}_ne_review.txt` (borderline lines) + run_stats `03_{source}_*.json` | Keeps lines with Devanagari ratio ≥0.70, routes 0.40–0.70 to review, drops the rest. **Deletes** the normalized-stage input after success. |
| 4 | `04_quality_filter.py` | `filtered/{source}_ne_filtered.txt` | `quality/{source}_ne_quality.txt`, `{source}_ne_quality_rejected.txt` + run_stats `04_{source}_*.json` | Length/URL/HTML/repeat-char/punctuation heuristics + optional safety-word blocklist. **Deletes** filtered-stage input + its review file after success. |
| 5 | `05_deduplicate.py` (or `05_deduplicate_mac.py`) | `quality/{source}_ne_quality.txt` for all 3 sources, together | `deduplicated/{source}_ne_deduplicated.txt` per source; report `nepali/metadata/deduplication_report.json` (`_mac.json` for the Mac variant — different filename, won't clobber) | Cross-source exact (SHA-256) + near-dup (MinHash/LSH) dedup in one shared pass; precedence Onlinekhabar > Khabarhub > IndicCorp. Mac variant uses in-memory LSH instead of SQLite, otherwise identical output. **Deletes** quality-stage input + rejected file per source after success. |
| 6 | `06_corpus_statistics.py` | `deduplicated/{source}_ne_deduplicated.txt` (all 3) | `nepali/metadata/corpus_statistics.json` (per-source + manual + grand-total stats) | Descriptive stats on the deduplicated corpus only: lengths, char composition, vocabulary, top-100 words. No cleanup. |
| 7 | `07_split_train_val_test.py` | `deduplicated/{source}_ne_deduplicated.txt` (all 3) | `split/{source}_train_candidate.txt` per source; **frozen** `final/validation.txt`, `final/test.txt` (combined all 3 sources); `nepali/metadata/split_statistics.json` | Deterministic hash-bucket split (~90/5/5) per source; validation/test combined and frozen from here on. No cleanup. |
| 8 | `08_decontaminate.py` (or `08_decontaminate_ram.py`) | `split/{source}_train_candidate.txt` + frozen `final/validation.txt` / `test.txt` | `decontaminated/{source}_train_decontam.txt` per source; report `nepali/metadata/decontamination_report.json` (**same filename for both variants** — second run overwrites first) | Removes exact/near-dup matches of held-out val/test from each source's train-candidate. RAM variant is identical except its held-out index is in-memory instead of SQLite. No cleanup. |
| 9 | `09_concatenate_final_corpus.py` | `decontaminated/{source}_train_decontam.txt` (all 3) | `final/{source}_train.txt` per source; `final/manual_train.txt` (onlinekhabar+khabarhub); `final/train.txt` (indiccorp+manual — **tokenizer trains on this**); `nepali/metadata/corpus_combination_report.json` | Pure streaming concatenation, no subsampling. Refuses to ever overwrite a held-out filename. No cleanup. |
| 10 | `10_train_tokenizer.py` | `final/train.txt` | `nepali/tokenizer/model/nepali_bpe_{vocab}.model/.vocab` per size; `nepali/tokenizer/vocab/...vocab`; `nepali/tokenizer/config/..._config.json`; report `nepali/metadata/token_count_check.json` (per-source + total token counts vs. 400M/100M/500M targets, shortfall) | Trains ONE SentencePiece BPE tokenizer from scratch per requested vocab size (default 6K/8K/10K/12K/14K/16K), then reloads the default-vocab model to report real token counts against project targets. Never retrains to close a shortfall. |
| 11 | `11_evaluate_tokenizer.py` | all `nepali_bpe_*.model`; `final/validation.txt` | `nepali/metadata/tokenizer_statistics.json`; plots `nepali/reports/figures/token_frequency_distribution.png` (always) and `tokenizer_comparison.png` (only if ≥2 variants trained) | Evaluates/compares every trained variant: fertility, UNK rate, chars/token, token frequency, worked tokenization examples. This is where the fertility-vs-tokenizer bar chart comes from. |
| 12 | `12_tokenize_corpus.py` | `final/train.txt`, `validation.txt`, `test.txt` + chosen `.model` | `nepali/data/tokenized/{train,validation,test}.bin` (uint32 token IDs); `tokenization_metadata.json`; `train_content_token_counts.json` | Converts final text splits to binary token-ID files, streaming with a self-verify re-read pass, reports train-split content tokens vs. the 500M target with per-source breakdown. Pipeline's last step. |

## Where to look for what

- **"Did stage N actually run and how much survived?"** → `nepali/data/run_stats/` (stages 02–04 only; one JSON per run, never overwritten).
- **"What are my final token counts / did I hit the targets?"** → `nepali/metadata/token_count_check.json` (from step 10) is the authoritative source — it's the one report that's re-measured directly from the trained tokenizer, not inferred.
- **"How does fertility compare across vocab sizes?"** → `nepali/reports/figures/tokenizer_comparison.png` (from step 11) — only exists if you trained 2+ vocab sizes.
- **"Is my corpus clean/healthy before I commit to the expensive stages?"** → `nepali/metadata/corpus_statistics.json` (step 6), run right after dedup, before split/decontam/tokenizer.
- Everything written under `nepali/metadata/` is a **report**, never a corpus file — safe to delete and regenerate by re-running the producing script.
- Everything written under `nepali/data/intermediate/{normalized,filtered,quality}/` is **transient and self-deleting** — don't expect these to still exist after the next stage runs successfully. `raw/`, `deduplicated/`, `split/`, `decontaminated/`, `final/` are never auto-deleted by any script.
