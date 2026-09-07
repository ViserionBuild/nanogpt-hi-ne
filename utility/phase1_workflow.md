# Phase 1 Workflow — Two Corpora and Two Tokenizers, From Nothing

How Phase 1 actually happened: what each pipeline stage produced, in what
order, and what it handed to Phase 2. Written from the verified metadata
JSON and the graded report, not from the original plan — this is a
workflow document, not a design document.

For the graded content (full numbers, figures, discussion), see
[report/phase1_report.md](../report/phase1_report.md). This file is the
workflow behind that report. For what Phase 2 built on top of this
output, see [phase2_workflow.md](phase2_workflow.md).

---

## 1. Where Phase 1 starts: the two languages

**Model H — Hindi.** Chosen as the higher-resource language. AI4Bharat's
IndicCorpV2 ships Hindi as three separate shards (`hi-1/2/3.txt`) because
a single file would be unwieldy — direct evidence of abundance. That
headroom let the manual lane be built for source *diversity*
(Wikipedia + news) rather than to hit a volume target.

**Model L — Nepali.** Chosen from the spec's allowed lower-resource list.
IndicCorpV2 ships Nepali as a **single** `ne.txt` file (~14.8 GB
upstream) against Hindi's three shards — a measurable resource gap
confirmed via the HuggingFace API tree listing and an HTTP Range probe.
Nepali has no usable Wikisource wiki, which forced its manual lane onto
news APIs only.

**Both are Devanagari**, deliberately. Sharing a script means any Model
H vs. Model L difference measured in Phase 2 traces to resource level
and language, not to script or encoding effects.

---

## 2. The pipeline: twelve stages, run independently per language

Both languages run through the **same twelve-stage design**, but as two
completely separate executions — no code path shares data, tokenizer, or
vocabulary between them. `shared/` holds only language-agnostic utility
functions (dedup, run-stats), never data.

| # | Stage | What it does | Hindi script | Nepali script |
|---|---|---|---|---|
| 01 | Download / collect | IndicCorpV2 via resumable HTTP range chunks (SHA-256 verified); manual lane via Wikipedia dump extraction / news APIs | `01_download_indiccorp.py` | `01_download_indiccorp.py` |
| 02 | Normalize | NFC normalization, whitespace/CRLF collapse, unsafe control-char stripping | `02_normalize.py` | `02_normalize.py` |
| 03 | Script filter | Devanagari-ratio gate; borderline lines routed to review, not silently dropped | `03_filter_hindi.py` | `03_filter_devanagari_script.py` |
| 04 | Quality filter | Length, URL-ratio, HTML-tag, repeated-char, punctuation, digit-ratio heuristics | `04_quality_filter.py` | `04_quality_filter.py` |
| 05 | Deduplication | Cross-source exact (SHA-256) + near-duplicate (MinHash/LSH), source precedence | `05_deduplicate.py` | `05_deduplicate.py` |
| 06 | Corpus statistics | Descriptive stats on the deduplicated corpus, before any irreversible step | `06_statistics.py` | `06_corpus_statistics.py` |
| 07 | Split | Deterministic hash-bucket 90/5/5 (`blake2b(seed:line_index)`, seed 42); val/test **frozen** from here | `07_split_dataset.py` | `07_split_train_val_test.py` |
| 08 | Decontaminate | Removes exact/near-dup matches of the frozen val/test sets from train candidates | `08_decontaminate.py` | `08_decontaminate.py` |
| 09/10 | Concatenate + train tokenizer | Streams per-source train files into `train.txt`; trains SentencePiece BPE on it | `09_train_tokenizer.py` → `10_concatenate_final_corpus.py` | `09_concatenate_final_corpus.py` → `10_train_tokenizer.py` |
| 11 | Evaluate tokenizer | Fertility, chars/token, token-frequency curves, worked examples — on held-out validation | `11_evaluate_tokenizer.py` | `11_evaluate_tokenizer.py` |
| 12 | Tokenize corpus | Streams final splits into `uint32` `.bin` files with a self-verifying re-read pass | `12_tokenize_corpus.py` | `12_tokenize_corpus.py` |

Two design choices shape how every downstream number should be read:

**Split happens before decontamination and before the tokenizer trains.**
`validation.txt` / `test.txt` are written once at stage 07 and never
touched again; stage 08 then removes their matches from the *train* side
only. This means the tokenizer never sees a single held-out line, and
Phase 2's held-out perplexity is measured against genuinely unseen text
— not text the tokenizer was fit on.

**Deduplication has source precedence, favoring the manual lane.** When
identical or near-identical text exists in both a manual source and
IndicCorp, the manual copy is kept and the IndicCorp copy is dropped
(Hindi: Wikipedia > News > IndicCorp; Nepali: Onlinekhabar > Khabarhub >
IndicCorp). This protects the manual-token share at the public lane's
expense — the right direction, given the ≥20% manual-fraction target.

---

## 3. What each stage actually removed

### Hindi

| Stage | IndicCorp | Wikipedia | News | Total |
|---|---|---|---|---|
| normalized (02) | 8,787,343 | 936,593 | 340,892 | 10,064,828 |
| post script-filter (03) | 8,779,002 | 884,851 | 283,287 | 9,947,140 |
| post quality-filter (04) | 8,742,935 | 820,086 | 277,305 | 9,840,326 |
| post dedup (05) | 8,520,630 | 672,411 | 147,487 | 9,340,528 |
| train candidate (07) | 7,668,144 | 605,562 | 132,808 | 8,406,514 |
| post decontam (08) | 7,169,358 | 575,485 | 85,172 | **7,830,015** |

Deduplication removed 5.08% (242,381 exact + 257,417 near-duplicates).
Decontamination then removed 576,499 near-duplicate matches of the frozen
val/test sets — News lost the most proportionally (132,808 → 85,172, 36%)
because news outlets syndicate and re-run stories, so a train-side
article is often a near-copy of a held-out one.

### Nepali

| Stage | IndicCorp | Onlinekhabar | Khabarhub | Total |
|---|---|---|---|---|
| normalized (02) | 14,946,717 | 297,891 | 257,041 | 15,501,649 |
| post script-filter (03) | 14,855,758 | 297,649 | 256,792 | 15,410,199 |
| post quality-filter (04) | 14,821,096 | 288,229 | 254,103 | 15,363,428 |
| post dedup (05) | 13,329,271 | 282,629 | 233,910 | 13,845,810 |
| train candidate (07) | 11,996,006 | 254,527 | 210,628 | 12,461,161 |
| post decontam (08) | 10,656,979 | 175,998 | 148,749 | **10,981,726** |

Deduplication removed 9.88% — roughly double Hindi's rate, the expected
signature of a smaller-language crawl (less unique text on the open web
means more of it gets scraped repeatedly). Both news sources lost ~30%
of their train candidates to decontamination, the same syndication
effect seen in Hindi.

---

## 4. What each pipeline handed to the tokenizer

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| IndicCorp tokens | 517,934,945 (81.9%) | 520,693,966 (81.7%) |
| Manual tokens | 114,417,599 (18.1%) | 116,842,022 (18.3%) |
| **Total train content tokens** | **632,352,544** | **637,535,988** |
| vs. 500M target | **+26.5%** | **+27.5%** |
| Held-out validation | 467,228 lines / 39,781,899 stored tokens | 692,772 lines / 43,712,387 stored tokens |
| Held-out test | 466,786 lines / 39,702,931 stored tokens | 691,877 lines / 43,779,875 stored tokens |

Both corpora cleared the ~500M-token target and the ≥100M-manual-token
sub-target by a wide margin (Hindi manual +14.4%, Nepali manual +16.8%).
The one requirement not met: **manual fraction landed at 18.09% (Hindi) /
18.33% (Nepali), both under the spec's ≥20% floor** — disclosed directly
in the corpus metadata (`meets_manual_fraction_requirement: false`)
rather than rounded up or hidden. This was a closed decision by the time
Phase 2 started, and Phase 2 treats it as a fixed input.

**Why manual tokens outweigh their document share.** News documents are
full articles (mean 1,499–2,494 characters); IndicCorp lines are
sentence-to-paragraph fragments (mean 218–286 characters). One manual
document is worth roughly 7–10 IndicCorp documents in token terms — which
is why the manual lane reaches ~18% of *tokens* from only ~3–8% of
*documents*. Every manual-fraction number here is token-weighted, not
document-weighted, because a document-count reading would be badly
misleading.

---

## 5. Tokenizer construction: one BPE model per language, same recipe

Identical hyperparameters for both languages, so that any difference in
the result is a property of the language, not the configuration:

| Setting | Value |
|---|---|
| Algorithm | SentencePiece BPE |
| Character coverage | 0.9995 |
| Byte fallback | enabled |
| Special tokens | `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3 |
| Seed | 42 |

A sweep of candidate vocabulary sizes was trained and evaluated on
held-out validation text for **both** languages, and **12,000 was
selected for both** — not defaulted, chosen by where the fertility curve
flattens:

| Vocab size | Hindi fertility | Nepali fertility |
|---|---|---|
| 8,000 | 1.3823 | 1.5484 |
| 10,000 | 1.3361 | 1.4852 |
| **12,000 (selected)** | **1.3042** | **1.4394** |
| 14,000 | 1.2804 | 1.4043 |
| 16,000 | 1.2619 | 1.3766 |

Going 10K → 12K buys 0.032 fertility in Hindi and 0.046 in Nepali; 14K →
16K buys only 0.019 and 0.028 — the marginal return has already shrunk by
12K. Since parameters spent on vocabulary in Phase 2 are parameters taken
directly out of the transformer body at a fixed ~25M budget, 12,000 is
where the trade-off stops paying for itself in either language.
**Using the same size for both matters for Phase 2**: a perplexity
difference between Model H and Model L should not be explainable by one
having a larger vocabulary.

Unknown-token rate is **0.0000% for every candidate size, in both
languages** — byte fallback means any uncovered character decomposes
into byte tokens rather than emitting `<unk>`, so vocabulary size was
selected on fertility and compression return instead.

**The two languages are not equally compressible, and that gap is
morphology, not tokenizer quality.** Nepali reads higher fertility at
every matched vocabulary size (1.4394 vs. 1.3042 at 12k) — its
agglutinative case-marking fuses postpositions onto stems, generating far
more distinct surface forms per lemma. This is the same fact reflected in
Nepali's larger unique-word count (6.80M vs. Hindi's 3.96M) despite being
the lower-resource language.

---

## 6. What Phase 1 actually handed to Phase 2

| Artifact | Path | Tracked in git |
|---|---|---|
| Tokenized train/val/test | `{lang}/data/tokenized/{train,validation,test}.bin` | no — Drive/Kaggle |
| Tokenizer model + vocab | `{lang}/tokenizer/model/*.model`, `{lang}/tokenizer/vocab/*.vocab` | **yes** |
| Corpus + tokenizer statistics | `{lang}/metadata/*.json` | **yes** |
| Per-stage run summaries | `{lang}/data/run_stats/*.json` | **yes** |
| Held-out UTF-8 byte counts | recorded in `{lang}/configs/{lang}_model.yaml` | **yes** |

The `.bin` files are what make Phase 2 possible without touching raw
text again: a flat `uint32` array, documents concatenated as
`[BOS,...,EOS][BOS,...,EOS]...`, read directly via `np.memmap` so a
multi-GB corpus is never fully loaded into RAM. Every number in
[phase2_workflow.md](phase2_workflow.md) §1 traces back to this table.

---

## 7. Deliverable checklist (spec §1)

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts (both languages) | `{hindi,nepali}/scripts/01_*.py` |
| 2 | Dataset preprocessing pipelines | `{hindi,nepali}/scripts/02-08_*.py` |
| 3 | Per-language dataset statistics reports | `{hindi,nepali}/metadata/*.json`, `report/phase1_report.md` |
| 4 | Per-language train/validation/test splits | `{hindi,nepali}/data/tokenized/*.bin` (Drive), split at stage 07 |
| 5 | Tokenizer training code (both languages) | `{hindi,nepali}/scripts/09-10_*.py` |
| 6 | Vocabulary files (one per language) | `{hindi,nepali}/tokenizer/vocab/*.vocab` |
| 7 | Tokenizer model files (one per language) | `{hindi,nepali}/tokenizer/model/*.model` |

---

## 8. Where automation substitutes for human judgment

Carried forward because it still applies to everything Phase 2 built on
top of this corpus. Every threshold in this pipeline was set
**algorithmically or by convention**, not by a fluent reader inspecting
kept and dropped text:

- Script-filter ratios (Hindi ≥0.60, Nepali ≥0.70 Devanagari) were round
  defaults, not tuned against sampled text. Review-band files were
  written but never manually read.
- The near-duplicate threshold (0.4 similarity, 32 MinHash permutations)
  is deliberately aggressive — it may remove text a human would consider
  distinct.
- **No human — fluent in Hindi or Nepali — read the kept text.** Volume
  and character-level plausibility are verified; content quality,
  factual coherence, and register are not.

This preserves scale and reproducibility but is not a substitute for a
fluent reader's judgment, and it is the caveat that carries forward
unchanged into every later phase's qualitative claims.
