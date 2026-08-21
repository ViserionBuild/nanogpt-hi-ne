[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Monolingual Transformer LMs — Hindi (Model H) & Nepali (Model L)

Two independently trained, from-scratch, decoder-only Transformer language
models (~25M params each), one for Hindi (higher-resource) and one for
Nepali (lower-resource). No pretrained weights, tokenizers, or shared
vocabulary between the two — completely separate pipelines under
`hindi/` and `nepali/`, sharing only generic, language-agnostic code in
`shared/` (model architecture classes, metric functions, generic data
utilities). See course spec (`LMA_Individual_Project_v1.pdf`) for full
requirements.

## Status of this repo

This is a **complete, runnable pipeline skeleton** — every stage (data
cleaning, tokenizer training, model, pretraining loop with checkpoint
resume, synthetic reasoning generation, finetuning, and all evaluation:
PPL/BPB, BLEU/chrF/ROUGE-L, diversity, attention heatmaps/entropy/distance)
is implemented and has been smoke-tested end-to-end on tiny sample data
(see `smoke_test_report.md`). What is **not** done yet, and requires your
own work:

- [ ] Real data collection at ~500M-token scale for both languages (public
      + your own manual OCR/scraping — see `hindi/scripts/collect_public.py`
      and `collect_manual.py`, and their Nepali equivalents)
- [ ] Actually running pretraining to convergence (needs a Colab GPU over
      real time — the loop supports resuming across sessions)
- [ ] Reviewing/refining the Hindi and Nepali reasoning-data templates in
      `*/finetune/generate_reasoning_data.py` for natural phrasing (they
      are functional but were not written by a native speaker)
- [ ] Uploading real checkpoints/datasets to Google Drive and filling in
      the links below
- [ ] Writing the actual per-phase reports in `report/`

Additional pieces implemented beyond the initial skeleton: an explicit
causal-masking verification script (`*/eval/verify_causal_masking.py`,
sweeps every sequence position and confirms perturbing token t+1 never
changes logits at position ≤ t — required by the spec), and a
bonus-ablation comparison script (`bonus_ablation/eval_ablation.py`) that
evaluates the standard vs. no-positional-embedding models side by side.

## Repository layout

```
LMA_IDV_Proj/
├── README.md                  # this file — reproduction steps + Drive links
├── requirements.txt
├── shared/                    # language-agnostic code only — no data, no weights
│   ├── data_utils.py            # generic corpus I/O helpers
│   ├── near_dup.py              # MinHash/LSH near-duplicate index (SQLite-backed)
│   ├── near_dup_mac.py          # same interface, in-memory index
│   ├── dedup_accounting.py      # dedup bookkeeping / reporting
│   ├── run_stats.py             # per-stage run-summary JSON writer
│   └── tokenizer_utils.py       # SentencePiece load/encode helpers
│
├── hindi/                     # Model H — fully self-contained
│   ├── configs/hindi_data.yaml   # every pipeline threshold, path, and target
│   ├── scripts/                  # 01–12, run in order (see Phase 1 below)
│   │   ├── 01_download_indiccorp.py … 12_tokenize_corpus.py
│   │   └── manual/               # Wikipedia dump + news-outlet collection
│   ├── tokenizer/
│   │   ├── model/                # *.model — trained SentencePiece BPE (tracked)
│   │   ├── vocab/                # *.vocab — vocabulary files (tracked)
│   │   └── config/               # per-vocab-size training config (tracked)
│   ├── metadata/                 # corpus/dedup/decontam/split/tokenizer reports (tracked)
│   ├── data/
│   │   ├── run_stats/            # one JSON per pipeline-stage run (tracked)
│   │   ├── intermediate/         # transient stage outputs (gitignored)
│   │   ├── final/                # train/validation/test .txt (gitignored → Drive)
│   │   └── tokenized/            # *.bin uint32 token IDs (gitignored → Drive)
│   └── reports/figures/          # per-language tokenizer plots
│
├── nepali/                    # Model L — same structure, own data/tokenizer/vocab
│
└── report/                    # GRADED FROM HERE
    ├── phase1_report.md
    └── figures/
```

**What is tracked vs. what is on Drive.** Code, configs, all statistics/report
JSON, tokenizer models and vocabularies, and the report with its figures are
committed. Corpora (`data/final/*.txt`), tokenized binaries
(`data/tokenized/*.bin`), raw downloads, and transient intermediate stages are
gitignored for size — the corpora are in the Drive folders linked below.

**Note on script numbering.** The two languages run the same twelve logical
stages, but Hindi and Nepali swap the order of *concatenate* and *train
tokenizer*: Hindi is `09_train_tokenizer` / `10_concatenate_final_corpus`,
Nepali is `09_concatenate_final_corpus` / `10_train_tokenizer`. Both train the
tokenizer on the final concatenated `train.txt`; only the filenames differ.
Hindi additionally has `manual/05_aggregate_news_corpus.py` to merge multiple
news outlets into one corpus, which Nepali does not need.

Only the Phase 1 pipeline exists in this repository so far; the
`train/`, `eval/`, `finetune/`, and `checkpoints/` directories referenced in the
Phase 2/3 commands below have not been created yet.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduction steps (per phase)

### Phase 1 — Data & Tokenizer

Each language has its own fully-streaming, from-scratch corpus pipeline
under `hindi/scripts/` and `nepali/scripts/`: download public data
(AI4Bharat IndicCorpV2), collect manual data (Hindi: Wikipedia + news
outlets; Nepali: news via a shared WordPress API), then run every source
through the same preprocessing stages — Unicode normalization,
language-script filtering, quality filtering, cross-source
deduplication (exact + MinHash/LSH near-dup) — before splitting into
train/validation/test, decontaminating the train split against the
frozen validation/test sets, and concatenating everything into the final
training corpus. A single SentencePiece BPE tokenizer is then trained
from scratch on that final corpus (never on validation/test), evaluated
across candidate vocabulary sizes (fertility, UNK rate, chars/token), and
used to tokenize the final corpus into binary token-id files. No
pretrained tokenizer or vocabulary is used, and Hindi/Nepali never share
data, tokenizer, or vocabulary at any stage.

Full step-by-step run instructions (exact commands, config keys, targets,
and pre-run checklists) are documented separately per language:

- **Hindi**: [`hindi/Hindi_Corpus_Pipeline_Run_Instructions.md`](hindi/Hindi_Corpus_Pipeline_Run_Instructions.md)
- **Nepali**: [`nepali/nepali_pipeline_run_guide.md`](nepali/nepali_pipeline_run_guide.md) 

## Google Drive links

Large artifacts (raw corpora, cleaned final corpora) are not committed to git —
see `.gitignore`. They live in the per-language Drive folders below.

### Phase 1 — datasets

| Artifact | Link |
|---|---|
| **Hindi (Model H)** — raw + final corpus | https://drive.google.com/drive/folders/1q2lJ2yE4S5MIJbcyKl9PsZo2VQP9qs7C?usp=sharing |
| **Nepali (Model L)** — raw + final corpus | https://drive.google.com/drive/folders/1YbEHnHQ9fM2apbD4US284-_USKcwmIz4?usp=sharing |

Each folder holds that language's raw collected data and the final
`train.txt` / `validation.txt` / `test.txt` splits. Tokenizer model and
vocabulary files are committed directly to this repository under
`<language>/tokenizer/`, so no Drive link is needed for them.

