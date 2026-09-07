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

**Phase 1 (data + tokenizer) is complete.** Both corpora are collected,
cleaned, split, decontaminated, and tokenized; both tokenizers are trained
from scratch and evaluated. See [`report/phase1_report.md`](report/phase1_report.md).

**Phase 2 (model + pretraining + evaluation) is implemented but not yet
trained.** The architecture, training loop, and full evaluation suite are
written and their correctness gates pass; no GPU run has happened yet.

| | Status |
|---|---|
| Corpora, splits, decontamination | done — 648.0M (Hindi) / 659.5M (Nepali) train tokens |
| Tokenizers (SentencePiece BPE, vocab 12,000 each) | done |
| Transformer implementation | done — **25,321,472** parameters, verified |
| Causal-masking verification | **passes**, max delta exactly 0.0 |
| Checkpoint/resume | **bit-exact**, verified by killing and resuming a run |
| Test suite | 22 gates, all passing (`tests/test_phase2.py`) |
| Pretraining Model H / Model L | **not started** — ~2-3 GPU-hours each on 2x T4 |
| Phase 2 evaluation results | pending training |
| Phase 3 (reasoning finetuning) | not started |
| Bonus ablation (no positional embeddings) | not started |

To run Phase 2, follow [`RUN_PHASE2.md`](RUN_PHASE2.md) — a step-by-step
checklist — with [`notebooks/phase2_kaggle.ipynb`](notebooks/phase2_kaggle.ipynb)
on Kaggle. [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) explains the reasoning and
covers troubleshooting.

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

Phase 2 adds `shared/model/`, `shared/train/`, and `shared/eval/` (all
language-agnostic — the two models share architecture *code* but no data,
tokenizer, vocabulary, or weights), plus per-language `train/`, `eval/`,
`checkpoints/`, and `logs/` directories. `finetune/` arrives in Phase 3.

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

