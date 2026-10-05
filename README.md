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
from scratch and evaluated. See [`report/phase1_report.md`](report/phase1_report.md)
and [`utility/phase1_workflow.md`](utility/phase1_workflow.md) for how it
was built.

**Phase 2 (model + pretraining + evaluation) is complete.** Both models
are trained end-to-end and evaluated on validation *and* test. See
[`report/phase2_report.md`](report/phase2_report.md) for the full results
and [`utility/phase2_workflow.md`](utility/phase2_workflow.md) for how it
was built.

| | Status |
|---|---|
| Corpora, splits, decontamination | done — 648.0M (Hindi) / 659.5M (Nepali) train tokens |
| Tokenizers (SentencePiece BPE, vocab 12,000 each) | done |
| Transformer implementation | done — **25,321,472** parameters, verified |
| Causal-masking verification | **passes**, max delta exactly 0.0 |
| Checkpoint/resume | **bit-exact**, verified by killing and resuming a run |
| Test suite | 22 gates, all passing (`tests/test_phase2.py`) |
| Pretraining Model H (Hindi) | **done** — 13,183 steps, ~2.6h on 2×T4 |
| Pretraining Model L (Nepali) | **done** — 13,417 steps, ~2.6h on 2×T4 |
| Phase 2 evaluation (validation + test, both models) | **done** — see [`report/phase2_report.md`](report/phase2_report.md) |
| Phase 3 synthetic reasoning dataset (both languages) | **done** — 2,400 train / 4×320 test examples each, see [`report/phase3_report.md`](report/phase3_report.md) §1 |
| Phase 3 reasoning finetuning (both models) | **done** — see [`report/phase3_report.md`](report/phase3_report.md) §2 |
| Phase 3 attention analysis (pretrained vs. finetuned) | **done** — see [`report/phase3_report.md`](report/phase3_report.md) §3 |
| Finetuned-checkpoint Drive links | **done** — see [Phase 3 — finetuned checkpoints](#phase-3--finetuned-checkpoints) below |
| Bonus ablation (no positional embeddings) | not started |

**Phase 3 (reasoning finetuning + attention analysis + final report) is
complete.** Both models are finetuned on their own synthetic comparative-
reasoning dataset and evaluated on the course's three-way accuracy rubric
(seen entity/wording, unseen entity, unseen wording). See
[`report/phase3_report.md`](report/phase3_report.md) for the full results
and [`utility/phase3_plan.md`](utility/phase3_plan.md) for the design
rationale, course PDFs consulted, and how a first attempt's memorization
failure was diagnosed and fixed.

## How to run Phase 2

Everything needed to reproduce the Phase 2 run lives under
[`utility/`](utility/):

| File | What it's for |
|---|---|
| [`utility/RUN_PHASE2.md`](utility/RUN_PHASE2.md) | step-by-step checklist, start here |
| [`utility/KAGGLE_SETUP.md`](utility/KAGGLE_SETUP.md) | why 2×T4 over other GPUs, dataset upload, troubleshooting |
| [`utility/phase2_workflow.md`](utility/phase2_workflow.md) | what Phase 1 handed over and how Phase 2 built on it |
| [`notebooks/phase2_kaggle.ipynb`](notebooks/phase2_kaggle.ipynb) | the notebook actually run on Kaggle |

The short version: upload `hindi/`, `nepali/`, and `shared/` as a Kaggle
Dataset (packaged by `utility/package_for_kaggle.sh`), attach it plus the
tokenized-corpus datasets to the notebook, set `LANGUAGE = 'hindi'` (or
`'nepali'`) in Cell 2, and Run All on a **GPU T4 x2** accelerator. Each
language takes one notebook session, ~2.6 hours. Locally-runnable
correctness gates (no GPU needed):

```bash
python hindi/train/count_params.py            # asserts 25,321,472 params
python hindi/eval/verify_causal_masking.py    # asserts max delta 0.0
python -m pytest tests/test_phase2.py -v      # 22 correctness gates
```

## How to run Phase 3

Runs entirely on CPU — the synthetic reasoning dataset is a few thousand
short sequences, not a several-hundred-million-token corpus, so no Kaggle
GPU session is needed (~15-25 minutes per language for finetuning, plus a
similar amount for evaluation, on an 8-core CPU with no GPU).

```bash
python hindi/reasoning_data/generate_dataset.py    # writes hindi/reasoning_data/*.json
python nepali/reasoning_data/generate_dataset.py
python hindi/finetune/finetune.py                  # resumes automatically from checkpoints_ft/latest.pt
python nepali/finetune/finetune.py
python hindi/eval/evaluate_reasoning.py --device cpu    # 3-way accuracy + perplexity before/after
python nepali/eval/evaluate_reasoning.py --device cpu
python hindi/eval/compare_attention.py --device cpu     # pretrained vs. finetuned attention
python nepali/eval/compare_attention.py --device cpu
python report/scripts/make_phase3_figures.py
python report/scripts/build_phase3_report.py       # report/phase3_report.md
python -m pytest tests/test_phase3.py -v           # 23 correctness gates
```

See [`report/phase3_report.md`](report/phase3_report.md) for the results
and [`utility/phase3_plan.md`](utility/phase3_plan.md) for the design
rationale — including a first finetuning attempt that memorized training
entities (0.000 accuracy on both held-out splits) and the fix that was
applied (entity pools enlarged 24→80 names, epochs/LR lowered, a per-epoch
unseen-entity accuracy probe added).

## Repository layout

```
LMA_IDV_Proj/
├── README.md                  # this file — reproduction steps + Drive links
├── requirements.txt
├── shared/                    # language-agnostic code only — no data, no weights
│   ├── data_utils.py, near_dup.py, dedup_accounting.py, run_stats.py,
│   │   tokenizer_utils.py       # Phase 1: corpus I/O, dedup, run-stats
│   ├── model/                    # Phase 2: attention.py, block.py, feedforward.py,
│   │                             #   transformer.py, positional.py, config.py
│   ├── train/                    # Phase 2: dataset.py, scheduler.py, checkpoint.py, trainer.py
│   ├── eval/                     # Phase 2/3: lm_metrics.py, gen_metrics.py, attention_tools.py,
│   │                             #   plots.py, reasoning_metrics.py (Phase 3: 3-way accuracy)
│   ├── reasoning/                 # Phase 3: entity pools, templates, chain sampling, dataset build
│   └── finetune/                  # Phase 3: SFT dataset (answer-only loss mask) + trainer
│
├── hindi/                     # Model H — fully self-contained
│   ├── configs/{hindi_data,hindi_model,hindi_finetune}.yaml  # every threshold, path, target
│   ├── scripts/                  # Phase 1: 01–12, run in order
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
│   ├── reasoning_data/             # Phase 3: lexicon.py + generated train/test JSON (tracked)
│   ├── finetune/finetune.py        # Phase 3: reasoning SFT entry point
│   ├── train/pretrain.py, count_params.py     # Phase 2
│   ├── eval/evaluate_lm.py, evaluate_generation.py, analyze_attention.py,
│   │   verify_causal_masking.py,              # Phase 2
│   │   evaluate_reasoning.py, compare_attention.py   # Phase 3
│   ├── eval/results/             # metric JSON (tracked) — Phase 2 + Phase 3
│   ├── checkpoints/               # Phase 2 best.pt, latest.pt (gitignored → Drive)
│   ├── checkpoints_ft/            # Phase 3 finetuned best.pt, latest.pt (gitignored → Drive)
│   └── logs/                     # training/finetune log CSVs + summaries (tracked)
│
├── nepali/                    # Model L — same structure, own data/tokenizer/vocab/weights
│
├── utility/                   # Kaggle run guides, packaging script, workflow docs
│   ├── RUN_PHASE2.md, KAGGLE_SETUP.md, package_for_kaggle.sh
│   └── phase1_workflow.md, phase2_workflow.md
│
├── notebooks/phase2_kaggle.ipynb  # the notebook run on Kaggle
│
└── report/                    # the report and its figures — read this first
    ├── phase1_report.md, phase2_report.md, phase3_report.md
    ├── scripts/                   # make_phase2/3_figures.py, build_phase2/3_report.py —
    │                               #   regenerate the figures/report from tracked logs and
    │                               #   eval JSON, no GPU needed for any of it
    └── figures/
        ├── hindi/, nepali/         # per-model figures (Phase 2 + Phase 3)
        └── comparison/             # both-model comparison figures (Phase 2 + Phase 3)
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

**`shared/model|train|eval/` is language-agnostic by construction, not just
by convention** — an AST-level test
(`tests/test_phase2.py::test_shared_code_never_names_a_language`) fails if
any file under `shared/` names either language in executable code. The two
models share architecture *code* only; data, tokenizer, vocabulary, and
weights stay fully separate in `hindi/` and `nepali/`. `finetune/` arrives
in Phase 3.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduction steps (per phase)

### Full pipeline, one command block

The consolidated end-to-end sequence, all three phases, per language
(replace `hindi` with `nepali` for Model L):

```bash
# Phase 1 (per language; replace hindi -> nepali for Model L)
python hindi/scripts/01_download_indiccorp.py  ...  python hindi/scripts/12_tokenize_corpus.py
  # (stages 01-12 in order; see hindi/Hindi_Corpus_Pipeline_Run_Instructions.md)
# Phase 2
python hindi/train/pretrain.py                       # resumes automatically from checkpoint
python hindi/eval/evaluate_lm.py --checkpoint hindi/checkpoints/best.pt
python hindi/eval/evaluate_generation.py --checkpoint hindi/checkpoints/best.pt
python hindi/eval/analyze_attention.py --checkpoint hindi/checkpoints/best.pt
# Phase 3
python hindi/reasoning_data/generate_dataset.py
python hindi/finetune/finetune.py                     # resumes automatically
python hindi/eval/evaluate_reasoning.py --device cpu  # 3-way accuracy + PPL before/after
python hindi/eval/compare_attention.py --device cpu
python report/scripts/make_phase2_figures.py && python report/scripts/make_phase3_figures.py
python report/scripts/build_phase2_report.py && python report/scripts/build_phase3_report.py
python -m pytest tests/test_phase2.py tests/test_phase3.py -v   # 22 + 23 correctness gates
```

Full per-phase step-by-step instructions, exact config keys, and all
Google Drive links (raw/final corpora, pretrained checkpoints, finetuned
checkpoints, both languages) are below and in the per-language run guides
linked throughout this file. Per-phase detail, all figures, and complete
tables live in [`report/phase1_report.md`](report/phase1_report.md),
[`report/phase2_report.md`](report/phase2_report.md),
[`report/phase3_report.md`](report/phase3_report.md).

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

### Phase 2 — checkpoints

| Artifact | Link |
|---|---|
| **Hindi (Model H)** — `best.pt`, `latest.pt` | https://drive.google.com/drive/folders/1lxbJCPTvOzcO2AihRrQJxxqBg58mVzXo?usp=sharing |
| **Nepali (Model L)** — `best.pt`, `latest.pt` | https://drive.google.com/drive/folders/18SYXQlJcohxqjVkoPfMMBfzXgjgUI5RN?usp=sharing |

Checkpoints are gitignored (~290 MB each) per the spec's "do not commit
large binary artifacts" rule. Training logs, loss curves, and every metric
JSON they produced are committed and do not require Drive —
see [`hindi/logs/`](hindi/logs/), [`nepali/logs/`](nepali/logs/), and
[`report/phase2_report.md`](report/phase2_report.md).

### Phase 3 — finetuned checkpoints

| Artifact | Link |
|---|---|
| **Hindi (Model H)** — finetuned `best.pt`, `latest.pt` | https://drive.google.com/drive/folders/1bCWxyz84Vt0Ppr0bZSSdKwVKfDpFgsD2?usp=sharing |
| **Nepali (Model L)** — finetuned `best.pt`, `latest.pt` | https://drive.google.com/drive/folders/1EPQZFpLFFBFa_yE1oonK8mQqk470M4sZ?usp=sharing |

Finetuned checkpoints are gitignored (~290 MB each, same rule as Phase 2)
and uploaded to the Drive folders above. Finetuning logs, loss curves, and
every metric JSON (reasoning accuracy, attention comparison) are committed
and do not require Drive — see [`hindi/logs/`](hindi/logs/),
[`nepali/logs/`](nepali/logs/), and
[`report/phase3_report.md`](report/phase3_report.md).

