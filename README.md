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
lma-project/
├── shared/            # model architecture, metrics, generic data/tokenizer utils (code only — no data/weights shared)
├── hindi/              # Model H — fully self-contained
│   ├── data/{raw,manual,processed,splits,stats}
│   ├── scripts/         # collection, cleaning, splitting
│   ├── tokenizer/        # SentencePiece training + artifacts
│   ├── train/            # pretrain.py (checkpoint-resumable)
│   ├── finetune/          # synthetic reasoning data + finetune.py
│   ├── eval/              # PPL/BPB, generation metrics, attention analysis, reasoning eval
│   ├── checkpoints/        # LOCAL — gitignored, Drive links below
│   └── configs/
├── nepali/             # Model L — identical structure
├── bonus_ablation/      # optional: no-positional-embedding ablation
└── report/              # phase reports + all figures/tables (graded from here)
```

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

### Phase 2 — Model, Pretraining, Evaluation
```bash
python -m hindi.train.pretrain --config hindi/configs/pretrain_config.yaml
python -m hindi.eval.compute_ppl_bpb --ckpt hindi/checkpoints/step_100000.pt --split test
python -m hindi.eval.generate_and_score --ckpt hindi/checkpoints/step_100000.pt
python -m hindi.eval.attention_analysis --ckpt hindi/checkpoints/step_100000.pt --tag pretrained
python -m hindi.eval.verify_causal_masking --ckpt hindi/checkpoints/step_100000.pt

# repeat with nepali.* equivalents
```

### Phase 3 — Reasoning Finetuning, Attention, Final Report
```bash
python -m hindi.finetune.generate_reasoning_data
python -m hindi.finetune.finetune --pretrained_ckpt hindi/checkpoints/step_100000.pt
python -m hindi.eval.reasoning_eval \
    --pretrained_ckpt hindi/checkpoints/step_100000.pt \
    --finetuned_ckpt hindi/checkpoints/finetune_epoch_9.pt
python -m hindi.eval.attention_analysis --ckpt hindi/checkpoints/finetune_epoch_9.pt --tag finetuned

# repeat with nepali.* equivalents
```

### Bonus (optional)
```bash
python -m bonus_ablation.train_ablation --config bonus_ablation/ablation_config.yaml
python -m bonus_ablation.eval_ablation \
    --standard_config hindi/configs/pretrain_config.yaml --standard_ckpt hindi/checkpoints/step_100000.pt \
    --ablation_config bonus_ablation/ablation_config.yaml --ablation_ckpt bonus_ablation/checkpoints/step_100000.pt
```

## Google Drive links (fill in before submission)

| Artifact | Link |
|---|---|
| Hindi raw/manual corpus (if large) | TODO |
| Hindi pretrained checkpoint | TODO |
| Hindi finetuned checkpoint | TODO |
| Nepali raw/manual corpus (if large) | TODO |
| Nepali pretrained checkpoint | TODO |
| Nepali finetuned checkpoint | TODO |

## Architecture summary

Decoder-only GPT-style Transformer, implemented from primitives
(`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` only — no
`nn.Transformer*`, no HuggingFace model classes):
- Token + learned positional embeddings, summed, then dropout.
- N pre-norm blocks: multi-head causal self-attention (explicit QKV
  projection, reshape into heads, scaled dot-product with additive causal
  mask, concat, output projection) + position-wise FFN (Linear-GELU-Linear).
- Final LayerNorm + linear LM head, weight-tied to the token embedding.
- Trained with causal cross-entropy (next-token prediction).

Not yet implemented — this architecture is currently descriptive only.
The model/training/eval code (`shared/attention.py`,
`shared/transformer_block.py`, `shared/gpt_model.py`, and everything
under Phase 2/3 below) still needs to be written; only the Phase 1 data
pipeline exists in this repo so far.

## Hard constraints honored

No pretrained models/tokenizers, no HuggingFace Transformer classes, no
`nn.Transformer*`. Model H and Model L never share data, tokenizer,
vocabulary, or weights (verify: `hindi/` and `nepali/` each have their
own `data/`, `tokenizer/`, `checkpoints/`; `shared/` contains only
architecture *code* and generic utility *functions*, never trained
weights or corpora).
