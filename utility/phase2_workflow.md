# Phase 2 Workflow — From Phase 1's Output to Two Trained Models

How Phase 2 actually happened: what Phase 1 handed over, what Phase 2 built
on top of it, and the order things ran in — including the parts that
diverged from the original plan. Written after both models finished
training, from the real artifacts in this repo, not from what was planned
in advance.

For the graded content (tables, figures, findings), see
[report/phase2_report.md](../report/phase2_report.md). This file is the
workflow behind that report. For how Phase 1 built the corpora and
tokenizers this phase consumes, see [phase1_workflow.md](phase1_workflow.md).

---

## 1. Where Phase 2 starts: what Phase 1 produced

Phase 1 (25 marks, complete, `phase-1` branch) built two independent
corpus + tokenizer pipelines. Its output is the entire input Phase 2 needs
— nothing else.

| Artifact | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| `data/tokenized/train.bin` | 648,012,574 tokens | 659,499,440 tokens |
| `data/tokenized/validation.bin` | 39,781,899 tokens | 43,712,387 tokens |
| `data/tokenized/test.bin` | 39,702,931 tokens | 43,779,875 tokens |
| Tokenizer | `hindi_bpe_12000.model` | `nepali_bpe_12000.model` |
| Vocab size | 12,000 | 12,000 |
| Manual fraction | 18.09% | 18.33% |
| Tokenizer fertility | 1.3042 | 1.4394 |
| Held-out bytes (validation) | 390,820,005 | 507,018,981 |

Two facts about this handover matter for everything that follows:

**The `.bin` format is what makes Phase 2 possible without touching raw
text.** Each file is a flat `uint32` array — tokenized documents
concatenated as `[BOS, ...content..., EOS][BOS, ...content..., EOS]...`,
no index, no padding. Phase 2's dataloader reads it directly via
`np.memmap`, so a 2.6 GB corpus is never fully loaded into RAM.

**The manual fraction fell short of the ≥20% spec target** (18.09% /
18.33%), disclosed in Phase 1's own metadata rather than hidden. This was
a closed decision by the time Phase 2 started — Phase 1 was already
graded — so Phase 2 treats it as a fixed input and does not revisit it.

---

## 2. What Phase 2 had to build

Phase 2 (40 marks) is three things: an architecture, a training run, and
an evaluation suite — for two completely independent models.

```
shared/model/      the Transformer, written once, used by both languages
shared/train/      dataset loader, scheduler, checkpointing, trainer
shared/eval/       LM metrics, generation, attention analysis, plots
hindi/, nepali/    per-language configs + thin entry-point scripts
```

`shared/` holds the architecture and training/eval *logic*; `hindi/` and
`nepali/` hold the *data, config, and results* — two separate corpora, two
separate tokenizers, two separate checkpoint trees, two separate weight
sets. The spec's hard constraint is on data/tokenizer/vocabulary/weights
independence, not on duplicating source code, so one shared implementation
with two configs is the design used here — duplicating the model into two
folders would let the two copies drift and introduce an architectural
confound into any H-vs-L comparison.

An AST-level test (`tests/test_phase2.py::test_shared_code_never_names_a_language`)
enforces this mechanically: nothing under `shared/` may name either
language in executable code. It caught three real leaks during
development before they shipped.

---

## 3. The architecture, in one pass

Built from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` only
— no `nn.Transformer*`, no `nn.MultiheadAttention`, no HuggingFace, and
deliberately no `F.scaled_dot_product_attention` (a fused kernel would
hide the exact mask/softmax mechanics the phase is about).

| Stage | File | What it does |
|---|---|---|
| Token + positional embedding | `shared/model/transformer.py` | learned absolute positions, added, then embedding dropout |
| Multi-head causal self-attention | `shared/model/attention.py` | Q/K/V, explicit `(B,T,d)→(B,h,T,d_k)` reshape, `1/√d_k` scaling, `-inf` causal mask before softmax |
| Transformer block | `shared/model/block.py` | pre-norm: `x + Attn(LN(x))`, `x + FFN(LN(x))` |
| Feed-forward | `shared/model/feedforward.py` | 512 → 2048 → 512, GELU |
| Output head | `shared/model/transformer.py` | tied to the token embedding |

Both models use the **identical** configuration (`n_layer=6, n_head=8,
d_model=512, d_ff=2048, block_size=512`), landing at exactly **25,321,472**
parameters — asserted, not eyeballed, by `hindi/train/count_params.py`.
Same vocab size (12,000) for both means the parameter count is identical
too, so any H-vs-L gap downstream is attributable to data and language,
not architecture.

**The causal mask is verified empirically, not just by code inspection** —
the spec requires this explicitly. `shared/eval/causal_check.py` perturbs
token *t+1* and confirms the logits at every position ≤ *t* are
bit-identical (max delta `0.0`), while the perturbed position's logits do
change (proving the test isn't vacuous).

---

## 4. Pretraining: what actually ran

### The plan vs. what happened

The original plan targeted a single
P100 GPU. That changed after checking the hardware: **P100 (2016,
Pascal) has no tensor cores**, ~19 TFLOPS fp16; **T4 (2018, Turing) does**,
~65 TFLOPS. Since training uses AMP fp16 throughout, a single T4 already
beats a P100 by roughly 3×. The trainer was extended with `DataParallel`
so it could use **2×T4** when both are visible, splitting each micro-batch
across the two cards. Checkpoints save the *unwrapped* module, so a
`state_dict` written on two GPUs has no `module.` prefix and loads on one
GPU without any special-casing.

### The run

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Optimizer steps | 13,183 | 13,417 |
| Tokens/step | 49,152 (32 × 512 × 3, split 16/16 across 2×T4) | same |
| Tokens seen | 647,970,816 | 659,472,384 |
| Epochs | 0.99993 | 0.99996 |
| Wall clock | ~2.6 h | ~2.6 h |
| Hardware | 2× Tesla T4 | 2× Tesla T4 |

The two step counts differ **on purpose**: Nepali's corpus is 11.5M
tokens larger, so matching *steps* between languages would mean
mismatched *data exposure*. `max_steps` is computed per language as
`floor(train_tokens / tokens_per_step)` — floor rather than ceil, so the
run stops just short of wrapping into a second epoch rather than
re-serving already-seen tokens.

**One epoch is enough, not a compromise.** At ~650M tokens against 25.3M
parameters, both runs land at ~26 tokens/parameter — past the ~20:1
Chinchilla-optimal ratio. Validation loss finished *below* training loss
in both runs, which is the expected signature of a single-pass run with
dropout on, not overfitting.

**Checkpoint-resume is mandatory per spec** ("a run that cannot resume
will lose marks") and was verified bit-exact before any real training
started: 40 uninterrupted steps produced weights identical (`0.000e+00`
difference) to 20 steps, a kill, and a resume. Every checkpoint carries
model weights, optimizer state, scheduler state, AMP scaler state, step,
tokens seen, the full config, and RNG state.

### Running two languages on two Kaggle accounts

Kaggle's GPU quota is 30 hours/week per account. To run both languages
without waiting on one 30-hour pool, Hindi ran under one account and
Nepali under a second, each with its own uploaded copy of the code and
data datasets and its own notebook (`LANGUAGE = 'hindi'` /
`LANGUAGE = 'nepali'` in an otherwise identical notebook). Both sessions
ran in parallel, so total wall time was one run's length rather than two.

---

## 5. Evaluation: what got measured, and a gap that was found and closed

The first evaluation pass revealed two real gaps, both closed in a second
run before results were finalized:

**Gap 1 — the test split was never evaluated.** The notebook's mount step
only symlinked `data/tokenized/*.bin`, so `evaluate_lm.py` only ever saw
`validation`. Since `best.pt` is *selected* by lowest validation loss,
reporting only validation isn't a clean held-out estimate. Fixed by also
symlinking `data/final/validation.txt` and `test.txt`, and changing the
eval cell to `--splits validation test`.

**Gap 2 — BPB fell back to config-recorded byte counts.** Because the
`.txt` corpora weren't mounted, `evaluate_lm.py`'s BPB calculation
silently fell back to byte counts recorded in the YAML config rather than
counting bytes from the actual mounted corpus. The same fix (mounting
`data/final/*.txt`) closed this too — the fallback is gone,
`bytes_source: "measured"` on every split now.

Both languages were retrained end-to-end with these fixes rather than
patched after the fact, so **every number in the report comes from one
consistent run per language** — no mixing of an old pretraining run with
a newer eval pass.

### What each metric measures

| Metric | Script | Measures |
|---|---|---|
| Cross-entropy, perplexity, BPB | `shared/eval/lm_metrics.py` | intrinsic LM quality, on validation and test |
| BLEU-4, chrF, ROUGE-L, Distinct-1/2, repetition rate | `shared/eval/gen_metrics.py` | generation quality at greedy + T∈{0.5,1.0,1.5} |
| Attention entropy, mean distance, head taxonomy | `shared/eval/attention_tools.py` | what the learned attention heads actually do |
| Causal-mask perturbation sweep | `shared/eval/causal_check.py` | the hard spec requirement, run once per model |

---

## 6. The result the whole design was built to produce

| | Model H (Hindi) | Model L (Nepali) | Gap |
|---|---|---|---|
| Perplexity (validation) | 26.21 | 41.52 | **+58.4%** |
| Bits-per-byte (validation) | 0.4797 | 0.4635 | **-3.4%** |

**The gap reverses between token space and byte space**, and it holds on
the test split too (+58.3% / -3.3%). Perplexity is denominated in each
model's own tokenizer; Nepali's fertility (1.4394 vs. Hindi's 1.3042)
means its tokenizer spends 10.4% more tokens on the same text, so at an
equal token budget Model L sees less linguistic content and its per-token
task is genuinely harder. BPB renormalizes onto UTF-8 bytes — a unit both
languages share — and once measured that way, Model L is modelling
Nepali slightly *better* than Model H is modelling Hindi.

This is exactly why the spec asks for BPB alongside perplexity: a reader
looking only at perplexity would conclude the lower-resource model failed
by a wide margin, when the actual gap is a tokenizer/morphology artifact.

---

## 7. Where the outputs live

```
hindi/checkpoints/    best.pt, latest.pt (+ .config.json each)
nepali/checkpoints/   same

hindi/logs/           pretrain_hindi.log, training_log_hindi.csv, training_summary_hindi.json
nepali/logs/          same

hindi/eval/results/   lm_metrics.json, generation_metrics.json, generation_samples.json,
                       attention_analysis.json, causal_mask_verification.json
nepali/eval/results/  same

report/figures/hindi/       per-model figures (loss, throughput, attention)
report/figures/nepali/      same
report/figures/comparison/  both-model comparison figures (loss/PPL/BPB overlays,
                             LR schedule, generation metrics, head taxonomy)

report/phase2_report.md     the graded document — every table and figure reference
                             resolves to one of the paths above
```

Figures and the report are regenerable from the tracked logs and eval
JSON without a GPU:

```bash
python report/scripts/make_phase2_figures.py
python report/scripts/build_phase2_report.py
```

Checkpoints (~290 MB each) are gitignored — the spec requires them on
Google Drive with links in the README, not committed as binaries.

---

## 8. Deliverable checklist (spec §2.3)

| # | Deliverable | Where |
|---|---|---|
| 1 | Transformer implementation | `shared/model/` |
| 2 | Model config files (H and L) | `hindi/configs/`, `nepali/configs/` |
| 3 | Parameter counts; training scripts; Drive links to checkpoints | `count_params.py`, `*/train/`, README |
| 4 | Training logs and loss curves | `*/logs/`, `report/figures/{hindi,nepali}/` |
| 5 | PPL/BPB tables; BLEU, chrF, ROUGE-L | `report/phase2_report.md` Tables T1, T4 |
| 6 | Generated samples; diversity/repetition | `*/eval/results/generation_samples.json` |
| 7 | Attention heatmaps + entropy/distance summaries | `report/figures/{hindi,nepali}/`, Table T6 |
| 8 | Resource-level comparison write-up | `report/phase2_report.md` §6, Table T7 |

**Not yet done**, tracked honestly rather than glossed over:
- Checkpoints not yet uploaded to Drive; README has Phase 1 corpus links
  only, not Phase 2 checkpoint links.
- Nothing yet committed to the `phase-2` branch.
- Bonus ablation (no positional embeddings) not run.

---

## 9. What Phase 3 picks up from here

Phase 3 (35 marks) finetunes each pretrained checkpoint independently on
a synthetic reasoning dataset, in that language's own script and
tokenizer, and reuses the Phase 2 attention toolkit to compare pretrained
vs. finetuned attention. It starts from exactly the two artifacts this
phase produces: `hindi/checkpoints/best.pt` and
`nepali/checkpoints/best.pt` — nothing else changes hands between the
phases.
