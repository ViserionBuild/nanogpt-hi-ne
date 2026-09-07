# Phase 2, Explained — What Was Required and What We Built

A study guide for my own understanding, written so I can defend every
line of this phase in a viva. Not a deliverable. The graded write-up is
[report/phase2_report.md](report/phase2_report.md).

Phase 2 is worth **40 of 100 marks**. Deadline 5 Sep 2026, branch `phase-2`.

Everything below is traceable: requirements are quoted from
`LMA_Individual_Project_v1.pdf` §2.1–§2.3, and every number was read back
out of the committed JSON/CSV rather than from memory.

---

## Contents

1. [The one-paragraph summary](#1-the-one-paragraph-summary)
2. [Requirement → implementation map](#2-requirement--implementation-map)
3. [The architecture, tensor by tensor](#3-the-architecture-tensor-by-tensor)
4. [Why each hyperparameter is what it is](#4-why-each-hyperparameter-is-what-it-is)
5. [Pretraining](#5-pretraining)
6. [Evaluation](#6-evaluation)
7. [The central finding](#7-the-central-finding-and-why-it-matters)
8. [The hard constraints and how we stay inside them](#8-the-hard-constraints)
9. [Questions I should be ready for](#9-questions-i-should-be-ready-for)
10. [What is genuinely not done](#10-what-is-genuinely-not-done)

---

## 1. The one-paragraph summary

We built one decoder-only Transformer implementation in `shared/`, and
instantiated it twice — **Model H** on Hindi (higher-resource) and
**Model L** on Nepali (lower-resource, from the spec's allowed list).
Identical architecture, 25,321,472 parameters each, separate corpora,
separate tokenizers, separate weights, never initialized from one another.
Both trained one epoch (~650M tokens) on Kaggle 2×T4 in about 2.6 hours
each. Model H reaches perplexity 26.21; Model L reaches 41.11 — a 56.8%
gap that **reverses** when measured in bits-per-byte (0.4797 vs 0.4622),
which is the finding the report is built around.

---

## 2. Requirement → implementation map

Spec §2.1 lists four forward-pass stages and a configuration requirement.
This table is the "where is it" index.

| Spec requirement | Where it lives | Status |
|---|---|---|
| **1. Token embeddings** (vocab × d_model) | [transformer.py](shared/model/transformer.py) | done |
| **1. Positional embeddings**, stated choice | [positional.py](shared/model/positional.py) | done — learned |
| **1. Combine + embedding dropout** | [transformer.py](shared/model/transformer.py) | done |
| **2. MHA from first principles** | [attention.py](shared/model/attention.py) | done |
| **2. Explicit reshape (B,T,d)→(B,h,T,d_k)** | [attention.py:121-126](shared/model/attention.py#L121-L126) | done |
| **2. 1/√d_k scaling, explained** | [attention.py:129-137](shared/model/attention.py#L129-L137) | done |
| **2. Causal mask, −∞ before softmax** | [attention.py:138-145](shared/model/attention.py#L138-L145) | done |
| **2. Empirical proof of no future leak** | [causal_check.py](shared/eval/causal_check.py) | done — max delta 0.0 |
| **3. Block: attn + FFN + residual + LN** | [block.py](shared/model/block.py) | done — pre-norm |
| **3. FFN with d_ff > d_model, GELU** | [feedforward.py](shared/model/feedforward.py) | done — 512→2048→512 |
| **4. Linear head to vocab; tying explained** | [transformer.py](shared/model/transformer.py) | done — tied |
| **4. Cross-entropy on position t → t+1** | [transformer.py](shared/model/transformer.py) | done |
| **Config: ~25M params, exact count** | [count_params.py](hindi/train/count_params.py) | done — 25,321,472 |
| **§2.2 AdamW + LR schedule + val during training** | [trainer.py](shared/train/trainer.py) | done |
| **§2.2 Resume-capable checkpoints** | [checkpoint.py](shared/train/checkpoint.py) | done — bit-exact |
| **§2.3 CE / PPL / BPB** | [lm_metrics.py](shared/eval/lm_metrics.py) | done |
| **§2.3 BLEU / chrF / ROUGE-L, 4 decode settings** | [gen_metrics.py](shared/eval/gen_metrics.py) | done |
| **§2.3 Repetition, Distinct-1/2** | [gen_metrics.py](shared/eval/gen_metrics.py) | done |
| **§2.3 Heatmaps, entropy, mean distance** | [attention_tools.py](shared/eval/attention_tools.py) | done |
| **§2.3 Resource-level comparison write-up** | [phase2_report.md](report/phase2_report.md) | done |
| Guideline 3: every plot titled + labelled | [plots.py](shared/eval/plots.py) | **enforced in code** |

The last row is worth knowing: the spec says *"Plots missing labels or
legends may receive zero marks for that component."* So `finalize_axes()`
**raises an exception** if a title or an axis label is missing. A figure
that would lose marks cannot be written to disk.

### The eight Phase 2 deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Transformer implementation | `shared/model/` |
| 2 | Model config files (H and L) | `hindi/configs/`, `nepali/configs/` |
| 3 | Parameter counts; training scripts; Drive links | `*/train/`, README |
| 4 | Training logs and loss curves | `*/logs/`, `report/figures/` |
| 5 | PPL/BPB tables; BLEU, chrF, ROUGE-L | `report/phase2_report.md` T3–T4 |
| 6 | Generated samples; diversity/repetition | `*/eval/results/generation_samples.json` |
| 7 | Heatmaps + entropy/distance summaries | `report/figures/` |
| 8 | Resource-level comparison write-up | `report/phase2_report.md` |

Deliverable 3's Drive links are still `_pending_` — see §10.

---

## 3. The architecture, tensor by tensor

### Configuration

| Field | Value | Why this value |
|---|---|---|
| `vocab_size` | 12,000 | Phase 1 fertility sweep; see below |
| `n_layer` | 6 | depth/width trade-off at 25M |
| `n_head` | 8 | → d_k = 64, the standard head width |
| `d_model` | 512 | |
| `d_ff` | 2,048 | 4 × d_model, GPT convention |
| `block_size` | 512 | max sequence length |
| `dropout` | 0.1 | attention, residual, embedding |
| `positional` | learned | |
| `norm_position` | pre | |
| `tie_weights` | True | |

### Parameter count — how 25,321,472 is reached

```
Token embedding      12000 x 512                6,144,000
Positional embedding   512 x 512                  262,144
Per block: attention   qkv + out proj           1,050,624
Per block: feed-forward 512->2048->512          2,099,712
Per block: layernorms  2 x LayerNorm                2,048
Per block total                                 3,152,384
All blocks (x6)                                18,914,304
Final LayerNorm        512                          1,024
Output head            tied                             0
------------------------------------------------------------
TOTAL                                          25,321,472
non-embedding                                  18,915,328
```

Regenerate with `python hindi/train/count_params.py`. It prints
`Target 25,321,472 — CONFIRMED` — the target is asserted, not eyeballed.

**Weight tying saves 6,144,000 parameters — 19.5% of the untied 31,465,472.**
The output head would be a `d_model × vocab_size` matrix, exactly the
transpose of the embedding's shape. Tying them is justified beyond
parameter count: both matrices learn "which vector represents this token",
one for lookup and one for scoring, so sharing them is a real regularizer,
not just a saving.

### The forward pass

```
ids                    (B, T)         int64
  ├─ token embedding   (B, T, 512)
  ├─ + positional      (B, T, 512)    learned, broadcast over batch
  └─ embd dropout      (B, T, 512)

  6 × Block:                          pre-norm
     x = x + attn(LN(x))
     x = x + ffn(LN(x))

final LayerNorm        (B, T, 512)
output head (tied)     (B, T, 12000)  logits
cross-entropy vs ids shifted by one
```

### Inside attention — the shapes that matter

```
x                  (B, T, 512)
qkv = x @ W_qkv    (B, T, 1536)     ONE fused projection, not three
split → q, k, v    (B, T, 512) each
view               (B, T, 8, 64)    split channels across heads
transpose(1,2)     (B, 8, T, 64)    heads become a batch dimension
q @ kᵀ             (B, 8, T, T)     every query against every key
× (1/√64)                           variance control
masked_fill −inf                    future positions
softmax(dim=−1)    (B, 8, T, T)     rows sum to 1
@ v                (B, 8, T, 64)
transpose(1,2)     (B, T, 8, 64)
.contiguous().view (B, T, 512)      concatenate heads
@ W_o              (B, T, 512)
```

Four things I must be able to explain on the spot:

**Why 1/√d_k.** For q, k with independent unit-variance components, their
dot product sums d_k products and so has variance d_k — standard deviation
8 at d_k = 64. Logits that large saturate softmax: the max goes to ~1, the
rest to ~0, and the gradient through softmax vanishes. Dividing by √d_k
restores unit variance and keeps softmax responsive.

**Why −inf and not a large negative number.** `exp(−inf) = 0` exactly.
A large finite number leaves a small but nonzero probability on future
tokens, which is a real information leak.

**Why mask before softmax, not after.** Masking after softmax would leave
the rows unnormalized — they'd no longer sum to 1 — and the future tokens
would already have influenced the denominator.

**Why `.contiguous()` before `.view()`.** `transpose` only changes strides;
it does not move memory. `view` reinterprets a contiguous buffer, so it
fails on a transposed tensor unless the memory is materialized first.

**Why one fused `qkv_proj` instead of three Linears.** Mathematically
identical — the weight is just their concatenation — but one large GEMM is
materially faster on GPU than three small ones.

### Design choices the spec asks me to state and justify

**Learned absolute positional embeddings.** Self-attention is permutation
invariant, so position must be injected explicitly. Learned embeddings are
a `block_size × d_model` table — simple, and at 262,144 parameters (1.0%
of the model) cheap. The cost is the hard limit: **the model cannot process
a sequence longer than 512 tokens**, because there is no row 513 in the
table. Sinusoidal or RoPE would extrapolate; learned does not. At a fixed
512-token context that limit never binds.

> A caveat I got wrong once and should not repeat: permutation invariance
> does **not** survive causal masking. The mask itself is position-dependent,
> so a causal model is already not permutation invariant. The argument for
> positional embeddings is about *unmasked* attention being invariant.

**Pre-norm** (`x + attn(LN(x))`) rather than post-norm. Pre-norm leaves the
residual stream a clean identity path from embedding to output, so gradients
reach early layers without passing through a LayerNorm at every step. This
is why pre-norm is more stable for deep stacks and usually trains without a
warmup-sensitivity cliff. Post-norm normalizes *after* the residual add,
which puts a normalization in the gradient path at every layer.

**GELU** in the feed-forward, 512→2048→512. d_ff must exceed d_model — the
FFN's job is to project into a wider space where features are more linearly
separable, apply a non-linearity, and project back.

**Residual-projection init scaling.** The `out_proj` of attention and the
second FFN linear are initialized at `0.02/√(2·n_layer)` rather than 0.02,
because 2·n_layer residual branches all add into one stream; without the
scaling, the residual variance grows with depth.

---

## 4. Why each hyperparameter is what it is

| Hyperparameter | Value | Reasoning |
|---|---|---|
| `learning_rate` | 6e-4 | standard for this size; grad norms stayed ~0.43, no instability |
| `min_learning_rate` | 6e-5 | cosine floor at 10% of peak |
| `weight_decay` | 0.1 | applied to 2-D weights only |
| `beta1, beta2` | 0.9, 0.95 | GPT-style; β₂ below 0.999 suits shorter runs |
| `grad_clip` | 1.0 | observed max grad norm 2.14, so it clips rarely |
| `warmup_ratio` | 0.02 | ~264 steps |
| `decay_style` | cosine | |
| `micro_batch_size` | 24 | fits 16 GB with headroom |
| `grad_accum_steps` | 4 | → 49,152 tokens per optimizer step |
| `eval_interval` | 500 | 26 validation points per run |
| `seed` | 42 | |
| `amp` | True | fp16 — the reason T4 beats P100 |

**Weight decay on 2-D weights only.** `configure_optimizers()` splits by
`p.dim() >= 2`. Matrices get decay; biases, LayerNorm gains, and embeddings
do not. Decaying a LayerNorm gain pulls it toward zero, which fights the
layer's purpose; decaying biases just shifts them off their useful values.

**The LR schedule is a pure function of the step**, not a stateful counter.
That is what makes resume exact: on restart the LR is recomputed from
`step`, so it cannot drift from the uninterrupted trajectory.

⚠️ **Never pass `--max-steps` to "finish faster".** `max_steps` defines the
cosine schedule's length, so changing it changes the learning rate at every
remaining step. The flag exists for smoke tests only, and it says so in
[pretrain.py:27](hindi/train/pretrain.py#L27).

---

## 5. Pretraining

### Data

`np.memmap` over `uint32` `.bin` files, cast to int64 at batch time for the
embedding lookup. Memmap means the 2.6 GB of tokens are never fully resident;
the OS pages in what each batch touches.

| | train tokens | max_steps | tokens seen | epochs |
|---|---|---|---|---|
| Hindi | 648,012,574 | 13,183 | 647,970,816 | 0.99993 |
| Nepali | 659,499,440 | 13,417 | 659,472,384 | 0.99996 |

`max_steps = floor(train_tokens / 49,152)`, computed per language.

**The step counts differ on purpose** — 13,183 vs 13,417 — and it is
flagged in both configs and in [dataset.py:129](shared/train/dataset.py#L129).
Nepali's corpus is 11.5M tokens larger; equal *steps* would mean unequal
*data*, leaving Nepali at 0.982 epochs while Hindi sat at ~1.0. Since the
whole point is a controlled H/L comparison, matching data exposure is what
keeps the control honest.

**Floor, not ceiling.** Ceiling would cross into epoch two by a fraction of
a step, and the dataloader would re-serve tokens the model had already seen.
The leftover is 41,758 tokens for Hindi (0.006%) and 27,056 for Nepali
(0.004%) — under one optimizer step each, which is exactly what flooring
discards. This preserves a clean "every token seen exactly once" guarantee.

### One epoch is sufficient — three independent reasons

1. **Past Chinchilla-optimal.** 25.6 and 26.0 tokens per parameter, against
   an optimum of ~20. Not under-trained.
2. **Nothing to memorize.** Val loss is *below* train loss (−0.125 Hindi,
   −0.149 Nepali). That's the signature of a single-pass run with dropout
   on: train is measured with dropout active, val without, and no token was
   ever seen twice. A second epoch would start closing and then crossing
   that gap — trading generalization for memorization.
3. **Marginal return already flat.** Hindi's last 3,500 steps bought 0.052
   nats, ~0.015 per 1,000 steps and decelerating, with LR already at 6e-5.

The rule that actually governs is the token/parameter ratio, not "one epoch
is always enough." With 25M params and only 50M tokens, one epoch would
badly under-train.

### Checkpointing — mandatory, "a run that cannot resume will lose marks"

Each checkpoint holds model weights, optimizer state, scheduler state,
scaler state, step, tokens_seen, the config, **and RNG state**. Written
atomically: to a `.tmp` file, then `os.replace`, which is atomic on POSIX —
so a crash mid-write cannot leave a truncated checkpoint.

**Verified bit-exact**: 40 uninterrupted steps produce weights identical to
20 steps + kill + resume, difference exactly `0.000e+00`.

> This test lied to me once. It showed 8.3e-03 divergence and I went hunting
> through RNG, dropout, and float64 before finding the real cause: *my test*
> used different `max_steps` for the two runs, which changed the cosine
> schedule. The model was always correct. Lesson — suspect the test.

### Multi-GPU

`DataParallel` when `torch.cuda.device_count() > 1`, splitting each
24-sequence micro-batch across cards. `_save()` deliberately writes
`self.model`, not the wrapper, so the `state_dict` has **no `module.` prefix**
and loads fine on a single GPU. Loss is reduced with
`if loss.dim() > 0: loss = loss.mean()` because DataParallel returns one
loss per device.

**T4×2 over P100** — I initially got this backwards. P100 (2016, Pascal)
has **no tensor cores**, ~19 TFLOPS fp16. T4 (2018, Turing) has them,
~65 TFLOPS. This training is AMP fp16, which is exactly what tensor cores
accelerate, so a single T4 beats a P100 by roughly 3×. Memory was never the
constraint — the job uses ~1.5 GB of 16 GB.

### Actual run

| | Model H | Model L |
|---|---|---|
| Steps | 13,183 | 13,417 |
| Wall clock | 2.58 h | 2.68 h |
| Final val loss | 3.2733 | 3.7262 |
| Best val loss | 3.2747 @ 13,000 | 3.7292 @ 13,000 |
| Mean grad norm | 0.427 | 0.429 |
| Max grad norm | 2.144 | 1.698 |

---

## 6. Evaluation

### Intrinsic (§2.3)

| Metric | Model H | Model L |
|---|---|---|
| Cross-entropy (nats) | 3.266259 | 3.716235 |
| **Perplexity** | **26.2131** | **41.1093** |
| Bits per token | 4.712215 | 5.361393 |
| **Bits per byte** | **0.47966** | **0.46223** |
| Tokens evaluated | 39,780,352 | 43,696,128 |

`PPL = exp(CE)`. BPB = `(CE_nats × n_tokens) / (ln2 × n_bytes)`.

**Why BPB matters.** Perplexity is per *token*, and the two models have
different tokenizers, so their tokens are not the same unit — comparing
their perplexities directly is comparing prices in different currencies.
Bytes are identical units across both languages, so BPB is the only
tokenizer-independent comparison available.

### Generation (§2.3) — 4 settings × 500 samples

| Setting | H BLEU-4 | H chrF | H ROUGE-L | H rep-3 | H dist-2 |
|---|---|---|---|---|---|
| greedy | 0.4369 | 6.85 | 6.75 | **0.684** | 0.190 |
| T=0.5 | 0.3968 | 8.03 | 7.35 | 0.315 | 0.419 |
| T=1.0 | 0.8162 | 14.11 | 8.77 | 0.021 | 0.727 |
| T=1.5 | **0.0000** | 18.18 | 5.38 | 0.001 | 0.947 |

| Setting | L BLEU-4 | L chrF | L ROUGE-L | L rep-3 | L dist-2 |
|---|---|---|---|---|---|
| greedy | 0.3477 | 7.24 | 4.85 | **0.765** | 0.169 |
| T=0.5 | 0.2346 | 8.18 | 5.42 | 0.382 | 0.444 |
| T=1.0 | 0.3148 | 14.43 | 5.48 | 0.016 | 0.841 |
| T=1.5 | 0.1235 | 21.60 | 3.96 | 0.000 | 0.960 |

Three readings the spec explicitly asks for ("explain why each metric is
or is not informative"):

**BLEU-4 is near-useless here.** Below 1 everywhere. At Hindi T=1.5 it is
*exactly* 0.0 — the 4-gram precision hit 0.0325→0.0 and BLEU's geometric
mean collapses to zero if any single order is zero. A 25M model doing
open-ended continuation against **one** reference cannot produce 4-gram
overlap. Reporting BLEU is required; reporting it *without* this explanation
would look like a broken pipeline.

**chrF is the informative one.** Character-level, so it gives partial credit
for correct morphology — which matters enormously in agglutinative Devanagari,
where a wrong suffix destroys a word-level match but preserves most characters.
It is also the only metric that *rises* with temperature (6.85 → 18.18).

**Greedy decoding degenerates.** Repetition rate 0.684 (H) and 0.765 (L) —
roughly three of every four 3-grams are repeats. Distinct-2 of 0.169–0.190
confirms it. This is the classic greedy failure mode, and the diversity
diagnostics catch what the overlap metrics miss entirely.

Tokenization for these metrics is **SentencePiece, not whitespace** — Devanagari
agglutination makes whitespace tokens far too coarse.

### Attention (§2.3)

| | Model H | Model L |
|---|---|---|
| Mean entropy (nats) | 1.3142 | 1.2238 |
| Mean distance (tokens) | 4.0077 | 3.8870 |
| Median entropy | 1.4186 | 1.3207 |
| Median distance | 4.02 | 3.7679 |

Per-layer entropy falls monotonically with depth in both models
(H: 1.714 → 1.053; L: 1.678 → 1.052) — early layers attend broadly, later
layers commit. Head taxonomy is computed against **each model's own median**,
so "local" and "sharp" are relative labels, not absolute thresholds:

| Type | Model H | Model L |
|---|---|---|
| local_sharp | 9 | 6 |
| local_diffuse | 15 | 18 |
| distant_sharp | 15 | 18 |
| distant_diffuse | 9 | 6 |

**Model L's attention is sharper and more local** — lower entropy, shorter
distance. Consistent with the fertility result: more within-word fragmentation
puts more of the predictive signal in immediately adjacent pieces.

### Causal masking proof (hard requirement)

The spec asks specifically: *"show that changing token t+1 does not change
the logits at position t."* [causal_check.py](shared/eval/causal_check.py)
perturbs future tokens and measures the logit delta at earlier positions.
**Max delta 0.0** — not "small", exactly zero, which is what `-inf` masking
guarantees.

---

## 7. The central finding, and why it matters

| | Model H | Model L | Gap |
|---|---|---|---|
| Perplexity (token space) | 26.21 | 41.11 | **+56.8%** |
| Bits per byte (byte space) | 0.4797 | 0.4622 | **−3.6%** |

**The ordering reverses.** In tokens, the lower-resource model looks 57%
worse. In bytes — the only unit both models share — it is 3.6% *better*.

The mechanism is tokenizer fertility, measured in Phase 1 at the chosen
12,000 vocab:

| | fertility | chars/token |
|---|---|---|
| Hindi | 1.3042 | 3.94 |
| Nepali | 1.4394 | 4.47 |

Nepali's tokenizer spends **10.4% more tokens** on the same text. Each
Nepali token therefore carries less information, and the model sees less
content per fixed-size batch. Per-token perplexity charges Model L for that
without acknowledging it; BPB divides it out.

The honest conclusion: **the H/L gap is largely a tokenizer/morphology
artifact, not a resource-level modelling failure.** A reader looking only at
perplexity would draw the opposite — and wrong — conclusion.

The reversal is visible at every evaluation from step 500 onward in
`bpb_curves_both.png`, so it is not an end-of-training accident.

**Why I will not "fix" the gap by giving Nepali a bigger vocabulary.** It
would work — 16k vocab drops fertility to 1.3766 — but it would destroy the
finding, since the evidence *is* the divergence between the two measurement
spaces. It would also break the controlled comparison the spec requires.

---

## 8. The hard constraints

The spec's final box lists constraints that carry a straight penalty.

| Constraint | How we comply |
|---|---|
| No pretrained models | trained from random init |
| No pretrained tokenizers | SentencePiece BPE trained in Phase 1 |
| No HuggingFace Transformer classes | never imported |
| No `nn.Transformer*` / `nn.MultiheadAttention` | never imported |
| Two **fully independent** monolingual LMs | separate data, tokenizer, vocab, weights |
| No shared vocabulary | 12,000 each, separately trained |
| No cross-language init | separate `pretrain.py` per language |
| MHA / positional / causal mask student-implemented | `shared/model/` |
| Reproducible | seeded; configs logged beside checkpoints |

Two judgment calls worth being able to defend:

**`F.scaled_dot_product_attention` is deliberately not used.** The spec bans
"any pre-built attention block". SDPA is a fused kernel that hides the mask,
the scaling, and the softmax. It is arguably permitted, but the *point* of
§2.1 is demonstrating I understand every tensor operation — so using it
would forfeit the intent even if it passed the letter.

**Shared code that names a language would violate independence.** There is
an AST-level test, `test_shared_code_never_names_a_language`, that parses
every file in `shared/` and fails if "hindi"/"nepali" appears in live code.
It strips comments via `tokenize` and docstrings via `ast`, so prose is
allowed and code is not. **It caught three real leaks in code I had just
written** — a language branch in `runner.py` and two hardcoded labels in
`plots.py`. That is also why `build_phase2_report.py` lives in
`report/scripts/` rather than `shared/`: it legitimately names both languages.

### Repository layout

Spec Guideline 7 illustrates per-language `model/` directories. We have one
`shared/model/` instead. This is compliant — the spec says *"subfolder names
inside each language directory may differ, but the two languages must remain
clearly separated."* Separation is about **data, tokenizer, vocabulary, and
weights**, all of which are per-language. Sharing the *class definition* is
what Guideline 1 asks for ("clear modular design"); duplicating the
Transformer into two folders would mean two copies to keep in sync and would
guarantee they drift.

### Test gates

`tests/test_phase2.py` — **22 tests**, run before spending GPU time:

- causal masking blocks future information (empirically, not by inspection)
- resume is bit-exact
- loss starts near `ln(12000) = 9.39` — an untrained model's uniform-prior entropy
- tied weights are actually the same tensor object
- parameter count equals the target exactly
- `shared/` never names a language

> Two of these tests were wrong when I first wrote them, and both failures
> were mine, not the model's. `test_loss_starts_near_uniform_entropy` passed
> `x` as its own target (a copy task). `test_no_positional_makes_model_
> permutation_invariant` asserted something conceptually false — permutation
> invariance does not survive causal masking — and had to be rewritten
> against unmasked attention.

---

## 9. Questions I should be ready for

**"What breaks if you remove the causal mask?"** Every position sees the
whole sequence, so predicting token t+1 becomes trivial — it is in the input.
Training loss collapses toward zero and the model learns nothing about
prediction. It stops being a language model.

**"What breaks without positional embeddings?"** Unmasked self-attention is
a permutation-invariant set operation, so word order would be invisible.
With causal masking the model retains *some* positional information from the
mask, which is exactly what the bonus ablation would quantify.

**"Why is validation loss below training loss?"** Dropout is active during
training and disabled at eval, and it is a single-epoch run so no token is
ever revisited. Not overfitting — the opposite.

**"Why does d_ff have to exceed d_model?"** The FFN projects into a wider
space where features are more linearly separable, applies the non-linearity
there, then projects back. With d_ff = d_model there is no expansion and the
layer loses most of its capacity.

**"Why is BLEU zero at temperature 1.5?"** BLEU-4 is the geometric mean of
1–4-gram precisions. The 4-gram precision reached exactly 0.0, and a
geometric mean with a zero factor is zero. Not a bug — a property of the
metric under high-entropy open-ended sampling.

**"Which checkpoint did you evaluate, and why?"** `best.pt` at step 13,000 —
lowest validation loss. Note it is not the *final* step (13,183/13,417),
because evaluation runs every 500 steps, so the last few hundred steps of
cosine decay are not in the evaluated weights.

---

## 10. What is genuinely not done

Being straight with myself about the gaps.

**1. The test split was never evaluated.** Both `lm_metrics.json` report
`splits: ['validation']` only. There is a 39.7M / 43.7M token test split
from Phase 1, untouched. This is the real gap: `best.pt` was *selected* by
validation loss, so validation is not a clean held-out estimate. One command
fixes it, no retraining:

```bash
python hindi/eval/evaluate_lm.py --checkpoint hindi/checkpoints/best.pt --splits validation test
```

**2. BPB byte counts came from a fallback path.** Both files carry
`"bytes_source": "config byte_counts (corpus .txt not mounted)"` — the `.txt`
splits were not mounted on Kaggle. The counts trace to Phase 1 and are very
likely right, but BPB *is* the central finding, so it should rest on measured
bytes rather than a self-documented fallback.

**3. Drive links for checkpoints are `_pending_`.** Deliverable 3 requires
them in the README. The checkpoints currently live only on the HDD at
`/media/vikash/Entertainment/LMA/Final/{Hindi,Nepali}/phase2/checkpoints/` —
the repo's `*/checkpoints/` holds only `.gitkeep`.

**4. Not committed to `phase-2`.** The spec grades the branch state at the
deadline and warns that graders will not open Drive to find figures.

**5. Bonus ablation (no positional embeddings) not run.** Optional, ~3 GPU-h,
and the better use of spare compute than a second epoch.

**6. No fluent-reader assessment of generated text.** The qualitative notes
are structural. Carried forward from the Phase 1 report's §9 caveat, and the
attention head labels are statistical rather than interpretive.

---

## Quick reference

```bash
# verify the architecture claims
python hindi/train/count_params.py            # → 25,321,472 CONFIRMED
python hindi/eval/verify_causal_masking.py    # → PASSED, max delta 0.0
python -m pytest tests/test_phase2.py -q      # → 22 passed

# regenerate every figure and the report (no GPU needed)
python report/scripts/make_phase2_figures.py
python report/scripts/build_phase2_report.py
```

| Fact | Value |
|---|---|
| Parameters | 25,321,472 |
| Baseline loss (untrained) | ln(12000) = 9.39 |
| Tokens per optimizer step | 49,152 |
| Steps | H 13,183 / L 13,417 |
| Final perplexity | H 26.21 / L 41.11 |
| Final BPB | H 0.4797 / L 0.4622 |
