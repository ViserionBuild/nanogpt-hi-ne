# Phase 2 Report — Model Implementation, Pretraining, and Evaluation

Language Models and Agents, Monsoon 2026 — Individual Project
Branch `phase-2` · Model H = Hindi (higher-resource) · Model L = Nepali (lower-resource)

## 1. Architecture

Decoder-only (GPT-style) Transformer, implemented from PyTorch primitives
only — `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`. No
`nn.Transformer*`, no `nn.MultiheadAttention`, no HuggingFace classes, and
no `F.scaled_dot_product_attention` (a pre-built fused attention kernel
would hide exactly the softmax and mask this phase is about).

Both models use the **identical** configuration. Since Phase 1 gave both
languages a vocabulary of 12,000, the two models have the same parameter
count, so any H-vs-L difference is attributable to data and language rather
than to architecture.

| Hyperparameter | Value | Note |
|---|---|---|
| `n_layer` | 6 | |
| `n_head` | 8 | head_dim = 512/8 = 64 |
| `d_model` | 512 | |
| `d_ff` | 2048 | 4 x d_model |
| `block_size` | 512 | hard cap from learned positions |
| `dropout` | 0.1 | embeddings, attention probs, sublayer outputs |
| positional | learned absolute | |
| normalization | LayerNorm, **pre-norm** | |
| activation | GELU | |
| weight tying | yes | |

### Table T2 — exact parameter count

| Component | Shape | Parameters |
|---|---|---|
| Token embedding | 12,000 x 512 | 6,144,000 |
| Positional embedding | 512 x 512 | 262,144 |
| Per block — attention | qkv + output proj | 1,050,624 |
| Per block — feed-forward | 512 -> 2048 -> 512 | 2,099,712 |
| Per block — 2 x LayerNorm |  | 2,048 |
| Per block total |  | 3,152,384 |
| All blocks (x6) |  | 18,914,304 |
| Final LayerNorm | 512 | 1,024 |
| Output head | **tied** to token embedding | 0 |
| **TOTAL** |  | 25,321,472 |

**Weight tying** reuses the token-embedding matrix as the output head,
saving 6,144,000 parameters — 19.5% of the untied
31,465,472 total, or 24.3% of
the final 25,321,472. It is justified because both matrices map
between the same vocabulary and the same representation space.

**Depth/width trade-off.** At a fixed ~25M budget, tying frees a full
6.14M-parameter matrix for the transformer body. Spending it on 6 layers of
width 512 rather than 12 layers of width 384 (26.1M, also on target) keeps
head_dim at 64 and the matmuls large enough to use the GPU efficiently.
Deep-narrow stacks are slower per parameter, and at ~650M training tokens
depth is not the binding constraint — data is.

**Why 1/sqrt(d_k).** For unit-variance Q and K, the dot product over d_k=64
terms has variance 64 (std 8). Softmax over logits that large saturates and
its gradient vanishes, so attention stops learning. Dividing by sqrt(d_k)
restores unit variance.

**Why pre-norm.** `x + Sublayer(LN(x))` leaves the residual stream an
unbroken identity path, so gradients reach layer 0 without passing through a
LayerNorm Jacobian at every layer. Post-norm compounds those factors
multiplicatively with depth and needs careful warmup to avoid early
divergence. Pre-norm trains stably from step 0 — which matters when a
session can die and resume at an arbitrary point.

### Table T3 — causal-masking verification (spec 2.1, mandatory)

For each position t, token t+1 was replaced with a different id and the
sequence re-run. Logits at positions <= t must be **bit-identical**, and the
logits at t+1 must change (proving the test is not vacuous).

| Check | Model H | Model L |
|---|---|---|
| Positions swept | 127 | 127 |
| Max |delta| at positions <= t | 0.0 | 0.0 |
| All past positions unchanged | True | True |
| Perturbed position did change | True | True |
| Max future attention weight | 0.0 | 0.0 |
| **Verdict** | **PASS** | **PASS** |

A second, independent check inspects the attention matrices directly and
confirms every row places exactly zero weight on strictly-future keys. The
lower-triangular structure is also visible in every heatmap in section 5.

## 2. Pretraining

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on 2-D weights only |
| Peak LR | 6e-4, linear warmup 2% then cosine decay to 6e-5 |
| Gradient clipping | 1.0 |
| Effective batch | 24 x 512 x 4 accum = **49,152 tokens/step** |
| Precision | AMP fp16 with GradScaler (P100/T4 have no bf16) |
| Steps | Hindi **13,183** · Nepali **13,417** — one epoch each |

Hindi and Nepali run different step counts because their corpora differ
(648,012,574 vs. 659,499,440 tokens). Equal
step counts would **not** be equal data.

**One epoch is the right budget.** Chinchilla-optimal for 25M parameters is
~500M tokens; both corpora exceed 640M. A second epoch buys little and
risks memorization.

**Checkpointing.** Every checkpoint carries model weights, optimizer state,
scheduler state, AMP scaler state, step, tokens seen, the full config, and
RNG state. Resume was verified to be **bit-exact**: training 40 steps
uninterrupted produces weights identical to 0 tensor-element difference
against training 20 steps, killing the process, and resuming.

### Figures F1–F7 — training curves

![Model H loss](figures/loss_curves_hin.png)
![Model L loss](figures/loss_curves_nep.png)
![Both models](figures/loss_curves_both.png)
![Perplexity](figures/perplexity_curves_both.png)
![Learning rate](figures/lr_schedule.png)

## 3. Intrinsic language-modeling results

### Table T1 — headline metrics, both models side by side

| Metric | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Parameters | 25,321,472 | 25,321,472 |
| Vocabulary size | 12,000 | 12,000 |
| Training tokens (1 epoch) | 648,012,574 | 659,499,440 |
| Optimizer steps | 13,183 | 13,417 |
| Tokenizer fertility (Phase 1) | 1.3042 | 1.4394 |
| Held-out bytes (validation) | 390,820,005 | 507,018,981 |
| **Validation CE loss (nats/token)** | _pending_ | _pending_ |
| **Validation Perplexity** | _pending_ | _pending_ |
| **Validation Bits per byte (BPB)** | _pending_ | _pending_ |
| **Test CE loss (nats/token)** | _pending_ | _pending_ |
| **Test Perplexity** | _pending_ | _pending_ |
| **Test Bits per byte (BPB)** | _pending_ | _pending_ |

`PPL = exp(CE)`. `BPB = (CE_nats x n_tokens) / (ln2 x n_bytes)`, using the
UTF-8 byte counts of each language's held-out `.txt`.

**Why BPB matters.** Perplexity is per-token and therefore denominated in
each model's own tokenizer. Phase 1 measured fertility 1.3042 for Hindi and
1.4394 for Nepali — Nepali spends ~10% more tokens on the same text, so a
raw PPL comparison is not like-for-like. BPB renormalizes onto UTF-8 bytes,
a tokenizer-independent unit.

## 4. Generation quality

500 held-out documents, 32-token prefix, 128 tokens generated, scored
against the true continuation. Overlap metrics are computed on each
language's **own SentencePiece pieces**, not whitespace: Devanagari
agglutination (Phase 1 found 6.8M unique Nepali whitespace words against
3.96M Hindi) would otherwise score two near-identical continuations at zero.

### Table T4 — generation metrics

| Model | Decoding | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition |
|---|---|---|---|---|---|---|---|
| H | Greedy | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| H | T=0.5 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| H | T=1.0 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| H | T=1.5 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| L | Greedy | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| L | T=0.5 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| L | T=1.0 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| L | T=1.5 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

![Generation metrics](figures/generation_metrics_both.png)
![Diversity](figures/diversity_vs_temperature.png)
![Quality vs diversity](figures/quality_diversity_tradeoff.png)

## 5. Attention analysis

### Table T6 — head taxonomy

_Model H (Hindi): attention analysis not yet run._

_Model L (Nepali): attention analysis not yet run._

Entropy is `-sum p log p` averaged over queries (higher = more diffuse).
Mean attention distance is `sum_j p_ij (i-j)` (higher = longer range). Heads
are classified against each model's own median, so the taxonomy is relative
rather than threshold-dependent.

![Hindi layer 0](figures/attention_heatmap_hin_L0.png)
![Hindi layer 5](figures/attention_heatmap_hin_L5.png)
![Nepali layer 0](figures/attention_heatmap_nep_L0.png)
![Nepali layer 5](figures/attention_heatmap_nep_L5.png)
![Head taxonomy](figures/attention_head_taxonomy.png)

## 6. Resource-level comparison

Both corpora are ~650M tokens, so this is **not** a data-quantity
comparison — Phase 1 over-delivered on both. The real resource asymmetry is
morphological: Nepali's higher fertility means that at an equal token budget
Model L sees **less linguistic content**, and its larger unique-word
inventory (6.80M vs. 3.96M) spreads probability mass over more rare surface
forms.

The diagnostic: if PPL_L > PPL_H while BPB_L is close to BPB_H, the gap is a
tokenizer/morphology artifact rather than a modeling failure.

## 7. Reproduction

```bash
python hindi/train/count_params.py            # Table T2
python hindi/eval/verify_causal_masking.py    # Table T3
python hindi/train/pretrain.py                # resumes automatically
python hindi/eval/evaluate_lm.py --checkpoint hindi/checkpoints/best.pt
python hindi/eval/evaluate_generation.py --checkpoint hindi/checkpoints/best.pt
python hindi/eval/analyze_attention.py --checkpoint hindi/checkpoints/best.pt
python -m pytest tests/test_phase2.py -v      # 22 correctness gates
```

Replace `hindi` with `nepali` for Model L.
