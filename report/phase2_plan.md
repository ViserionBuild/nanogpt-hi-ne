# Phase 2 Plan — Model Implementation, Pretraining, and Evaluation

**Marks:** 40 | **Deadline:** 5 September 2026, 11:59 P.M. | **Branch:** `phase-2`
**Compute target:** Kaggle Notebooks (P100 16GB or 2×T4 15GB, 12-hour sessions, 30 GPU-hrs/week quota)

This plan covers §2.1–2.3 of `LMA_Individual_Project_v1.pdf` and the optional
positional-embedding ablation bonus. Phase 1 is complete; every artifact this
phase consumes already exists and has been verified.

---

## 0. What Phase 1 hands over

Verified directly from the binaries, not from the metadata JSON:

| Artifact | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| `data/tokenized/train.bin` | 648,012,574 tokens (2.59 GB) | 659,499,440 tokens (2.64 GB) |
| `data/tokenized/validation.bin` | 39,781,899 tokens (0.16 GB) | 43,712,387 tokens (0.17 GB) |
| `data/tokenized/test.bin` | 39,702,931 tokens (0.16 GB) | 43,779,875 tokens (0.17 GB) |
| Tokenizer | `hindi_bpe_12000.model` | `nepali_bpe_12000.model` |
| Vocab size | 12,000 | 12,000 |
| Max observed token id | 11,999 | 11,999 |
| Special tokens | pad=0, unk=1, bos=2, eos=3 | pad=0, unk=1, bos=2, eos=3 |
| Held-out bytes (validation) | 390,820,005 | 507,018,981 |
| Held-out bytes (test) | 390,355,261 | 507,858,111 |

**`.bin` format.** Flat `uint32` array, documents concatenated as
`[BOS, ...content..., EOS][BOS, ...content..., EOS]...`. No document index,
no padding, no separator beyond BOS/EOS. A random-offset window sampler reads
it directly via `np.memmap` with zero preprocessing.

**Byte counts are needed for BPB** and are recorded above because the
`.txt` files stay on Drive/Kaggle rather than in git. BPB uses the UTF-8 byte
count of the held-out text against the token-space cross-entropy — Devanagari
is 3 bytes/char, so this correction is what makes Model H and Model L
comparable despite their different fertilities (1.3042 vs 1.4394).

---

## 1. Repository layout to build

Follows the spec's §Guidelines-7 layout. `shared/` holds language-agnostic
code only — the spec forbids sharing *data, tokenizer, vocabulary, and
weights*, not architecture source. Two configs and two checkpoint trees keep
the independence visible.

```
shared/
  model/
    __init__.py
    config.py            # GPTConfig dataclass; serialized into every checkpoint
    positional.py        # LearnedPositional | SinusoidalPositional | NoPositional
    attention.py         # MultiHeadCausalSelfAttention — hand-written
    feedforward.py       # position-wise FFN (GELU, 4x inner dim)
    block.py             # TransformerBlock: pre-norm + MHA + FFN + residuals
    transformer.py       # DecoderLM: embed -> N blocks -> final LN -> tied head
  train/
    __init__.py
    dataset.py           # memmap window sampler over *.bin
    scheduler.py         # linear warmup + cosine decay
    checkpoint.py        # save/load weights+optim+sched+step+config+RNG
    trainer.py           # AdamW loop, grad accum, AMP, periodic val, logging
  eval/
    __init__.py
    lm_metrics.py        # CE loss, PPL, BPB
    generate.py          # greedy + temperature sampling (0.5 / 1.0 / 1.5)
    gen_metrics.py       # BLEU-4, chrF, ROUGE-L, distinct-1/2, repetition
    attention_tools.py   # weight capture, entropy, mean attention distance
    plots.py             # loss curves, heatmaps, metric bars — all labelled

hindi/
  configs/hindi_model.yaml
  train/pretrain.py
  eval/{evaluate_lm.py, evaluate_generation.py, analyze_attention.py,
        verify_causal_masking.py}
  checkpoints/          # gitignored -> Drive
  logs/                 # training logs + loss-curve CSV (TRACKED)

nepali/                 # identical structure, own config/checkpoints/logs

bonus_ablation/         # no-positional-embedding rerun (one language)

report/
  phase2_report.md
  figures/              # all Phase 2 plots
```

### 1.1 Layout compliance and model independence

The spec's §Guidelines-7 tree is prefixed **"Example:"** and qualified in the
next paragraph: *"Subfolder names inside each language directory **may differ**,
but the two languages must remain clearly separated."* The binding constraint is
separation, not mirroring — and Phase 1 already diverged from the example tree
(`scripts/`, `metadata/`, `tokenizer/model|vocab|config`) and was graded. The
hard constraints on p.11 enumerate **data, tokenizer, vocabulary, and weights**;
architecture source is not among them.

**Why the model lives in `shared/` rather than being copied per language.**
Duplicating it into `hindi/model/` and `nepali/model/` would make independence
more visually obvious to a grader skimming the tree, but the two copies would
drift — and then an H-vs-L perplexity difference carries an architectural
confound that cannot be ruled out. A single shared implementation with two
separate configs and two separate checkpoint trees is the stronger experimental
design, and it is the same control argument Phase 1 makes for using vocab
12,000 in both tokenizers. **State this explicitly in the report** rather than
leaving the grader to infer it.

**No circular dependency is possible, by construction.** The graph is a DAG:

```
shared/model/ --+
shared/train/ --+--> hindi/train/pretrain.py
shared/eval/  --+    nepali/train/pretrain.py
```

Arrows only ever point from `<lang>/` into `shared/`. A cycle would require
`shared/` to import from `hindi/` or `nepali/`, which it has no reason to do —
it never knows which language it is serving. Everything language-specific
arrives as config values at the call site: paths, vocab size, tokenizer file.
This is the pattern Phase 1 already runs across twelve stages
(`shared/near_dup.py`, `shared/run_stats.py` are imported by both pipelines and
import neither) and it has not produced a cycle.

The rule that keeps it true: **no module under `shared/` may reference a
language by name in executable code.** That is mechanically checkable, so it is
enforced as a test rather than left to discipline — `tests/test_shared_agnostic.py`
strips comments (via `tokenize`) and docstrings (via `ast`) and then asserts
neither "hindi" nor "nepali" appears in what remains. Comments naming the
calling scripts are legitimate and must not be flagged; a naive substring scan
over the raw source produces 8 false positives today, which is why the test
parses instead.

> **One real violation exists and must be fixed in Step 2.**
> [`shared/run_stats.py:63`](../shared/run_stats.py#L63) hardcodes a language in
> a default argument:
> ```python
> d = repo_root / config.get("paths", {}).get("run_stats", "nepali/data/run_stats")
> ```
> Harmless today because every caller passes a config, but it is exactly the
> latent coupling the test exists to catch. Change the default to `None` and
> raise if unset, so a missing config fails loudly instead of silently writing
> Hindi statistics into the Nepali tree.

**Config self-containment.** Verified: neither `hindi_data.yaml` nor
`nepali_data.yaml` contains an `include` / `inherit` / `extends` / `defaults`
directive — each is a complete, standalone description of its pipeline. The new
`hindi_model.yaml` and `nepali_model.yaml` follow the same rule: each repeats
`vocab_size: 12000` and its own `tokenizer_model:` path rather than importing a
shared block. The duplication is deliberate — a grader can read one file and
know exactly what that model was, with nothing resolved from elsewhere.

**Tracked vs. Drive.** Code, configs, training logs, loss-curve CSVs, all
metric JSON, all figures, and the report are committed. Checkpoints
(`*.pt`, ~300 MB each) and `.bin` corpora are gitignored and go to Drive with
shareable links in the README — the spec is explicit that graders will not
open external dashboards, so every *number and figure* must be in the branch
even though the weights are not.

---

## 2. Architecture (§2.1)

### 2.1 Chosen configuration — identical for both languages

| Hyperparameter | Value | Rationale |
|---|---|---|
| `vocab_size` | 12,000 | from Phase 1; same for both languages |
| `n_layer` | 6 | depth/width trade-off, see below |
| `n_head` | 8 | head_dim = 64, the standard efficient size |
| `d_model` | 512 | divisible by n_head |
| `d_ff` | 2048 | 4x d_model, conventional |
| `block_size` | 512 | context length; caps positional table |
| `dropout` | 0.1 | embeddings, attention probs, sublayer outputs |
| `weight_tying` | true | embedding <-> output head shared |
| activation | GELU | |
| norm | LayerNorm, **pre-norm** | |

**Parameter count: 25.32M** (verified by direct computation)
- token embedding 12,000x512 = 6,144,000
- positional embedding 512x512 = 262,144
- per block 3,152,384 x 6 = 18,914,304
  (attention 4x512x512 + biases = 1,050,624; FFN 2x512x2048 + biases = 2,099,712; 2 LayerNorms = 2,048)
- final LayerNorm = 1,024
- output head = **0** (tied to the token embedding)

Total **25,321,472 ~= 25.3M**, on the ~25M target. Verified by constructing the
real `nn.Module` and summing `parameters()` with the tied weight counted once —
not by arithmetic alone. Body 18,915,328 + embeddings 6,406,144 = 25,321,472.
`hindi/train/count_params.py` asserts this exact integer so a later edit to the
config cannot silently drift off target.

The same config for both languages is deliberate and must be stated in the
report: Hindi and Nepali have the same vocab size, so any H-vs-L difference in
perplexity or generation quality is attributable to **data and language**, not
to architecture. This is the same control argument Phase 1 makes for using
vocab 12,000 in both tokenizers.

**Depth/width justification (the spec asks for this explicitly).** At a fixed
~25M budget, tying the head frees a full 6.14M matrix — that is 24% of the
budget returned to the transformer body. Spending it on 6 layers of width 512
rather than, say, 12 layers of width 384 (26.1M, also on target) keeps
head_dim at 64 and the matmuls large enough to use the GPU efficiently; deep-
narrow stacks are slower per parameter on a P100 and, at only ~650M training
tokens, depth is not the binding constraint — data is.

### 2.2 Forward pass — the five required stages

Implemented from primitives only: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`nn.Dropout`, `nn.Parameter`. **No `nn.Transformer*`, no `nn.MultiheadAttention`,
no HuggingFace, no `F.scaled_dot_product_attention`** — the spec forbids
pre-built attention blocks and the point is to own every tensor op.

**1. Input representation** (`transformer.py`)
- token embedding: `(B,T) -> (B,T,d)` via `nn.Embedding(V,d)`
- positional: learned absolute `nn.Embedding(block_size,d)`, added
- embedding dropout after the sum
- *Choice to state in the report:* learned absolute. It is the simplest scheme
  that satisfies the requirement, and it **hard-caps the sequence length at
  `block_size`=512** — position ids >= 512 have no row in the table. Sinusoidal
  and RoPE both extrapolate; learned does not. This constraint is explicitly
  what the spec asks to be explained, and `positional.py` will implement
  sinusoidal and a no-op variant alongside it so the claim is demonstrable
  and the bonus ablation is a one-line config change.

**2. Multi-head causal self-attention** (`attention.py`)
- `Q,K,V = X @ W_q, X @ W_k, X @ W_v`, each `(B,T,d)`
  (single fused `nn.Linear(d, 3d)` then `.split`, documented as such)
- reshape `(B,T,d) -> (B,T,h,d_k) -> transpose(1,2) -> (B,h,T,d_k)`,
  `d_k = d/h = 64`; every reshape/transpose commented with its shapes
- `scores = Q @ K.transpose(-2,-1) / sqrt(d_k)` -> `(B,h,T,T)`
- **1/sqrt(d_k) explanation for the report:** for unit-variance Q,K the dot
  product over `d_k` terms has variance `d_k`, so at d_k=64 the logits have
  std 8; softmax over such logits saturates and its gradient vanishes. Dividing
  by sqrt(d_k) restores unit variance and keeps softmax in its responsive range.
- **causal mask:** a registered `torch.tril` buffer `(1,1,block,block)`;
  `scores.masked_fill_(mask[:,:,:T,:T]==0, float('-inf'))` **before** softmax
- softmax -> attention dropout -> `@ V` -> `(B,h,T,d_k)`
- transpose back, `.contiguous().view(B,T,d)`, output projection `W_o`, dropout
- `forward(..., return_attn=False)` optionally returns the `(B,h,T,T)` weights,
  which is how Phase 2's attention analysis and Phase 3's comparison get their data

**3. Transformer block** (`block.py`) — pre-norm:
```
x = x + Dropout(Attn(LN1(x)))
x = x + Dropout(FFN(LN2(x)))
```
*Pre-norm justification for the report:* the residual stream stays an identity
path from input to output, so gradients reach early layers without passing
through a LayerNorm's Jacobian at every step. Post-norm needs a warmup schedule
to avoid early divergence; pre-norm trains stably from step 0, which matters
when a Kaggle session can die and restart.

**4. Output head** (`transformer.py`)
- final LayerNorm, then `logits = x @ W_emb.T` (tied)
- *Tying saving to state:* 12,000x512 = **6,144,000 params**. Untied the model
  would be 31,465,472; tied it is 25,321,472 — so tying removes **19.5% of the
  untied total**, equivalently **24.3% of the final 25.3M budget**, and is what
  brings the model onto the ~25M target at this depth/width. Quote whichever
  denominator you state explicitly; they describe the same 6.14M matrix.
  Justified because input and output live in the same vocabulary space.
- loss: `F.cross_entropy(logits[:,:-1].reshape(-1,V), targets[:,1:].reshape(-1))`,
  mean over all positions

**5. Initialization**
- normal(0, 0.02) for Linear/Embedding weights; zeros for biases
- residual-projection scaling: `W_o` and FFN output weights initialized with
  std `0.02/sqrt(2*n_layer)` so the residual stream variance does not grow
  with depth

### 2.3 Causal-masking verification (hard requirement)

`*/eval/verify_causal_masking.py` — the spec names this test explicitly
("show that changing token t+1 does not change the logits at position t"):

1. take a held-out sequence of length T=512
2. for each position t, perturb token `t+1` to a different random id
3. assert `logits[:, :t+1, :]` are **bit-identical** to the unperturbed run
4. sweep every t, report pass/fail and max observed delta into a JSON

Run this **before** any GPU time is spent. It catches an off-by-one in the mask
or a mask applied after softmax — the two bugs that silently produce a model
that looks like it trains but has leaked the future.

---

## 3. Pretraining (§2.2)

### 3.1 Data pipeline

`shared/train/dataset.py`:
- `np.memmap(path, dtype=np.uint32, mode='r')` — never loads 2.6 GB into RAM
- sample a random start offset `i` in `[0, N - block_size - 1]`;
  `x = data[i : i+block]`, `y = data[i+1 : i+1+block]`
- cast `uint32 -> int64` per batch (torch has no uint32 embedding index type);
  ids are all < 12,000 so the cast is lossless
- validation uses a **fixed seed and fixed offsets** so every eval measures the
  same tokens and the loss curve is comparable across steps and across resumes

Windows cross document boundaries. This is standard for LM pretraining and is
what the concatenated `.bin` layout is for — BOS/EOS are in the stream, so the
model learns document boundaries as ordinary tokens.

### 3.2 Hyperparameters

| Setting | Value | Note |
|---|---|---|
| optimizer | AdamW | spec-required |
| betas | (0.9, 0.95) | standard for LM pretraining |
| weight decay | 0.1 | on 2-D params only; **not** on LayerNorm/bias/embeddings |
| peak LR | 6e-4 | |
| schedule | linear warmup 2% of steps -> cosine decay to 6e-5 | spec-required LR scheduling |
| grad clip | 1.0 | |
| batch (micro) | 24 x 512 tokens | tuned to fit P100 16GB |
| grad accum | 4 | -> effective ~49,152 tokens/step |
| precision | AMP fp16 + GradScaler | P100 has no bf16; T4 also fp16 |
| eval interval | every 500 steps, 100 fixed val batches | |
| checkpoint interval | every 1000 steps + always at session end | |

### 3.3 Token budget — one epoch, and why

~49,152 tokens/step (micro-batch 24 x block 512 x accum 4) gives one full
pass in **13,184 steps for Hindi** and **13,418 steps for Nepali** — the two
differ because the corpora differ, so each language's config sets `max_steps`
from its own token count rather than sharing one number.

This is the right budget on two independent grounds. **Chinchilla-optimal for
25M params is ~500M tokens** — you already have 650M, so a single epoch is
already past compute-optimal and a second epoch buys little while risking
memorization on a corpus this size. And it fits the compute: on a P100 a 25M
model at block 512 with AMP runs ~30-45K tokens/s, so one epoch is **~4.5-6
hours per language**, comfortably inside Kaggle's 12-hour session limit and
inside the 30 GPU-hr/week quota for both languages plus the ablation.

### 3.4 Checkpointing (mandatory — "a run that cannot resume will lose marks")

`shared/train/checkpoint.py` saves, per the spec's enumerated list:

```python
{
  'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
  'scaler_state_dict',          # AMP loss scaler — resume diverges without it
  'step', 'epoch', 'best_val_loss',
  'config',                     # full GPTConfig + train config, spec-required
  'torch_rng_state', 'numpy_rng_state',
  'loss_history',
}
```

Two files kept: `latest.pt` (rolling, for resume) and `best.pt` (lowest val
loss). Written to `/kaggle/working/` then copied to Drive — Kaggle wipes
`/kaggle/working` when the session ends.

**Resume must be tested by actually killing a run**, locally on CPU with a tiny
config, before any GPU time: train 50 steps, kill, resume, confirm step count,
LR value, and loss continue smoothly rather than restarting or spiking.

### 3.5 Kaggle execution specifics

- **Datasets:** upload the 6 `.bin` files (5.6 GB total) as one private Kaggle
  Dataset, plus the two `.model` tokenizer files. Kaggle Datasets allow 20 GB
  per dataset — this fits in one. Do this first; it is the long pole.
- **Repo on Kaggle:** clone from GitHub in the notebook, or attach the repo as
  a second small dataset. Keep the notebook thin — it should import from
  `shared/` and call `pretrain.py`, not contain the model inline.
- **Checkpoints out:** save to `/kaggle/working/` (20 GB, persists between a
  notebook's own commits) and push to Drive at each checkpoint, so a session
  timeout never costs more than 1000 steps.
- **GPU choice:** prefer **P100** over 2xT4. A single 16 GB P100 avoids
  DataParallel entirely; the model is far too small to need two GPUs, and
  single-device keeps the checkpoint/resume path simple.
- **Quota:** 30 GPU-hrs/week. Budget ~6h Hindi + ~6h Nepali + ~6h ablation +
  ~4h evaluation = ~22h, leaving headroom for one failed session.

### 3.6 Logging

- per-step: step, LR, train loss, tokens seen, elapsed, tokens/s -> CSV in
  `<lang>/logs/` (**tracked in git** — the spec wants loss curves in the branch)
- per-eval: step, val loss, val PPL -> same CSV
- the full resolved config dumped as JSON alongside the checkpoint, so a run is
  reproducible from the README

---

## 4. Evaluation (§2.3)

Every metric reported for **both models side by side**, each on its own
held-out set.

### 4.1 Intrinsic language-modeling metrics

| Metric | Definition | Where |
|---|---|---|
| Validation CE loss | mean cross-entropy, nats/token | `lm_metrics.py` |
| Perplexity | `exp(CE)` | |
| **Bits-per-byte** | `(CE_nats * n_tokens) / (ln2 * n_bytes)` | |

BPB uses the **UTF-8 byte counts recorded in §0** (Hindi val 390,820,005;
Nepali val 507,018,981), not token counts. This is the metric that makes the
two languages comparable when their tokenizers differ, and it is the one that
will show whether Nepali's higher fertility genuinely costs the model or merely
inflates its token-space perplexity. **Expect Nepali PPL > Hindi PPL in token
space while the BPB gap is narrower** — if so, that is a tokenizer effect, and
saying so with the numbers is the §2.3 "discuss the gap" deliverable.

Run on validation (for model selection) and once on test (final number only).

### 4.2 Generation quality

For each model, generate continuations from **held-out prefixes** at:
greedy, and temperature **0.5, 1.0, 1.5** (all four required).

Protocol: sample N=500 held-out documents, use the first 32 tokens as prefix,
generate 128 tokens, compare against the true continuation as reference.

| Metric | Note |
|---|---|
| BLEU-4 | corpus-level; n-gram order stated as required |
| chrF | character-level — the right choice for Devanagari morphology |
| ROUGE-L | longest-common-subsequence F |
| Distinct-1 / Distinct-2 | unique unigrams/bigrams over generated tokens |
| Repetition rate | fraction of repeated n-grams |

Implement these directly (`sacrebleu` for BLEU/chrF is acceptable — it is a
metric library, not a pretrained model). Tokenize for BLEU/ROUGE with the
language's **own SentencePiece tokenizer**, not whitespace — whitespace
tokenization on Devanagari understates overlap.

**Required written analysis:** why each metric is or is not informative for
open-ended LM generation. The honest answer, to be argued with the numbers:
BLEU-4 will be near zero because there is exactly one reference and open-ended
continuation has no single right answer; chrF is the most informative of the
three because Devanagari morphology means a model can be substantively right
while missing whole-word matches; ROUGE-L sits between. Diversity metrics
matter more than overlap metrics here — a degenerate model that repeats one
phrase can score respectably on BLEU while being useless, and Distinct-1/2 plus
repetition rate is what catches that.

Plus qualitative notes on coherence and language correctness, with real
generated samples in the report.

### 4.3 Attention analysis

`shared/eval/attention_tools.py`, driven by `return_attn=True`:

- **Heatmaps** — at least one early layer (0) and one late layer (5), multiple
  heads, query vs. key positions, colour-scaled, with the example sentence in
  the model's own language printed alongside. Devanagari must render: set a
  matplotlib font that has the glyphs (e.g. Noto Sans Devanagari) or the axis
  labels come out as boxes.
- **Attention entropy** per head/layer: `-sum p log p` averaged over queries.
  Higher = more diffuse.
- **Mean attention distance** per head/layer: `sum_j p_ij * (i - j)` averaged
  over queries — how far back each head looks.
- **Discussion:** which heads are local/positional (low distance, low entropy),
  which are content-based (high distance, higher entropy), and how H and L
  differ. Expect early layers local, later layers mixed.

All plots titled, axis-labelled, and legended — **the spec says plots missing
labels or legends may receive zero for that component.**

---

## 4bis. Required plots and results — complete specification

This section enumerates **every figure and every table** that must exist in
`report/phase2_report.md` for the §2.3 deliverables to be complete, with the
exact file name, the script that produces it, and the axes/labels each needs.

> **Zero-marks rule.** The spec states plainly: *"Each plot must include a
> title, x-label, y-label, and legend (if applicable). Plots missing labels or
> legends may receive zero marks for that component."* Every entry below lists
> its required labels. `shared/eval/plots.py` applies them through a single
> `finalize_axes(ax, title, xlabel, ylabel, legend=True)` helper so no figure
> can ship without them, and a `verify_figures.py` check asserts each saved
> PNG's source call passed a non-empty title/xlabel/ylabel.

### 4bis.0 Figure conventions

Carried over from Phase 1 so both phases' figures look like one document.

| Convention | Value |
|---|---|
| Output dir | `report/figures/` (tracked in git) |
| Naming | `<plot>_<lang>.png`, lang in `{hin, nep, abl}` — matches Phase 1's `tokenizer_comparison_hin.png` |
| Format / DPI | PNG, 150 dpi |
| Colour | Model H and Model L keep **one consistent colour each** across every figure |
| Comparison style | H and L on the **same axes** wherever units allow, per "report every metric for both models side by side" |

**Devanagari font — verified on this machine, do not skip.** matplotlib's
default DejaVu Sans has **no Devanagari glyphs** and drops them silently, so
`भारत एक विशाल देश है।` renders as empty boxes with only a warning. Tested here:

| Font | Result |
|---|---|
| `Lohit Devanagari` | **renders correctly** — installed and in matplotlib's cache |
| `Noto Sans Devanagari` | installed at `/usr/share/fonts/truetype/noto/` but **not in matplotlib's cache** — `findfont` fails |
| `DejaVu Sans` (default) | **glyphs missing** |

So `plots.py` sets `matplotlib.rcParams["font.family"] = "Lohit Devanagari"`
with a fallback chain, and on Kaggle (a different font set) it must
re-verify at import and `fm.fontManager.addfont(...)` the Noto TTF if needed.
Any figure carrying Devanagari — attention heatmaps especially — is unreadable
without this. **Check one heatmap visually before generating the full set.**

---

### 4bis.1 Training / pretraining plots

Produced from the tracked per-step CSV in `<lang>/logs/`, so they are
regenerable without a GPU and survive a lost checkpoint.

| # | File | Content | x-label | y-label | Legend |
|---|---|---|---|---|---|
| **F1** | `loss_curves_hin.png` | Hindi train + validation loss vs. step | Training step | Cross-entropy loss (nats/token) | train / validation |
| **F2** | `loss_curves_nep.png` | Nepali train + validation loss vs. step | Training step | Cross-entropy loss (nats/token) | train / validation |
| **F3** | `loss_curves_both.png` | **H and L overlaid** — the headline figure | Training step | Cross-entropy loss (nats/token) | Model H / Model L, train / val |
| **F4** | `perplexity_curves_both.png` | Validation PPL vs. step, both models | Training step | Validation perplexity | Model H / Model L |
| **F5** | `bpb_curves_both.png` | Validation **BPB** vs. step, both models | Training step | Bits per byte | Model H / Model L |
| **F6** | `lr_schedule.png` | Realised LR vs. step (warmup + cosine) | Training step | Learning rate | — (single series) |
| **F7** | `throughput_hin.png`, `throughput_nep.png` | tokens/s vs. step | Training step | Tokens / second | — |

**F3 and F5 together are the core Phase 2 scientific claim.** F3 in token space
will likely show Nepali with the higher loss; F5 in byte space should narrow or
invert that gap. Whichever way it lands, that contrast *is* the "discuss the
gap between Model H and Model L" deliverable, and it needs both panels to make.

> **Plot loss against tokens-seen as a secondary x-axis, not only steps.** Hindi
> runs 13,184 steps and Nepali 13,418 for one epoch each, so equal step counts
> are *not* equal data. A twin x-axis in tokens keeps the comparison honest.

### 4bis.2 Intrinsic language-modeling results

**Table T1 — headline LM metrics (both models side by side).**
Final numbers from `best.pt`, on validation *and* test.

| Metric | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Parameters | 25,321,472 | 25,321,472 |
| Vocab size | 12,000 | 12,000 |
| Training tokens (1 epoch) | 648,012,574 | 659,499,440 |
| Steps | 13,184 | 13,418 |
| Final train loss | *fill* | *fill* |
| **Validation CE loss** (nats/token) | *fill* | *fill* |
| **Validation perplexity** | *fill* | *fill* |
| **Validation BPB** | *fill* | *fill* |
| **Test CE loss** | *fill* | *fill* |
| **Test perplexity** | *fill* | *fill* |
| **Test BPB** | *fill* | *fill* |
| Held-out bytes (val) | 390,820,005 | 507,018,981 |
| Tokenizer fertility (Phase 1) | 1.3042 | 1.4394 |

`PPL = exp(CE)`; `BPB = (CE_nats x n_tokens) / (ln 2 x n_bytes)` using the byte
counts above. Report test **once**, at the end — validation drives all selection.

**Table T2 — parameter breakdown** (spec: *"report the exact parameter count"*).

| Component | Shape | Params |
|---|---|---|
| Token embedding | 12,000 x 512 | 6,144,000 |
| Positional embedding | 512 x 512 | 262,144 |
| Per block: attention (qkv + proj, w/ bias) | — | 1,050,624 |
| Per block: FFN (512->2048->512, w/ bias) | — | 2,099,712 |
| Per block: 2 x LayerNorm | — | 2,048 |
| Blocks total | x6 | 18,914,304 |
| Final LayerNorm | 512 | 1,024 |
| Output head | **tied** | 0 |
| **Total** | | **25,321,472** |

Identical for both models (same vocab size), which is the point — it is what
licenses attributing every H-vs-L difference to data and language.

**Table T3 — causal-masking verification** (hard spec requirement).
Positions swept, max absolute logit delta at positions <= t (must be `0.0`),
pass/fail. Emitted as `<lang>/eval/causal_mask_verification.json` and pasted in.

### 4bis.3 Generation quality plots and results

Protocol: 500 held-out documents, 32-token prefix, 128 tokens generated,
reference = true continuation. Four decoding settings: **greedy, T=0.5, T=1.0,
T=1.5** — all four are named in the spec, so all four appear in every table.

**Table T4 — generation metrics, both models x four decoding settings.**

| Model | Decoding | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition rate |
|---|---|---|---|---|---|---|---|
| H | greedy | | | | | | |
| H | T=0.5 | | | | | | |
| H | T=1.0 | | | | | | |
| H | T=1.5 | | | | | | |
| L | greedy | | | | | | |
| L | T=0.5 | | | | | | |
| L | T=1.0 | | | | | | |
| L | T=1.5 | | | | | | |

State the n-gram order explicitly (**BLEU-4**) as the spec requires. Tokenize
for BLEU/ROUGE with **each language's own SentencePiece model**, not
whitespace — whitespace tokenization on agglutinative Devanagari understates
overlap, and Phase 1 measured Nepali fertility at 1.4394 precisely because
surface forms are long.

| # | File | Content | x-label | y-label | Legend |
|---|---|---|---|---|---|
| **F8** | `generation_metrics_both.png` | Grouped bars: BLEU-4 / chrF / ROUGE-L, H vs L, per decoding setting | Decoding setting | Metric score | Model H / Model L |
| **F9** | `diversity_vs_temperature.png` | Distinct-1 and Distinct-2 vs. temperature, both models | Sampling temperature | Distinct-n ratio | H/L x Distinct-1/2 |
| **F10** | `repetition_vs_temperature.png` | Repetition rate vs. temperature, both models | Sampling temperature | Repeated n-gram fraction | Model H / Model L |
| **F11** | `quality_diversity_tradeoff.png` | chrF (y) against Distinct-2 (x), one point per setting | Distinct-2 | chrF | Model H / Model L |

**F11 is the figure that carries the written argument** the spec asks for
("briefly explain why each metric is or is not informative"). Expect greedy to
sit high-chrF/low-diversity and T=1.5 the reverse — showing directly that no
single overlap metric captures generation quality here.

**Table T5 — qualitative samples.** At least 3 prompts x 4 decoding settings
per model, showing prefix / generated continuation / reference, in Devanagari,
with a one-line note on coherence and language correctness. Required by the
spec's "qualitative notes" bullet and by deliverable 6.

> **Expected result, stated in advance so it is not mistaken for a bug.**
> BLEU-4 will be near zero. A 25M model trained one epoch, continuing
> open-ended text against a *single* reference, has no path to 4-gram overlap.
> chrF will be the most informative of the three because Devanagari morphology
> lets a model be substantively right while missing whole-word matches, and
> ROUGE-L will sit between. This is the analysis the deliverable wants —
> report the numbers **and** why they are what they are.

### 4bis.4 Attention analysis plots and results

Driven by `forward(..., return_attn=True)` returning `(B, h, T, T)`.
Spec minimum: **one early layer, one late layer, multiple heads** — here layer
0 and layer 5, all 8 heads, on example sentences in each model's own language.

| # | File | Content | x-label | y-label | Colorbar |
|---|---|---|---|---|---|
| **F12** | `attention_heatmap_hin_L0.png` | Hindi layer 0, 8-head grid | Key position | Query position | Attention weight |
| **F13** | `attention_heatmap_hin_L5.png` | Hindi layer 5, 8-head grid | Key position | Query position | Attention weight |
| **F14** | `attention_heatmap_nep_L0.png` | Nepali layer 0, 8-head grid | Key position | Query position | Attention weight |
| **F15** | `attention_heatmap_nep_L5.png` | Nepali layer 5, 8-head grid | Key position | Query position | Attention weight |
| **F16** | `attention_entropy_heatmap.png` | Layer x head entropy grid, H and L panels | Head index | Layer index | Mean entropy (nats) |
| **F17** | `attention_distance_heatmap.png` | Layer x head mean-distance grid, H and L panels | Head index | Layer index | Mean distance (tokens) |
| **F18** | `attention_entropy_by_layer.png` | Entropy vs. layer, per head | Layer index | Attention entropy (nats) | head 0..7 |
| **F19** | `attention_distance_by_layer.png` | Mean distance vs. layer, per head | Layer index | Mean attention distance (tokens) | head 0..7 |
| **F20** | `attention_head_taxonomy.png` | Scatter: entropy (x) vs. distance (y), one point per (layer, head), H and L | Attention entropy (nats) | Mean attention distance (tokens) | Model H / Model L |

**Heatmap requirements.** The token pieces must be printed as tick labels in
Devanagari (hence the font check in §4bis.0), the lower-triangular causal
structure must be visible — that is a second, visual confirmation of the mask —
and the example sentence must appear in the caption in the model's own language.

**Table T6 — attention summary, per layer x head, both models.** Entropy and
mean distance for all 6 x 8 = 48 heads per model; classify each as *local /
positional* (low distance, low entropy) or *content-based* (high distance,
higher entropy). Emitted as `attention_summary_{hin,nep}.json`.

**F20 is what the required discussion is built on** — "which heads look
local/positional, which look content-based, and how Model H and Model L
differ." A scatter makes head specialisation visible at a glance in a way the
per-layer line plots do not.

### 4bis.5 Resource-level comparison (deliverable 8)

**Table T7 — the summary table the whole phase builds toward.**

| Dimension | Model H (Hindi) | Model L (Nepali) | Interpretation |
|---|---|---|---|
| Train tokens | 648,012,574 | 659,499,440 | near-identical — *not* the differentiator |
| Unique words (Phase 1) | 3,956,492 | 6,800,829 | Nepali morphologically richer |
| Tokenizer fertility | 1.3042 | 1.4394 | Nepali needs ~10% more tokens/word |
| Val perplexity | *fill* | *fill* | token-space |
| Val BPB | *fill* | *fill* | byte-space, tokenizer-fair |
| chrF (greedy) | *fill* | *fill* | |
| Mean attention distance | *fill* | *fill* | |

**The key analytical point to make.** Both corpora are ~650M tokens, so this is
*not* a data-quantity comparison — Phase 1 over-delivered on both. The real
resource asymmetry shows up as **fertility and morphology**: Nepali spends more
tokens per word, so at equal token budget Model L sees **less linguistic
content**, and its higher unique-word count means more of its probability mass
goes to rare surface forms. If PPL_L > PPL_H while BPB_L ~= BPB_H, the gap is a
tokenizer/morphology artifact rather than a modeling failure — and that
distinction is the most interesting finding Phase 2 can produce.

### 4bis.6 Bonus ablation plots (if run)

Reuses every script above with `lang` suffix `abl`; the spec requires the
**full** Phase 2 suite on the ablated model.

| # | File | Content |
|---|---|---|
| **F21** | `ablation_loss_curves.png` | Standard vs. no-positional loss vs. step |
| **F22** | `ablation_metrics_comparison.png` | Grouped bars: PPL/BPB/BLEU/chrF/ROUGE-L, both variants |
| **F23** | `ablation_attention_heatmap.png` | Layer 0 + 5 heatmaps without positions |
| **F24** | `ablation_distance_entropy.png` | Entropy/distance grids, both variants |

F23 is the most informative: without positional information, attention should
lose its positional/diagonal structure entirely — self-attention is permutation
invariant, so the model can only learn a bag-of-tokens distribution.

### 4bis.7 Figure/table inventory

Everything must be **in the branch** — "Graders will not open external
dashboards or Drive folders to find figures; if it is not in the branch, it is
not graded."

| Group | Figures | Tables |
|---|---|---|
| Training | F1–F7 (7) | — |
| Intrinsic LM | — | T1, T2, T3 |
| Generation | F8–F11 (4) | T4, T5 |
| Attention | F12–F20 (9) | T6 |
| Comparison | — | T7 |
| **Core total** | **20 figures** | **7 tables** |
| Bonus ablation | F21–F24 (4) | (folded into T4/T7) |

**Missing dependencies.** `sacrebleu` (BLEU + chrF) and `rouge_score` are
**not installed** in `.venv` — verified. Add both to `requirements.txt` in
Step 2. These are metric libraries, not pretrained models, so they are
permitted under the "no pretrained models" constraint. `matplotlib`, `seaborn`,
`numpy`, `pandas`, `sentencepiece`, and `torch 2.13.0+cu130` are present.

---

## 5. Bonus — no-positional-embedding ablation

`positional.py` already carries a `NoPositional` variant, so this is a config
flag, not new code. Pick **one** language (recommend **Hindi**, the
higher-resource one — cleaner signal), retrain with positions removed, and run
the **full** Phase 2 eval suite: PPL/BPB, BLEU/chrF/ROUGE-L, diversity,
attention heatmaps/entropy/distance.

Expected result to document: loss plateaus well above the baseline. Self-
attention is permutation-invariant, so without positions the model can only
learn a bag-of-tokens distribution — it cannot represent word order at all.
Attention distance/entropy plots should show the difference structurally, which
is the more interesting half of the finding.

Bonus marks recover marks lost elsewhere, so this is worth doing if the
schedule holds.

---

## 6. Deliverables checklist (spec §2.3)

Each deliverable mapped to the artifact that satisfies it (see §4bis).

| # | Deliverable | Satisfied by | Done |
|---|---|---|---|
| 1 | Transformer implementation (MHA, positional embeddings, causal mask) | `shared/model/*` + **T3** causal-mask verification | [ ] |
| 2 | Model configuration files (H and L) | `hindi/configs/hindi_model.yaml`, `nepali/configs/nepali_model.yaml` | [ ] |
| 3 | Parameter counts; training scripts; Drive links to checkpoints | **T2**, `<lang>/train/pretrain.py`, README links | [ ] |
| 4 | Training logs and loss curves | `<lang>/logs/*.csv` + **F1–F7** | [ ] |
| 5 | PPL / BPB tables; BLEU, chrF, ROUGE-L results | **T1**, **T4**, **F4–F5**, **F8** | [ ] |
| 6 | Generated samples and diversity / repetition stats | **T5**, **F9–F11** | [ ] |
| 7 | Attention heatmaps plus entropy / distance summaries | **F12–F20**, **T6** | [ ] |
| 8 | Resource-level comparison write-up | **T7** + §2.3 gap discussion | [ ] |
| — | Every figure/table **in the branch**, not on Drive | `report/figures/` tracked | [ ] |
| — | README reproduction steps + Drive links | `README.md` | [ ] |
| bonus | Ablation: full eval suite on no-positional model | **F21–F24** | [ ] |

**20 core figures, 7 core tables.** Every figure carries a title, x-label,
y-label, and legend where applicable — enforced by `finalize_axes()` and
checked by `verify_figures.py`, because the spec allows **zero marks** for a
plot missing them.

---

## 7. Execution order

Ordered so that everything cheap and failure-prone happens before any GPU
time is spent.

**Step 1 — Kaggle dataset upload (start immediately, runs in background).**
6 `.bin` files + 2 tokenizer models -> one private Kaggle Dataset. 5.6 GB.
Everything else waits on this, so it starts first.

**Step 2 — Repo hygiene.** Branch `phase-2` off `main`. Add `sacrebleu` and
`rouge_score` to `requirements.txt` (both missing — verified), and fix the
`shared/run_stats.py:63` language leak flagged in §1.1. Rewrite the README
section that describes `train/`, `eval/`, `finetune/`, `bonus_ablation/`, and
`smoke_test_report.md` as existing — none of them do yet. Fix the broken
`Report.md` link (the file is `report/phase1_report.md`). Re-verify the Hindi
`tokenization_metadata.json`, whose train/validation entries are `null` with
`"from_existing_bin": true` while Nepali's are fully populated.

**Step 3 — Model code.** `shared/model/*`. Then a parameter-count script that
asserts 25,321,472 and prints the per-component breakdown for the report.

**Step 4 — Correctness gates, on CPU, before any GPU time.**
- causal-masking sweep passes bit-identically
- overfit a single batch to near-zero loss (proves model + loss + backward)
- shape assertions at every reshape/transpose in attention

**Step 5 — Trainer + resume.** Build it, then **kill a local CPU run and
resume it**, confirming step/LR/loss continuity. The spec makes this
mandatory; testing it locally costs minutes and testing it on Kaggle costs a
session.

**Step 6 — Pretrain Hindi** on Kaggle P100, ~13,200 steps, ~5-6h.

**Step 7 — Pretrain Nepali**, same config, same budget.

**Step 8 — Evaluation suite** (writable while training runs, against a random-
init model so the plumbing is proven before real checkpoints land): LM metrics
(T1), generation (T4/T5), attention (T6), and all 20 figures. Verify the
Devanagari font on **one** heatmap before generating the rest.

**Step 9 — Ablation** if quota and schedule allow.

**Step 10 — `report/phase2_report.md`**, in the same candid register as the
Phase 1 report — including a §"where automation substitutes for human
judgment" equivalent, since no fluent reader will be assessing the generated
Hindi or Nepali samples either.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Kaggle session dies mid-run | Checkpoint every 1000 steps, push to Drive each time; resume path tested locally first |
| 30 GPU-hr/week quota exhausted | Budget ~22h; single epoch not multiple; run the ablation last as the droppable item |
| Attention bug that silently leaks the future | Causal-masking sweep as a hard gate before Step 6 |
| fp16 instability on P100 | GradScaler + grad clip 1.0; `scaler_state_dict` in the checkpoint so resume does not diverge |
| Devanagari renders as boxes in heatmaps | Install/point matplotlib at Noto Sans Devanagari; verify one figure before generating all |
| Generation metrics near zero and look like a bug | They are expected for a 25M single-epoch model; report them with the analysis of *why*, per §4.2 |
| OOM on P100 at batch 24 | Fall back to micro-batch 16 with grad accum 6 — same effective batch |
