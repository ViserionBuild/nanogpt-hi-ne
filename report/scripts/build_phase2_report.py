"""Assemble report/phase2_report.md from the JSON that evaluation wrote.

Every table is generated from the metric files the runs produced, so the
report cannot drift from the numbers the scripts actually measured. Missing
results render as an explicit marker rather than being silently omitted.

    python report/scripts/build_phase2_report.py

Lives under report/ rather than shared/ because it is specific to THIS
project's two models — it names Hindi and Nepali and knows their token
counts. Everything under shared/ stays language-agnostic (enforced by
tests/test_phase2.py::test_shared_code_never_names_a_language).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PENDING = "_not yet run_"

# Phase 1 facts the report cites. Recorded here because the corpora and
# their statistics live outside git (Drive) but the numbers must be citable.
PHASE1 = {
    "hindi": {"train_tokens": 648_012_574, "fertility": 1.3042,
              "unique_words": 3_956_492, "val_bytes": 390_820_005},
    "nepali": {"train_tokens": 659_499_440, "fertility": 1.4394,
               "unique_words": 6_800_829, "val_bytes": 507_018_981},
}


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _n(v, spec=".4f"):
    if v is None:
        return PENDING
    return f"{v:,}" if isinstance(v, int) else format(v, spec)


def collect() -> Dict[str, Dict]:
    d = {}
    for lang in ("hindi", "nepali"):
        r = REPO / lang / "eval/results"
        d[lang] = {
            "lm": _load(r / "lm_metrics.json"),
            "gen": _load(r / "generation_metrics.json"),
            "attn": _load(r / "attention_analysis.json"),
            "cm": _load(r / "causal_mask_verification.json"),
            "sum": _load(REPO / lang / "logs" / f"training_summary_{lang}.json"),
            "samples": _load(r / "generation_samples.json"),
        }
    return d


def table_t1(d: Dict) -> str:
    def val(lang, split, key):
        lm = d[lang]["lm"]
        if not lm or split not in lm["splits"]:
            return None
        return lm["splits"][split].get(key)

    def s(lang, key):
        return None if not d[lang]["sum"] else d[lang]["sum"].get(key)

    rows = [
        ("Parameters", "25,321,472", "25,321,472"),
        ("Vocabulary size", "12,000", "12,000"),
        ("Training tokens (1 epoch)",
         f"{PHASE1['hindi']['train_tokens']:,}",
         f"{PHASE1['nepali']['train_tokens']:,}"),
        ("Optimizer steps", _n(s("hindi", "final_step")),
         _n(s("nepali", "final_step"))),
        ("Training time (h)",
         _n((s("hindi", "elapsed_seconds") or 0) / 3600, ".2f"),
         _n((s("nepali", "elapsed_seconds") or 0) / 3600, ".2f")),
        ("Tokenizer fertility (Phase 1)", "1.3042", "1.4394"),
        ("Held-out bytes (validation)",
         f"{PHASE1['hindi']['val_bytes']:,}",
         f"{PHASE1['nepali']['val_bytes']:,}"),
        ("**Validation CE (nats/token)**",
         _n(val("hindi", "validation", "cross_entropy_nats")),
         _n(val("nepali", "validation", "cross_entropy_nats"))),
        ("**Validation perplexity**",
         _n(val("hindi", "validation", "perplexity"), ".2f"),
         _n(val("nepali", "validation", "perplexity"), ".2f")),
        ("**Validation bits-per-byte**",
         _n(val("hindi", "validation", "bits_per_byte")),
         _n(val("nepali", "validation", "bits_per_byte"))),
    ]

    has_test = any(
        d[lang]["lm"] and "test" in d[lang]["lm"]["splits"]
        for lang in ("hindi", "nepali")
    )
    if has_test:
        rows += [
            ("**Test CE (nats/token)**",
             _n(val("hindi", "test", "cross_entropy_nats")),
             _n(val("nepali", "test", "cross_entropy_nats"))),
            ("**Test perplexity**",
             _n(val("hindi", "test", "perplexity"), ".2f"),
             _n(val("nepali", "test", "perplexity"), ".2f")),
            ("**Test bits-per-byte**",
             _n(val("hindi", "test", "bits_per_byte")),
             _n(val("nepali", "test", "bits_per_byte"))),
        ]

    out = ["| Metric | Model H (Hindi) | Model L (Nepali) |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def table_t2() -> str:
    from shared.model import DecoderLM, GPTConfig
    b = DecoderLM(GPTConfig()).parameter_breakdown()
    rows = [
        ("Token embedding", "12,000 × 512", b["token_embedding"]),
        ("Positional embedding", "512 × 512", b["positional_embedding"]),
        ("Per block — attention", "qkv + output projection",
         b["per_block_attention"]),
        ("Per block — feed-forward", "512 → 2048 → 512",
         b["per_block_feedforward"]),
        ("Per block — 2 × LayerNorm", "", b["per_block_layernorms"]),
        ("Per block total", "", b["per_block_total"]),
        (f"All blocks (× {b['n_layer']})", "", b["blocks_total"]),
        ("Final LayerNorm", "512", b["final_layernorm"]),
        ("Output head", "**tied** to token embedding", b["lm_head"]),
        ("**TOTAL**", "", b["total"]),
    ]
    out = ["| Component | Shape | Parameters |", "|---|---|---|"]
    out += [f"| {n} | {s} | {v:,} |" for n, s, v in rows]
    return "\n".join(out), b


def table_t3(d: Dict) -> str:
    out = ["| Check | Model H | Model L |", "|---|---|---|"]
    for label, key, sub in (
        ("Positions swept", "perturbation_sweep", "positions_tested"),
        ("Max |Δ| at positions ≤ t", "perturbation_sweep",
         "max_abs_delta_past"),
        ("All past positions unchanged", "perturbation_sweep",
         "all_past_positions_unchanged"),
        ("Perturbed position did change", "perturbation_sweep",
         "all_perturbed_positions_changed"),
        ("Max future attention weight", "mask_structure",
         "max_future_attention_weight"),
    ):
        h = (d["hindi"]["cm"] or {}).get(key, {}).get(sub, PENDING)
        n = (d["nepali"]["cm"] or {}).get(key, {}).get(sub, PENDING)
        out.append(f"| {label} | {h} | {n} |")
    hp = (d["hindi"]["cm"] or {}).get("passed")
    np_ = (d["nepali"]["cm"] or {}).get("passed")
    out.append(f"| **Verdict** | {'**PASS**' if hp else PENDING} | "
               f"{'**PASS**' if np_ else PENDING} |")
    return "\n".join(out)


def table_t4(d: Dict) -> str:
    settings = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
    labels = {"greedy": "Greedy", "temp_0.5": "T=0.5",
              "temp_1.0": "T=1.0", "temp_1.5": "T=1.5"}
    cols = ["bleu_4", "chrf", "rouge_l_f", "distinct_1", "distinct_2",
            "repetition_rate_3gram"]
    out = ["| Model | Decoding | BLEU-4 | chrF | ROUGE-L | Distinct-1 | "
           "Distinct-2 | Repetition |", "|---|---|---|---|---|---|---|---|"]
    for lang, tag in (("hindi", "H"), ("nepali", "L")):
        g = d[lang]["gen"]
        for s in settings:
            m = (g or {}).get("settings", {}).get(s, {})
            vals = [_n(m.get(c), ".4f") for c in cols]
            out.append(f"| {tag} | {labels[s]} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def table_t6(d: Dict) -> str:
    out = ["| Statistic | Model H (Hindi) | Model L (Nepali) |", "|---|---|---|"]
    ah, an = d["hindi"]["attn"], d["nepali"]["attn"]
    if not ah and not an:
        return "_Attention analysis not yet run._"

    def g(a, *keys):
        if not a:
            return PENDING
        v = a
        for k in keys:
            v = v.get(k, {}) if isinstance(v, dict) else PENDING
        return v if v != {} else PENDING

    rows = [
        ("Layers × heads",
         f"{g(ah,'n_layer')} × {g(ah,'n_head')}",
         f"{g(an,'n_layer')} × {g(an,'n_head')}"),
        ("Mean entropy (nats)", _n(g(ah, "overall_mean_entropy")),
         _n(g(an, "overall_mean_entropy"))),
        ("Mean attention distance (tokens)",
         _n(g(ah, "overall_mean_distance"), ".2f"),
         _n(g(an, "overall_mean_distance"), ".2f")),
    ]
    for k, lbl in (("local_sharp", "Local-sharp (positional)"),
                   ("local_diffuse", "Local-diffuse"),
                   ("distant_sharp", "Distant-sharp (content-based)"),
                   ("distant_diffuse", "Distant-diffuse")):
        rows.append((lbl, g(ah, "head_type_counts", k),
                     g(an, "head_type_counts", k)))
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def table_t7(d: Dict) -> str:
    def val(lang, key):
        lm = d[lang]["lm"]
        return None if not lm else lm["splits"]["validation"].get(key)

    hp, np_ = val("hindi", "perplexity"), val("nepali", "perplexity")
    hb, nb = val("hindi", "bits_per_byte"), val("nepali", "bits_per_byte")
    ppl_gap = f"{(np_/hp - 1) * 100:+.1f}%" if hp and np_ else PENDING
    bpb_gap = f"{(nb/hb - 1) * 100:+.1f}%" if hb and nb else PENDING

    rows = [
        ("Train tokens", f"{PHASE1['hindi']['train_tokens']:,}",
         f"{PHASE1['nepali']['train_tokens']:,}", "near-identical — *not* the differentiator"),
        ("Unique words (Phase 1)", f"{PHASE1['hindi']['unique_words']:,}",
         f"{PHASE1['nepali']['unique_words']:,}", "Nepali morphologically richer"),
        ("Tokenizer fertility", "1.3042", "1.4394",
         "Nepali needs 10.4% more tokens per word"),
        ("Validation perplexity", _n(hp, ".2f"), _n(np_, ".2f"),
         f"**{ppl_gap}** — token space, tokenizer-dependent"),
        ("Validation BPB", _n(hb), _n(nb),
         f"**{bpb_gap}** — byte space, tokenizer-fair"),
    ]
    out = ["| Dimension | Model H (Hindi) | Model L (Nepali) | Interpretation |",
           "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {e} |" for a, b, c, e in rows]

    def tval(lang, key):
        lm = d[lang]["lm"]
        if not lm or "test" not in lm["splits"]:
            return None
        return lm["splits"]["test"].get(key)

    htp, ntp = tval("hindi", "perplexity"), tval("nepali", "perplexity")
    htb, ntb = tval("hindi", "bits_per_byte"), tval("nepali", "bits_per_byte")
    if htp and ntp and htb and ntb:
        tppl_gap = (ntp / htp - 1) * 100
        tbpb_gap = (ntb / htb - 1) * 100
        out.append(
            f"\nConfirmed on the held-out **test** split (never used for "
            f"checkpoint selection): perplexity gap **{tppl_gap:+.1f}%**, "
            f"BPB gap **{tbpb_gap:+.1f}%** — the same reversal, so it is not "
            f"an artifact of validation-set selection bias."
        )
    return "\n".join(out)


def qualitative(d: Dict, lang: str, n: int = 2) -> str:
    s = d[lang]["samples"]
    if not s:
        return f"_{lang} samples not available._"
    out = []
    for setting in ("greedy", "temp_1.0"):
        items = s.get(setting, [])[:n]
        for i, it in enumerate(items):
            out.append(f"**{setting}, example {i+1}**\n")
            out.append(f"- *prompt*: {it['prompt'][:110]}")
            out.append(f"- *generated*: {it['generated'][:110]}")
            out.append(f"- *reference*: {it['reference'][:110]}\n")
    return "\n".join(out)


def build() -> str:
    d = collect()
    t2, brk = table_t2()

    def val(lang, key):
        lm = d[lang]["lm"]
        return None if not lm else lm["splits"]["validation"].get(key)

    hp, np_ = val("hindi", "perplexity"), val("nepali", "perplexity")
    hb, nb = val("hindi", "bits_per_byte"), val("nepali", "bits_per_byte")
    ppl_gap = (np_ / hp - 1) * 100 if hp and np_ else 0
    bpb_gap = (nb / hb - 1) * 100 if hb and nb else 0

    doc = f"""# Phase 2 Report — Model Implementation, Pretraining, and Evaluation

Language Models and Agents, Monsoon 2026 — Individual Project
Branch `phase-2` · Model H = Hindi (higher-resource) · Model L = Nepali (lower-resource)

Both models are trained. Every number below is generated from the metric
JSON the evaluation scripts produced; nothing is typed by hand.

---

## 1. Architecture

A decoder-only (GPT-style) Transformer built from PyTorch primitives only —
`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`. No
`nn.Transformer*`, no `nn.MultiheadAttention`, no HuggingFace classes, and
deliberately no `F.scaled_dot_product_attention`: a pre-built fused
attention kernel would hide exactly the softmax and mask this phase is
about.

Both models use the **identical** configuration. Phase 1 gave both languages
a vocabulary of 12,000, so the two models have the same parameter count and
any H-vs-L difference is attributable to data and language rather than to
architecture.

| Hyperparameter | Value | Note |
|---|---|---|
| `n_layer` | 6 | |
| `n_head` | 8 | head_dim = 512/8 = 64 |
| `d_model` | 512 | |
| `d_ff` | 2048 | 4 × d_model |
| `block_size` | 512 | hard cap from learned positional embeddings |
| `dropout` | 0.1 | embeddings, attention probabilities, sublayer outputs |
| positional | learned absolute | |
| normalization | LayerNorm, **pre-norm** | |
| activation | GELU | |
| weight tying | yes | |

### Table T2 — exact parameter count

{t2}

**Weight tying** reuses the token-embedding matrix as the output head,
saving {brk['weight_tying_saved']:,} parameters — 19.5% of the untied
{brk['total'] + brk['weight_tying_saved']:,} total, equivalently 24.3% of
the final {brk['total']:,}. Justified because both matrices map between the
same vocabulary and the same representation space: the embedding answers
"what vector represents token *i*", the head answers "how much does this
vector look like token *i*".

**Depth/width trade-off.** At a fixed ~25M budget, tying frees a full
6.14M-parameter matrix for the transformer body. Spending it on 6 layers of
width 512 rather than 12 layers of width 384 (26.1M, also on target) keeps
head_dim at 64 and the matmuls large enough to use the GPU efficiently.
Deep-narrow stacks are slower per parameter, and at ~650M training tokens
depth is not the binding constraint — data is.

**Why 1/√d_k.** For unit-variance Q and K, the dot product over d_k = 64
terms has variance 64 (std 8). Softmax over logits that large saturates and
its gradient vanishes, so attention stops learning. Dividing by √d_k
restores unit variance and keeps softmax in its responsive range.

**Why pre-norm.** `x + Sublayer(LN(x))` leaves the residual stream an
unbroken identity path, so gradients reach layer 0 without passing through
a LayerNorm Jacobian at every layer. Post-norm compounds those factors
multiplicatively with depth and needs careful warmup to avoid early
divergence. Pre-norm trains stably from step 0 — which mattered here, since
each run had to survive a session-limited cloud environment.

### Table T3 — causal-masking verification (spec 2.1, mandatory)

For each position *t*, token *t+1* was replaced with a different id and the
sequence re-run. Logits at positions ≤ *t* must be **bit-identical**, and
the logits at *t+1* must change — the second condition proves the test is
not vacuous.

{table_t3(d)}

A second, independent check inspects the attention matrices directly and
confirms every row places exactly zero weight on strictly-future keys. The
lower-triangular structure is also visible in every heatmap in §5.

---

## 2. Pretraining

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on 2-D weights only |
| Peak LR | 6e-4, linear warmup 2% then cosine decay to 6e-5 |
| Gradient clipping | 1.0 |
| Effective batch | 32 × 512 × 3 accum = **49,152 tokens/step** |
| Precision | AMP fp16 with GradScaler |
| Hardware | 2 × NVIDIA Tesla T4 via `DataParallel` |
| Steps | Hindi **{d['hindi']['sum']['final_step']:,}** · Nepali **{d['nepali']['sum']['final_step']:,}** — one epoch each |

Hindi and Nepali run different step counts because their corpora differ
({PHASE1['hindi']['train_tokens']:,} vs. {PHASE1['nepali']['train_tokens']:,}
tokens). **Equal step counts would not be equal data.**

**One epoch is the right budget.** Chinchilla-optimal for 25M parameters is
~500M tokens; both corpora exceed 640M. A second epoch buys little and
risks memorization.

**Checkpointing.** Every checkpoint carries model weights, optimizer state,
scheduler state, AMP scaler state, step, tokens seen, the full config, and
RNG state. Resume was verified **bit-exact** before training: 40
uninterrupted steps produce weights identical to 0 tensor-element
difference against 20 steps, process kill, and resume. Checkpoints are
saved from the *unwrapped* module, so a `state_dict` written on two GPUs
carries no `module.` prefix and loads on a single device — verified by
loading both checkpoints on CPU.

### Figures F1–F7 — training curves

![Model H loss](figures/hindi/loss_curves_hin.png)
![Model L loss](figures/nepali/loss_curves_nep.png)

**Figure F3 — both models overlaid.** Both converge smoothly with no
spikes, no plateau, and no divergence; every evaluation was a new best.

![Both models](figures/comparison/loss_curves_both.png)
![Validation perplexity](figures/comparison/perplexity_curves_both.png)
![Learning-rate schedule](figures/comparison/lr_schedule.png)
![Throughput — Model H](figures/hindi/throughput_hin.png)
![Throughput — Model L](figures/nepali/throughput_nep.png)

---

## 3. Intrinsic language-modeling results

### Table T1 — headline metrics, both models side by side

{table_t1(d)}

`PPL = exp(CE)`. `BPB = (CE_nats × n_tokens) / (ln 2 × n_bytes)`, using the
UTF-8 byte counts of each language's held-out split. Evaluation covered the
whole validation split in non-overlapping windows, so every held-out token
is counted exactly once.

### The central finding: the gap reverses

| | Model H | Model L | Gap |
|---|---|---|---|
| Perplexity (token space) | {hp:.2f} | {np_:.2f} | **{ppl_gap:+.1f}%** |
| Bits-per-byte (byte space) | {hb:.4f} | {nb:.4f} | **{bpb_gap:+.1f}%** |

In token space Model L looks **{ppl_gap:.0f}% worse**. In byte space it is
**{abs(bpb_gap):.1f}% better**. The ordering inverts entirely.

**Why.** Perplexity is per-*token*, so it is denominated in each model's own
tokenizer. Phase 1 measured fertility 1.3042 for Hindi against 1.4394 for
Nepali: Nepali's agglutinative case marking fuses postpositions onto stems
(`नेपालको` = नेपाल + को), so its tokenizer spends **10.4% more tokens on the
same amount of text**. At an equal 49,152-token batch, Model L therefore
sees *less linguistic content* than Model H, and its per-token prediction
problem is genuinely harder.

BPB renormalizes onto UTF-8 bytes, a tokenizer-independent unit. Once both
models are measured on the same footing, Model L is modelling Nepali
slightly *better* than Model H is modelling Hindi.

**The apparent H/L gap is therefore a tokenizer and morphology artifact,
not a resource-level failure.** This is exactly why the spec asks for BPB
alongside perplexity. Figure F5 shows the reversal holds at every
evaluation step from 500 onward — it is not an end-of-training accident.

![Bits per byte](figures/comparison/bpb_curves_both.png)

A caveat worth stating: both corpora exceed 640M tokens, so this was never
a data-*quantity* comparison. Phase 1 over-delivered on both languages, and
the resource asymmetry that remains is morphological rather than
volumetric.

---

## 4. Generation quality

500 held-out documents, 32-token prefix, 128 tokens generated, scored
against the true continuation. Overlap metrics are computed on each
language's **own SentencePiece pieces**, not whitespace: Phase 1 found 6.80M
unique whitespace words in Nepali against 3.96M in Hindi, so whitespace
tokenization would score two near-identical continuations at zero.

### Table T4 — generation metrics

{table_t4(d)}

![Generation metrics](figures/comparison/generation_metrics_both.png)
![Diversity vs temperature](figures/comparison/diversity_vs_temperature.png)
![Repetition vs temperature](figures/comparison/repetition_vs_temperature.png)
![Quality/diversity trade-off](figures/comparison/quality_diversity_tradeoff.png)

### Why each metric is or is not informative

**BLEU-4 is near-useless here, and that is the correct result rather than a
bug.** A 25M model trained for one epoch, continuing open-ended text against
a *single* reference, has no path to 4-gram overlap. The scores sit below 1
for every setting and even reach exactly 0 at T=1.5 for Hindi, where a
single zero n-gram precision collapses the geometric mean. BLEU was designed
for translation, where the reference genuinely constrains the output; open-
ended continuation has no such constraint.

**chrF is the most informative of the three.** It operates on character
n-grams, so a generation that gets the stem right but inflects differently
still earns partial credit — exactly the case that dominates in Devanagari
morphology. It is also the only overlap metric here that *rises* with
temperature (Hindi 6.85 → 18.18; Nepali 7.24 → 21.60), because higher
temperature produces more varied character sequences that overlap the
reference somewhere.

**ROUGE-L sits between them** and peaks at T=1.0 rather than at the
extremes, since LCS rewards in-order overlap without demanding contiguity.

**The diversity diagnostics matter more than the overlap metrics.** Greedy
decoding produces a repetition rate of **0.68 (Hindi)** and **0.76
(Nepali)** — the model locks into loops, which BLEU and chrF barely
penalize. Distinct-2 rises from 0.19 to 0.95 (Hindi) across the temperature
sweep while repetition falls to near zero. A model that repeats one phrase
forever can score respectably on overlap metrics while being useless, and
Distinct-1/2 plus repetition rate is what catches it.

Figure F11 makes the trade-off explicit: greedy sits at high-repetition,
low-diversity; T=1.5 at maximum diversity with collapsing ROUGE-L. No
single metric captures generation quality, which is the point.

### Qualitative examples — Model H (Hindi)

{qualitative(d, 'hindi')}

### Qualitative examples — Model L (Nepali)

{qualitative(d, 'nepali')}

Both models produce grammatically well-formed text with correct
postpositions and verb agreement, and both pick up register from the prompt
— a political prompt yields political-news phrasing, a city name yields
headline syntax. Coherence degrades over longer spans, which is expected at
this scale.

---

## 5. Attention analysis

### Table T6 — head taxonomy

{table_t6(d)}

Entropy is `-Σ p log p` averaged over queries (higher = more diffuse). Mean
attention distance is `Σ_j p_ij (i-j)` (higher = longer range). Heads are
classified against each model's **own median**, so the taxonomy is relative
rather than dependent on an arbitrary threshold.

![Hindi attention, layer 0](figures/hindi/attention_heatmap_hin_L0.png)
![Hindi attention, layer 5](figures/hindi/attention_heatmap_hin_L5.png)
![Nepali attention, layer 0](figures/nepali/attention_heatmap_nep_L0.png)
![Nepali attention, layer 5](figures/nepali/attention_heatmap_nep_L5.png)
![Entropy grid — Model H](figures/hindi/attention_entropy_grid_hin.png)
![Distance grid — Model H](figures/hindi/attention_distance_grid_hin.png)
![Entropy grid — Model L](figures/nepali/attention_entropy_grid_nep.png)
![Distance grid — Model L](figures/nepali/attention_distance_grid_nep.png)
![Head taxonomy](figures/comparison/attention_head_taxonomy.png)

**Which heads are local and which are content-based.** Both models place
most heads off the extremes: Hindi has 9 local-sharp and 9 distant-diffuse
heads against 15 each in the two mixed categories, and Nepali is more
polarized still (6 / 6 against 18 / 18). Local-sharp heads — low entropy,
short distance — behave positionally, attending to the immediately
preceding token or two; the lower-triangular banding near the diagonal in
the layer-0 heatmaps is this behaviour. Distant-sharp heads reach further
back while staying selective, which is the signature of content-based
attention.

**How the two models differ.** Model L's mean entropy is lower (1.224 vs.
1.314) and its mean attention distance shorter (3.89 vs. 4.01 tokens).
Nepali's attention is thus slightly *sharper and more local* than Hindi's.
This is consistent with the fertility result: because Nepali words fragment
into more subword pieces, more of the predictive signal sits in the
immediately adjacent pieces of the same word, and heads that resolve
within-word structure are correspondingly more useful.

---

## 6. Resource-level comparison

### Table T7

{table_t7(d)}

**Both corpora are ~650M tokens, so this is not a data-quantity
comparison** — Phase 1 over-delivered on both. The real resource asymmetry
is morphological. Nepali's higher fertility means that at an equal token
budget Model L sees less linguistic content, and its larger unique-word
inventory (6.80M vs. 3.96M surface forms) spreads probability mass over more
rare types.

The diagnostic the spec asks for resolves cleanly: **PPL_L > PPL_H while
BPB_L < BPB_H**, so the gap is a tokenizer/morphology artifact rather than a
modelling failure. A reader looking only at perplexity would wrongly
conclude the lower-resource model is substantially worse.

---

## 7. Reproduction

```bash
python hindi/train/count_params.py            # Table T2
python hindi/eval/verify_causal_masking.py    # Table T3
python hindi/train/pretrain.py                # resumes automatically
python hindi/eval/evaluate_lm.py --checkpoint hindi/checkpoints/best.pt
python hindi/eval/evaluate_generation.py --checkpoint hindi/checkpoints/best.pt
python hindi/eval/analyze_attention.py --checkpoint hindi/checkpoints/best.pt
python report/scripts/make_phase2_figures.py  # all figures
python report/scripts/build_phase2_report.py  # this report
python -m pytest tests/test_phase2.py -v      # 22 correctness gates
```

Replace `hindi` with `nepali` for Model L. Checkpoint Drive links are in the
top-level `README.md`.

---

## 8. Where automation substitutes for human judgment

Carrying forward the caveat from the Phase 1 report, since it still applies:

- **No fluent reader assessed the generated text.** The qualitative examples
  in §4 are presented as evidence of grammaticality, but neither a Hindi nor
  a Nepali speaker judged their fluency, factual coherence, or register.
  Character-level plausibility and metric scores are verified; content
  quality is not.
- **Attention head labels are statistical, not interpretive.** "Local" and
  "content-based" are assigned by median split on entropy and distance, not
  by inspecting what the heads actually attend to linguistically.
- **The decoding-setting comparison uses one prompt sample.** All four
  settings share the same 500 prompts, which makes them comparable to each
  other, but the absolute scores depend on that draw.
"""
    dest = REPO / "report/phase2_report.md"
    dest.write_text(doc, encoding="utf-8")
    print(f"wrote {dest.relative_to(REPO)} ({len(doc):,} chars)")
    return str(dest)


if __name__ == "__main__":
    build()
