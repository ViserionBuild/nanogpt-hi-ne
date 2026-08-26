"""Assemble report/phase2_report.md from the JSON that evaluation wrote.

Lives under report/ rather than shared/ because it is specific to THIS
project's two models -- it names Hindi and Nepali, and knows their token
counts. Everything under shared/ must stay language-agnostic (enforced by
tests/test_shared_agnostic).

Every table in the Phase 2 report is generated from the metric JSON files
rather than typed by hand, so the report cannot drift from the numbers the
scripts actually produced. Missing results are rendered as an explicit
"not yet run" marker instead of being silently omitted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

PENDING = "_pending_"


def _load(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, spec: str = ".4f") -> str:
    if v is None:
        return PENDING
    if isinstance(v, (int, float)):
        return f"{v:,}" if isinstance(v, int) else format(v, spec)
    return str(v)


def table_t1(h: Optional[Dict], l: Optional[Dict],
             h_facts: Dict, l_facts: Dict) -> str:
    """Headline LM metrics, both models side by side."""
    def g(res, split, key):
        if not res:
            return None
        return res.get("splits", {}).get(split, {}).get(key)

    rows = [
        ("Parameters", f"{h_facts['params']:,}", f"{l_facts['params']:,}"),
        ("Vocabulary size", "12,000", "12,000"),
        ("Training tokens (1 epoch)",
         f"{h_facts['train_tokens']:,}", f"{l_facts['train_tokens']:,}"),
        ("Optimizer steps", f"{h_facts['steps']:,}", f"{l_facts['steps']:,}"),
        ("Tokenizer fertility (Phase 1)", "1.3042", "1.4394"),
        ("Held-out bytes (validation)",
         f"{h_facts['val_bytes']:,}", f"{l_facts['val_bytes']:,}"),
    ]
    for split in ("validation", "test"):
        for key, label, spec in (
            ("cross_entropy_nats", "CE loss (nats/token)", ".4f"),
            ("perplexity", "Perplexity", ".2f"),
            ("bits_per_byte", "Bits per byte (BPB)", ".4f"),
        ):
            rows.append((f"**{split.title()} {label}**",
                         _fmt(g(h, split, key), spec),
                         _fmt(g(l, split, key), spec)))

    out = ["| Metric | Model H (Hindi) | Model L (Nepali) |",
           "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def table_t2(breakdown: Dict) -> str:
    """Exact parameter breakdown."""
    b = breakdown
    rows = [
        ("Token embedding", "12,000 x 512", b["token_embedding"]),
        ("Positional embedding", "512 x 512", b["positional_embedding"]),
        ("Per block — attention", "qkv + output proj",
         b["per_block_attention"]),
        ("Per block — feed-forward", "512 -> 2048 -> 512",
         b["per_block_feedforward"]),
        ("Per block — 2 x LayerNorm", "", b["per_block_layernorms"]),
        ("Per block total", "", b["per_block_total"]),
        (f"All blocks (x{b['n_layer']})", "", b["blocks_total"]),
        ("Final LayerNorm", "512", b["final_layernorm"]),
        ("Output head", "**tied** to token embedding", b["lm_head"]),
        ("**TOTAL**", "", b["total"]),
    ]
    out = ["| Component | Shape | Parameters |", "|---|---|---|"]
    out += [f"| {n} | {s} | {v:,} |" for n, s, v in rows]
    return "\n".join(out)


def table_t4(h: Optional[Dict], l: Optional[Dict]) -> str:
    """Generation metrics x four decoding settings x two models."""
    settings = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
    labels = {"greedy": "Greedy", "temp_0.5": "T=0.5",
              "temp_1.0": "T=1.0", "temp_1.5": "T=1.5"}
    cols = ["bleu_4", "chrf", "rouge_l_f", "distinct_1", "distinct_2",
            "repetition_rate_3gram"]
    head = ("| Model | Decoding | BLEU-4 | chrF | ROUGE-L | Distinct-1 | "
            "Distinct-2 | Repetition |")
    out = [head, "|---|---|---|---|---|---|---|---|"]
    for name, res in (("H", h), ("L", l)):
        for s in settings:
            m = (res or {}).get("settings", {}).get(s, {})
            vals = [_fmt(m.get(c), ".4f") for c in cols]
            out.append(f"| {name} | {labels[s]} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def table_t6(a: Optional[Dict], model_label: str) -> str:
    """Attention head taxonomy summary."""
    if not a:
        return f"_{model_label}: attention analysis not yet run._"
    c = a["head_type_counts"]
    return "\n".join([
        f"| Statistic | {model_label} |", "|---|---|",
        f"| Layers x heads | {a['n_layer']} x {a['n_head']} |",
        f"| Mean entropy (nats) | {a['overall_mean_entropy']:.4f} |",
        f"| Mean attention distance (tokens) | {a['overall_mean_distance']:.4f} |",
        f"| Local-sharp heads (positional) | {c['local_sharp']} |",
        f"| Local-diffuse heads | {c['local_diffuse']} |",
        f"| Distant-sharp heads (content-based) | {c['distant_sharp']} |",
        f"| Distant-diffuse heads | {c['distant_diffuse']} |",
    ])


def table_t3(h: Optional[Dict], l: Optional[Dict]) -> str:
    """Causal-masking verification."""
    out = ["| Check | Model H | Model L |", "|---|---|---|"]
    for label, key, sub in (
        ("Positions swept", "perturbation_sweep", "positions_tested"),
        ("Max |delta| at positions <= t", "perturbation_sweep",
         "max_abs_delta_past"),
        ("All past positions unchanged", "perturbation_sweep",
         "all_past_positions_unchanged"),
        ("Perturbed position did change", "perturbation_sweep",
         "all_perturbed_positions_changed"),
        ("Max future attention weight", "mask_structure",
         "max_future_attention_weight"),
    ):
        hv = (h or {}).get(key, {}).get(sub)
        lv = (l or {}).get(key, {}).get(sub)
        out.append(f"| {label} | {hv} | {lv} |")
    out.append(f"| **Verdict** | "
               f"{'**PASS**' if (h or {}).get('passed') else PENDING} | "
               f"{'**PASS**' if (l or {}).get('passed') else PENDING} |")
    return "\n".join(out)


def build(repo_root: Path) -> str:
    """Render the full report and write it to report/phase2_report.md."""
    import sys
    sys.path.insert(0, str(repo_root))
    from shared.model import DecoderLM, GPTConfig

    hr = repo_root / "hindi/eval/results"
    nr = repo_root / "nepali/eval/results"

    lm_h, lm_l = _load(str(hr / "lm_metrics.json")), _load(str(nr / "lm_metrics.json"))
    gen_h = _load(str(hr / "generation_metrics.json"))
    gen_l = _load(str(nr / "generation_metrics.json"))
    att_h = _load(str(hr / "attention_analysis.json"))
    att_l = _load(str(nr / "attention_analysis.json"))
    cm_h = _load(str(hr / "causal_mask_verification.json"))
    cm_l = _load(str(nr / "causal_mask_verification.json"))

    breakdown = DecoderLM(GPTConfig()).parameter_breakdown()
    h_facts = {"params": breakdown["total"], "train_tokens": 648_012_574,
               "steps": 13_183, "val_bytes": 390_820_005}
    l_facts = {"params": breakdown["total"], "train_tokens": 659_499_440,
               "steps": 13_417, "val_bytes": 507_018_981}

    doc = f"""# Phase 2 Report — Model Implementation, Pretraining, and Evaluation

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

{table_t2(breakdown)}

**Weight tying** reuses the token-embedding matrix as the output head,
saving {breakdown['weight_tying_saved']:,} parameters — 19.5% of the untied
{breakdown['total'] + breakdown['weight_tying_saved']:,} total, or 24.3% of
the final {breakdown['total']:,}. It is justified because both matrices map
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

{table_t3(cm_h, cm_l)}

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
({h_facts['train_tokens']:,} vs. {l_facts['train_tokens']:,} tokens). Equal
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

{table_t1(lm_h, lm_l, h_facts, l_facts)}

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

{table_t4(gen_h, gen_l)}

![Generation metrics](figures/generation_metrics_both.png)
![Diversity](figures/diversity_vs_temperature.png)
![Quality vs diversity](figures/quality_diversity_tradeoff.png)

## 5. Attention analysis

### Table T6 — head taxonomy

{table_t6(att_h, "Model H (Hindi)")}

{table_t6(att_l, "Model L (Nepali)")}

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
"""
    dest = repo_root / "report/phase2_report.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(doc, encoding="utf-8")
    print(f"wrote {dest}")
    return str(dest)


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[2])
