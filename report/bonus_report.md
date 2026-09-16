# Bonus Report — No-Positional-Embeddings Ablation (Model H / Hindi)

Optional bonus (spec p. 12): one language retrained with positional
embeddings removed, evaluated with the full Phase 2 suite, compared
against the standard model for that language. See
[`utility/bonus_plan.md`](../utility/bonus_plan.md) for the design and
[`notebooks/bonus_ablation_kaggle.ipynb`](../notebooks/bonus_ablation_kaggle.ipynb)
for the exact run.

## 1. Setup

Hindi (Model H) was chosen (either language was valid; Hindi's Phase 2/3
material is what the rest of this repo's reports already build on — see
`utility/bonus_plan.md` §3). A single config change from
[`hindi/configs/hindi_model.yaml`](../hindi/configs/hindi_model.yaml):

```yaml
model:
  positional: none            # was: learned
paths:
  checkpoints: bonus_ablation/hindi/checkpoints   # was: hindi/checkpoints
  logs: bonus_ablation/hindi/logs
  eval_results: bonus_ablation/hindi/eval_results
  figures: bonus_ablation/hindi/figures
```

Everything else — vocabulary, tokenizer, corpus, architecture
(`n_layer=6, n_head=8, d_model=512, d_ff=2048, block_size=512`),
optimizer, LR schedule, batch size, seed (42), and step count (13,183 —
one epoch over the same 648M-token corpus) — is byte-identical to
standard Model H. This isolates positional information as the single
ablated variable.

**Parameter count.** `NoPositional` (`shared/model/positional.py`) holds
no `nn.Parameter` of its own — it returns an all-zero tensor in place of
the position embedding — so the ablated model has **25,059,328**
parameters, exactly 262,144 fewer than standard Model H's 25,321,472 (the
512×512 learned positional table). Verified directly against the running
model (`shared.train.runner.print_parameter_report`), not assumed: this
was in fact where two earlier attempts at this run failed, because an
initial version of this report/notebook incorrectly asserted the ablation
would have the *same* parameter count as standard Model H. It does not —
removing the positional table necessarily removes exactly those
parameters. The corrected assertion (`expect=25_059_328`) is what
ultimately ran successfully.

**Training.** 13,183 steps, 2.62 h on 2×T4 (Kaggle) — matching standard
Model H's own 2.62 h almost exactly, since compute cost is dominated by
`d_model`/`n_layer`/sequence length, none of which changed. Causal-masking
verification (`verify_causal_masking.py`) still passes — masking is
independent of the positional scheme.

## 2. Side-by-side results

### 2.1 Language modeling (validation / test)

| Metric | Standard Model H | Ablated Model H | Δ |
|---|---|---|---|
| Val cross-entropy (nats) | 3.2662 | 3.3206 | +0.0544 |
| Val perplexity | 26.213 | 27.677 | **+1.46 (+5.6%)** |
| Val bits-per-byte | 0.47966 | 0.48764 | +0.00798 |
| Test cross-entropy (nats) | 3.2653 | 3.3195 | +0.0543 |
| Test perplexity | 26.187 | 27.648 | **+1.46 (+5.6%)** |
| Test bits-per-byte | 0.47913 | 0.48710 | +0.00796 |

The ablated model is consistently, measurably worse on the direct
next-token-prediction objective — a small but real and repeatable gap
(nearly identical on val and test), not noise.

### 2.2 Generation quality (500 held-out prefixes; greedy + T∈{0.5,1.0,1.5})

| Setting | Metric | Standard | Ablated | Δ |
|---|---|---|---|---|
| greedy | BLEU-4 | 0.5138 | 0.4578 | -0.056 |
| greedy | chrF | 7.162 | 6.718 | -0.443 |
| greedy | ROUGE-L F | 6.927 | 6.736 | -0.191 |
| greedy | repetition-rate (3-gram) | 0.7012 | 0.6960 | -0.005 |
| T=0.5 | BLEU-4 | 0.3530 | 0.3710 | +0.018 |
| T=1.0 | BLEU-4 | 0.7625 | 0.8440 | +0.082 |
| T=1.5 | BLEU-4 | 0.2040 | 0.1957 | -0.008 |
| all settings | Distinct-1 / Distinct-2 | ~identical | ~identical | ≤0.02 |

Unlike the LM-loss gap, **generation-quality deltas are small and
inconsistent in sign** — the ablated model is sometimes marginally worse
(greedy chrF/ROUGE-L), sometimes marginally *better* (T=1.0 BLEU-4) than
standard Model H. At this model scale (~25M params, one epoch), BLEU/chrF
on open-ended generation are already close to floor for both models (see
`report/phase2_report.md`'s own discussion of why BLEU-4 is uninformative
here) — the ablation's effect on the LM objective does not visibly
propagate through to these corpus-level generation metrics. This is a
genuine (if slightly surprising) finding, not a discrepancy to explain
away: **greedy decoding degenerates for both models almost identically**.
For the exact same prompt (*"...पर्यटक आना पसंद"*), standard Model H's
greedy output loops the same clause six times in a row
(*"यहां पर कई तरह के स्थानीय पर्यटक आते हैं।"* repeated); the ablated
model's greedy output on a different prompt shows the same kind of
short-cycle repetition (*"जनता के विश्वास को...जनता ने जनता के विश्वास
को..."*). Repetition-rate-3gram under greedy decoding is nearly identical
between the two (0.696 ablated vs. 0.701 standard) — this looks like a
property of greedy decoding at this model scale generally, not something
positional embeddings specifically prevent.

### 2.3 Attention structure

| Metric | Standard Model H | Ablated Model H |
|---|---|---|
| Overall mean entropy (nats) | 1.311 | 1.483 |
| Overall mean attention distance (tokens) | 3.982 | 3.104 |
| Head types (sharp/diffuse × local/distant) | 11 / 13 / 13 / 11 | 16 / 8 / 8 / 16 |
| Per-layer entropy trend (L0→L5) | 1.65 → 1.43 → 1.39 → 1.25 → 1.12 → 1.03 (monotonic decrease) | 1.42 → 1.52 → 1.76 → 1.55 → 1.18 → 1.47 (non-monotonic) |
| Per-layer mean distance (L0→L5) | 4.02 → 4.52 → 4.25 → 2.84 → 3.45 → 4.81 | 2.38 → 3.25 → 3.54 → 3.06 → 2.58 → 3.82 |

This is where the ablation's effect is clearest and most interpretable.
**Standard Model H's attention entropy decreases smoothly and
monotonically from layer 0 to layer 5** — a textbook pattern where early
layers attend diffusely (broad context-gathering) and late layers
specialize into sharp, focused heads. **The ablated model's entropy
trend is scrambled** (rises through layers 0-2, dips, rises again) —
there is no clean early-diffuse/late-sharp progression. The ablated
model's attention is also, on average, higher-entropy (more diffuse,
1.48 vs. 1.31) yet shorter-range (mean distance 3.10 vs. 3.98 tokens) and
skews more heavily toward local heads (16 local_sharp + 16 local_diffuse
= 32/48 heads local, vs. 24/48 in the standard model). Without positional
embeddings, attention has no signal for *how far away* a token is except
what it can infer from content and the causal mask's implicit
"more-recent-tokens-are-more-numerous" bias — a plausible explanation for
attention collapsing toward short-range, less-specialized patterns.

Heatmaps: [`bonus_ablation/hindi/figures/attention_heatmap_hin_ablation_L0.png`](../bonus_ablation/hindi/figures/attention_heatmap_hin_ablation_L0.png),
[`..._L5.png`](../bonus_ablation/hindi/figures/attention_heatmap_hin_ablation_L5.png)
— compare against standard Model H's
[`report/figures/hindi/attention_heatmap_hin_L0.png`](figures/hindi/attention_heatmap_hin_L0.png) /
[`..._L5.png`](figures/hindi/attention_heatmap_hin_L5.png).
Entropy/distance-by-layer plots:
[`attention_entropy_by_layer_hin_ablation.png`](../bonus_ablation/hindi/figures/attention_entropy_by_layer_hin_ablation.png),
[`attention_distance_by_layer_hin_ablation.png`](../bonus_ablation/hindi/figures/attention_distance_by_layer_hin_ablation.png).

**Known figure caveat**: the Kaggle session's Devanagari font check
reported `renders Devanagari: False` for this run (the `apt-get
fonts-lohit-deva fonts-indic` step did not take effect before the
heatmaps were generated) — axis tick labels showing Hindi tokens on the
attention heatmaps may render as missing-glyph boxes rather than
Devanagari text. The underlying attention-weight data and all numeric
metrics above are unaffected; only the heatmap PNGs' token-label text may
be affected.

## 3. What breaks without position information — prediction vs. observation

**Prediction (from theory).** Self-attention is permutation-invariant:
`Attention(QP, KP, VP)` for permutation `P` returns the permuted output
(`shared/model/positional.py` docstring). With `positional: none`, token
embeddings are the only per-position signal reaching the first layer, and
they are identical for a repeated token regardless of where it appears.
The causal mask still restricts *which* prior tokens a position may
attend to, but gives no representation of *how far back* — the model
should be unable to reliably distinguish order-dependent meaning (e.g.
"A से B बड़ा है" vs. "B से A बड़ा है" — reversed comparatives with the
same token multiset), predicting degraded performance especially on
comparative-reasoning material.

**What was actually observed.**
- **Confirmed, cleanly**: the causal language-modeling objective is
  measurably and consistently worse (+5.6% perplexity, both splits) —
  the model is a worse next-token predictor without positional
  information, exactly as predicted, though the gap is a moderate
  degradation rather than a collapse. The causal mask evidently *does*
  supply a usable, if crude, implicit position signal (a token's
  attendable-context size still grows monotonically with its position),
  which is likely why this is not closer to a bag-of-tokens collapse.
- **Confirmed, structurally**: attention's per-layer specialization
  pattern breaks down — the clean early-diffuse/late-sharp hierarchy
  standard Model H learns is absent in the ablated model, replaced by a
  non-monotonic, more locally-biased pattern. This is a direct, visible
  symptom of the model having no positional axis to organize a
  depth-wise attention hierarchy around.
- **Not confirmed at the generation-metric level**: BLEU/chrF/ROUGE-L and
  diversity/repetition are statistically indistinguishable between the
  two models. This project's models are small (~25M params) and trained
  for one epoch, and greedy decoding already degenerates similarly for
  both models (see §2.2) — the corpus-level generation metrics are not
  sensitive enough at this scale to pick up the LM-loss-level and
  attention-structure-level effects that clearly are present. This
  should be read as a property of the *measurement* (BLEU/chrF on
  short, one-epoch, ~25M-parameter open-ended generation, already
  discussed as a weak signal in `report/phase2_report.md`), not as
  evidence that positional embeddings don't matter.

## 4. Summary

Removing positional embeddings from Model H produces a real, reproducible
degradation in language-modeling quality (+5.6% perplexity on both
validation and test) and a clear, interpretable disruption of learned
attention structure (the standard model's clean layer-wise
diffuse-to-sharp entropy progression collapses into a non-monotonic,
more locally-biased pattern) — while leaving corpus-level generation
metrics (BLEU/chrF/ROUGE-L/diversity) essentially unchanged, because
those metrics are not sensitive at this model scale regardless of
positional information. The clearest, most direct evidence for what
positional embeddings contribute in this project is the perplexity/BPB
gap and the attention-entropy-by-layer comparison, not the generation
benchmark scores.

## Reproduction

```
python hindi/train/pretrain.py --config hindi/configs/hindi_model_ablation.yaml
python hindi/eval/evaluate_lm.py --config hindi/configs/hindi_model_ablation.yaml --checkpoint bonus_ablation/hindi/checkpoints/best.pt --splits validation test
python hindi/eval/evaluate_generation.py --config hindi/configs/hindi_model_ablation.yaml --checkpoint bonus_ablation/hindi/checkpoints/best.pt
python hindi/eval/analyze_attention.py --config hindi/configs/hindi_model_ablation.yaml --checkpoint bonus_ablation/hindi/checkpoints/best.pt --device cpu
```

Run via [`notebooks/bonus_ablation_kaggle.ipynb`](../notebooks/bonus_ablation_kaggle.ipynb)
on Kaggle (2×T4). Raw metrics: `bonus_ablation/hindi/eval_results/`.
Figures: `bonus_ablation/hindi/figures/`. Training log:
`bonus_ablation/hindi/logs/pretrain_hindi.log`. Checkpoint (not committed
to git — see README for the Drive link): `bonus_ablation/hindi/checkpoints/best.pt`.
