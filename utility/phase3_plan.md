# Phase 3 Plan — Reasoning Finetuning, Attention Analysis, and Final Report

Status: **planning only — no implementation started.** This document exists
to fix the design before writing any code, so Phase 3 (35 marks, final
deadline 16 Sep 2026) can be built against a settled plan rather than
improvised script-by-script. Nothing below is final until confirmed.

For what Phase 1/2 already produced and how, see
[phase1_workflow.md](phase1_workflow.md) and
[phase2_workflow.md](phase2_workflow.md). For the graded requirements
verbatim, see `LMA_Individual_Project_v1.pdf` §3 (pp. 9–12).

---

## 1. What the spec requires (§3, pp. 9–12)

Three deliverables, in order:

1. **Reasoning finetuning (§3.1).** Finetune Model H and Model L
   independently, each starting from its own Phase 2 pretrained checkpoint,
   on a **synthetic** dataset built in that language — not a downloaded
   benchmark. Focus: comparative and basic reasoning, especially
   transitive/ordering comparisons (`A > B, B > C ⇒ ?`; comparisons over
   height/age/price/quantity for two or three entities). Requirements:
   - Text in the target language's own script and natural phrasing (not
     English prompts with foreign names dropped in).
   - Train/validation/test splits; report size, template variety, and how
     train/test leakage is avoided (held-out entities and/or held-out
     relation patterns).
   - Generation scripts released so the dataset is reproducible.
   - Protocol: start from the language's own pretrained checkpoint, keep
     that language's tokenizer/vocabulary fixed, document every
     hyperparameter, save finetuned checkpoints in the same resumable
     format as pretraining, Drive-link them in the README.
   - Report pretrained-vs-finetuned accuracy/exact-match on the synthetic
     test set, qualitative successes/failures, and an H-vs-L comparison.

2. **Attention analysis, post-finetune (§3.2).** Reuse the Phase 2
   attention toolkit on the **finetuned** checkpoints. Compare
   pretrained-vs-finetuned heatmaps for at least one early and one late
   layer per model. Discuss whether finetuning changed local-vs-long-range
   attention or head specialization, especially on comparative-reasoning
   prompts.

3. **Final report (§3.3).** Consolidates the whole project (pulling in key
   Phase 1/2 figures/tables as needed) and must answer, explicitly:
   1. How did data scale and quality differ between Model H and Model L?
   2. How do language-modeling and reasoning results compare across the
      two resource tiers?
   3. What tokenizer/corpus factors most affected the lower-resource model?
   4. What evidence explains the observed differences?

Deliverables checklist (spec, p. 10): reasoning data-prep/generation
scripts, finetune scripts + configs, finetuned checkpoints (Drive links,
H and L), finetuning logs/metrics/qualitative examples, pretrain-vs-finetune
attention comparison, final comparison tables + error analysis, complete
final report + reproduction README.

---

## 2. What the course PDFs add on top of the spec

Five course PDFs were read in full for this phase:
`CL3410_L9_Supervised_finetuning`, `CL3410_L10_Post_trainingV3`,
`SFT Ultimate Guide`, `CL3410_M3_L6_Pretraining_Transformers`,
`CL3410_M4_L8_Context_Engineering` (all in `course_pdfs/`). The findings
below are what should ground the *implementation choices* the spec leaves
open — this is the "understand and apply course concepts" grading axis.

### 2.1 The load-bearing finding

L9's own final slide (p. 34) is close to a literal specification for
§3.1's methodology, and L9's worked example (pp. 3–4) is a miniature
prototype of exactly this phase's task: an **824,064-parameter, 4-layer
model, trained on a synthetic Hindi transitive-ordering corpus**
("Bhavana is older than Vaishali. Vaishali is older than Shalini. Who is
the youngest?"). The pretrained-only base model does not answer — it just
continues the passage. SFT is what installs the "answer directly" behavior.

L9 p. 34 prescribes, concretely:
- Generate ordering/comparison data from templates.
- Train on one entity pool, test on a **disjoint** pool.
- Write **four surface-form templates** (paraphrases of the same
  underlying task); train on three, **hold the fourth out entirely**.
- Report **three accuracy numbers separately** (see 2.2), with **more than
  one seed**.
- Report **pretraining validation perplexity before and after** finetuning.

### 2.2 The three-way evaluation rubric (L9 p. 27 / L10 p. 15)

| Test set | What a high score tells you |
|---|---|
| Seen entities, seen wording | training fit the distribution at all |
| Unseen entities, seen wording | generalizes past the exact training examples |
| Unseen wording (the held-out 4th template) | the behavior survives a rewording, not just memorized phrasing |

"The three degrade independently. Report all three." This maps directly
onto §3.1's "held-out entity names or held-out relation patterns"
requirement and gives it a concrete, gradeable structure rather than a
single blended accuracy number.

### 2.3 Loss masking mechanics (L9 pp. 6–11, L10 pp. 3–5)

Pretraining sums cross-entropy over every position; SFT sums only over the
**answer/completion token positions**, normalized by `|A|`:

```
L_SFT = -(1/|A|) * sum_{t in A} log p_theta(x_t | x_<t)
```

The model still *attends to* the full sequence (prompt + answer) in the
forward pass — masking applies to the **loss**, not to attention. An
explicit stop token (`<eos>`) is required in the sequence so the model
learns where to stop generating; without it, generation runs on past the
answer. This is already available in this project (`eos_id=3` is canonical
from Phase 1) and the loss function already supports selective masking —
`DecoderLM.forward` computes cross-entropy with `ignore_index=pad_id`
(`shared/model/transformer.py:161`), so a finetuning example can mask
prompt tokens by setting their target label to `pad_id` (or a dedicated
`ignore_id`) without touching the model or its forward pass.

**Course-flagged pitfall to avoid** (L9 p. 22): the demo model trained 3
epochs at lr 3e-4 on 3,000 synthetic examples, hit training loss 0.27, and
scored **0.000 exact-match on unseen names** — pure memorization of the 20
training names, not the underlying comparison logic. The corresponding
perplexity swing (1.33 → 194.7) is explicitly *not* to be read as
"finetuning made the model worse" — perplexity measures generic
next-token prediction, not task accuracy; the two are reported side by
side, never one as a proxy for the other. This is the direct cautionary
tale for building the entity pools and template set — see §4 below.

### 2.4 Data-quality checklist (L9/L10 "best practices" slide)

| Criterion | Question | Failure mode if skipped |
|---|---|---|
| Coverage | which behaviors must the model have? | an untrained behavior never appears |
| Diversity of form | how many wordings per behavior? | model learns the wording, not the behavior |
| Correctness | is every label actually right? | model learns the error fluently |
| Separation | does any test item leak into training? | inflated, undetectable overfitting |

Plus, from the SFT Ultimate Guide (§3.3, p. 20–21): **~1,000 examples** is
cited as a practical lower bound for effective finetuning; watch for
**rare-case coverage** (ties, longer chains); avoid a fixed answer phrase
repeated verbatim across too many examples, which teaches parroting rather
than reasoning (SFT Guide §7.6.1, p. 60).

### 2.5 What SFT reliably changes vs. does not (L9 p. 32 / L10 p. 12)

| Changes reliably | Changes unreliably |
|---|---|
| Output format | facts absent from pretraining |
| Register/style | reasoning the base model cannot do at all |
| Task behavior | — |

Relevant framing for explaining any Model H vs. Model L reasoning-accuracy
gap in the final report: if a capability isn't latent in the pretrained
model in some form, SFT mostly teaches a confidently-formatted wrong
answer, not the missing capability itself.

### 2.6 Attention analysis: not directly taught — say so

None of the five PDFs formally define attention entropy, attention
distance, or a local/content-based head taxonomy — those are this
project's own Phase 2 toolkit (`shared/eval/attention_tools.py`), reused
here. What the Pretraining deck (M3/L6) *does* teach and directly
underwrites this analysis: causal masking mechanics (Q·Kᵀ, upper-triangular
positions set to −∞ pre-softmax), and the parameter-distribution fact that
~65% of parameters sit in feed-forward layers vs. ~32% in attention
projections — i.e. attention is specifically the cross-token mixing
mechanism, which is the reason a pretrained-vs-finetuned *attention*
comparison (rather than a weight-norm comparison) is the informative one
to run. The Phase 3 report should say plainly that entropy/distance/head
taxonomy are a defensible extension of course material, not slide content
verbatim.

### 2.7 Context engineering notes relevant to template/eval design (M4/L8)

- **Ordering effects** ("Lost in the Middle", p. 26): put constraints
  early, keep the question near the evidence it depends on, put uncertain
  material in the middle rather than at the boundaries. Relevant to how
  multi-fact comparison chains are laid out in a template (state the
  facts, then ask the question at the end, adjacent to the last fact).
- **Devanagari token cost** (pp. 22–24): Devanagari characters are ~3
  bytes in UTF-8 against 1 for ASCII, so a fixed token budget carries less
  content in Hindi/Nepali than in English. Both models are already
  BPE-tokenized in-script (Phase 1), so this mainly matters for choosing
  how long the reasoning-prompt sequences can get before hitting
  `block_size=512`, and is a citable point for the "why fewer/shorter
  examples than an English SFT set would need" discussion.
- **Instruction-following as a pipeline product** (p. 15): pretraining
  supplies world/language knowledge, post-training supplies the
  disposition to answer directly, context supplies the specific request.
  Frames why the pretrained-only baseline is expected to fail this task
  (as in the L9 Hindi demo) and SFT is what's expected to fix it.

---

## 3. How this fits the existing codebase

Phase 1/2 conventions (to preserve exactly): `shared/` is architecture and
training/eval *logic*, language-agnostic by construction and enforced by
`tests/test_phase2.py::test_shared_code_never_names_a_language`; each
language directory is self-contained with its own configs/data/checkpoints;
every script is a thin per-language wrapper around a shared entry point;
every checkpoint carries model weights, optimizer state, scheduler state,
step, and config for exact resume; every metric is written to JSON, every
figure goes through `shared/eval/plots.finalize_axes` (title/xlabel/ylabel
mandatory), and the report is assembled from those JSON files by a script,
never typed by hand.

Proposed new pieces (naming only — not yet built):

```
shared/
  reasoning/           # NEW — language-agnostic synthetic-data machinery
    entities.py           # entity pools, disjoint train/test split logic
    templates.py           # template rendering, surface-form engine
    generate.py             # chain sampling (pairwise / transitive), label derivation
  finetune/            # NEW — SFT loop, reusing shared/train primitives
    dataset.py              # (prompt, answer) -> input_ids, loss-mask labels
    trainer.py               # thin wrapper/subclass over shared/train/trainer.py
  eval/
    reasoning_metrics.py  # NEW — 3-way accuracy (seen/seen, unseen-entity, unseen-wording)

hindi/
  reasoning_data/       # NEW — generated train/val/test JSON + generation config
  finetune/             # NEW — configs + entry-point scripts (mirrors train/)
  checkpoints_ft/       # NEW — finetuned checkpoints (gitignored -> Drive)

nepali/                 # mirrors hindi/ above

report/
  phase3_report.md       # NEW
  scripts/
    build_phase3_report.py  # NEW — assembles from JSON, mirrors phase2's script
```

Reused as-is, no changes needed: `shared/model/*` (same architecture, same
tokenizer/vocab per spec §3.1 "keep tokenizer/vocabulary fixed"),
`shared/train/checkpoint.py` (resumable format already generic),
`shared/eval/attention_tools.py` + `shared/eval/plots.py` (attention
comparison and figures), `DecoderLM.forward`'s existing `ignore_index`
support (loss masking, no model change required).

Open question this plan does **not** resolve yet (holding per your
instruction): exact task/template variety, exact number of templates and
whether to follow L9 p. 34's "4 templates, 3 train + 1 held-out" literally,
and how the attention comparison is framed in the report. These will be
decided in a follow-up planning pass before any code is written.

---

## 4. Risks / things to get right early

- **Memorization, not reasoning** (L9's own cautionary result). Entity
  pools must be large enough and disjoint enough between train/test that
  exact-match accuracy on unseen entities is a meaningful number, not
  another 0.000.
- **Tokenizer/vocab must stay frozen** — reasoning data is generated as
  *text*, then encoded with each language's existing Phase 1 SentencePiece
  model. No retraining, no new special tokens beyond what Phase 1 already
  defined (`pad=0, unk=1, bos=2, eos=3`), unless a real gap is found (e.g.
  an explicit answer-start marker) — to be decided before generation code
  is written.
- **Independence between H and L must hold in Phase 3 too**: separate
  synthetic corpora, separate finetuning runs, no shared checkpoints —
  same rule as Phases 1–2, just extended.
- **Report traceability**: every Phase 3 number must come from a JSON file
  the finetuning/eval scripts wrote, per the project's existing rule that
  the report is generated, not hand-typed.

---

## 5. Next step

Do not start implementation. Next planning pass (on request) should nail
down: task/template inventory and count, entity-pool sizes per language,
exact finetuning hyperparameters (LR relative to pretraining LR, epochs,
batch size), and how the attention pre/post comparison is presented —
then, only after that's confirmed, move to a `phase-3` branch and start
building `shared/reasoning/`.
