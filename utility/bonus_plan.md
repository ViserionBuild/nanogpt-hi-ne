# Bonus Plan — No-Positional-Embeddings Ablation

Status: **complete.** Ablated **Hindi (Model H)**, trained on Kaggle
(2×T4, 13,183 steps, 2.62h), evaluated with the full Phase 2 suite, and
compared against standard Model H. Results and analysis:
[`report/bonus_report.md`](../report/bonus_report.md). Headline finding:
+5.6% perplexity (both val/test) and a scrambled attention-entropy
layer-wise trend vs. standard Model H's clean monotonic decrease;
generation-quality metrics (BLEU/chrF/ROUGE-L/diversity) were
statistically indistinguishable between the two — see the report's §3 for
why. This document fixes the design that was followed, on branch `bonus`
(based on `phase-3`, which is graded and closed — this work does not
touch anything under the phase-1/2/3 grading scope). For the graded
requirement verbatim, see `utility/LMA_Individual_Project_v1.pdf` p. 12
("Bonus (optional)").

---

## 1. What the spec requires (p. 12)

> As an optional bonus in the **same** project repository, pick **one** of
> your two languages and retrain a model with positional embeddings
> removed. Run the **full Phase 2 evaluation suite** on that ablated model
> (perplexity/BPB, generation metrics including BLEU/chrF/ROUGE-L,
> diversity diagnostics, and attention heatmaps/entropy/distance) and
> compare it to your standard model for that language. Bonus marks from
> this ablation can be used to recover marks lost elsewhere in the
> project; they do not raise the score above the project maximum of 100.
> Document the ablation setup, results, and a short explanation of what
> breaks without position information.

Three things this implies that are easy to miss:

1. **One language only** — not both. The comparison is ablated-H (or
   ablated-L) vs. that same language's standard Phase 2 model, not H vs. L.
2. **The full Phase 2 suite**, not a subset — same metrics, same plots,
   same attention analysis already built for Phase 2, just pointed at a
   second checkpoint.
3. **Bonus, not required** — hard constraints (p. 11) explicitly separate
   this from the main graded project ("the optional bonus ablation below
   is separate"). It must not touch Model H/L's existing checkpoints,
   configs, or results.

---

## 2. What already exists to reuse

Nothing here needs new code — `positional: none` is already implemented
and wired through the model factory:

- [`shared/model/positional.py`](../shared/model/positional.py) — three
  schemes (`learned`, `sinusoidal`, `none`), selected by a single config
  string. `NoPositional` (line 98) returns an all-zero tensor for the
  positional term, making the model permutation-invariant: causal masking
  still limits *which* tokens can be attended to, but the model has no
  signal for *where* in the sequence a token sits.
- [`hindi/train/pretrain.py`](../hindi/train/pretrain.py) /
  [`nepali/train/pretrain.py`](../nepali/train/pretrain.py) — take
  `--config <path>`, `--no-resume`, `--device`, `--micro-batch-size`,
  `--grad-accum-steps`. No code change needed — only a config with a
  different `positional` value and different output paths.
- Phase 2 eval suite, already built and already run once per language:
  `evaluate_lm.py` (PPL/BPB), `evaluate_generation.py` (BLEU/chrF/ROUGE-L
  + diversity), `analyze_attention.py` (heatmaps/entropy/distance),
  `verify_causal_masking.py`. All take `--config` and `--checkpoint`, so
  pointing them at an ablated checkpoint + ablated config is the entire
  integration.
- `utility/RUN_PHASE2.md` §"Optional — Bonus ablation" already sketches
  this exact approach ("Set `positional: none` in a copied config and
  rerun the full evaluation suite") and budgets **~3h** for it on Kaggle's
  2×T4, consistent with the ~2.6h each standard model took to pretrain.

So this bonus is a **config fork + a training run + eval-suite reruns +
one comparison writeup** — not new implementation.

---

## 3. Language choice: Hindi (Model H) — confirmed

Picking Hindi over Nepali:

- Model H is the higher-resource model (648.0M train tokens vs. Nepali's
  659.5M — comparable scale, so this isn't about corpus size). The choice
  is really about which language's *standard* results are the clearer
  baseline to ablate against: Hindi's Phase 2/3 reports are the ones
  referenced first in `README.md` and `report/phase2_report.md`, so the
  ablated-vs-standard contrast sits next to material a grader will already
  have open.
- No technical difference in risk or runtime — both models are the same
  architecture/size (~25.3M params) and took ~2.6h to pretrain, so the
  ablation is expected to cost about the same either way.

This is a judgment call, not a constraint from the spec (either language
is valid). Reasoning stated here so it's not silently arbitrary.

---

## 4. Plan

### 4.1 New, isolated config (no shared state with Model H)

Add `hindi/configs/hindi_model_ablation.yaml`: a full copy of
`hindi_model.yaml` with exactly these changes —

```yaml
model:
  positional: none            # was: learned

paths:
  checkpoints: bonus_ablation/hindi/checkpoints
  logs: bonus_ablation/hindi/logs
  eval_results: bonus_ablation/hindi/eval_results
  figures: bonus_ablation/hindi/figures
```

Everything else (`vocab_size`, `n_layer`, `n_head`, `d_model`, `d_ff`,
`block_size`, dropout, `tie_weights`, all of `training:`, all of
`evaluation:`) stays byte-identical to `hindi_model.yaml`, so the ablation
isolates exactly one variable: positional information. Same tokenizer,
same train/val/test data, same seed, same step count, same optimizer
schedule.

Output paths route into the **already-created, empty**
[`bonus_ablation/`](../bonus_ablation/) directory (top-level, sibling to
`hindi/`/`nepali/`) rather than into `hindi/checkpoints/` — this keeps the
bonus artifact physically separate from Model H's graded Phase 2
checkpoint, so there is never a risk of the ablation overwriting or being
confused with graded work. Layout:

```
bonus_ablation/
  hindi/
    checkpoints/     # best.pt, latest.pt, *.config.json — same resumable
                      # format as Phase 2 (mandatory per p. 4: "A run that
                      # cannot resume after interruption will lose marks")
    logs/
    eval_results/     # evaluate_lm.py / evaluate_generation.py JSON output
    figures/           # attention heatmaps, comparison plots
```

### 4.2 Retrain

```
python hindi/train/pretrain.py --config hindi/configs/hindi_model_ablation.yaml
```

Same 13,183 steps / cosine schedule / AdamW / checkpoint-resume behavior
as the standard run — only the model's `forward()` differs, because
`build_positional("none", ...)` is selected instead of `"learned"`.
Expect (per `RUN_PHASE2.md`'s existing note) validation loss to plateau
well above Model H's baseline: with no position signal, the only order
information available to the model is whatever the causal mask implies
positionally-blind attention (i.e. none directly — the mask restricts
*which* tokens are visible, not *where* they are), so the model degrades
toward a bag-of-tokens-per-prefix-length model.

### 4.3 Run the full Phase 2 evaluation suite on the ablated checkpoint

Same four scripts, same flags, just `--config`/`--checkpoint` pointed at
the ablation's own paths — no script edits:

```
python hindi/eval/verify_causal_masking.py --config hindi/configs/hindi_model_ablation.yaml --checkpoint bonus_ablation/hindi/checkpoints/best.pt
python hindi/eval/evaluate_lm.py           --config hindi/configs/hindi_model_ablation.yaml --checkpoint bonus_ablation/hindi/checkpoints/best.pt --splits validation test
python hindi/eval/evaluate_generation.py   --config hindi/configs/hindi_model_ablation.yaml --checkpoint bonus_ablation/hindi/checkpoints/best.pt --split validation
python hindi/eval/analyze_attention.py     --config hindi/configs/hindi_model_ablation.yaml --checkpoint bonus_ablation/hindi/checkpoints/best.pt
```

This produces, per the spec's explicit list: perplexity/BPB, BLEU/chrF/
ROUGE-L, diversity diagnostics (Distinct-1/2, repetition rate), and
attention heatmaps/entropy/mean-attention-distance — the same shapes of
output Phase 2 already produced for standard Model H, just against the
ablated checkpoint, written under `bonus_ablation/hindi/eval_results/`
and `bonus_ablation/hindi/figures/` instead of `hindi/eval/results/` and
`report/figures/hindi/`.

### 4.4 Comparison + writeup

New `bonus_ablation/report.md` (or `report/bonus_report.md` — final
location TBD, see open question below), covering exactly what p. 12 asks
for:

1. **Setup**: config diff (only `positional: none` + output paths changed),
   confirming everything else held fixed.
2. **Side-by-side results table**: standard Model H vs. ablated Model H —
   val/test PPL, BPB, BLEU-4, chrF(++), ROUGE-L, Distinct-1/2, repetition
   rate.
3. **Attention comparison**: heatmaps for one early + one late layer,
   ablated vs. standard, plus entropy/mean-attention-distance — expect the
   ablated model's attention to show no learned locality/positional
   structure (uniform or content-only patterns, since nothing in the
   ablated model's inputs varies with position).
4. **Explanation of what breaks**: self-attention is permutation-invariant
   (`shared/model/positional.py` docstring, line 3–6) — `Attention(QP, KP,
   VP)` for permutation `P` returns the permuted output. With `positional:
   none`, token embeddings are the *only* per-position signal into the
   first layer, and they are identical for a repeated token regardless of
   where it appears. The model can still use the causal mask to restrict
   *which* prior tokens it may look at, but cannot represent order among
   them — e.g. it cannot distinguish "A से B बड़ा है" from "B से A बड़ा है"
   ("A is bigger than B" vs. the reverse) if the token multiset is
   identical, which directly predicts degraded performance on the
   comparative-reasoning material this project's reasoning finetuning
   (Phase 3) depends on. This should be stated as a **prediction check**
   against the observed generation/attention results, not asserted
   without evidence.

### 4.5 README + Drive links

Update the top-level `README.md` status table row ("Bonus ablation (no
positional embeddings) | not started") to reflect completion, add the
ablated checkpoint's Drive link alongside the existing Model H / Model L
checkpoint links (per p. 2: "Large artifacts ... must be uploaded to
Google Drive ... Do not commit large binary artifacts to git" — the
ablated `best.pt` is the same size class as the standard checkpoints and
must not be committed to git either).

### 4.6 Running it on Kaggle (confirmed compute target)

A **dedicated notebook**,
[`notebooks/bonus_ablation_kaggle.ipynb`](../notebooks/bonus_ablation_kaggle.ipynb),
rather than reusing `phase2_kaggle.ipynb` — the two runs need different
datasets attached (Hindi only, no Nepali) and this run has no
language-toggle cell (it is always Hindi, always the ablated config), so
folding it into the Phase 2 notebook would add a branch path to code that
is otherwise deliberately uniform across H/L. Same infrastructure
otherwise: same three-dataset-mount pattern, same 2×T4 accelerator, same
Cell 1 (environment) copied verbatim from `phase2_kaggle.ipynb` since the
code-mounting logic doesn't change.

**Before packaging**: `hindi/configs/hindi_model_ablation.yaml` must be
committed (or at least `git add`-ed) on the `bonus` branch — the code
archive is built from `git ls-files`
(`./package_for_kaggle.sh code`), so an untracked config silently would
not ship to Kaggle. `bonus_ablation/` (currently empty) ships as an empty
directory; the training cell creates the actual output subdirectories at
runtime, same as Phase 2's `hindi/checkpoints/` does.

**Correctness note discovered while building the notebook**:
`hindi/train/count_params.py` does **not** accept `--config` — it has no
`argparse` at all and hardcodes `hindi/configs/hindi_model.yaml`
internally (it's a one-off script for the report's Table T2). Passing
`--config` to it is silently ignored, so it would report the *standard*
model's breakdown regardless of what's asked — a trap for this ablation
specifically, since the check must run against the ablated config to be
meaningful. The notebook works around this by calling
`shared.train.runner.print_parameter_report(CONFIG, ...)` directly instead
of shelling out to `count_params.py`. Every other reused script
(`pretrain.py`, `evaluate_lm.py`, `evaluate_generation.py`,
`analyze_attention.py`, `verify_causal_masking.py`) does take `--config`
correctly — verified by reading each script's `argparse` block.

**Gates cell**: `pytest tests/test_phase2.py` (unchanged — nothing in it
is Hindi-standard-config-specific), `print_parameter_report` against the
ablation config (see above — expect **25,059,328**: standard Model H's
25,321,472 minus the 262,144-parameter learned positional table
(512 x 512) that `NoPositional` removes entirely, since it holds no
`nn.Parameter` of its own. That exact 262,144 difference — and nothing
else — is itself worth stating in the writeup as a sanity check that this
is a clean single-variable ablation, not a differently-sized model), then
`verify_causal_masking.py --config hindi/configs/hindi_model_ablation.yaml`.

**Train cell**: `pretrain.py --config hindi/configs/hindi_model_ablation.yaml`.
Same resume behavior as Phase 2: re-running this exact command after a
Kaggle session dies resumes from
`bonus_ablation/hindi/checkpoints/latest.pt` automatically (checkpoint
format unchanged — model weights, optimizer state, scheduler state, step,
config, per p. 4's mandatory-checkpointing requirement).

**Backup cell**: zips `bonus_ablation/hindi/{checkpoints,logs}` to
`/kaggle/working` for download → Google Drive, same pattern as Phase 2's
Cell 5.

**Evaluate cell**: the four Phase 2 eval scripts against the ablated
checkpoint, per §4.3 above, run with `--config
hindi/configs/hindi_model_ablation.yaml --checkpoint
bonus_ablation/hindi/checkpoints/best.pt`.

**Artifacts cell**: zips `bonus_ablation/hindi/{figures,eval_results,logs}`
for download and commit to the `bonus` branch — same "if it's not in the
branch it's not graded" rule applies to the bonus's figures/tables as to
the main project's (p. 2).

**Budget**: `RUN_PHASE2.md` already budgets **~3h** for this ablation
inside the existing 30 GPU-hours/week allowance, alongside the ~10h
already spent on the two standard models + evaluation — leaving headroom
even after this run.

---

## 5. What this explicitly does NOT do

- Does not touch `hindi/checkpoints/`, `hindi/configs/hindi_model.yaml`,
  or any Phase 1–3 graded artifact — the ablation is fully additive, under
  `bonus_ablation/` and one new sibling config file.
- Does not run on Nepali/Model L — spec says pick **one** language.
- Does not finetune the ablated model on the reasoning dataset — the spec
  scopes the bonus to "the full Phase 2 evaluation suite," which is
  pretraining-stage evaluation only, not Phase 3's reasoning finetuning.
- Does not attempt to make the ablated model competitive — a worse model
  here is the expected and correct outcome; the deliverable is the
  documented comparison and explanation, not a fix.

---

## 6. Decisions

1. **Language**: Hindi (Model H) — confirmed (§3).
2. **Compute**: Kaggle, 2×T4, same workflow as Phase 2 — confirmed (§4.6).
3. **Report location**: `report/bonus_report.md`, alongside
   `phase2_report.md`/`phase3_report.md` — not a path under
   `bonus_ablation/`. p. 3's repo-layout guidance is explicit that
   "every figure and table that you want graded" must be inside
   `report/` (or notebooks) on the branch; `bonus_ablation/` holds run
   artifacts (checkpoints, logs, raw eval JSON, figures), and
   `report/bonus_report.md` is the human-readable writeup that cites
   them, matching how `phase2_report.md`/`phase3_report.md` already cite
   `report/figures/`.

## 7. Implementation order

1. ~~Create `hindi/configs/hindi_model_ablation.yaml` (§4.1).~~ **Done.**
2. ~~Create `notebooks/bonus_ablation_kaggle.ipynb` (§4.6).~~ **Done.**
3. ~~`git add` the config and notebook so `package_for_kaggle.sh code`
   picks them up.~~ **Done** (also fixed a real bug found along the way:
   `package_for_kaggle.sh` resolved its repo root from its own script
   location, so running it after the script moved to `utility/` packaged
   only 8 files from that subfolder — fixed to resolve via
   `git rev-parse --show-toplevel`).
4. ~~Run gates locally first.~~ **Done** — caught and fixed the
   25,321,472-vs-25,059,328 parameter-count bug described below before
   burning a third Kaggle run on it.
5. ~~Package and upload the updated code dataset to Kaggle.~~ **Done** —
   `vikashkumar999/lma-code`, `lma-data-hindi` (no `lma-data-nepali`
   needed).
6. ~~Run `notebooks/bonus_ablation_kaggle.ipynb` on Kaggle.~~ **Done** —
   kernel `vikashkumar999/lma-bonus-ablation`, version 4 (versions 1-3
   failed on the parameter-count assertion — see below), completed in
   2.62h. All cells ran cleanly through Cell 8; causal-masking
   verification passed; training reached all 13,183 steps.
7. ~~Download and copy `bonus_ablation/hindi/{eval_results,figures,logs}`
   into the repo and to the external Drive backup.~~ **Done** — copied
   into `bonus_ablation/hindi/` in this repo (checkpoint `.pt` files
   intentionally excluded per the project's large-artifact convention;
   `.gitignore`'s `bonus_ablation/checkpoints/**/*.pt` pattern was also
   fixed to `bonus_ablation/*/checkpoints/**/*.pt` since the real path
   includes a language segment the original pattern missed) and to
   `/media/vikash/Entertainment/LMA/Final/Hindi/bonus_ablation/`
   (`checkpoints_and_logs.zip`, `eval_artifacts.zip`, both checksum-
   verified against the Kaggle kernel's output). **Still pending**:
   uploading the checkpoint to Google Drive and linking it from
   `README.md` — the checkpoint itself is only backed up locally
   (repo-excluded `.pt`, plus the two zips above) for now.
8. ~~Write `report/bonus_report.md`.~~ **Done** — see the report for the
   full analysis; headline results are summarized at the top of this
   document.
9. ~~Update `README.md`'s bonus status row.~~ **Done.**

### What actually went wrong along the way (for the record)

Two of the four Kaggle kernel runs failed on the exact same
`AssertionError: expected 25,321,472 parameters, got 25,059,328` in the
gates cell, despite two attempts to fix it:

1. First fix attempt updated `hindi/configs/hindi_model_ablation.yaml`'s
   sibling files and the **`lma-code` dataset** (an *input* the notebook
   copies `shared/`/`hindi/` code from at runtime) — but never touched
   the **kernel's own notebook file**, which is a separately-versioned
   artifact holding the actual executing cells. Restarting the kernel
   just re-ran the same stale, unfixed cell again.
2. Second attempt (`kaggle kernels push` with the corrected notebook as
   the kernel's `code_file`) actually replaced the executing code —
   confirmed by pulling the live kernel back down and grep-checking for
   the corrected constant before considering it fixed. That run
   (version 4) succeeded end-to-end.

The underlying constant was always correct in intent — `NoPositional`
adds zero parameters relative to itself — the bug was comparing against
the wrong baseline: it *replaces* `LearnedPositional`, a 262,144-parameter
table (512×512), so the correct expected total for the ablated config is
25,059,328, not standard Model H's 25,321,472. This was caught early by
the assertion itself (as designed) rather than silently producing a
mis-sized model.
