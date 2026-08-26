# Phase 2 — Step-by-step

Follow these in order. Tick each box as you go. Everything before Step 5
runs on your laptop and takes minutes; Steps 5–7 are the two long Kaggle
sessions.

Companion files:
- [`notebooks/phase2_kaggle.ipynb`](notebooks/phase2_kaggle.ipynb) — the notebook to upload
- [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) — reference for *why*, and troubleshooting
- [`report/phase2_plan.md`](report/phase2_plan.md) — the full design and figure/table spec

---

## Step 1 — Verify locally (2 min)

Prove the code is sound before anything is uploaded.

```bash
cd ~/Desktop/git_folders/LMA_IDV_Proj

python -m pytest tests/test_phase2.py -q       # expect: 22 passed
python hindi/train/count_params.py             # expect: 25,321,472 — CONFIRMED
python hindi/eval/verify_causal_masking.py     # expect: PASSED, max delta 0.0
python nepali/eval/verify_causal_masking.py    # expect: PASSED, max delta 0.0
```

- [ ] 22 tests pass
- [ ] Parameter count is exactly **25,321,472** for both languages
- [ ] Causal masking passes with max delta **0.0** (not "close to zero")

> If any of these fail, stop and fix it here. They take one minute locally
> and would otherwise waste a six-hour GPU session.

---

## Step 2 — Build the archives (~10 min, mostly the data zip)

```bash
./package_for_kaggle.sh          # first time: code + both data archives
```

Produces:

| File | Size | Re-upload when |
|---|---|---|
| `dist/lma-code.zip` | ~4.4 MB | any code change |
| `dist/lma-data-hindi.zip` | ~1.3 GB | never |
| `dist/lma-data-nepali.zip` | ~1.3 GB | never |

- [ ] All three archives exist in `dist/`

> The `.bin` files compress ~52% (uint32 ids in 0–11,999 leave the high
> bytes zero), so 5.5 GB of corpora upload as ~2.6 GB. Verified lossless.

---

## Step 3 — Create three Kaggle datasets (~30–60 min, upload-bound)

At [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**,
once per archive:

| Dataset title | Upload this |
|---|---|
| `lma-code` | `dist/lma-code.zip` |
| `lma-data-hindi` | `dist/lma-data-hindi.zip` |
| `lma-data-nepali` | `dist/lma-data-nepali.zip` |

Set all three **Private**. Kaggle unzips automatically.

- [ ] `lma-code` created
- [ ] `lma-data-hindi` created
- [ ] `lma-data-nepali` created

> **Later code changes**: `./package_for_kaggle.sh code`, then the dataset's
> **New Version** button. Seconds, not another gigabyte.

---

## Step 4 — Create the notebook (5 min)

**New Notebook** → **File → Import Notebook** → upload
`notebooks/phase2_kaggle.ipynb`.

Then in the right-hand sidebar:

- [ ] **Input → Add Input**: add all three datasets
- [ ] **Accelerator**: **GPU P100**
- [ ] **Persistence**: **Files only**
- [ ] **Internet**: **On**

> Choose P100 over T4 x2. A 25.3M-parameter model fits in a fraction of one
> 16 GB card, so a second GPU adds DataParallel overhead and a harder
> resume path for no speedup.

---

## Step 5 — Train Model H, Hindi (~6 h)

In the notebook, Cell 2:

```python
LANGUAGE = 'hindi'
```

Then **Run All**. Watch for:

| Cell | Expect |
|---|---|
| 1 | `GPU: Tesla P100-PCIE-16GB`, `hindi: 648,012,574 train tokens` |
| 3 | `22 passed`, `25,321,472 — CONFIRMED`, `PASSED` |
| 4 | loss falling from ~9.4 (= `ln 12000`, the uniform baseline) |
| 5 | a `hindi_phase2_*.zip` in the Output tab |

- [ ] Training reached 13,183 steps
- [ ] Downloaded the checkpoint zip and uploaded it to Google Drive
- [ ] Downloaded the artifacts zip from Cell 7

> **If the session dies**, re-run the identical notebook. Resume is
> bit-exact. Do **not** add `--max-steps` to "finish faster" — that changes
> the cosine schedule and therefore the learning rate at every remaining
> step.
>
> **If CUDA OOM**, use the fallback cell:
> `--micro-batch-size 16 --grad-accum-steps 6` (same 49,152-token batch).

---

## Step 6 — Train Model L, Nepali (~6 h, new session)

Same notebook, Cell 2:

```python
LANGUAGE = 'nepali'
```

**Run All.** Nepali runs **13,417** steps — more than Hindi, because its
corpus is larger (659,499,440 vs. 648,012,574 tokens). Equal step counts
would not be equal data.

- [ ] Training reached 13,417 steps
- [ ] Checkpoint zip on Drive
- [ ] Artifacts zip downloaded

---

## Step 7 — Commit the results (30 min)

Unzip both artifact archives into the repo and commit. **This is the step
that determines your marks** — the spec says graders will not open Drive or
external dashboards to find figures.

```bash
cd ~/Desktop/git_folders/LMA_IDV_Proj
unzip -o ~/Downloads/hindi_artifacts_*.zip
unzip -o ~/Downloads/nepali_artifacts_*.zip

python report/scripts/build_phase2_report.py    # fills every table from the JSON

git add report/ hindi/logs hindi/eval/results nepali/logs nepali/eval/results
git commit -m "Phase 2 results: trained checkpoints, metrics, and figures"
git push origin phase-2
```

- [ ] `report/figures/` holds the loss curves, generation plots, and heatmaps
- [ ] `*/eval/results/*.json` committed
- [ ] `*/logs/training_log_*.csv` committed (these are the loss curves)
- [ ] `report/phase2_report.md` regenerated with real numbers
- [ ] Drive links for both checkpoints pasted into `README.md`
- [ ] Pushed to the **`phase-2`** branch

---

## Step 8 — Fill in the analysis

`build_phase2_report.py` generates every table, but three things need you
to write them:

- [ ] **The H vs. L gap.** Compare PPL and BPB. If `PPL_L > PPL_H` while
      `BPB_L ≈ BPB_H`, the gap is a tokenizer/morphology artifact rather
      than a modeling failure — Nepali's fertility is 1.4394 against Hindi's
      1.3042, so at an equal token budget Model L sees less linguistic
      content. That contrast is the most interesting finding available.
- [ ] **Why each generation metric is or is not informative.** BLEU-4 will
      be near zero and that is correct, not a bug: a 25M model trained one
      epoch, continuing open-ended text against a *single* reference, has no
      path to 4-gram overlap. chrF is the informative one for Devanagari;
      Distinct-1/2 and repetition rate are what catch degenerate output.
- [ ] **Which heads are local vs. content-based**, from the entropy/distance
      scatter, and how the two models differ.

---

## Optional — Bonus ablation (~6 h)

Recovers marks lost elsewhere. Set `positional: none` in a copied config
and rerun the full evaluation suite. Expect the loss to plateau well above
baseline: without positions, the only order information left is what the
causal mask implies.

---

## Budget

Kaggle gives **30 GPU-hours/week**, 12 h per session.

| | Hours |
|---|---|
| Hindi | ~6 |
| Nepali | ~6 |
| Evaluation | ~1 |
| Ablation (optional) | ~6 |
| **Total** | **~19**, leaving ~11 h spare |

---

## If something breaks

| Symptom | Fix |
|---|---|
| `FileNotFoundError` on a `.bin` | Cell 1 prints where it looked; check the dataset is attached |
| CUDA OOM | fallback cell: `--micro-batch-size 16 --grad-accum-steps 6` |
| Session died | re-run the same notebook — resume is bit-exact |
| Heatmap labels are boxes | re-run the `apt-get` in Cell 1, then the font check in Cell 6 |
| BLEU is ~0 | expected — see Step 8 |
| Gates fail | **do not train**; fix locally first |

Fuller explanations in [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) §5.
