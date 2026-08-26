# Running Phase 2 on Kaggle

Everything needed to get from this laptop to two trained models. Read §0
first — the one decision that matters is made there.

---

## 0. The one thing to get right

**Package the code and the data separately, and upload the data once.**

| Archive | Size | Changes | Upload |
|---|---|---|---|
| `dist/lma-code.zip` | **4.4 MB** | every time you fix a line | freely, as often as you like |
| `dist/lma-data-hindi.zip` | **~1.3 GB** | never (Phase 1 output, frozen) | **once** |
| `dist/lma-data-nepali.zip` | **~1.3 GB** | never | **once** |

The `.bin` files are 5.5 GB raw but compress to ~2.6 GB, because `uint32`
token ids in the range 0–11,999 leave the high bytes almost always zero.
Measured: 200 MB → 95 MB, a 52% reduction at `zip -1`. Verified lossless —
a compressed-then-extracted `validation.bin` memmaps to byte-identical
tokens.

If you bundle everything into one archive, every one-line fix costs a
multi-gigabyte re-upload. Don't.

---

## 1. On your laptop — build the archives

```bash
cd ~/Desktop/git_folders/LMA_IDV_Proj

./package_for_kaggle.sh          # both code and data (first time)
./package_for_kaggle.sh code     # code only (every time after)
```

The code archive is built from `git ls-files`, so it cannot drift from the
repo: anything committed is included, anything gitignored (checkpoints,
`.bin`, raw corpora) is excluded automatically.

**What goes in the 4.4 MB code archive** (169 files):

```
shared/model/       transformer, attention, blocks, positional encodings
shared/train/       dataset, trainer, scheduler, checkpointing
shared/eval/        LM metrics, generation, attention analysis, plots
hindi/  nepali/     configs, train/ and eval/ entry points
        tokenizer/  the .model and .vocab files (~8 MB across all sizes)
        metadata/   Phase 1 statistics JSON
tests/              22 correctness gates
report/             the Phase 1 report and the Phase 2 plan
```

**What is deliberately excluded**: the `.bin` corpora (separate archive),
`.venv/`, `.git/`, checkpoints, and the spec PDF.

---

## 2. On Kaggle — create the datasets

Go to **kaggle.com/datasets → New Dataset**, and make **three** of them:

| Dataset title | Upload | Slug it will get |
|---|---|---|
| `lma-code` | `dist/lma-code.zip` | `<username>/lma-code` |
| `lma-data-hindi` | `dist/lma-data-hindi.zip` | `<username>/lma-data-hindi` |
| `lma-data-nepali` | `dist/lma-data-nepali.zip` | `<username>/lma-data-nepali` |

Kaggle unzips uploaded archives automatically, so each dataset appears at
`/kaggle/input/<slug>/` with the directory structure preserved.

Set them **Private**. They stay private for grading — the TAs get the
Google Drive links from the README, not Kaggle.

> **Updating the code dataset later**: open the dataset → **New Version** →
> upload the new `lma-code.zip`. Notebooks pick up the new version on their
> next run. This is the loop you will use dozens of times; it takes seconds.

---

## 3. On Kaggle — create the notebook

**New Notebook**, then in the right-hand sidebar:

- **Input → Add Input** → add all three datasets
- **Accelerator → GPU P100**
- **Persistence → Files only** (keeps `/kaggle/working` between runs)
- **Internet → On** (only needed if you `pip install sacrebleu`)

### Why P100 and not T4 x2

The model is 25.3M parameters. It fits in a fraction of one 16 GB card, so
a second GPU adds `DataParallel` overhead, gradient-sync complexity, and a
harder checkpoint/resume path — for no speedup. Pick the single P100.

### Cell 1 — set up

```python
import os, shutil, sys, subprocess

# Kaggle mounts datasets read-only. Copy the code somewhere writable so
# logs, checkpoints and figures can be written next to it.
REPO = "/kaggle/working/LMA_IDV_Proj"
if os.path.exists(REPO):
    shutil.rmtree(REPO)
shutil.copytree("/kaggle/input/lma-code", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

# Symlink the corpora into the paths the configs expect. Symlinks, not
# copies: the .bin files are 5.5 GB uncompressed and /kaggle/working has
# a 20 GB budget that the checkpoints also need.
for lang in ("hindi", "nepali"):
    dst = f"{REPO}/{lang}/data/tokenized"
    os.makedirs(dst, exist_ok=True)
    src = f"/kaggle/input/lma-data-{lang}/{lang}/data/tokenized"
    for f in os.listdir(src):
        link = os.path.join(dst, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(src, f), link)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sacrebleu", "rouge-score"], check=False)

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)
      if torch.cuda.is_available() else 0)
```

Expected: `GPU: Tesla P100-PCIE-16GB`, `VRAM GB: 17.1`.

### Cell 2 — gates before spending GPU time

```python
!python -m pytest tests/test_phase2.py -q
!python hindi/train/count_params.py
!python hindi/eval/verify_causal_masking.py
```

Expect `22 passed`, `Target 25,321,472 — CONFIRMED`, and `PASSED` with max
delta `0.0`. **If any of these fail, stop.** They take under a minute and
they catch the bugs that would otherwise waste a six-hour session.

### Cell 3 — train

```python
!python hindi/train/pretrain.py
```

Runs 13,183 steps (~5–6 h on P100). Re-running the exact same command after
a session dies resumes from `hindi/checkpoints/latest.pt` automatically.

Then, in a later session, `!python nepali/train/pretrain.py` (13,417 steps).

### Cell 4 — save the checkpoint off Kaggle

```python
!cd /kaggle/working/LMA_IDV_Proj && zip -q -1 -r /kaggle/working/hindi_ckpt.zip \
    hindi/checkpoints hindi/logs
print("download /kaggle/working/hindi_ckpt.zip from the Output tab")
```

**Do this every session.** `/kaggle/working` survives between runs of the
*same* notebook with Persistence on, but is not a backup — download the zip
and put it on Drive, then paste the link into the README.

### Cell 5 — evaluate

```python
for lang in ("hindi", "nepali"):
    ck = f"{lang}/checkpoints/best.pt"
    if os.path.exists(ck):
        !python {lang}/eval/evaluate_lm.py --checkpoint {ck} --splits validation
        !python {lang}/eval/evaluate_generation.py --checkpoint {ck}
        !python {lang}/eval/analyze_attention.py --checkpoint {ck}

!python report/scripts/build_phase2_report.py
```

Then zip `report/figures/`, `*/eval/results/` and `*/logs/` and download
them — those are the graded artifacts and they must end up committed to the
`phase-2` branch. **The spec says graders will not open Drive or external
dashboards to find figures; if it is not in the branch, it is not graded.**

---

## 4. Session budget

Kaggle gives **30 GPU-hours/week**, 12 h maximum per session.

| Work | Hours |
|---|---|
| Hindi pretraining | ~6 |
| Nepali pretraining | ~6 |
| Evaluation, both models | ~1 |
| Bonus ablation (optional) | ~6 |
| **Total** | **~19**, leaving ~11 h of headroom |

One epoch per language is the right budget on its own merits, not just to
fit: Chinchilla-optimal for 25M parameters is ~500M tokens and both corpora
exceed 640M, so a second epoch buys little and risks memorization.

---

## 5. Troubleshooting

**CUDA out of memory.** Lower the micro-batch and raise accumulation to
keep the effective batch at 49,152 tokens:

```python
!python hindi/train/pretrain.py --micro-batch-size 16 --grad-accum-steps 6
```

**Session died mid-run.** Re-run the identical command. Resume is
bit-exact — verified that 40 uninterrupted steps produce weights identical
to 20 steps + kill + resume, to zero difference. Do **not** pass
`--max-steps` to "finish faster": `max_steps` defines the cosine schedule,
so changing it changes the learning rate at every remaining step.

**Devanagari renders as boxes in the heatmaps.** Kaggle's font set differs
from a local Linux box. Install fonts before the eval cell:

```python
!apt-get install -qq -y fonts-lohit-deva fonts-indic > /dev/null
import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)
```

`shared/eval/plots.py` searches for a Devanagari-capable font and registers
it by file path if it is not in matplotlib's cache. Check one heatmap
visually before generating the rest — matplotlib drops missing glyphs with
only a warning.

**`FileNotFoundError` on a `.bin`.** The symlinks in Cell 1 did not resolve.
Check the real path with `!ls /kaggle/input/lma-data-hindi/hindi/data/tokenized/`
— if Kaggle flattened the archive, the files may sit directly at
`/kaggle/input/lma-data-hindi/` and the symlink source needs adjusting.

**Generation metrics are near zero.** Expected, not a bug. A 25M model
trained one epoch, continuing open-ended text against a *single* reference,
cannot achieve 4-gram overlap. chrF is the informative metric for
Devanagari; Distinct-1/2 and repetition rate are what catch degenerate
output. Report the numbers *and* the reason.

---

## 6. Quick reference

```bash
# laptop
./package_for_kaggle.sh code            # after any code change
```

```python
# kaggle, per session
!python -m pytest tests/test_phase2.py -q          # gates
!python hindi/train/pretrain.py                    # trains / resumes
!python hindi/eval/evaluate_lm.py --checkpoint hindi/checkpoints/best.pt
!python report/scripts/build_phase2_report.py      # regenerate the report
```
