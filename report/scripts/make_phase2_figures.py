"""Generate every Phase 2 figure from the results the runs produced.

Reads the training CSVs and metric JSON written by pretraining/evaluation
and emits the figures the Phase 2 report references. Regenerable without a
GPU, so a lost figure never means re-running training.

    python report/scripts/make_phase2_figures.py

Lives under report/ rather than shared/ because it names this project's two
specific languages; everything under shared/ stays language-agnostic.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shared.eval import plots  # noqa: E402

FIGDIR = str(REPO / "report/figures")
FIGDIR_HIN = str(REPO / "report/figures/hindi")
FIGDIR_NEP = str(REPO / "report/figures/nepali")
FIGDIR_CMP = str(REPO / "report/figures/comparison")
FIGDIR_BY_CODE = {"hin": FIGDIR_HIN, "nep": FIGDIR_NEP}

# UTF-8 byte counts of each held-out split, from Phase 1. Needed for BPB and
# recorded here because the .txt corpora are gitignored (they live on Drive).
VAL_BYTES = {"hindi": 390_820_005, "nepali": 507_018_981}

LANGS = [
    ("hindi", "hin", "Model H (Hindi)", plots.COLOR_H),
    ("nepali", "nep", "Model L (Nepali)", plots.COLOR_L),
]


def load(path: Path):
    if not path.exists():
        print(f"  [skip] missing {path.relative_to(REPO)}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_bpb_column(df: pd.DataFrame, lang: str, n_val_tokens: int) -> pd.DataFrame:
    """Derive a per-eval BPB column from val_loss.

    BPB = (CE_nats * n_tokens) / (ln2 * n_bytes). Both counts describe the
    same held-out text, so the ratio n_tokens/n_bytes is a constant of the
    split and BPB is just val_loss scaled by it.
    """
    n_bytes = VAL_BYTES[lang]
    scale = n_val_tokens / (math.log(2) * n_bytes)
    df = df.copy()
    df["bits_per_byte"] = df["val_loss"] * scale
    return df


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(FIGDIR_HIN, exist_ok=True)
    os.makedirs(FIGDIR_NEP, exist_ok=True)
    os.makedirs(FIGDIR_CMP, exist_ok=True)
    font = plots.configure_devanagari_font()
    if font and not plots.verify_devanagari_rendering():
        print("  [warn] font cannot render Devanagari — labels may be boxes")

    logs, lm, gen, attn = {}, {}, {}, {}
    for lang, code, label, _color in LANGS:
        csv = REPO / lang / "logs" / f"training_log_{lang}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            m = load(REPO / lang / "eval/results/lm_metrics.json")
            n_tok = (m or {}).get("splits", {}).get(
                "validation", {}).get("n_tokens_in_split", 0)
            logs[lang] = add_bpb_column(df, lang, n_tok) if n_tok else df
            lm[lang] = m
        gen[lang] = load(REPO / lang / "eval/results/generation_metrics.json")
        attn[lang] = load(REPO / lang / "eval/results/attention_analysis.json")

    made = []

    # ---- F1/F2: per-language loss curves -------------------------------
    for lang, code, label, color in LANGS:
        if lang in logs:
            made.append(plots.plot_loss_curves(
                str(REPO / lang / "logs" / f"training_log_{lang}.csv"),
                label, f"loss_curves_{code}.png", FIGDIR_BY_CODE[code], color))

    h_csv = str(REPO / "hindi/logs/training_log_hindi.csv")
    n_csv = str(REPO / "nepali/logs/training_log_nepali.csv")

    # ---- F3: both models overlaid --------------------------------------
    if logs:
        made.append(plots.plot_loss_curves_both(
            h_csv, n_csv, "loss_curves_both.png", FIGDIR_CMP,
            "Model H (Hindi)", "Model L (Nepali)"))

    # ---- F4/F5: perplexity and BPB vs step ------------------------------
    # Written to temp CSVs so the shared plotting helper (which reads a
    # path, not a frame) can pick up the derived bits_per_byte column.
    tmp = REPO / "report/figures/.tmp"
    tmp.mkdir(exist_ok=True)
    paths = {}
    for lang in logs:
        p = tmp / f"{lang}.csv"
        logs[lang].to_csv(p, index=False)
        paths[lang] = str(p)

    if paths:
        made.append(plots.plot_metric_curves_both(
            paths.get("hindi", ""), paths.get("nepali", ""),
            "val_perplexity", "Validation perplexity",
            "Validation perplexity during pretraining",
            "perplexity_curves_both.png", FIGDIR_CMP,
            "Model H (Hindi)", "Model L (Nepali)"))
        made.append(plots.plot_metric_curves_both(
            paths.get("hindi", ""), paths.get("nepali", ""),
            "bits_per_byte", "Bits per byte",
            "Validation bits-per-byte — tokenizer-independent comparison",
            "bpb_curves_both.png", FIGDIR_CMP,
            "Model H (Hindi)", "Model L (Nepali)"))

    # ---- F6/F7: LR schedule and throughput -------------------------------
    if "hindi" in logs:
        made.append(plots.plot_lr_schedule(h_csv, "lr_schedule.png", FIGDIR_CMP))
    for lang, code, label, color in LANGS:
        if lang in logs:
            made.append(plots.plot_throughput(
                str(REPO / lang / "logs" / f"training_log_{lang}.csv"),
                label, f"throughput_{code}.png", FIGDIR_BY_CODE[code], color))

    # ---- F8-F11: generation quality --------------------------------------
    gm = {}
    for lang, code, label, _c in LANGS:
        if gen.get(lang):
            gm[label] = gen[lang]["settings"]
    if gm:
        made.append(plots.plot_generation_metrics(
            gm, "generation_metrics_both.png", FIGDIR_CMP))
        made.append(plots.plot_diversity_vs_temperature(
            gm, "diversity_vs_temperature.png", FIGDIR_CMP))
        made.append(plots.plot_repetition_vs_temperature(
            gm, "repetition_vs_temperature.png", FIGDIR_CMP))
        made.append(plots.plot_quality_diversity_tradeoff(
            gm, "quality_diversity_tradeoff.png", FIGDIR_CMP))

    # ---- F16/F17: layer x head entropy and distance grids ------------------
    for lang, code, label, _c in LANGS:
        a = attn.get(lang)
        if not a:
            continue
        made.append(plots.plot_layer_head_grid(
            np.array(a["entropy_by_layer_head"]),
            f"{label} — attention entropy by layer and head",
            "Mean entropy (nats)", f"attention_entropy_grid_{code}.png",
            FIGDIR_BY_CODE[code]))
        made.append(plots.plot_layer_head_grid(
            np.array(a["distance_by_layer_head"]),
            f"{label} — mean attention distance by layer and head",
            "Mean distance (tokens)", f"attention_distance_grid_{code}.png",
            FIGDIR_BY_CODE[code], cmap="magma"))

    # ---- F20: head taxonomy scatter ----------------------------------------
    if attn.get("hindi") or attn.get("nepali"):
        made.append(plots.plot_head_taxonomy(
            attn.get("hindi"), attn.get("nepali"),
            "attention_head_taxonomy.png", FIGDIR_CMP,
            "Model H (Hindi)", "Model L (Nepali)"))

    for p in tmp.glob("*.csv"):
        p.unlink()
    tmp.rmdir()

    print(f"\n{len(made)} figures written to report/figures/")
    return made


if __name__ == "__main__":
    main()
