"""All Phase 2 figures.

The spec is blunt about labelling: "Each plot must include a title,
x-label, y-label, and legend (if applicable). Plots missing labels or
legends may receive zero marks for that component." Every figure here goes
through :func:`finalize_axes`, which refuses to save an axes without a
title and both axis labels, so a missing label is a crash rather than a
silently unlabelled figure.

**Devanagari fonts.** matplotlib's default DejaVu Sans contains no
Devanagari glyphs and drops them silently, rendering every token label as
an empty box. :func:`configure_devanagari_font` searches the installed
fonts for one that actually has the glyphs and registers it. Verified on
this machine: ``Lohit Devanagari`` works; ``Noto Sans Devanagari`` is
installed but not in matplotlib's cache and must be added by file path.
Call it once at import; check one heatmap visually before generating a set.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm      # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

DPI = 150
FIGDIR = "report/figures"

# One colour per model, held constant across every figure so the two are
# visually identifiable at a glance throughout the report.
COLOR_H = "#1f77b4"      # first model (higher-resource)
COLOR_L = "#d62728"      # second model (lower-resource)
COLOR_ABL = "#7f7f7f"    # ablation

# Fonts known to carry Devanagari, most preferred first.
_DEVANAGARI_CANDIDATES = [
    "Noto Sans Devanagari", "Lohit Devanagari", "Samyak Devanagari",
    "Mangal", "Nirmala UI", "Kohinoor Devanagari", "FreeSerif",
]
_DEVANAGARI_SEARCH_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"), "/kaggle/input",
]


def configure_devanagari_font(verbose: bool = True) -> Optional[str]:
    """Find and register a font with Devanagari coverage.

    Returns the family name it selected, or None if none was found (in
    which case Devanagari axis labels will render as boxes and the caller
    should fall back to token ids).
    """
    available = {f.name for f in fm.fontManager.ttflist}

    for name in _DEVANAGARI_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            if verbose:
                print(f"[plots] Devanagari font: {name}")
            return name

    # Not in the cache -- look for the actual files and register them.
    for root in _DEVANAGARI_SEARCH_DIRS:
        if not os.path.isdir(root):
            continue
        for path in Path(root).rglob("*.ttf"):
            if any(k in path.name.lower()
                   for k in ("devanagari", "lohit-deva", "mangal", "samyak")):
                try:
                    fm.fontManager.addfont(str(path))
                    name = fm.FontProperties(fname=str(path)).get_name()
                    plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                    if verbose:
                        print(f"[plots] Devanagari font registered: {name} "
                              f"({path})")
                    return name
                except Exception:
                    continue

    if verbose:
        print("[plots] WARNING: no Devanagari font found — token labels will "
              "render as boxes. Install fonts-noto or fonts-lohit-deva.")
    return None


def verify_devanagari_rendering(text: str = "भारत एक विशाल देश है।") -> bool:
    """Return True if the current font can actually draw *text*.

    matplotlib only warns about missing glyphs, so this promotes that
    warning into a checkable boolean.
    """
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.text(0.1, 0.5, text)
        ax.axis("off")
        fig.canvas.draw()
        plt.close(fig)
    return not any("missing from font" in str(w.message) for w in caught)


def finalize_axes(ax, title: str, xlabel: str, ylabel: str,
                  legend: bool = False, grid: bool = True) -> None:
    """Apply the mandatory labels; refuse to proceed without them."""
    if not title or not xlabel or not ylabel:
        raise ValueError(
            "every figure needs a title, an x-label and a y-label — the spec "
            "allows zero marks for a plot missing them"
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if legend:
        ax.legend(fontsize=9, framealpha=0.9)
    if grid:
        ax.grid(alpha=0.3, linestyle="--")


def _save(fig, name: str, outdir: str = FIGDIR) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path}")
    return path


# ------------------------------------------------------- F1-F7 training curves

def plot_loss_curves(log_csv: str, language: str, out_name: str,
                     outdir: str = FIGDIR, color: str = COLOR_H) -> str:
    """F1/F2: train and validation loss vs. step for one model."""
    import pandas as pd
    df = pd.read_csv(log_csv)
    tr = df.dropna(subset=["train_loss"])
    va = df.dropna(subset=["val_loss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tr["step"], tr["train_loss"], color=color, alpha=0.6, lw=1,
            label="Training loss")
    if len(va):
        ax.plot(va["step"], va["val_loss"], color=color, lw=2,
                marker="o", ms=4, label="Validation loss")
    finalize_axes(ax, f"{language} — training and validation loss",
                  "Training step", "Cross-entropy loss (nats/token)",
                  legend=True)
    return _save(fig, out_name, outdir)


def plot_loss_curves_both(log_h: str, log_l: str, out_name: str,
                          outdir: str = FIGDIR,
                          label_h: str = "Model H",
                          label_l: str = "Model L") -> str:
    """F3: both models overlaid — the headline comparison figure."""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for path, color, label in ((log_h, COLOR_H, label_h),
                               (log_l, COLOR_L, label_l)):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        tr = df.dropna(subset=["train_loss"])
        va = df.dropna(subset=["val_loss"])
        ax.plot(tr["step"], tr["train_loss"], color=color, alpha=0.35, lw=1,
                label=f"{label} — train")
        if len(va):
            ax.plot(va["step"], va["val_loss"], color=color, lw=2,
                    marker="o", ms=4, label=f"{label} — validation")
    finalize_axes(ax, "Model H vs. Model L — pretraining loss",
                  "Training step", "Cross-entropy loss (nats/token)",
                  legend=True)
    return _save(fig, out_name, outdir)


def plot_metric_curves_both(log_h: str, log_l: str, column: str,
                            ylabel: str, title: str, out_name: str,
                            outdir: str = FIGDIR,
                            label_h: str = "Model H",
                            label_l: str = "Model L") -> str:
    """F4/F5: validation perplexity or BPB vs. step, both models."""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for path, color, label in ((log_h, COLOR_H, label_h),
                               (log_l, COLOR_L, label_l)):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if column not in df.columns:
            continue
        d = df.dropna(subset=[column])
        if len(d):
            ax.plot(d["step"], d[column], color=color, lw=2,
                    marker="o", ms=4, label=label)
    finalize_axes(ax, title, "Training step", ylabel, legend=True)
    return _save(fig, out_name, outdir)


def plot_lr_schedule(log_csv: str, out_name: str = "lr_schedule.png",
                     outdir: str = FIGDIR) -> str:
    """F6: realised learning rate — shows warmup then cosine decay."""
    import pandas as pd
    df = pd.read_csv(log_csv).dropna(subset=["learning_rate"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["step"], df["learning_rate"], color="#2ca02c", lw=2,
            label="Learning rate")
    finalize_axes(ax, "Learning-rate schedule (linear warmup + cosine decay)",
                  "Training step", "Learning rate", legend=True)
    return _save(fig, out_name, outdir)


def plot_throughput(log_csv: str, language: str, out_name: str,
                    outdir: str = FIGDIR, color: str = COLOR_H) -> str:
    """F7: training throughput in tokens/second."""
    import pandas as pd
    df = pd.read_csv(log_csv).dropna(subset=["tokens_per_sec"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["step"], df["tokens_per_sec"], color=color, lw=1, alpha=0.7,
            label="Tokens / second")
    finalize_axes(ax, f"{language} — training throughput",
                  "Training step", "Tokens / second", legend=True)
    return _save(fig, out_name, outdir)


# -------------------------------------------------- F8-F11 generation quality

def plot_generation_metrics(metrics: Dict[str, Dict], out_name: str,
                            outdir: str = FIGDIR) -> str:
    """F8: grouped bars of BLEU-4 / chrF / ROUGE-L per decoding setting.

    ``metrics`` maps model label -> {setting -> metric dict}.
    """
    settings = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
    names = [("bleu_4", "BLEU-4"), ("chrf", "chrF"), ("rouge_l_f", "ROUGE-L")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(settings))
    width = 0.35

    for ax, (key, label) in zip(axes, names):
        for i, (model_label, per_setting) in enumerate(metrics.items()):
            vals = [per_setting.get(s, {}).get(key, 0.0) for s in settings]
            color = COLOR_H if i == 0 else COLOR_L
            ax.bar(x + (i - 0.5) * width, vals, width, label=model_label,
                   color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["Greedy", "T=0.5", "T=1.0", "T=1.5"])
        finalize_axes(ax, f"{label} by decoding setting",
                      "Decoding setting", f"{label} score", legend=True)
    return _save(fig, out_name, outdir)


def plot_diversity_vs_temperature(metrics: Dict[str, Dict], out_name: str,
                                  outdir: str = FIGDIR) -> str:
    """F9: Distinct-1 and Distinct-2 against temperature."""
    settings = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
    xs = [0.0, 0.5, 1.0, 1.5]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, (label, per_setting) in enumerate(metrics.items()):
        color = COLOR_H if i == 0 else COLOR_L
        for key, style, mark in (("distinct_1", "-", "o"),
                                 ("distinct_2", "--", "s")):
            ys = [per_setting.get(s, {}).get(key, 0.0) for s in settings]
            ax.plot(xs, ys, style, color=color, marker=mark, lw=2,
                    label=f"{label} — {key.replace('_', '-')}")
    finalize_axes(ax, "Generation diversity vs. sampling temperature",
                  "Sampling temperature (0 = greedy)",
                  "Distinct-n ratio", legend=True)
    return _save(fig, out_name, outdir)


def plot_repetition_vs_temperature(metrics: Dict[str, Dict], out_name: str,
                                   outdir: str = FIGDIR) -> str:
    """F10: repeated 3-gram fraction against temperature."""
    settings = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
    xs = [0.0, 0.5, 1.0, 1.5]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, per_setting) in enumerate(metrics.items()):
        ys = [per_setting.get(s, {}).get("repetition_rate_3gram", 0.0)
              for s in settings]
        ax.plot(xs, ys, color=COLOR_H if i == 0 else COLOR_L,
                marker="o", lw=2, label=label)
    finalize_axes(ax, "Repetition rate vs. sampling temperature",
                  "Sampling temperature (0 = greedy)",
                  "Repeated 3-gram fraction", legend=True)
    return _save(fig, out_name, outdir)


def plot_quality_diversity_tradeoff(metrics: Dict[str, Dict], out_name: str,
                                    outdir: str = FIGDIR) -> str:
    """F11: chrF against Distinct-2 — why no single metric suffices."""
    settings = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
    labels = {"greedy": "Greedy", "temp_0.5": "T=0.5",
              "temp_1.0": "T=1.0", "temp_1.5": "T=1.5"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (model_label, per_setting) in enumerate(metrics.items()):
        color = COLOR_H if i == 0 else COLOR_L
        xs = [per_setting.get(s, {}).get("distinct_2", 0.0) for s in settings]
        ys = [per_setting.get(s, {}).get("chrf", 0.0) for s in settings]
        ax.plot(xs, ys, "-o", color=color, lw=1.5, ms=9, alpha=0.85,
                label=model_label)
        for s, x, y in zip(settings, xs, ys):
            ax.annotate(labels[s], (x, y), textcoords="offset points",
                        xytext=(6, 5), fontsize=8, color=color)
    finalize_axes(ax, "Quality vs. diversity trade-off across decoding settings",
                  "Distinct-2 (higher = more diverse)",
                  "chrF (higher = closer to reference)", legend=True)
    return _save(fig, out_name, outdir)


# ------------------------------------------------------ F12-F20 attention

def plot_attention_heatmap(attn: np.ndarray, tokens: Sequence[str],
                           layer: int, language: str, out_name: str,
                           outdir: str = FIGDIR, max_heads: int = 8,
                           max_tokens: int = 24) -> str:
    """F12-F15: one panel per head for a single layer.

    ``attn`` is ``(n_head, T, T)``. The lower-triangular structure visible
    in every panel is a second, visual confirmation that the causal mask
    is working.
    """
    n_head = min(attn.shape[0], max_heads)
    T = min(attn.shape[-1], max_tokens, len(tokens))
    ncols = 4
    nrows = int(np.ceil(n_head / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for h in range(n_head):
        ax = axes[h]
        im = ax.imshow(attn[h, :T, :T], cmap="viridis", aspect="auto",
                       vmin=0, vmax=float(attn[h, :T, :T].max()))
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(tokens[:T], rotation=90, fontsize=6)
        ax.set_yticklabels(tokens[:T], fontsize=6)
        finalize_axes(ax, f"Layer {layer}, head {h}",
                      "Key position (attended to)",
                      "Query position (attending from)", grid=False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="Attention weight")

    for ax in axes[n_head:]:
        ax.axis("off")

    fig.suptitle(f"{language} — attention weights, layer {layer}",
                 fontsize=13, fontweight="bold", y=1.005)
    return _save(fig, out_name, outdir)


def plot_layer_head_grid(matrix: np.ndarray, title: str, cbar_label: str,
                         out_name: str, outdir: str = FIGDIR,
                         cmap: str = "viridis") -> str:
    """F16/F17: a layer x head grid of entropy or mean distance."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] < matrix.mean() else "black")
    finalize_axes(ax, title, "Head index", "Layer index", grid=False)
    fig.colorbar(im, ax=ax, label=cbar_label)
    return _save(fig, out_name, outdir)


def plot_by_layer(matrix: np.ndarray, title: str, ylabel: str,
                  out_name: str, outdir: str = FIGDIR) -> str:
    """F18/F19: one line per head, across layers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in range(matrix.shape[1]):
        ax.plot(range(matrix.shape[0]), matrix[:, h], marker="o", lw=1.5,
                alpha=0.8, label=f"head {h}")
    ax.plot(range(matrix.shape[0]), matrix.mean(axis=1), "k--", lw=2.5,
            label="mean")
    finalize_axes(ax, title, "Layer index", ylabel, legend=True)
    return _save(fig, out_name, outdir)


def plot_head_taxonomy(analysis_h: Dict, analysis_l: Optional[Dict],
                       out_name: str, outdir: str = FIGDIR,
                       label_h: str = "Model H",
                       label_l: str = "Model L") -> str:
    """F20: entropy vs. distance scatter — head specialisation at a glance."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for analysis, color, label in ((analysis_h, COLOR_H, label_h),
                                   (analysis_l, COLOR_L, label_l)):
        if not analysis:
            continue
        heads = analysis["heads"]
        xs = [h["entropy_nats"] for h in heads]
        ys = [h["mean_distance_tokens"] for h in heads]
        # marker size encodes depth, so early vs. late layers are separable
        sizes = [22 + 13 * h["layer"] for h in heads]
        ax.scatter(xs, ys, s=sizes, c=color, alpha=0.65, edgecolors="white",
                   linewidth=0.5, label=f"{label} (size = layer index)")
    finalize_axes(ax,
                  "Attention head taxonomy: local/positional vs. content-based",
                  "Attention entropy (nats) — higher = more diffuse",
                  "Mean attention distance (tokens) — higher = longer range",
                  legend=True)
    return _save(fig, out_name, outdir)


# --------------------------------------------------- Phase 3 pretrain-vs-finetune

def plot_entropy_distance_before_after(
    analysis_before: Dict, analysis_after: Dict, out_name: str,
    outdir: str = FIGDIR, language: str = "",
) -> str:
    """F23: mean entropy and mean distance per layer, pretrained vs. finetuned.

    Phase 3 3.2: "Compare pretrained vs. finetuned heatmaps... Comment on
    whether finetuning changed local vs. long-range attention or head
    specialization." The heatmaps themselves (plot_attention_heatmap,
    called once per checkpoint) show this per-head; this figure shows the
    same shift summarized per layer so a systematic sharpen/broaden trend
    is visible without inspecting every head individually.
    """
    ent_b = np.array(analysis_before["mean_entropy_per_layer"])
    ent_a = np.array(analysis_after["mean_entropy_per_layer"])
    dist_b = np.array(analysis_before["mean_distance_per_layer"])
    dist_a = np.array(analysis_after["mean_distance_per_layer"])
    layers = range(len(ent_b))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(layers, ent_b, "o--", color=COLOR_ABL, lw=2, label="Pretrained")
    axes[0].plot(layers, ent_a, "o-", color=COLOR_H, lw=2, label="Finetuned")
    finalize_axes(axes[0], f"{language} — attention entropy, pretrained vs. finetuned",
                 "Layer index", "Mean entropy (nats)", legend=True)

    axes[1].plot(layers, dist_b, "o--", color=COLOR_ABL, lw=2, label="Pretrained")
    axes[1].plot(layers, dist_a, "o-", color=COLOR_H, lw=2, label="Finetuned")
    finalize_axes(axes[1], f"{language} — mean attention distance, pretrained vs. finetuned",
                 "Layer index", "Mean distance (tokens)", legend=True)
    return _save(fig, out_name, outdir)


def plot_head_taxonomy_shift(analysis_before: Dict, analysis_after: Dict,
                             out_name: str, outdir: str = FIGDIR,
                             language: str = "") -> str:
    """F24: per-head entropy/distance, arrows from pretrained to finetuned
    position — shows which heads moved and in which direction, not just
    the per-layer mean (two heads can shift oppositely and cancel out in
    the layer mean)."""
    heads_b = {(h["layer"], h["head"]): h for h in analysis_before["heads"]}
    heads_a = {(h["layer"], h["head"]): h for h in analysis_after["heads"]}

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for key in heads_b:
        if key not in heads_a:
            continue
        b, a = heads_b[key], heads_a[key]
        ax.annotate(
            "", xy=(a["entropy_nats"], a["mean_distance_tokens"]),
            xytext=(b["entropy_nats"], b["mean_distance_tokens"]),
            arrowprops=dict(arrowstyle="->", color=COLOR_H, alpha=0.55, lw=1.3),
        )
    ax.scatter([h["entropy_nats"] for h in analysis_before["heads"]],
              [h["mean_distance_tokens"] for h in analysis_before["heads"]],
              c=COLOR_ABL, s=30, alpha=0.7, label="Pretrained", zorder=3)
    ax.scatter([h["entropy_nats"] for h in analysis_after["heads"]],
              [h["mean_distance_tokens"] for h in analysis_after["heads"]],
              c=COLOR_H, s=30, alpha=0.85, label="Finetuned", zorder=3)
    finalize_axes(
        ax, f"{language} — per-head attention shift after finetuning",
        "Attention entropy (nats) — higher = more diffuse",
        "Mean attention distance (tokens) — higher = longer range", legend=True,
    )
    return _save(fig, out_name, outdir)


# -------------------------------------------------------- Phase 3 finetuning

def plot_finetune_loss_curves(log_csv: str, language: str, out_name: str,
                              outdir: str = FIGDIR, color: str = COLOR_H) -> str:
    """SFT train/val loss vs. step — the finetuning analogue of F1/F2."""
    import pandas as pd
    df = pd.read_csv(log_csv)
    tr = df.dropna(subset=["train_loss"])
    va = df.dropna(subset=["val_loss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tr["step"], tr["train_loss"], color=color, alpha=0.5, lw=1,
           marker=".", ms=3, label="Training loss (masked, answer tokens only)")
    if len(va):
        ax.plot(va["step"], va["val_loss"], color=color, lw=2,
               marker="o", ms=4, label="Validation loss")
    finalize_axes(ax, f"{language} — reasoning SFT loss",
                 "Finetuning step", "Masked cross-entropy loss (nats/answer-token)",
                 legend=True)
    return _save(fig, out_name, outdir)


def plot_three_way_accuracy(results: Dict[str, Dict], out_name: str,
                            outdir: str = FIGDIR) -> str:
    """Grouped bars: 3-way accuracy (seen/unseen-entity/unseen-wording),
    pretrained-only vs. finetuned, per model.

    ``results`` maps model label -> {"before": {split: mean_accuracy},
    "after": {split: mean_accuracy}}.
    """
    splits = ["test_seen", "test_unseen_entity", "test_unseen_wording"]
    split_labels = ["Seen entity\n+ seen wording", "Unseen entity\n+ seen wording",
                    "Unseen entity\n+ unseen wording"]
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5))
    axes = np.atleast_1d(axes)

    for ax, (model_label, r) in zip(axes, results.items()):
        x = np.arange(len(splits))
        width = 0.35
        before = [r["before"].get(s, 0.0) for s in splits]
        after = [r["after"].get(s, 0.0) for s in splits]
        ax.bar(x - width / 2, before, width, color=COLOR_ABL, alpha=0.85,
              label="Pretrained only")
        ax.bar(x + width / 2, after, width, color=COLOR_H, alpha=0.85,
              label="Finetuned")
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        finalize_axes(ax, f"{model_label} — reasoning accuracy",
                     "Test split (course 3-way rubric)", "Exact-match accuracy",
                     legend=True)
    return _save(fig, out_name, outdir)


def plot_perplexity_before_after(results: Dict[str, Dict], out_name: str,
                                 outdir: str = FIGDIR) -> str:
    """Pretraining-validation perplexity before/after SFT, per model.

    Course point (L9 p.22): this number is EXPECTED to move and must not
    be read as "finetuning made the model worse" — it measures a
    distribution shift onto the narrow synthetic task, not task ability.
    """
    labels = list(results.keys())
    before = [results[m]["before"] for m in labels]
    after = [results[m]["after"] for m in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, before, width, color=COLOR_ABL, alpha=0.85, label="Pretrained")
    ax.bar(x + width / 2, after, width, color=COLOR_H, alpha=0.85, label="Finetuned")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    finalize_axes(ax, "Pretraining-validation perplexity before/after reasoning SFT",
                 "Model", "Perplexity (original pretraining validation set)",
                 legend=True)
    return _save(fig, out_name, outdir)


# --------------------------------------------------------------- ablation

def plot_ablation_comparison(metrics: Dict[str, Dict], out_name: str,
                             outdir: str = FIGDIR) -> str:
    """F22: standard vs. no-positional across every headline metric."""
    keys = [("perplexity", "Perplexity"), ("bits_per_byte", "BPB"),
            ("bleu_4", "BLEU-4"), ("chrf", "chrF"), ("rouge_l_f", "ROUGE-L")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4.3))
    for ax, (key, label) in zip(np.atleast_1d(axes), keys):
        labels = list(metrics.keys())
        vals = [metrics[m].get(key) or 0.0 for m in labels]
        ax.bar(labels, vals, color=[COLOR_H, COLOR_ABL][:len(labels)],
               alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        finalize_axes(ax, label, "Model variant", label)
    return _save(fig, out_name, outdir)
