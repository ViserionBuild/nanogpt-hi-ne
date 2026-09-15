"""Generate every Phase 3 figure from the results the finetuning/eval runs produced.

Reads finetune_log_*.csv and reasoning_metrics.json/attention_comparison.json
written by hindi/nepali's finetune + eval scripts. The pretrained-vs-finetuned
attention heatmaps and the per-model entropy/distance-shift figures are
already written eagerly by run_attention_comparison(); this script covers
the figures that need BOTH languages together (SFT loss curves overlay,
3-way accuracy comparison, perplexity before/after comparison) which is why
it lives here rather than inside the per-language eval scripts, exactly the
same division Phase 2 uses (make_phase2_figures.py owns comparison figures).

    python report/scripts/make_phase3_figures.py

Lives under report/ rather than shared/ because it names this project's two
specific languages; everything under shared/ stays language-agnostic.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.eval import plots  # noqa: E402

FIGDIR_CMP = str(REPO / "report/figures/comparison")

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


def main() -> None:
    os.makedirs(FIGDIR_CMP, exist_ok=True)
    made = []

    # --- per-model SFT loss curves ---------------------------------------
    for lang, code, label, color in LANGS:
        log_csv = REPO / lang / "logs" / f"finetune_log_{lang}.csv"
        figdir = REPO / "report" / "figures" / lang
        if log_csv.exists():
            made.append(plots.plot_finetune_loss_curves(
                str(log_csv), label, f"finetune_loss_{code}.png", str(figdir), color,
            ))
        else:
            print(f"  [skip] missing {log_csv.relative_to(REPO)}")

    # --- 3-way accuracy + perplexity before/after, both models -----------
    reasoning = {}
    ppl = {}
    for lang, code, label, color in LANGS:
        rpath = REPO / lang / "eval" / "results" / "reasoning_metrics.json"
        r = load(rpath)
        if r is None:
            continue
        before_acc = {s: r["before_finetuning"]["three_way_accuracy"][s]["mean_accuracy"]
                     for s in ("test_seen", "test_unseen_entity", "test_unseen_wording")}
        after_acc = {s: r["after_finetuning"]["three_way_accuracy"][s]["mean_accuracy"]
                    for s in ("test_seen", "test_unseen_entity", "test_unseen_wording")}
        reasoning[label] = {"before": before_acc, "after": after_acc}
        ppl[label] = {
            "before": r["before_finetuning"]["pretraining_perplexity"]["perplexity"],
            "after": r["after_finetuning"]["pretraining_perplexity"]["perplexity"],
        }

    if reasoning:
        made.append(plots.plot_three_way_accuracy(
            reasoning, "reasoning_accuracy_3way.png", FIGDIR_CMP,
        ))
    if ppl:
        made.append(plots.plot_perplexity_before_after(
            ppl, "pretraining_perplexity_before_after.png", FIGDIR_CMP,
        ))

    print(f"\n[make_phase3_figures] wrote {len(made)} figures:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
