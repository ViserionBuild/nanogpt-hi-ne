"""Attention analysis for Model H (Hindi) — Table T6, figures F12-F19.

Entropy and mean attention distance per (layer, head), a head taxonomy,
and heatmaps for an early and a late layer with the example sentence
rendered in Devanagari.

    python hindi/eval/analyze_attention.py --checkpoint hindi/checkpoints/best.pt
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.eval.runner import analyze_attention   # noqa: E402

DEFAULT_EXAMPLE = "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं।"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "hindi/configs/hindi_model.yaml"))
    ap.add_argument("--checkpoint", default=str(REPO / "hindi/checkpoints/best.pt"))
    ap.add_argument("--device", default="cpu",
                    help="CPU is fine — this is a handful of forward passes")
    ap.add_argument("--text", default=DEFAULT_EXAMPLE,
                    help="example sentence in the model's own language")
    a = ap.parse_args()
    analyze_attention(a.config, REPO, a.checkpoint, a.device, a.text)
