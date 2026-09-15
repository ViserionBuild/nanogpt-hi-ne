"""Pretrained-vs-finetuned attention comparison for Model H (Hindi) — Phase 3 §3.2.

Reuses the Phase 2 attention toolkit unchanged on both hindi/checkpoints/best.pt
(pretrained) and hindi/checkpoints_ft/best.pt (finetuned), for the same
comparative-reasoning example sentence.

    python hindi/eval/compare_attention.py
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.finetune.runner import run_attention_comparison   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "hindi/configs/hindi_finetune.yaml"))
    ap.add_argument("--model-config", default=str(REPO / "hindi/configs/hindi_model.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="finetuned checkpoint (default: hindi/checkpoints_ft/best.pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--text", default=None,
                    help="example sentence (default: first test_seen prompt)")
    a = ap.parse_args()

    run_attention_comparison(a.config, a.model_config, REPO,
                             checkpoint=a.checkpoint, device=a.device,
                             example_text=a.text)
