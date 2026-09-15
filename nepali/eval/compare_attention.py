"""Pretrained-vs-finetuned attention comparison for Model L (Nepali) — Phase 3 §3.2.

Mirrors hindi/eval/compare_attention.py — see there for what this reports.

    python nepali/eval/compare_attention.py
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.finetune.runner import run_attention_comparison   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "nepali/configs/nepali_finetune.yaml"))
    ap.add_argument("--model-config", default=str(REPO / "nepali/configs/nepali_model.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="finetuned checkpoint (default: nepali/checkpoints_ft/best.pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--text", default=None,
                    help="example sentence (default: first test_seen prompt)")
    a = ap.parse_args()

    run_attention_comparison(a.config, a.model_config, REPO,
                             checkpoint=a.checkpoint, device=a.device,
                             example_text=a.text)
