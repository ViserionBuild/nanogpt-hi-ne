"""Reasoning-task accuracy for Model L (Nepali) — Phase 3 §3.1.

Mirrors hindi/eval/evaluate_reasoning.py — see there for what this reports.

    python nepali/eval/evaluate_reasoning.py
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.finetune.runner import run_reasoning_evaluation   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "nepali/configs/nepali_finetune.yaml"))
    ap.add_argument("--model-config", default=str(REPO / "nepali/configs/nepali_model.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="finetuned checkpoint (default: nepali/checkpoints_ft/best.pt)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_reasoning_evaluation(args.config, args.model_config, REPO,
                             checkpoint=args.checkpoint, device=args.device)
