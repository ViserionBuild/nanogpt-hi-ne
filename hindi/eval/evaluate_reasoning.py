"""Reasoning-task accuracy for Model H (Hindi) — Phase 3 §3.1.

3-way accuracy (seen-entity/seen-wording, unseen-entity, unseen-wording),
each over multiple seeds, for BOTH the pretrained-only and the finetuned
checkpoint, plus pretraining-validation perplexity before/after.

    python hindi/eval/evaluate_reasoning.py
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.finetune.runner import run_reasoning_evaluation   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "hindi/configs/hindi_finetune.yaml"))
    ap.add_argument("--model-config", default=str(REPO / "hindi/configs/hindi_model.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="finetuned checkpoint (default: hindi/checkpoints_ft/best.pt)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_reasoning_evaluation(args.config, args.model_config, REPO,
                             checkpoint=args.checkpoint, device=args.device)
