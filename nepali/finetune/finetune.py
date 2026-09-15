"""Reasoning-finetune Model L (Nepali) — Phase 3.

Starts from nepali/checkpoints/best.pt (Phase 2 pretrained) and finetunes
on nepali/reasoning_data/train.json. Resumes automatically from
nepali/checkpoints_ft/latest.pt if present.

    python nepali/finetune/finetune.py
    python nepali/finetune/finetune.py --no-resume
    python nepali/finetune/finetune.py --device cpu   # smoke test
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.finetune.runner import run_finetuning   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reasoning-finetune Model L (Nepali)")
    ap.add_argument("--config", default=str(REPO / "nepali/configs/nepali_finetune.yaml"))
    ap.add_argument("--model-config", default=str(REPO / "nepali/configs/nepali_model.yaml"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    run_finetuning(args.config, args.model_config, REPO, resume=not args.no_resume)
