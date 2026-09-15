"""Reasoning-finetune Model H (Hindi) — Phase 3.

Starts from hindi/checkpoints/best.pt (Phase 2 pretrained) and finetunes
on hindi/reasoning_data/train.json. Resumes automatically from
hindi/checkpoints_ft/latest.pt if present.

    python hindi/finetune/finetune.py
    python hindi/finetune/finetune.py --no-resume
    python hindi/finetune/finetune.py --device cpu   # smoke test
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.finetune.runner import run_finetuning   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reasoning-finetune Model H (Hindi)")
    ap.add_argument("--config", default=str(REPO / "hindi/configs/hindi_finetune.yaml"))
    ap.add_argument("--model-config", default=str(REPO / "hindi/configs/hindi_model.yaml"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    run_finetuning(args.config, args.model_config, REPO, resume=not args.no_resume)
