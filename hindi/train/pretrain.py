"""Pretrain Model H (Hindi) — Phase 2.

Resumes automatically from hindi/checkpoints/latest.pt if present, so a
killed Kaggle session is recovered by re-running the same command.

    python hindi/train/pretrain.py
    python hindi/train/pretrain.py --no-resume
    python hindi/train/pretrain.py --max-steps 200 --device cpu   # smoke test
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.train.runner import run_pretraining   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Pretrain Model H (Hindi)")
    ap.add_argument("--config",
                    default=str(REPO / "hindi/configs/hindi_model.yaml"))
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing checkpoint and start over")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override max_steps (smoke tests only — changing it "
                         "changes the cosine schedule)")
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--micro-batch-size", type=int, default=None,
                    help="lower this if the GPU runs out of memory")
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    args = ap.parse_args()

    override = {k: v for k, v in [
        ("max_steps", args.max_steps),
        ("device", args.device),
        ("micro_batch_size", args.micro_batch_size),
        ("grad_accum_steps", args.grad_accum_steps),
    ] if v is not None}

    run_pretraining(args.config, REPO, resume=not args.no_resume,
                    override=override)


if __name__ == "__main__":
    main()
