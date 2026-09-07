"""Verify causal masking for Model L (Nepali).

Spec 2.1 stage 2 requires showing empirically that changing token t+1 does
not change the logits at position t. Run this BEFORE spending GPU time: it
catches an off-by-one mask or a mask applied after softmax, the two bugs
that produce a model which appears to train while having leaked the future.

    python nepali/eval/verify_causal_masking.py
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.train.runner import run_causal_verification   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "nepali/configs/nepali_model.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="optional trained checkpoint; masking is structural "
                         "so random weights are sufficient")
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    out = run_causal_verification(args.config, REPO, args.checkpoint,
                                  args.seq_len)
    print("\nPASSED" if out["passed"] else "\nFAILED")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
