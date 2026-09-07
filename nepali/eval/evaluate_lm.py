"""Cross-entropy / perplexity / BPB for Model L (Nepali) — Table T1.

    python nepali/eval/evaluate_lm.py --checkpoint nepali/checkpoints/best.pt
    python nepali/eval/evaluate_lm.py --checkpoint ... --splits validation test
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.eval.runner import evaluate_language_model   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "nepali/configs/nepali_model.yaml"))
    ap.add_argument("--checkpoint", default=str(REPO / "nepali/checkpoints/best.pt"))
    ap.add_argument("--splits", nargs="+", default=["validation"],
                    help="validation and/or test; report test ONCE, at the end")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="limit batches (quick check); omit for the full split")
    a = ap.parse_args()
    evaluate_language_model(a.config, REPO, a.checkpoint, a.splits,
                            a.device, a.max_batches)
