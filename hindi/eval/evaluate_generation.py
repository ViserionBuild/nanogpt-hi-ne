"""Generation metrics for Model H (Hindi) — Tables T4/T5.

Greedy plus temperatures 0.5 / 1.0 / 1.5, scored against held-out
references with BLEU-4, chrF, ROUGE-L, Distinct-1/2 and repetition rate.

    python hindi/eval/evaluate_generation.py --checkpoint hindi/checkpoints/best.pt
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.eval.runner import evaluate_generation   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "hindi/configs/hindi_model.yaml"))
    ap.add_argument("--checkpoint", default=str(REPO / "hindi/checkpoints/best.pt"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=None)
    a = ap.parse_args()
    evaluate_generation(a.config, REPO, a.checkpoint, a.split, a.device,
                        a.n_samples)
