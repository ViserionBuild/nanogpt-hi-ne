#!/usr/bin/env python3
"""
11_evaluate_tokenizer.py — Evaluate and Compare Tokenizer Variants

Computes fertility, UNK rate, characters-per-token, token frequency
distribution, and generates comparison tables and plots across all
trained tokenizer variants (e.g. 16K / 32K / 48K).

Usage:
    python hindi/scripts/11_evaluate_tokenizer.py \
        --config hindi/configs/hindi_data.yaml
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configure and return a logger."""
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("11_evaluate_tokenizer")


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

HINDI_EXAMPLES: List[str] = [
    "भारत एक विशाल देश है।",
    "भारतीय अंतरिक्ष अनुसंधान संगठन",
    "पर्यावरणीय परिवर्तन",
    "जलवायु परिवर्तन",
    "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग",
    "संविधान में मौलिक अधिकारों की रक्षा का प्रावधान है।",
]


def evaluate_tokenizer(
    sp: spm.SentencePieceProcessor,
    eval_path: str,
    logger: logging.Logger,
    progress_interval: int = 500_000,
) -> Dict[str, Any]:
    """Compute all evaluation metrics for a single tokenizer on *eval_path*."""
    vocab_size = sp.GetPieceSize()
    unk_id = sp.unk_id()

    total_tokens = 0
    total_unk = 0
    total_chars = 0
    total_words = 0
    token_freq: Counter = Counter()

    t0 = time.time()
    line_count = 0

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            line_count += 1

            ids = sp.EncodeAsIds(line)
            total_tokens += len(ids)
            total_unk += sum(1 for i in ids if i == unk_id)
            total_chars += len(line)
            total_words += len(line.split())
            token_freq.update(ids)

            if line_count % progress_interval == 0:
                logger.info(f"  Evaluated {line_count:,} lines, {total_tokens:,} tokens …")

    elapsed = time.time() - t0
    logger.info(f"  Evaluation done: {line_count:,} lines in {elapsed:.1f}s")

    # Derived metrics
    fertility = total_tokens / total_words if total_words > 0 else 0.0
    unk_rate = total_unk / total_tokens if total_tokens > 0 else 0.0
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0.0

    # Top-100 tokens
    top100 = token_freq.most_common(100)
    top100_decoded = [
        {"rank": i + 1, "id": tid, "piece": sp.IdToPiece(tid), "count": cnt}
        for i, (tid, cnt) in enumerate(top100)
    ]

    # Rare tokens (count == 1)
    rare_count = sum(1 for _, c in token_freq.items() if c == 1)

    # Tokens appearing in the eval set
    used_ids = set(token_freq.keys())
    unused_vocab = vocab_size - len(used_ids)

    # Tokenization examples
    examples = []
    for sent in HINDI_EXAMPLES:
        pieces = sp.EncodeAsPieces(sent)
        ids = sp.EncodeAsIds(sent)
        examples.append({
            "text": sent,
            "pieces": pieces,
            "ids": ids,
            "num_tokens": len(ids),
            "num_words": len(sent.split()),
            "fertility": len(ids) / max(len(sent.split()), 1),
        })

    return {
        "vocab_size": vocab_size,
        "eval_file": eval_path,
        "eval_lines": line_count,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "fertility": round(fertility, 4),
        "unk_rate": round(unk_rate, 6),
        "unk_count": total_unk,
        "chars_per_token": round(chars_per_token, 4),
        "top_100_tokens": top100_decoded,
        "rare_tokens_count": rare_count,
        "unused_vocab_entries": unused_vocab,
        "used_vocab_entries": len(used_ids),
        "tokenization_examples": examples,
        "evaluation_time_seconds": round(elapsed, 2),
        "_token_freq_for_plot": token_freq,  # excluded from JSON
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_frequency_distribution(
    results: List[Dict[str, Any]], figures_dir: str
) -> None:
    """Log-log token frequency distribution for each variant."""
    plt.figure(figsize=(10, 6))

    for res in results:
        freq = res["_token_freq_for_plot"]
        counts = sorted(freq.values(), reverse=True)
        ranks = np.arange(1, len(counts) + 1)
        label = f"V={res['vocab_size']:,}"
        plt.loglog(ranks, counts, label=label, alpha=0.7)

    plt.xlabel("Token Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.title("Token Frequency Distribution (Zipf Plot)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(figures_dir, "token_frequency_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_comparison_bars(
    results: List[Dict[str, Any]], figures_dir: str
) -> None:
    """Bar charts comparing fertility, chars/token, UNK rate across variants."""
    labels = [f"{r['vocab_size']//1000}K" for r in results]
    fertility = [r["fertility"] for r in results]
    cpt = [r["chars_per_token"] for r in results]
    unk = [r["unk_rate"] * 100 for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Fertility
    bars = axes[0].bar(labels, fertility, color="#4C72B0", alpha=0.85)
    axes[0].set_title("Fertility (tokens / words)")
    axes[0].set_ylabel("Fertility")
    for bar, val in zip(bars, fertility):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Chars per token
    bars = axes[1].bar(labels, cpt, color="#55A868", alpha=0.85)
    axes[1].set_title("Characters / Token")
    axes[1].set_ylabel("Chars/Token")
    for bar, val in zip(bars, cpt):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # UNK rate
    bars = axes[2].bar(labels, unk, color="#C44E52", alpha=0.85)
    axes[2].set_title("UNK Rate (%)")
    axes[2].set_ylabel("UNK %")
    for bar, val in zip(bars, unk):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                     f"{val:.4f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Tokenizer Variant Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(figures_dir, "tokenizer_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare trained tokenizer variants."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--tokenizer-dir", type=str, help="Override tokenizer model dir")
    parser.add_argument("--eval-file", type=str, help="Override evaluation text file")
    parser.add_argument("--output", type=str, help="Override output JSON path")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    progress_interval = config.get("logging", {}).get("progress_interval", 1_000_000)

    repo_root = Path(args.config).resolve().parent.parent.parent

    # Resolve paths
    if args.tokenizer_dir:
        tok_dir = args.tokenizer_dir
    else:
        tok_dir = str(repo_root / config["paths"]["tokenizer_model"])

    if args.eval_file:
        eval_file = args.eval_file
    else:
        eval_file = str(repo_root / config["paths"]["final"] / "validation.txt")

    if args.output:
        output_path = args.output
    else:
        output_path = str(repo_root / config["paths"]["metadata"] / "tokenizer_statistics.json")

    figures_dir = str(repo_root / config["paths"]["figures"])
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(eval_file):
        logger.error(f"Evaluation file not found: {eval_file}")
        sys.exit(1)

    # Discover trained models
    model_files = sorted(glob.glob(os.path.join(tok_dir, "hindi_bpe_*.model")))
    if not model_files:
        logger.error(f"No tokenizer models found in {tok_dir}")
        sys.exit(1)

    logger.info(f"Found {len(model_files)} tokenizer model(s): {model_files}")
    logger.info(f"Evaluation file: {eval_file}")

    # Evaluate each variant
    all_results: List[Dict[str, Any]] = []

    for model_path in model_files:
        # Extract vocab size from filename
        match = re.search(r"hindi_bpe_(\d+)\.model", os.path.basename(model_path))
        if not match:
            logger.warning(f"Skipping unrecognized model file: {model_path}")
            continue

        vs = int(match.group(1))
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_path} (vocab_size={vs})")
        logger.info(f"{'='*60}")

        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)

        result = evaluate_tokenizer(sp, eval_file, logger, progress_interval)
        result["model_path"] = model_path
        all_results.append(result)

    # ---- Comparison table ----------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("TOKENIZER COMPARISON TABLE")
    logger.info(f"{'='*70}")
    logger.info(
        f"{'Vocabulary':>12} {'Fertility':>12} {'Chars/Token':>12} "
        f"{'UNK Rate':>12} {'UNK Count':>12}"
    )
    logger.info("-" * 62)
    for r in all_results:
        logger.info(
            f"{r['vocab_size']:>12,} {r['fertility']:>12.4f} "
            f"{r['chars_per_token']:>12.4f} {r['unk_rate']:>12.6f} "
            f"{r['unk_count']:>12,}"
        )
    logger.info("-" * 62)

    # ---- Tokenization examples -----------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("TOKENIZATION EXAMPLES")
    logger.info(f"{'='*70}")
    for r in all_results:
        logger.info(f"\n--- Vocab Size: {r['vocab_size']:,} ---")
        for ex in r["tokenization_examples"]:
            logger.info(f"  Text:   {ex['text']}")
            logger.info(f"  Pieces: {ex['pieces']}")
            logger.info(f"  Tokens: {ex['num_tokens']}  Words: {ex['num_words']}  "
                        f"Fertility: {ex['fertility']:.2f}")
            logger.info("")

    # ---- Generate plots ------------------------------------------------------
    if len(all_results) > 0:
        logger.info("Generating plots …")
        p1 = plot_frequency_distribution(all_results, figures_dir)
        logger.info(f"  Saved: {p1}")
        if len(all_results) > 1:
            p2 = plot_comparison_bars(all_results, figures_dir)
            logger.info(f"  Saved: {p2}")

    # ---- Save JSON (strip internal freq counter) -----------------------------
    json_results = []
    for r in all_results:
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        json_results.append(out)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"tokenizer_evaluations": json_results},
            f, indent=2, ensure_ascii=False, default=str,
        )
    logger.info(f"Statistics saved to {output_path}")
    logger.info("Tokenizer evaluation complete.")


if __name__ == "__main__":
    main()
