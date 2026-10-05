#!/usr/bin/env python3
"""
07_split_train_val_test.py — Train-candidate / Validation / Test Split (streaming, hash-based)

LEAKAGE-SAFE ORDERING (CL3410 Lecture 3, "Decontamination")
=============================================================
This script now runs EARLY in the pipeline — directly on the DEDUPLICATED
per-source corpora (script 05 output) — BEFORE corpus concatenation
(script 09) and BEFORE the single final tokenizer is trained (script 10).

This matters because:
  • Corpus concatenation (09) and the final tokenizer (10) both compute
    statistics (vocabulary, BPE content-token counts) over whatever
    corpus they are given. If that corpus still contained the eventual
    validation/test text, those statistics — and therefore the training
    data selected using them — would be indirectly informed by the
    held-out data.
  • Holding out validation/test FIRST, before those statistics exist,
    structurally rules this out.

THREE SOURCES: PUBLIC (IndicCorp) + MANUAL (Onlinekhabar + Khabarhub)
=============================================================
05_deduplicate.py produces three separate, never-merged deduplicated files
(indiccorp_ne_deduplicated.txt, onlinekhabar_ne_deduplicated.txt,
khabarhub_ne_deduplicated.txt). This script splits each of the three
independently using the SAME deterministic hash-bucketing mechanism as
before: each line is bucketed by line_hash(seed, per-source line_index),
so splitting stays fully streaming and memory-bounded — no permutation
arrays, no NumPy index vectors, no loading the corpus.

Outputs:
  • paths.split/{source}_train_candidate.txt   for source in (indiccorp,
    onlinekhabar, khabarhub) — still needs decontamination
    (script 08) + concatenation (script 09) — NOT final.
  • paths.final/validation.txt   (all three sources combined, FROZEN from here on)
  • paths.final/test.txt         (all three sources combined, FROZEN from here on)

validation.txt and test.txt are written once here and must never be
touched by any later pipeline stage (no dedup, no selection, no tokenizer
training) — only read for evaluation (script 11) or for building the
decontamination index (script 08).

FAILURE SAFETY: a manual subsource that is ENABLED in config
(manual.news.onlinekhabar/khabarhub.enabled) but whose deduplicated input
file is missing is a hard ERROR — this script refuses to silently turn a
missing manual source into an empty corpus (the old two-bucket
"public"/"manual" version of this script pointed at a
manual_deduplicated.txt file that 05_deduplicate.py never produces, which
silently dropped the entire manual corpus; this version cannot repeat that
mistake because each source's file is checked individually). A source that
is DISABLED in config and absent is expected and not an error — same
enabled/disabled distinction already used by 09_concatenate_final_corpus.py.

Usage:
    python nepali/scripts/07_split_train_val_test.py --config nepali/configs/nepali_data.yaml

    # Explicit per-source overrides:
    python nepali/scripts/07_split_train_val_test.py --config nepali/configs/nepali_data.yaml \\
        --indiccorp-input nepali/data/intermediate/deduplicated/indiccorp_ne_deduplicated.txt \\
        --onlinekhabar-input nepali/data/intermediate/deduplicated/onlinekhabar_ne_deduplicated.txt \\
        --khabarhub-input nepali/data/intermediate/deduplicated/khabarhub_ne_deduplicated.txt
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.tokenizer.data_utils import count_lines, line_hash


# ---------------------------------------------------------------------------
# Source configuration (mirrors 09_concatenate_final_corpus.py)
# ---------------------------------------------------------------------------

PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("onlinekhabar", "khabarhub")
ALL_SOURCES = (PUBLIC_SOURCE,) + MANUAL_SOURCES


def source_enabled_map(config: Dict[str, Any]) -> Dict[str, bool]:
    """A manual source is 'required' (its absence is a hard FAIL, not a
    silent skip) iff it is enabled in config. indiccorp has no
    enable/disable switch anywhere in this project, so it is always
    required."""
    manual_cfg = config.get("manual", {})
    news_cfg = manual_cfg.get("news", {})
    return {
        PUBLIC_SOURCE: True,
        "onlinekhabar": news_cfg.get("onlinekhabar", {}).get("enabled", True) and news_cfg.get("enabled", True),
        "khabarhub": news_cfg.get("khabarhub", {}).get("enabled", True) and news_cfg.get("enabled", True),
    }


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
    return logging.getLogger("07_split_train_val_test")


def line_stats(line: str):
    """Return (chars, words) for a single line (no newline)."""
    return len(line), len(line.split())


def split_one_source(
    source_name: str,
    input_path: str,
    train_out_path: str,
    val_writer,
    test_writer,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    logger: logging.Logger,
    progress_interval: int,
) -> Dict[str, Any]:
    """Hash-bucket-split a single source into train-candidate / validation /
    test. Validation and test lines are appended to the shared, already-open
    *val_writer* / *test_writer* file handles (combining sources into one
    final validation.txt / test.txt). Train-candidate lines are written to
    this source's own file (kept separate for decontamination/selection).

    Fully streaming: one pass over the input, O(1) memory.
    """
    if not os.path.exists(input_path):
        logger.warning(f"[{source_name}] input not found, skipping: {input_path}")
        if os.path.exists(train_out_path):
            logger.warning(
                f"[{source_name}] removing stale train-candidate output from "
                f"a prior run (its input no longer exists): {train_out_path}"
            )
            os.remove(train_out_path)
        return {}

    n_lines = count_lines(input_path)
    logger.info(f"[{source_name}] input: {input_path} ({n_lines:,} lines)")

    counts = {"train_candidate": 0, "validation": 0, "test": 0}
    char_counts = {"train_candidate": 0, "validation": 0, "test": 0}
    word_counts = {"train_candidate": 0, "validation": 0, "test": 0}

    os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(train_out_path, "w", encoding="utf-8", buffering=1 << 20) as train_f:
        for idx, raw in enumerate(fin):
            line = raw.rstrip("\r\n")
            h = line_hash(seed, idx)
            if h < train_ratio:
                bucket = "train_candidate"
                train_f.write(line + "\n")
            elif h < train_ratio + val_ratio:
                bucket = "validation"
                val_writer.write(line + "\n")
            else:
                bucket = "test"
                test_writer.write(line + "\n")

            chars, words = line_stats(line)
            counts[bucket] += 1
            char_counts[bucket] += chars
            word_counts[bucket] += words

            if (idx + 1) % progress_interval == 0:
                logger.info(f"[{source_name}] processed {idx + 1:,} / {n_lines:,} lines")

    elapsed = time.time() - t0
    logger.info(f"[{source_name}] split completed in {elapsed:.1f}s: {counts}")

    return {
        "input_file": input_path,
        "train_candidate_file": train_out_path,
        "total_lines": n_lines,
        "counts": counts,
        "characters": char_counts,
        "approx_words": word_counts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split deduplicated per-source corpora (IndicCorp, "
                     "Onlinekhabar, Khabarhub) into train-candidate "
                     "/ validation / test BEFORE tokenizer training and "
                     "corpus selection (leakage-safe ordering)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    for source in ALL_SOURCES:
        parser.add_argument(f"--{source}-input", type=str,
                             help=f"Override {source} deduplicated corpus path")
    parser.add_argument("--output-dir", type=str, help="Override train-candidate output directory (paths.split)")
    parser.add_argument("--final-dir", type=str, help="Override validation/test output directory (paths.final)")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    repo_root = Path(args.config).resolve().parent.parent.parent

    dedup_dir = repo_root / config["paths"]["deduplicated"]
    input_path_by_source = {
        source: (getattr(args, f"{source}_input", None) or str(dedup_dir / f"{source}_ne_deduplicated.txt"))
        for source in ALL_SOURCES
    }
    enabled = source_enabled_map(config)

    split_dir = args.output_dir or str(repo_root / config["paths"]["split"])
    final_dir = args.final_dir or str(repo_root / config["paths"]["final"])

    seed = args.seed if args.seed is not None else config["split"]["random_seed"]
    train_ratio = config["split"]["train"]
    val_ratio = config["split"]["validation"]
    test_ratio = config["split"]["test"]
    split_method = config.get("split", {}).get("method", "hash")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    val_path = os.path.join(final_dir, "validation.txt")
    test_path = os.path.join(final_dir, "test.txt")
    train_candidate_path_by_source = {
        source: os.path.join(split_dir, f"{source}_train_candidate.txt") for source in ALL_SOURCES
    }

    if not args.overwrite:
        for p in (val_path, test_path):
            if os.path.exists(p):
                logger.error(
                    f"Output file {p} already exists. Refusing to silently "
                    f"append to a held-out split. Use --overwrite to replace."
                )
                sys.exit(1)

    progress_interval = config.get("logging", {}).get("progress_interval", 1_000_000)

    logger.info("=" * 70)
    logger.info("SPLITTING DEDUPLICATED CORPORA INTO TRAIN-CANDIDATE / VALIDATION / TEST")
    logger.info("(this runs BEFORE decontamination, tokenizer training, and selection)")
    logger.info("=" * 70)
    logger.info(f"IndicCorp (public) input: {input_path_by_source[PUBLIC_SOURCE]}")
    for source in MANUAL_SOURCES:
        logger.info(f"{source} (manual) input: {input_path_by_source[source]}"
                     f"{'' if enabled[source] else ' [disabled in config]'}")
    logger.info(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Seed:   {seed}   Method: {split_method}")

    per_source_reports = {}
    with open(val_path, "w", encoding="utf-8", buffering=1 << 20) as val_f, \
         open(test_path, "w", encoding="utf-8", buffering=1 << 20) as test_f:

        for source in ALL_SOURCES:
            per_source_reports[source] = split_one_source(
                source, input_path_by_source[source], train_candidate_path_by_source[source],
                val_f, test_f, seed, train_ratio, val_ratio, test_ratio, logger, progress_interval,
            )

    if not any(per_source_reports.values()):
        logger.error("No source input was found (indiccorp/onlinekhabar/khabarhub all missing). Nothing to split.")
        sys.exit(1)

    # FAILURE SAFETY: a source that is enabled in config but whose input
    # file was missing is a hard error, never a silent skip — this is what
    # prevents the manual corpus from silently disappearing (the bug in
    # the old public/manual-only version of this script).
    missing_required_sources = [
        source for source in ALL_SOURCES
        if not per_source_reports.get(source) and enabled[source]
    ]
    if missing_required_sources:
        for source in missing_required_sources:
            logger.error(
                f"REQUIRED source missing: '{source}' is enabled in config "
                f"but its deduplicated input file was not found: "
                f"{input_path_by_source[source]}. Refusing to silently "
                f"produce an incomplete corpus."
            )
        sys.exit(1)

    total_val = count_lines(val_path)
    total_test = count_lines(test_path)
    total_train_by_source = {
        source: per_source_reports.get(source, {}).get("counts", {}).get("train_candidate", 0)
        for source in ALL_SOURCES
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"VALIDATION SET CREATED: {val_path} ({total_val:,} lines)")
    logger.info(f"TEST SET CREATED:       {test_path} ({total_test:,} lines)")
    logger.info("=" * 70)
    logger.info(
        "validation.txt and test.txt are now FROZEN — no later pipeline "
        "stage may write to, select from, dedup into, or train on them."
    )
    logger.info("Train-candidate (still requires decontamination + selection):")
    logger.info(f"  IndicCorp train candidate:    {total_train_by_source[PUBLIC_SOURCE]:,} lines -> {train_candidate_path_by_source[PUBLIC_SOURCE]}")
    logger.info(f"  Onlinekhabar train candidate: {total_train_by_source['onlinekhabar']:,} lines -> {train_candidate_path_by_source['onlinekhabar']}")
    logger.info(f"  Khabarhub train candidate:    {total_train_by_source['khabarhub']:,} lines -> {train_candidate_path_by_source['khabarhub']}")
    logger.info(f"  Total validation: {total_val:,} lines")
    logger.info(f"  Total test:       {total_test:,} lines")

    # ---- Metadata --------------------------------------------------------
    meta = {
        "seed": seed,
        "method": split_method,
        "hash": "blake2b(seed:line_index) normalized to [0,1)",
        "ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        "note": (
            "This split runs BEFORE the final tokenizer is trained, corpus "
            "selection, and decontamination. validation.txt/test.txt are "
            "combined across all three sources and frozen; "
            "{source}_train_candidate.txt files still require "
            "decontamination (script 08) before any tokenizer or selection "
            "step reads them."
        ),
        "public_source": PUBLIC_SOURCE,
        "manual_sources": list(MANUAL_SOURCES),
        "source_enabled": enabled,
        "sources": per_source_reports,
        "combined": {
            "validation": {"file": val_path, "lines": total_val},
            "test": {"file": test_path, "lines": total_test},
            "train_candidate_lines": sum(total_train_by_source.values()),
        },
    }
    meta_path = os.path.join(str(repo_root / config["paths"]["metadata"]), "split_statistics.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Split metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
