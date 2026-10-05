#!/usr/bin/env python3
"""
09_concatenate_final_corpus.py — Combine Decontaminated Sources Into final/train.txt

SINGLE-TOKENIZER ARCHITECTURE
==============================
There is only ONE BPE tokenizer in this pipeline (trained by script 10, on
`final/train.txt`). There is no temporary/pre-selection tokenizer, so this
script cannot (and does not) measure or select by actual BPE content-token
counts — that measurement is only possible AFTER script 10 trains the one
final tokenizer, and is performed there (streaming token-count pass over
each final/*_train.txt file / `train.txt`, reported against the 400M
IndicCorp / 100M manual targets — see script 10).

-----------------------------------------------------------------------------
THREE SOURCES: PUBLIC (IndicCorp) + MANUAL (Onlinekhabar + Khabarhub)
-----------------------------------------------------------------------------
The Nepali corpus has exactly TWO top-level sources — indiccorp (public) and
manual — but manual itself is TWO subcategories, each collected and
decontaminated independently: Onlinekhabar and Khabarhub (see
nepali/scripts/manual/collect_news.py). Manual is NEVER just one of the
two — this script combines both, in this fixed order, to build the manual
corpus:

    final/onlinekhabar_train.txt  (manual/news, unchanged content)
    final/khabarhub_train.txt     (manual/news, unchanged content)
            |
            v  concatenated, in that order
    final/manual_train.txt

and separately:

    final/indiccorp_train.txt     (public/IndicCorp, unchanged content)

Then the two are concatenated (indiccorp first, then manual) into the
single file the final tokenizer and model are actually trained on:

    final/train.txt

This script does ONE thing: combine the DECONTAMINATED train-candidate
corpora (script 08 output, one file per source — i.e. AFTER validation/test
have been split off (07) and AFTER exact/near-duplicate overlap with them
has been removed (08)) into the six files above. No subsampling/trimming
happens here: since no tokenizer exists yet, there is no trustworthy way to
hit an exact token target by fractional subsampling, and CL3410's rule
against silently duplicating data to hit a target has no equivalent
"silently drop data to avoid a shortfall" analog — so ALL decontaminated
data from EVERY present source is carried forward exactly once (each
source's content is read once and reused via file-level concatenation for
manual_train.txt/train.txt, never re-read/re-copied from the original
decontaminated input — this is what keeps output free of any duplication,
not just an absence of intentional dedup logic). If the resulting token
counts (measured by script 10 after training) fall short of the 400M /
100M targets, that shortfall is reported, not papered over by trimming
here or by training a second tokenizer.

A source whose decontaminated input file does not exist yet is treated as
EMPTY for this run (warned, not fatal) — this keeps the script usable at
every stage of standing up each source's pipeline (e.g. before Onlinekhabar/
Khabarhub have been collected/decontaminated at all), the same graceful-
degradation convention already used by 05_deduplicate.py for a not-yet-
collected source. The run only fails if ALL THREE sources are missing.

Design guarantees:
  • Fully streaming (line-by-line copy / file-level concatenation) — no
    tokenizer, no per-line arrays, no full-corpus buffering. Memory use
    O(1) regardless of corpus size.
  • Deterministic — plain ordered concatenation of the same inputs always
    produces byte-identical output.
  • Validation/test isolation preserved: every resolved input path (and
    the combined output path) is checked against a held-out-filename
    guard before anything is read/written.

Usage:
    python nepali/scripts/09_concatenate_final_corpus.py --config nepali/configs/nepali_data.yaml

    # Explicit per-source overrides (defaults shown above are normally enough):
    python nepali/scripts/09_concatenate_final_corpus.py --config nepali/configs/nepali_data.yaml \\
        --indiccorp-input nepali/data/intermediate/decontaminated/indiccorp_train_decontam.txt \\
        --onlinekhabar-input nepali/data/intermediate/decontaminated/onlinekhabar_train_decontam.txt \\
        --khabarhub-input nepali/data/intermediate/decontaminated/khabarhub_train_decontam.txt
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.tokenizer.data_utils import count_lines

# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

PUBLIC_SOURCE = "indiccorp"

# FIXED order — this is exactly how final/manual_train.txt is built
# (requirement: onlinekhabar, then khabarhub). Manual is never just one of
# the two; both are always considered.
MANUAL_SOURCES = ("onlinekhabar", "khabarhub")

ALL_SOURCES = (PUBLIC_SOURCE,) + MANUAL_SOURCES

HELD_OUT_BASENAMES = ("validation.txt", "test.txt", "val.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("09_concatenate_final_corpus")


def refuse_if_held_out(path: str, logger: logging.Logger) -> None:
    """Fail loudly (never silently) if *path* is a held-out validation/test
    file — applied to every resolved input AND output path so validation/
    test isolation holds regardless of which source it is."""
    if os.path.basename(path).lower() in HELD_OUT_BASENAMES:
        logger.error(f"REFUSING to touch held-out file: {path}")
        sys.exit(1)


def stream_copy_with_stats(input_path: str, output_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Stream-copy *input_path* to *output_path*, counting lines/words/chars.
    O(1) memory. Returns zeroed stats (and no output file) if the input is
    missing, so a missing source degrades gracefully rather than crashing.

    If *output_path* already exists from a PREVIOUS run (e.g. this source
    was present last time but is missing now), it is removed rather than
    left in place — otherwise a later concat_files() call would silently
    pick up that stale file and leak a previous run's content into
    manual_train.txt/train.txt as if the source were still present."""
    if not os.path.exists(input_path):
        if os.path.exists(output_path):
            logger.warning(
                f"  Input not found: {input_path} — removing stale prior "
                f"output so it isn't silently reused: {output_path}"
            )
            os.remove(output_path)
        else:
            logger.warning(f"  Input not found, skipping: {input_path}")
        return {"input_file": input_path, "lines": 0, "words": 0, "characters": 0, "present": False}

    lines = words = chars = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    t0 = time.time()
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8", buffering=1 << 20) as fout:
        for raw in fin:
            line = raw.rstrip("\r\n")
            fout.write(line + "\n")
            lines += 1
            words += len(line.split())
            chars += len(line)
    elapsed = time.time() - t0
    logger.info(f"  {input_path} -> {output_path}: {lines:,} lines in {elapsed:.1f}s")
    return {"input_file": input_path, "output_file": output_path,
            "lines": lines, "words": words, "characters": chars, "present": True}


def concat_files(sources: List[str], combined_output: str, logger: logging.Logger) -> int:
    """Stream-concatenate *sources* (in order) into *combined_output*.
    O(1) memory. Returns total lines written. Sources are read from
    already-materialized per-source output files (never the original
    decontaminated inputs a second time) so no source's content is ever
    duplicated into the output."""
    os.makedirs(os.path.dirname(combined_output) or ".", exist_ok=True)
    total = 0
    with open(combined_output, "w", encoding="utf-8", buffering=1 << 20) as fout:
        for src in sources:
            if not os.path.exists(src):
                continue
            with open(src, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine decontaminated IndicCorp (public) + Onlinekhabar "
                     "+ Khabarhub (manual) train-candidate corpora "
                     "into final/train.txt (no tokenizer, no trimming — see "
                     "module docstring)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--indiccorp-input", "--public-input", dest="indiccorp_input", type=str,
                         help="Path to decontaminated IndicCorp (public) train-candidate corpus")
    parser.add_argument("--onlinekhabar-input", dest="onlinekhabar_input", type=str,
                         help="Path to decontaminated Onlinekhabar (manual/news) train-candidate corpus")
    parser.add_argument("--khabarhub-input", dest="khabarhub_input", type=str,
                         help="Path to decontaminated Khabarhub (manual/news) train-candidate corpus")
    parser.add_argument("--output", type=str, help="Override combined output file (final/train.txt)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    repo_root = Path(args.config).resolve().parent.parent.parent

    decontam_dir = repo_root / config["paths"]["decontaminated"]
    input_path_by_source = {
        source: (getattr(args, f"{source}_input", None) or str(decontam_dir / f"{source}_train_decontam.txt"))
        for source in ALL_SOURCES
    }

    final_dir = repo_root / config["paths"]["final"]
    combined_output = Path(args.output) if args.output else final_dir / "train.txt"
    manual_output = final_dir / "manual_train.txt"
    output_path_by_source = {source: final_dir / f"{source}_train.txt" for source in ALL_SOURCES}

    for path in list(input_path_by_source.values()) + [str(combined_output), str(manual_output)] + \
            [str(p) for p in output_path_by_source.values()]:
        refuse_if_held_out(path, logger)

    all_output_paths = [combined_output, manual_output] + list(output_path_by_source.values())
    for out_path in all_output_paths:
        if out_path.exists() and not args.overwrite:
            logger.error(f"Output exists: {out_path}. Use --overwrite to redo.")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("COMBINING DECONTAMINATED SOURCES (no tokenizer, no trimming)")
    logger.info("=" * 70)
    logger.info(f"IndicCorp (public) input:            {input_path_by_source[PUBLIC_SOURCE]}")
    for source in MANUAL_SOURCES:
        logger.info(f"{source.capitalize()} (manual) input:"
                    f"{' ' * max(1, 12 - len(source))}{input_path_by_source[source]}")

    # ---- Stream each present source to its own final/<source>_train.txt --
    stats_by_source: Dict[str, Dict[str, Any]] = {}
    for source in ALL_SOURCES:
        stats_by_source[source] = stream_copy_with_stats(
            input_path_by_source[source], str(output_path_by_source[source]), logger
        )

    if not any(stats_by_source[source]["present"] for source in ALL_SOURCES):
        logger.error(
            "No source was found — checked IndicCorp (public) and both "
            "manual sources (Onlinekhabar, Khabarhub). Nothing to combine."
        )
        sys.exit(1)

    missing = [s for s in ALL_SOURCES if not stats_by_source[s]["present"]]
    if missing:
        logger.warning(
            f"Proceeding with {len(ALL_SOURCES) - len(missing)}/{len(ALL_SOURCES)} sources present. "
            f"Missing (treated as empty): {', '.join(missing)}"
        )

    # ---- manual_train.txt = onlinekhabar + khabarhub, in that
    # fixed order — built from the just-written per-source files (never
    # re-reading the original decontaminated inputs), so manual content is
    # never duplicated into the output. ------------------------------------
    logger.info(f"Concatenating manual sources (onlinekhabar, khabarhub) into {manual_output} …")
    manual_sources_files = [str(output_path_by_source[s]) for s in MANUAL_SOURCES]
    manual_lines = concat_files(manual_sources_files, str(manual_output), logger)
    manual_lines_expected = sum(stats_by_source[s]["lines"] for s in MANUAL_SOURCES)
    assert manual_lines == manual_lines_expected, (
        f"manual_train.txt line count mismatch: wrote {manual_lines}, "
        f"sources sum {manual_lines_expected}"
    )

    # ---- train.txt = indiccorp + manual, built the same file-level way ---
    logger.info(f"Concatenating IndicCorp + manual into {combined_output} …")
    total_lines = concat_files(
        [str(output_path_by_source[PUBLIC_SOURCE]), str(manual_output)], str(combined_output), logger
    )
    combined_lines_check = count_lines(str(combined_output))
    expected_total = stats_by_source[PUBLIC_SOURCE]["lines"] + manual_lines
    assert combined_lines_check == total_lines == expected_total, (
        f"Combined line count mismatch: wrote {total_lines}, on-disk {combined_lines_check}, "
        f"indiccorp + manual sum {expected_total}"
    )

    manual_words = sum(stats_by_source[s]["words"] for s in MANUAL_SOURCES)
    manual_chars = sum(stats_by_source[s]["characters"] for s in MANUAL_SOURCES)

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMBINE COMPLETE")
    logger.info("=" * 70)
    for source in ALL_SOURCES:
        s = stats_by_source[source]
        logger.info(f"{source.capitalize():12s}: {s['lines']:,} lines, {s['words']:,} words, "
                    f"{s['characters']:,} characters -> {output_path_by_source[source]}")
    logger.info(f"{'Manual':12s}: {manual_lines:,} lines, {manual_words:,} words, "
                f"{manual_chars:,} characters -> {manual_output} "
                f"(= onlinekhabar + khabarhub)")
    logger.info(f"{'Combined':12s}: {total_lines:,} lines -> {combined_output} (= indiccorp + manual)")
    logger.info(
        "NOTE: actual BPE content-token counts (and the 400M/100M target "
        "shortfall check) are only measured AFTER the single final "
        "tokenizer is trained on this file — see script 10."
    )

    meta_dir = repo_root / config["paths"]["metadata"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "note": (
            "Word/line/character counts only — this pipeline has a single "
            "tokenizer (script 10), so no BPE token counts exist yet at this "
            "stage. See metadata/token_count_check.json (written by script "
            "11) for actual BPE content-token counts per source and vs. "
            "targets."
        ),
        "manual_composition": list(MANUAL_SOURCES),
        "sources": stats_by_source,
        "manual_combined": {
            "output_file": str(manual_output),
            "composition": list(MANUAL_SOURCES),
            "lines": manual_lines,
            "words": manual_words,
            "characters": manual_chars,
        },
        "combined": {
            "output_file": str(combined_output),
            "composition": [PUBLIC_SOURCE, "manual"],
            "lines": total_lines,
            "words": stats_by_source[PUBLIC_SOURCE]["words"] + manual_words,
            "characters": stats_by_source[PUBLIC_SOURCE]["characters"] + manual_chars,
        },
        "missing_sources": missing,
        "subsampling_applied": False,
    }
    report_path = meta_dir / "corpus_combination_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Combination report saved to {report_path}")


if __name__ == "__main__":
    main()
