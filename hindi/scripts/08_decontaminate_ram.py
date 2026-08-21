#!/usr/bin/env python3
"""
08_decontaminate_ram.py — Remove Validation/Test Leakage From the Training Corpus

Implements the "Decontamination" procedure from CL3410 Lecture 3:

  1. Hold out the test set first (done by script 07, which runs BEFORE
     this script).
  2. Search each source's candidate training corpus for EXACT and
     NEAR-DUPLICATE matches of the held-out validation/test data.
  3. Remove matches from the TRAINING corpus. validation.txt/test.txt are
     never modified — only read.
  4. Do this before the single final tokenizer (09) and corpus selection
     (10) compute any statistics from the training corpus.

Near-duplicate matching uses MinHash + LSH banding (shared/near_dup.py) so
paraphrased / near-identical test items are caught, not just byte-identical
lines — exact string matching alone is NOT sufficient per the professor's
slide.

THREE SOURCES: PUBLIC (IndicCorp) + MANUAL (Wikipedia + News)
=============================================================
Each of the three sources produced by script 07 (indiccorp, wikipedia,
news — split/{source}_train_candidate.txt) is decontaminated
INDEPENDENTLY here, using the exact same algorithm and the SAME held-out
index built once from validation.txt/test.txt. The three sources are
never merged in this script — outputs are
decontaminated/{source}_train_decontam.txt, matching exactly what
10_concatenate_final_corpus.py expects.

News is itself the pre-merged combination of every ENABLED news outlet
(done as early as possible by manual/05_aggregate_news_corpus.py -> 02/03/04/05/07);
individual outlets are NEVER a separate decontamination source or output
here. Wikipedia is its own separate file/source and is never merged with
news at any pipeline stage.

There is no per-line provenance sidecar anywhere in this script: since
wikipedia and news are now separate real top-level files (source ==
file), there is no sub-source-within-one-merged-file breakdown left to
track — unlike the old two-bucket "indiccorp/manual" model, where a
single combined manual file needed a sidecar to keep the Wikipedia/news
distinction visible after merging.

FAILURE SAFETY: a manual subsource (wikipedia or news) that is ENABLED in
config but whose train-candidate input file is missing is a hard ERROR,
not a silent skip — same enabled/disabled distinction used by
07_split_dataset.py, so wikipedia or news can never silently vanish
between script 07 and script 10. Wikipedia and news are each required
independently of one another.

Memory guarantees (hundreds of millions of lines):
  • The held-out validation/test corpus is small relative to the training
    corpus (~5%+5% split) and its exact-hashes / LSH band-keys are the
    ONLY thing held in a RAM-backed exact-hash and RAM-backed LSH index,
    built once and reused across all three sources.
  • Each source's training corpus is streamed exactly once, line by line.
    It is NEVER loaded into RAM, and its own hashes/signatures are never
    persisted — only looked up against the held-out RAM index.

Usage:
    python hindi/scripts/08_decontaminate_ram.py \\
        --config hindi/configs/hindi_data.yaml

    # Explicit per-source overrides:
    python hindi/scripts/08_decontaminate_ram.py --config hindi/configs/hindi_data.yaml \\
        --indiccorp-input hindi/data/intermediate/split/indiccorp_train_candidate.txt \\
        --wikipedia-input hindi/data/intermediate/split/wikipedia_train_candidate.txt \\
        --news-input hindi/data/intermediate/split/news_train_candidate.txt
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.data_utils import count_lines, iter_lines
from shared.near_dup import MinHasher, line_band_keys


# ---------------------------------------------------------------------------
# Source configuration (mirrors 07_split_dataset.py / 10_concatenate_final_corpus.py)
# ---------------------------------------------------------------------------

PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("wikipedia", "news")
ALL_SOURCES = (PUBLIC_SOURCE,) + MANUAL_SOURCES


def source_enabled_map(config: Dict[str, Any]) -> Dict[str, bool]:
    """Mirrors 07_split_dataset.py's / 08_decontaminate.py's identical
    logic: a manual source is 'required' iff it is enabled in config.
    wikipedia is required iff manual.wikipedia.enabled; news is required
    iff manual.news.enabled (news-GLOBALLY-enabled only — individual
    outlets are already merged into one file by
    manual/05_aggregate_news_corpus.py before this script ever runs, so their
    individual enabled/disabled flags are not re-consulted here)."""
    manual_cfg = config.get("manual", {})
    news_cfg = manual_cfg.get("news", {})
    return {
        PUBLIC_SOURCE: True,
        "wikipedia": manual_cfg.get("wikipedia", {}).get("enabled", True),
        "news": news_cfg.get("enabled", True),
    }


def parse_source_overrides(pairs, valid_sources, flag_name: str, logger: logging.Logger) -> Dict[str, str]:
    """Parse repeated `--input-override SOURCE=PATH` CLI arguments into
    {source: path} — see 07_split_dataset.py's identical helper."""
    result: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            logger.error(f"{flag_name} must be SOURCE=PATH, got: {item!r}")
            sys.exit(1)
        name, path = item.split("=", 1)
        if name not in valid_sources:
            logger.error(f"{flag_name}: unknown source {name!r}. Valid sources: {', '.join(valid_sources)}")
            sys.exit(1)
        result[name] = path
    return result


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
    return logging.getLogger("08_decontaminate_ram")


def exact_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def build_holdout_index(
    holdout_files: List[str],
    exact_match: bool,
    near_duplicate: bool,
    minhasher: Optional[MinHasher],
    bands: int,
    rows: int,
    shingle_size: int,
    logger: logging.Logger,
) -> tuple[set[bytes], set[str], Dict[str, int]]:
    """Build the held-out exact/LSH lookup index entirely in RAM.

    The large training corpus is still streamed from disk. Only the
    held-out validation/test signatures are retained in memory.
    """
    exact_hashes: set[bytes] = set()
    band_keys: set[str] = set()
    counts: Dict[str, int] = {}
    n_bands_inserted = 0
    n_hashes_inserted = 0

    for path in holdout_files:
        if not os.path.exists(path):
            logger.warning(f"Held-out file not found, skipping: {path}")
            counts[path] = 0
            continue

        n = 0
        for line in iter_lines(path):
            if not line:
                continue
            n += 1

            if exact_match:
                digest = hashlib.sha256(line.encode("utf-8")).digest()
                if digest not in exact_hashes:
                    exact_hashes.add(digest)
                    n_hashes_inserted += 1

            if near_duplicate:
                for key in line_band_keys(
                    line, minhasher, bands, rows, shingle_size
                ):
                    if key not in band_keys:
                        band_keys.add(key)
                        n_bands_inserted += 1

            if n % 200_000 == 0:
                logger.info(f"  Indexed {n:,} held-out lines from {path}")

        counts[path] = n
        logger.info(f"  Indexed {n:,} held-out lines from {path}")

    logger.info(
        f"  In-memory held-out index built: "
        f"{n_hashes_inserted:,} unique exact hashes, "
        f"{n_bands_inserted:,} unique LSH band keys"
    )
    return exact_hashes, band_keys, counts


def decontaminate_file(
    exact_hashes: set[bytes],
    band_keys: set[str],
    input_path: str,
    output_path: str,
    exact_match: bool,
    near_duplicate: bool,
    minhasher: Optional[MinHasher],
    bands: int,
    rows: int,
    shingle_size: int,
    logger: logging.Logger,
    progress_interval: int,
) -> Dict[str, Any]:
    """Stream *input_path*, drop lines that exact- or near-duplicate the
    held-out index, write survivors to *output_path*. One pass, O(1) memory
    beyond the RAM index (which holds only held-out signatures)."""
    empty_result = {
        "input_lines": 0, "exact_removed": 0, "near_dup_removed": 0,
        "kept": 0, "kept_words": 0,
    }
    if not os.path.exists(input_path):
        logger.warning(f"Train-candidate file not found, skipping: {input_path}")
        if os.path.exists(output_path):
            logger.warning(
                f"Removing stale output from a prior run (its input no "
                f"longer exists): {output_path}"
            )
            os.remove(output_path)
        return dict(empty_result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = 0
    exact_removed = 0
    near_dup_removed = 0
    kept = 0
    kept_words = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8", buffering=1 << 20) as out_f:
        for line in iter_lines(input_path):
            total += 1
            dropped = False

            if exact_match:
                digest = hashlib.sha256(line.encode("utf-8")).digest()
                if digest in exact_hashes:
                    exact_removed += 1
                    dropped = True

            if not dropped and near_duplicate and line:
                for key in line_band_keys(line, minhasher, bands, rows, shingle_size):
                    if key in band_keys:
                        near_dup_removed += 1
                        dropped = True
                        break

            if not dropped:
                out_f.write(line + "\n")
                kept += 1
                kept_words += len(line.split())

            if total % progress_interval == 0:
                elapsed = time.time() - t0
                speed = total / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {total:,} lines scanned | kept {kept:,} | "
                    f"exact removed {exact_removed:,} | near-dup removed {near_dup_removed:,} | "
                    f"{speed:,.0f} lines/s"
                )

    elapsed = time.time() - t0
    logger.info(
        f"  {os.path.basename(input_path)}: {total:,} lines -> {kept:,} kept "
        f"({exact_removed:,} exact + {near_dup_removed:,} near-dup removed) in {elapsed:.1f}s"
    )
    return {
        "input_lines": total,
        "exact_removed": exact_removed,
        "near_dup_removed": near_dup_removed,
        "kept": kept,
        "kept_words": kept_words,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decontaminate the IndicCorp, Wikipedia, and News training corpora against held-out validation/test."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--indiccorp-input", "--public-input", dest="indiccorp_input", type=str,
                         help="Override IndicCorp (public) train-candidate path")
    parser.add_argument("--wikipedia-input", type=str,
                         help="Override Wikipedia train-candidate path")
    parser.add_argument("--news-input", type=str,
                         help="Override News (all enabled outlets, combined) train-candidate path")
    parser.add_argument("--input-override", action="append", default=[], metavar="SOURCE=PATH",
                         help="Override the train-candidate input path for 'indiccorp', "
                              "'wikipedia', or 'news' (e.g. --input-override news=/path/to/file.txt). "
                              "Equivalent to --indiccorp-input/--wikipedia-input/--news-input.")
    parser.add_argument("--validation-input", type=str, help="Override validation.txt path")
    parser.add_argument("--test-input", type=str, help="Override test.txt path")
    parser.add_argument("--output-dir", type=str, help="Override output directory (paths.decontaminated)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    progress_interval = config.get("logging", {}).get("progress_interval", 1_000_000)

    repo_root = Path(args.config).resolve().parent.parent.parent
    decon_cfg = config.get("decontamination", {})

    enabled = decon_cfg.get("enabled", True)
    use_validation = decon_cfg.get("validation", True)
    use_test = decon_cfg.get("test", True)
    exact_match = decon_cfg.get("exact_match", True)
    near_duplicate = decon_cfg.get("near_duplicate", True)
    nd_cfg = decon_cfg.get("near_duplicate_config", {})
    shingle_size = nd_cfg.get("shingle_size", 5)
    num_perm = nd_cfg.get("num_minhash_permutations", 16)
    bands = nd_cfg.get("lsh_bands", 4)
    rows = nd_cfg.get("lsh_rows", 4)
    nd_seed = nd_cfg.get("seed", config.get("split", {}).get("random_seed", 42))

    input_overrides = parse_source_overrides(args.input_override, ALL_SOURCES, "--input-override", logger)
    split_dir = repo_root / config["paths"]["split"]
    input_path_by_source = {
        source: (
            input_overrides.get(source)
            or getattr(args, f"{source}_input", None)
            or str(split_dir / f"{source}_train_candidate.txt")
        )
        for source in ALL_SOURCES
    }
    val_input = args.validation_input or str(repo_root / config["paths"]["final"] / "validation.txt")
    test_input = args.test_input or str(repo_root / config["paths"]["final"] / "test.txt")
    output_dir = args.output_dir or str(repo_root / config["paths"]["decontaminated"])

    output_path_by_source = {
        source: os.path.join(output_dir, f"{source}_train_decontam.txt") for source in ALL_SOURCES
    }
    enabled_map = source_enabled_map(config)

    if not args.overwrite and any(os.path.exists(p) for p in output_path_by_source.values()):
        logger.error(f"Output already exists in {output_dir}. Use --overwrite.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DECONTAMINATION STARTED")
    logger.info("=" * 70)
    logger.info(f"enabled={enabled}  exact_match={exact_match}  near_duplicate={near_duplicate}")
    logger.info(f"validation={use_validation}  test={use_test}")
    logger.info("Lookup mode: RAM-backed exact hashes + RAM-backed LSH band keys")
    logger.info("Training corpus remains streamed from disk; only held-out signatures stay in RAM.")

    if not enabled:
        logger.warning(
            "decontamination.enabled=false in config — copying train-candidate "
            "files through UNCHANGED. This is NOT leakage-safe; only use for "
            "debugging/ablation."
        )
        report = {"enabled": False}
        for source in ALL_SOURCES:
            in_path = input_path_by_source[source]
            out_path = output_path_by_source[source]
            if os.path.exists(in_path):
                shutil.copyfile(in_path, out_path)
                report[source] = {"kept": count_lines(out_path)}
            elif enabled_map[source]:
                logger.error(
                    f"REQUIRED source missing: '{source}' is required "
                    f"(enabled in config) but its train-candidate input was "
                    f"not found: {in_path}."
                )
                sys.exit(1)
        meta_path = str(repo_root / config["paths"]["metadata"] / "decontamination_report.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Decontamination report saved to {meta_path}")
        return

    holdout_files = []
    if use_validation:
        holdout_files.append(val_input)
    if use_test:
        holdout_files.append(test_input)

    if not holdout_files:
        logger.error(
            "decontamination.enabled=true but both validation and test are "
            "disabled — nothing to decontaminate against. Refusing to run."
        )
        sys.exit(1)

    for p in holdout_files:
        if not os.path.exists(p):
            logger.error(
                f"Held-out file not found: {p}\n"
                "Run script 07 (split) first — validation/test must be "
                "established BEFORE decontamination can run."
            )
            sys.exit(1)

    # FAILURE SAFETY: a source that is required but whose train-candidate
    # input is missing is a hard error, BEFORE any work happens — never
    # interpret a missing-but-required wikipedia or news input as an
    # empty corpus.
    missing_required_sources = [
        source for source in ALL_SOURCES
        if not os.path.exists(input_path_by_source[source]) and enabled_map[source]
    ]
    if missing_required_sources:
        for source in missing_required_sources:
            logger.error(
                f"REQUIRED source missing: '{source}' is enabled in config "
                f"but its train-candidate input was not found: "
                f"{input_path_by_source[source]}. Refusing to silently "
                f"produce an incomplete corpus. Run script 07 (split) first."
            )
        sys.exit(1)

    # ---- Build the held-out index entirely in RAM --------------------------
    # Training files remain streamed from disk. Only held-out hashes and LSH
    # band keys stay in RAM for fast membership lookups.
    minhasher = MinHasher(num_perm=num_perm, seed=nd_seed) if near_duplicate else None

    logger.info(f"Building in-memory held-out index from: {holdout_files}")
    exact_hashes, band_keys, holdout_counts = build_holdout_index(
        holdout_files, exact_match, near_duplicate, minhasher,
        bands, rows, shingle_size, logger,
    )


    validation_examples = holdout_counts.get(val_input, 0) if use_validation else 0
    test_examples = holdout_counts.get(test_input, 0) if use_test else 0

    # ---- Decontaminate each source independently --------------------------
    results = {}
    for source in ALL_SOURCES:
        in_path = input_path_by_source[source]
        out_path = output_path_by_source[source]
        logger.info(f"Decontaminating {source} train-candidate: {in_path}")
        results[source] = decontaminate_file(
            exact_hashes, band_keys, in_path, out_path,
            exact_match, near_duplicate, minhasher,
            bands, rows, shingle_size, logger, progress_interval,
        )

    for ext in ("-wal", "-shm"):
        if os.path.exists(p):
            os.remove(p)

    total_exact_removed = sum(r["exact_removed"] for r in results.values())
    total_near_dup_removed = sum(r["near_dup_removed"] for r in results.values())
    total_kept = sum(r["kept"] for r in results.values())
    total_kept_words = sum(r["kept_words"] for r in results.values())

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"EXACT MATCHES REMOVED: {total_exact_removed:,}")
    logger.info(f"NEAR DUPLICATES REMOVED: {total_near_dup_removed:,}")
    logger.info(f"FINAL TRAINING CORPUS AFTER DECONTAMINATION: {total_kept:,} lines "
                f"({total_kept_words:,} words)")
    logger.info("=" * 70)

    report = {
        "enabled": True,
        "config": {
            "exact_match": exact_match,
            "near_duplicate": near_duplicate,
            "validation": use_validation,
            "test": use_test,
            "near_duplicate_config": {
                "shingle_size": shingle_size,
                "num_minhash_permutations": num_perm,
                "lsh_bands": bands,
                "lsh_rows": rows,
                "seed": nd_seed,
            },
        },
        "held_out": {
            "validation_examples": validation_examples,
            "test_examples": test_examples,
            "validation_file": val_input if use_validation else None,
            "test_file": test_input if use_test else None,
        },
        "sources": results,
        "totals": {
            "exact_matches_removed": total_exact_removed,
            "near_duplicates_removed": total_near_dup_removed,
            "final_training_lines": total_kept,
            "final_training_words": total_kept_words,
        },
        "public_source": PUBLIC_SOURCE,
        "manual_sources": list(MANUAL_SOURCES),
        "source_enabled": enabled_map,
        "outputs": output_path_by_source,
    }
    meta_path = str(repo_root / config["paths"]["metadata"] / "decontamination_report.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Decontamination report saved to {meta_path}")


if __name__ == "__main__":
    main()
