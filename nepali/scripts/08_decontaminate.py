#!/usr/bin/env python3
"""
08_decontaminate.py — Remove Validation/Test Leakage From the Training Corpus

Implements the "Decontamination" procedure from CL3410 Lecture 3:

  1. Hold out the test set first (done by script 07, which now runs BEFORE
     this script).
  2. Search each source's candidate training corpus for EXACT and
     NEAR-DUPLICATE matches of the held-out validation/test data.
  3. Remove matches from the TRAINING corpus. validation.txt/test.txt are
     never modified — only read.
  4. Do this before the single final tokenizer (10) and corpus selection
     (also script 10) compute any statistics from the training corpus.

Near-duplicate matching uses MinHash + LSH banding (shared/near_dup.py) so
paraphrased / near-identical test items are caught, not just byte-identical
lines — exact string matching alone is NOT sufficient per the professor's
slide.

THREE SOURCES: PUBLIC (IndicCorp) + MANUAL (Onlinekhabar + Khabarhub)
=============================================================
Each of the three sources produced by script 07 (indiccorp, onlinekhabar,
khabarhub — split/{source}_train_candidate.txt) is decontaminated
INDEPENDENTLY here, using the exact same algorithm and the SAME held-out
index built once from validation.txt/test.txt. The three sources are never
merged in this script — outputs are
decontaminated/{source}_train_decontam.txt, matching exactly what
09_concatenate_final_corpus.py already expects by default (indiccorp_train_decontam.txt,
onlinekhabar_train_decontam.txt, khabarhub_train_decontam.txt) — see
09_concatenate_final_corpus.py's module docstring.

FAILURE SAFETY: a manual subsource that is ENABLED in config but whose
train-candidate input file is missing is a hard ERROR, not a silent skip —
same enabled/disabled distinction used by 07_split_train_val_test.py, so
the manual corpus can never silently vanish between 07 and 09.

Memory guarantees (hundreds of millions of lines):
  • The held-out validation/test corpus is small relative to the training
    corpus (~5%+5% split) and its exact-hashes / LSH band-keys are the
    ONLY thing held in a (disk-backed, WAL-mode SQLite) index, built once
    and reused across all three sources.
  • Each source's training corpus is streamed exactly once, line by line.
    It is NEVER loaded into RAM, and its own hashes/signatures are never
    persisted — only looked up against the held-out index.

Usage:
    python nepali/scripts/08_decontaminate.py \\
        --config nepali/configs/nepali_data.yaml

    # Explicit per-source overrides:
    python nepali/scripts/08_decontaminate.py --config nepali/configs/nepali_data.yaml \\
        --indiccorp-input nepali/data/intermediate/split/indiccorp_train_candidate.txt \\
        --onlinekhabar-input nepali/data/intermediate/split/onlinekhabar_train_candidate.txt \\
        --khabarhub-input nepali/data/intermediate/split/khabarhub_train_candidate.txt
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.data_utils import count_lines, iter_lines
from shared.near_dup import MinHasher, line_band_keys


# ---------------------------------------------------------------------------
# Source configuration (mirrors 07_split_train_val_test.py / 09_concatenate_final_corpus.py)
# ---------------------------------------------------------------------------

PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("onlinekhabar", "khabarhub")
ALL_SOURCES = (PUBLIC_SOURCE,) + MANUAL_SOURCES


def source_enabled_map(config: Dict[str, Any]) -> Dict[str, bool]:
    """Mirrors 07_split_train_val_test.py's identical logic: a manual
    source is 'required' iff enabled in config."""
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
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("08_decontaminate")


def exact_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def build_holdout_index(
    conn: sqlite3.Connection,
    holdout_files: List[str],
    exact_match: bool,
    near_duplicate: bool,
    minhasher: Optional[MinHasher],
    bands: int,
    rows: int,
    shingle_size: int,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Stream *holdout_files* (validation/test) into the exact-hash and/or
    LSH band-key SQLite index. Returns example counts per file."""
    counts = {}
    cur = conn.cursor()
    cur.execute("BEGIN;")
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
                cur.execute(
                    "INSERT OR IGNORE INTO exact_hashes (hash) VALUES (?)",
                    (exact_hash(line),),
                )
                n_hashes_inserted += cur.rowcount
            if near_duplicate:
                for key in line_band_keys(line, minhasher, bands, rows, shingle_size):
                    cur.execute(
                        "INSERT OR IGNORE INTO band_keys (key) VALUES (?)", (key,)
                    )
                    n_bands_inserted += cur.rowcount
            if n % 200_000 == 0:
                conn.commit()
                cur.execute("BEGIN;")
        counts[path] = n
        logger.info(f"  Indexed {n:,} held-out lines from {path}")
    conn.commit()
    logger.info(
        f"  Held-out index built: {n_hashes_inserted:,} unique exact hashes, "
        f"{n_bands_inserted:,} unique LSH band keys"
    )
    return counts


def decontaminate_file(
    conn: sqlite3.Connection,
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
) -> Dict[str, int]:
    """Stream *input_path*, drop lines that exact- or near-duplicate the
    held-out index, write survivors to *output_path*. One pass, O(1) memory
    beyond the SQLite index (which holds only held-out signatures)."""
    if not os.path.exists(input_path):
        logger.warning(f"Train-candidate file not found, skipping: {input_path}")
        if os.path.exists(output_path):
            logger.warning(
                f"Removing stale output from a prior run (its input no "
                f"longer exists): {output_path}"
            )
            os.remove(output_path)
        return {"input_lines": 0, "exact_removed": 0, "near_dup_removed": 0, "kept": 0, "kept_words": 0}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cur = conn.cursor()

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
                cur.execute(
                    "SELECT 1 FROM exact_hashes WHERE hash=? LIMIT 1", (exact_hash(line),)
                )
                if cur.fetchone() is not None:
                    exact_removed += 1
                    dropped = True

            if not dropped and near_duplicate and line:
                for key in line_band_keys(line, minhasher, bands, rows, shingle_size):
                    cur.execute("SELECT 1 FROM band_keys WHERE key=? LIMIT 1", (key,))
                    if cur.fetchone() is not None:
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
        description="Decontaminate the training corpus against held-out validation/test."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    for source in ALL_SOURCES:
        parser.add_argument(f"--{source}-input", type=str,
                             help=f"Override {source} train-candidate path")
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

    split_dir = repo_root / config["paths"]["split"]
    input_path_by_source = {
        source: (getattr(args, f"{source}_input", None) or str(split_dir / f"{source}_train_candidate.txt"))
        for source in ALL_SOURCES
    }
    val_input = args.validation_input or str(repo_root / config["paths"]["final"] / "validation.txt")
    test_input = args.test_input or str(repo_root / config["paths"]["final"] / "test.txt")
    output_dir = args.output_dir or str(repo_root / config["paths"]["decontaminated"])

    output_path_by_source = {
        source: os.path.join(output_dir, f"{source}_train_decontam.txt") for source in ALL_SOURCES
    }
    enabled = source_enabled_map(config)

    if not args.overwrite and any(os.path.exists(p) for p in output_path_by_source.values()):
        logger.error(f"Output already exists in {output_dir}. Use --overwrite.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DECONTAMINATION STARTED")
    logger.info("=" * 70)
    logger.info(f"enabled={enabled}  exact_match={exact_match}  near_duplicate={near_duplicate}")
    logger.info(f"validation={use_validation}  test={use_test}")

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
            elif enabled[source]:
                logger.error(
                    f"REQUIRED source missing: '{source}' is enabled in "
                    f"config but its train-candidate input was not found: "
                    f"{in_path}."
                )
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

    # ---- Build the (small) held-out index --------------------------------
    db_path = os.path.join(output_dir, "decontamination_index.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("CREATE TABLE IF NOT EXISTS exact_hashes (hash TEXT PRIMARY KEY);")
    conn.execute("CREATE TABLE IF NOT EXISTS band_keys (key TEXT PRIMARY KEY);")
    conn.commit()

    minhasher = MinHasher(num_perm=num_perm, seed=nd_seed) if near_duplicate else None

    logger.info(f"Building held-out index from: {holdout_files}")
    holdout_counts = build_holdout_index(
        conn, holdout_files, exact_match, near_duplicate, minhasher,
        bands, rows, shingle_size, logger,
    )

    validation_examples = holdout_counts.get(val_input, 0) if use_validation else 0
    test_examples = holdout_counts.get(test_input, 0) if use_test else 0

    # ---- Decontaminate each train-candidate source independently ---------
    results = {}
    for source in ALL_SOURCES:
        in_path = input_path_by_source[source]
        out_path = output_path_by_source[source]
        logger.info(f"Decontaminating {source} train-candidate: {in_path}")
        results[source] = decontaminate_file(
            conn, in_path, out_path, exact_match, near_duplicate, minhasher,
            bands, rows, shingle_size, logger, progress_interval,
        )

    # FAILURE SAFETY: a source that is enabled in config but whose
    # train-candidate input was missing (kept=0, input_lines=0 from
    # decontaminate_file's own missing-file branch) is a hard error, never
    # a silent skip — prevents the manual corpus from silently vanishing
    # between script 07 and script 09.
    missing_required_sources = [
        source for source in ALL_SOURCES
        if not os.path.exists(input_path_by_source[source]) and enabled[source]
    ]
    if missing_required_sources:
        conn.close()
        for source in missing_required_sources:
            logger.error(
                f"REQUIRED source missing: '{source}' is enabled in config "
                f"but its train-candidate input was not found: "
                f"{input_path_by_source[source]}. Refusing to silently "
                f"produce an incomplete corpus."
            )
        sys.exit(1)

    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in ("-wal", "-shm"):
        p = db_path + ext
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
        "source_enabled": enabled,
        "outputs": output_path_by_source,
    }
    meta_path = str(repo_root / config["paths"]["metadata"] / "decontamination_report.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Decontamination report saved to {meta_path}")


if __name__ == "__main__":
    main()
