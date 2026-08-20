#!/usr/bin/env python3
"""
05_deduplicate_mac.py — Mac-Optimized Cross-Source Exact + Near-Duplicate +
Sub-Document Deduplication (IndicCorp + Onlinekhabar + Khabarhub)

SEPARATE FROM nepali/scripts/05_deduplicate.py. This file exists purely as
a faster alternative for a machine with abundant RAM (developed against
Apple M4 / 24 GB) — it does NOT replace, modify, or get imported by
05_deduplicate.py, and 05_deduplicate.py is unaffected by anything in this
file. Use 05_deduplicate.py on constrained (~8 GB) machines; use this
script on the Mac.

METHODOLOGY — IDENTICAL to 05_deduplicate.py, only the near-duplicate
INDEX STORAGE differs:
  1. EXACT duplicate detection — SHA-256 line hash, shared across all three
     sources (deduplicate_exact_cross_source() below is a verbatim copy of
     05_deduplicate.py's function of the same name — not reimplemented,
     not altered).
  2. NEAR-DUPLICATE / SUB-DOCUMENT detection — word shingles -> MinHash
     signature -> LSH banding for candidate generation -> Jaccard estimate
     (near-duplicate) / overlap-containment estimate (sub-document
     duplicate) for confirmation. Uses the EXACT SAME primitives as
     05_deduplicate.py: shared/near_dup.py's MinHasher,
     signature_and_shingle_count, lsh_band_keys, jaccard_estimate,
     overlap_estimate — imported unchanged, never reimplemented here.

THE ONLY ALGORITHMIC-INFRASTRUCTURE CHANGE: the near-duplicate pass uses
shared/near_dup_mac.py's InMemoryMinHashDedupIndex (a plain-dict-backed
LSH index) instead of shared/near_dup.py's SQLite-backed
MinHashDedupIndex. Candidate-generation and confirmation semantics
(Jaccard, overlap/containment, source precedence, sub-document detection)
are unchanged — see shared/near_dup_mac.py's module docstring for the
memory model. The corpus text itself is still streamed from disk line by
line and never loaded into memory; only compact per-KEPT-line signatures/
shingle-counts/band-keys are held in RAM.

Cross-source deduplication and source precedence (Onlinekhabar > Khabarhub
> IndicCorp) are identical to 05_deduplicate.py — see that
script's module docstring for the full rationale, which applies unchanged
here.

Mac-specific default parameters (configurable via nepali_data.yaml's
`deduplication_mac` block, NOT hard-coded — see resolve_near_cfg() below):
    shingle_type: word
    shingle_size: 5
    num_minhash_permutations: 64
    lsh_bands: 16
    lsh_rows: 4
    similarity_threshold: 0.40
    sub_document_overlap_threshold: 0.85
    sub_document_length_ratio_max: 0.70
    min_document_words: 4
    seed: 42
These differ from 05_deduplicate.py's Linux defaults (32 perm / 16 bands /
2 rows / 0.30 threshold / 3-word shingles) deliberately — they were NOT
copied from any third-party reference implementation; they are new values
chosen for this project, using a RAM-backed index inspired by (but not a
copy of) the general "use an in-memory LSH structure when RAM allows it"
idea. Jaccard confirmation, sub-document/overlap detection, cross-source
dedup, and source precedence are all still present and unchanged.

Compatibility with the rest of the pipeline (07 -> 08 -> 09 -> 11):
  * Same input filenames as 05_deduplicate.py: quality.<source>_output_dir
    / quality.<source>_output_filename, defaulting to
    `{source}_ne_quality.txt` in quality.output_dir.
  * Same output filenames as 05_deduplicate.py: paths.deduplicated/
    `{source}_ne_deduplicated.txt` for indiccorp/onlinekhabar/khabarhub —
    never merged into one file. Downstream scripts (07/08/09/
    11) read these exact filenames regardless of which dedup script
    produced them, so this script is a drop-in alternative to
    05_deduplicate.py, not a fork of the downstream pipeline.
  * Same CLI surface: --config, --{source}-input/--{source}-output,
    --resume, --overwrite, --in-memory (exact pass), --skip-near-dup.
  * Mac-only additions: --sample-lines N (cap each source to at most N
    lines, for benchmarking on a 500K-1M-line sample before running the
    full corpus) and richer benchmark logging (candidates generated,
    comparisons performed, avg candidates/doc, memory usage).

Usage:
    python nepali/scripts/05_deduplicate_mac.py --config nepali/configs/nepali_data.yaml

    # Benchmark on a 1M-line-per-source sample first:
    python nepali/scripts/05_deduplicate_mac.py --config nepali/configs/nepali_data.yaml \\
        --sample-lines 1000000 --overwrite
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.data_utils import iter_lines, count_lines
from shared.run_stats import cleanup_prior_stage_files
from shared.near_dup import (
    MinHasher,
    jaccard_estimate,
    lsh_band_keys,
    overlap_estimate,
    signature_and_shingle_count,
)
from shared.near_dup_mac import InMemoryMinHashDedupIndex
# Same accounting-reconciliation check the Linux script uses — reused, not
# reimplemented, so there is one validation implementation shared by both.
from shared.dedup_accounting import validate_dedup_accounting

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None


# ---------------------------------------------------------------------------
# Mac-tuned near-duplicate defaults (see module docstring). Every value is
# still overridable via nepali_data.yaml's `deduplication_mac.near_duplicate`
# block — these are just the fallback defaults, not hard-coded behavior.
# ---------------------------------------------------------------------------
MAC_NEAR_DUP_DEFAULTS: Dict[str, Any] = {
    "shingle_type": "word",
    "shingle_size": 5,
    "num_minhash_permutations": 64,
    "lsh_bands": 16,
    "lsh_rows": 4,
    "similarity_threshold": 0.40,
    "sub_document_overlap_threshold": 0.85,
    "sub_document_length_ratio_max": 0.70,
    "min_document_words": 4,
    "seed": 42,
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    logging_config = config.get("logging", {})
    level_str = logging_config.get("level", "INFO")
    fmt = logging_config.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level_str), format=fmt)
    return logging.getLogger("05_deduplicate_mac")


def peak_memory_mb() -> Optional[float]:
    """Best-effort peak RSS in MB, for benchmark logging. Returns None if
    unavailable (non-POSIX platform) — never raises."""
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports it in KB.
    divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
    return usage / divisor


def capped_lines(path: str, limit: Optional[int]) -> Iterator[str]:
    """Like shared.data_utils.iter_lines(path), but stops after *limit*
    lines when limit is not None. Used only for --sample-lines
    benchmarking; the full run (limit=None) streams every line exactly as
    05_deduplicate.py does."""
    if limit is None:
        yield from iter_lines(path)
        return
    n = 0
    for line in iter_lines(path):
        if n >= limit:
            return
        n += 1
        yield line


# ---------------------------------------------------------------------------
# Pass 1 (cross-source): EXACT duplicate removal. Verbatim copy of
# 05_deduplicate.py's deduplicate_exact_cross_source() — the exact pass is
# not this script's optimization target (SQLite/in-memory-dict hashing of
# short strings is already cheap; the near-duplicate LSH index is where
# repeated disk round-trips actually hurt), so it is reproduced unchanged
# rather than reimplemented, to guarantee identical exact-dedup behavior.
# ---------------------------------------------------------------------------

def deduplicate_exact_cross_source(
    sources,
    output_paths: Dict[str, str],
    logger: logging.Logger,
    progress_interval: int,
    resume: bool,
    in_memory: bool = False,
    db_path: str = None,
    sample_lines: Optional[int] = None,
) -> Dict[str, Any]:
    if not resume:
        for label, _ in sources:
            p = output_paths.get(label)
            if p and os.path.exists(p):
                os.remove(p)
        if not in_memory and db_path and os.path.exists(db_path):
            os.remove(db_path)

    conn = None
    cursor = None
    mem_hash_source: Dict[str, str] = {}

    if in_memory:
        if resume:
            for label, _ in sources:
                p = output_paths.get(label)
                if p and os.path.exists(p):
                    for line in iter_lines(p):
                        h = hashlib.sha256(line.encode("utf-8")).hexdigest()
                        mem_hash_source.setdefault(h, label)
    else:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_hashes (hash TEXT PRIMARY KEY, source TEXT NOT NULL);"
        )
        conn.commit()
        cursor = conn.cursor()
        if resume:
            for label, _ in sources:
                p = output_paths.get(label)
                if p and os.path.exists(p):
                    for line in iter_lines(p):
                        h = hashlib.sha256(line.encode("utf-8")).hexdigest()
                        cursor.execute(
                            "INSERT OR IGNORE INTO seen_hashes (hash, source) VALUES (?, ?)", (h, label)
                        )
            conn.commit()

    mode = "a" if resume else "w"
    out_files: Dict[str, Any] = {}
    for label, _ in sources:
        p = output_paths.get(label)
        if not p:
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        out_files[label] = open(p, mode, encoding="utf-8", buffering=8192)

    stats = {
        label: {"input_lines": 0, "duplicates": 0, "uniques": 0, "cross_source_duplicates": 0}
        for label, _ in sources
    }
    total_processed = 0
    t0 = time.time()

    logger.info(f"Source precedence (ties keep the EARLIER source): {[label for label, _ in sources]}")

    try:
        for label, path in sources:
            if not path or not os.path.exists(path):
                logger.warning(f"[{label}] exact-dedup input not found — treating as an empty source: {path}")
                continue
            out_f = out_files.get(label)
            for line in capped_lines(path, sample_lines):
                stats[label]["input_lines"] += 1
                total_processed += 1
                h = hashlib.sha256(line.encode("utf-8")).hexdigest()

                if in_memory:
                    origin = mem_hash_source.get(h)
                    is_new = origin is None
                    if is_new:
                        mem_hash_source[h] = label
                else:
                    cursor.execute(
                        "INSERT OR IGNORE INTO seen_hashes (hash, source) VALUES (?, ?)", (h, label)
                    )
                    is_new = cursor.rowcount > 0
                    origin = None
                    if not is_new:
                        cursor.execute("SELECT source FROM seen_hashes WHERE hash=?", (h,))
                        row = cursor.fetchone()
                        origin = row[0] if row else None

                if is_new:
                    stats[label]["uniques"] += 1
                    if out_f is not None:
                        out_f.write(line + "\n")
                else:
                    stats[label]["duplicates"] += 1
                    if origin is not None and origin != label:
                        stats[label]["cross_source_duplicates"] += 1

                if not in_memory and total_processed % 100_000 == 0:
                    conn.commit()

                if total_processed % progress_interval == 0:
                    elapsed = time.time() - t0
                    speed = total_processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  [{label}] {stats[label]['input_lines']:,} lines | "
                        f"{total_processed:,} total across sources so far | {speed:,.0f} lines/s"
                    )
    finally:
        if conn is not None:
            conn.commit()
            conn.close()
        for f in out_files.values():
            f.close()

    combined = {
        "input_lines": sum(s["input_lines"] for s in stats.values()),
        "duplicates": sum(s["duplicates"] for s in stats.values()),
        "uniques": sum(s["uniques"] for s in stats.values()),
        "cross_source_duplicates": sum(s["cross_source_duplicates"] for s in stats.values()),
        "by_source": stats,
    }
    ratio = (combined["duplicates"] / combined["input_lines"] * 100) if combined["input_lines"] > 0 else 0
    logger.info(
        f"Finished cross-source exact pass. Total lines: {combined['input_lines']:,}, "
        f"Duplicates: {combined['duplicates']:,} ({ratio:.2f}%) "
        f"[of which {combined['cross_source_duplicates']:,} were cross-source], "
        f"Unique: {combined['uniques']:,}"
    )
    return combined


# ---------------------------------------------------------------------------
# Pass 2 (cross-source): NEAR-DUPLICATE / SUB-DOCUMENT duplicate removal —
# THE MAC OPTIMIZATION. Uses InMemoryMinHashDedupIndex (shared/near_dup_mac.py)
# instead of MinHashDedupIndex (shared/near_dup.py, SQLite-backed). All
# candidate-generation and confirmation logic (LSH banding, Jaccard
# estimate, overlap/containment estimate, sub-document detection, source
# precedence) is identical to 05_deduplicate.py's
# deduplicate_near_cross_source() — only the index storage call sites
# (index.find_candidates / index.add) changed target class.
# ---------------------------------------------------------------------------

def resolve_near_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read nepali_data.yaml's `deduplication_mac.near_duplicate` block,
    falling back to MAC_NEAR_DUP_DEFAULTS for any key not present — so the
    Mac parameters are configurable (edit the yaml) rather than hard-coded,
    while still working out of the box with the parameters specified for
    this task. Does NOT read the Linux `deduplication` block — the two
    scripts' tuning is deliberately independent."""
    mac_cfg = config.get("deduplication_mac", {}).get("near_duplicate", {})
    return {**MAC_NEAR_DUP_DEFAULTS, **mac_cfg}


def deduplicate_near_cross_source_mac(
    sources,
    output_paths: Dict[str, str],
    near_cfg: Dict[str, Any],
    logger: logging.Logger,
    progress_interval: int,
    sample_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """NEAR-DUPLICATE / SUB-DOCUMENT pass across multiple labeled sources,
    RAM-backed LSH index. Same source-precedence semantics as
    05_deduplicate.py: the first source to KEEP a given (near-)unique line
    wins; a later source's near/sub-document duplicate of it is removed."""
    shingle_type = near_cfg.get("shingle_type", "word")
    shingle_size = near_cfg.get("shingle_size", 5)
    num_perm = near_cfg.get("num_minhash_permutations", 64)
    bands = near_cfg.get("lsh_bands", 16)
    rows = near_cfg.get("lsh_rows", 4)
    similarity_threshold = near_cfg.get("similarity_threshold", 0.40)
    sub_doc_overlap_threshold = near_cfg.get("sub_document_overlap_threshold", 0.85)
    sub_doc_length_ratio_max = near_cfg.get("sub_document_length_ratio_max", 0.70)
    min_document_words = near_cfg.get("min_document_words", 4)
    seed = near_cfg.get("seed", 42)

    logger.info("=" * 70)
    logger.info("CROSS-SOURCE NEAR-DUPLICATE DEDUPLICATION STARTED (Mac / in-memory LSH)")
    logger.info("=" * 70)
    logger.info(f"Source precedence (ties keep the EARLIER source): {[label for label, _ in sources]}")
    logger.info(
        f"shingle_type={shingle_type} shingle_size={shingle_size} "
        f"num_perm={num_perm} lsh_bands={bands} lsh_rows={rows}"
    )
    logger.info(
        f"similarity_threshold={similarity_threshold} "
        f"sub_document_overlap_threshold={sub_doc_overlap_threshold} "
        f"min_document_words={min_document_words}"
    )
    if sample_lines is not None:
        logger.info(f"BENCHMARK MODE: capped at {sample_lines:,} lines per source")

    minhasher = MinHasher(num_perm=num_perm, seed=seed)
    index = InMemoryMinHashDedupIndex(num_perm=num_perm)

    out_files: Dict[str, Any] = {}
    for label, _ in sources:
        p = output_paths.get(label)
        if not p:
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        out_files[label] = open(p, "w", encoding="utf-8", buffering=1 << 20)

    stats = {
        label: {
            "input_lines": 0,
            "near_duplicates_removed": 0,
            "sub_document_duplicates_removed": 0,
            "kept": 0,
            "kept_words": 0,
            "short_lines_kept": 0,
            # Blank lines are read (counted in input_lines) but neither
            # kept, dropped, nor short-line-kept — tracked explicitly so
            # input_lines = kept + near_duplicates_removed +
            # sub_document_duplicates_removed + blank_lines_skipped always
            # reconciles (see main()'s post-run accounting validation).
            "blank_lines_skipped": 0,
        }
        for label, _ in sources
    }

    next_line_id = 0
    total = 0
    total_candidates = 0
    total_comparisons = 0
    lines_evaluated = 0  # lines that reached candidate generation (excludes short-line fast path)
    t0 = time.time()

    try:
        for label, path in sources:
            if not path or not os.path.exists(path):
                logger.warning(f"[{label}] near-dup input not found — treating as an empty source: {path}")
                continue
            out_f = out_files.get(label)
            for line in capped_lines(path, sample_lines):
                total += 1
                stats[label]["input_lines"] += 1
                if not line:
                    stats[label]["blank_lines_skipped"] += 1
                    continue

                word_count = len(line.split())
                if word_count < min_document_words:
                    out_f.write(line + "\n")
                    stats[label]["kept"] += 1
                    stats[label]["kept_words"] += word_count
                    stats[label]["short_lines_kept"] += 1
                    continue

                sig, shingle_count = signature_and_shingle_count(
                    line, minhasher, shingle_size, shingle_type
                )
                band_keys = lsh_band_keys(sig, bands, rows)
                candidates = index.find_candidates(band_keys)
                lines_evaluated += 1
                total_candidates += len(candidates)

                verdict = None
                for _cid, csig, ccount in candidates:
                    total_comparisons += 1
                    j = jaccard_estimate(sig, csig)
                    shorter, longer = sorted((shingle_count, ccount))
                    length_ratio = shorter / longer if longer > 0 else 1.0

                    if length_ratio <= sub_doc_length_ratio_max:
                        ov = overlap_estimate(j, shingle_count, ccount)
                        if ov >= sub_doc_overlap_threshold:
                            verdict = ("sub_document", ov)
                            break

                    if j >= similarity_threshold and verdict is None:
                        verdict = ("near_duplicate", j)

                if verdict is not None:
                    kind, _score = verdict
                    if kind == "sub_document":
                        stats[label]["sub_document_duplicates_removed"] += 1
                    else:
                        stats[label]["near_duplicates_removed"] += 1
                else:
                    out_f.write(line + "\n")
                    stats[label]["kept"] += 1
                    stats[label]["kept_words"] += word_count
                    index.add(next_line_id, sig, shingle_count, band_keys)
                    next_line_id += 1

                if total % progress_interval == 0:
                    elapsed = time.time() - t0
                    speed = total / elapsed if elapsed > 0 else 0
                    avg_candidates = total_candidates / lines_evaluated if lines_evaluated > 0 else 0
                    mem = peak_memory_mb()
                    mem_str = f"{mem:,.0f} MB" if mem is not None else "n/a"
                    logger.info(
                        f"  [{label}] {total:,} lines scanned (all sources) | "
                        f"kept so far (this source) {stats[label]['kept']:,} | "
                        f"{speed:,.0f} lines/s | avg candidates/doc={avg_candidates:.2f} | "
                        f"peak RSS={mem_str}"
                    )
    finally:
        for f in out_files.values():
            f.close()

    combined = {
        "input_lines": sum(s["input_lines"] for s in stats.values()),
        "near_duplicates_removed": sum(s["near_duplicates_removed"] for s in stats.values()),
        "sub_document_duplicates_removed": sum(s["sub_document_duplicates_removed"] for s in stats.values()),
        "kept": sum(s["kept"] for s in stats.values()),
        "kept_words": sum(s["kept_words"] for s in stats.values()),
        "short_lines_kept": sum(s["short_lines_kept"] for s in stats.values()),
        "blank_lines_skipped": sum(s["blank_lines_skipped"] for s in stats.values()),
        "by_source": stats,
        # Benchmark-oriented extras (not present in 05_deduplicate.py's
        # report — additive, does not change any existing field).
        "benchmark": {
            "candidates_generated": total_candidates,
            "comparisons_performed": total_comparisons,
            "lines_evaluated": lines_evaluated,
            "avg_candidates_per_document": round(total_candidates / lines_evaluated, 4) if lines_evaluated > 0 else 0.0,
            "index_memory_report": index.memory_report(),
            "peak_rss_mb": peak_memory_mb(),
        },
    }

    elapsed = time.time() - t0
    speed = combined["input_lines"] / elapsed if elapsed > 0 else 0
    logger.info("=" * 70)
    logger.info(f"NEAR DUPLICATES REMOVED (all sources): {combined['near_duplicates_removed']:,}")
    logger.info(f"SUB-DOCUMENT DUPLICATES REMOVED (all sources): {combined['sub_document_duplicates_removed']:,}")
    logger.info(f"CANDIDATES GENERATED: {total_candidates:,}  COMPARISONS PERFORMED: {total_comparisons:,}")
    logger.info(f"AVG CANDIDATES/DOCUMENT: {combined['benchmark']['avg_candidates_per_document']:.4f}")
    mem = peak_memory_mb()
    if mem is not None:
        logger.info(f"PEAK RSS: {mem:,.1f} MB")
    logger.info(
        f"CROSS-SOURCE NEAR-DUPLICATE DEDUPLICATION FINISHED: {combined['input_lines']:,} lines scanned, "
        f"{combined['kept']:,} kept in {elapsed:.1f}s ({speed:,.0f} lines/s)"
    )
    logger.info("=" * 70)

    return combined


# ---------------------------------------------------------------------------
# Source configuration — identical set/order/precedence to 05_deduplicate.py.
# ---------------------------------------------------------------------------

SOURCE_PRECEDENCE_ORDER = ("onlinekhabar", "khabarhub", "indiccorp")

DEFAULT_QUALITY_FILENAME = {name: f"{name}_ne_quality.txt" for name in SOURCE_PRECEDENCE_ORDER}
DEFAULT_DEDUP_FILENAME = {name: f"{name}_ne_deduplicated.txt" for name in SOURCE_PRECEDENCE_ORDER}

# Mirrors 05_quality_filter.py's _REJECTED_FILENAME_CONFIG_KEY — needed
# here ONLY to compute each source's quality-stage rejected-file path, so
# cleanup after a real (non --sample-lines) run can delete it alongside
# {source}_ne_quality.txt (same lifecycle — rejected files are not
# preserved independently of their stage's main output).
_QUALITY_REJECTED_FILENAME_CONFIG_KEY = {
    "indiccorp": "rejected_filename",
    "onlinekhabar": "onlinekhabar_rejected_filename",
    "khabarhub": "khabarhub_rejected_filename",
}


def resolve_source_paths(config: Dict[str, Any], repo_root: Path, args) -> tuple:
    """Identical resolution logic to 05_deduplicate.py's
    resolve_source_paths() — same config keys, same defaults — so this
    script reads/writes the exact same files 05_deduplicate.py would,
    making it a true drop-in alternative for downstream stages (07/08/09/11)."""
    quality_cfg = config.get("quality", {})
    shared_quality_dir = quality_cfg.get("output_dir", "nepali/data/intermediate/quality")
    dedup_dir = config["paths"]["deduplicated"]

    input_path_by_label: Dict[str, str] = {}
    output_path_by_label: Dict[str, str] = {}

    for name in SOURCE_PRECEDENCE_ORDER:
        cli_input = getattr(args, f"{name}_input", None)
        cli_output = getattr(args, f"{name}_output", None)

        in_dir = quality_cfg.get(f"{name}_output_dir", shared_quality_dir)
        in_filename = quality_cfg.get(f"{name}_output_filename", DEFAULT_QUALITY_FILENAME[name])
        input_path_by_label[name] = cli_input or str(repo_root / in_dir / in_filename)

        out_filename = DEFAULT_DEDUP_FILENAME[name]
        output_path_by_label[name] = cli_output or str(repo_root / dedup_dir / out_filename)

    return input_path_by_label, output_path_by_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mac-optimized (in-memory LSH index) cross-source exact + "
                     "near-duplicate + sub-document deduplication across "
                     "IndicCorp, Onlinekhabar, and Khabarhub. "
                     "Separate from 05_deduplicate.py (Linux/8GB) — same "
                     "methodology and output filenames, RAM-backed near-dup "
                     "index instead of SQLite."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    for name in SOURCE_PRECEDENCE_ORDER:
        input_aliases = [f"--{name}-input"] + (["--input"] if name == "indiccorp" else [])
        output_aliases = [f"--{name}-output"] + (["--output"] if name == "indiccorp" else [])
        parser.add_argument(*input_aliases, dest=f"{name}_input", type=str,
                             help=f"Override {name} quality-stage input file path")
        parser.add_argument(*output_aliases, dest=f"{name}_output", type=str,
                             help=f"Override {name} deduplicated-stage output file path")
    parser.add_argument("--resume", action="store_true", help="Resume the EXACT pass from previous run (near-dup pass always runs fully)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output (ignores resume)")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory dict instead of SQLite for the EXACT pass (the near-dup pass is already in-memory in this script)")
    parser.add_argument("--skip-near-dup", action="store_true", help="Force-disable the near-duplicate pass regardless of config")
    parser.add_argument("--sample-lines", type=int, default=None,
                         help="Benchmark mode: cap each source to at most N lines "
                              "(e.g. 500000-1000000) instead of processing the full corpus.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    progress_interval = config.get("logging", {}).get("progress_interval", 1000000)
    dedup_cfg = config.get("deduplication_mac", config.get("deduplication", {}))
    exact_enabled = dedup_cfg.get("exact", {}).get("enabled", True)
    near_cfg = resolve_near_cfg(config)
    near_enabled = dedup_cfg.get("near_duplicate", {}).get("enabled", True) and not args.skip_near_dup
    keep_intermediate = config.get("storage", {}).get("keep_intermediate_files", True)

    repo_root = Path(args.config).resolve().parent.parent.parent

    input_path_by_label, output_paths_by_label = resolve_source_paths(config, repo_root, args)

    for p in output_paths_by_label.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)

    resume = args.resume and not args.overwrite

    existing_inputs = [name for name in SOURCE_PRECEDENCE_ORDER if os.path.exists(input_path_by_label[name])]
    if not existing_inputs:
        logger.error(
            "No source input was found. Checked:\n" + "\n".join(
                f"  {name}: {input_path_by_label[name]}" for name in SOURCE_PRECEDENCE_ORDER
            )
        )
        sys.exit(1)

    sources_in_precedence_order = [(name, input_path_by_label[name]) for name in SOURCE_PRECEDENCE_ORDER]

    for name in SOURCE_PRECEDENCE_ORDER:
        found = "FOUND" if name in existing_inputs else "NOT FOUND — treated as empty source"
        logger.info(f"{name} input path:  {input_path_by_label[name]}  [{found}]")
        logger.info(f"{name} output path: {output_paths_by_label[name]}")
    logger.info(f"exact.enabled={exact_enabled}  near_duplicate.enabled={near_enabled}  (Mac/in-memory LSH)")
    logger.info(
        "Cross-source dedup: " + " + ".join(SOURCE_PRECEDENCE_ORDER) +
        " are deduplicated TOGETHER in one pass "
        "(source precedence: " + " > ".join(SOURCE_PRECEDENCE_ORDER) + " on ties)."
    )

    run_t0 = time.time()

    exact_output_paths = (
        {
            name: os.path.join(
                os.path.dirname(output_paths_by_label[name]), f"{name}_ne_exact_deduplicated.txt"
            )
            for name in SOURCE_PRECEDENCE_ORDER
        }
        if near_enabled
        else output_paths_by_label
    )

    exact_stats = {"input_lines": 0, "duplicates": 0, "uniques": 0, "cross_source_duplicates": 0, "by_source": {}}
    if exact_enabled:
        logger.info("=" * 70)
        logger.info("CROSS-SOURCE EXACT DEDUPLICATION STARTED")
        logger.info("=" * 70)
        db_path = os.path.join(os.path.dirname(output_paths_by_label["indiccorp"]), "cross_source_exact_dedup_mac.sqlite")
        if not args.in_memory:
            logger.info(f"Database path: {db_path}")
        exact_stats = deduplicate_exact_cross_source(
            sources_in_precedence_order,
            exact_output_paths,
            logger,
            progress_interval,
            resume,
            in_memory=args.in_memory,
            db_path=db_path,
            sample_lines=args.sample_lines,
        )
        logger.info(f"EXACT DUPLICATES REMOVED (all sources): {exact_stats['duplicates']:,}")
    else:
        logger.info("Exact dedup disabled by config — copying inputs through unchanged.")
        by_source = {}
        for label, in_path in sources_in_precedence_order:
            out_path = exact_output_paths.get(label)
            n = 0
            if in_path and os.path.exists(in_path) and out_path:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as out_f:
                    for line in capped_lines(in_path, args.sample_lines):
                        out_f.write(line + "\n")
                        n += 1
            by_source[label] = {"input_lines": n, "duplicates": 0, "uniques": n, "cross_source_duplicates": 0}
        exact_stats = {
            "input_lines": sum(s["input_lines"] for s in by_source.values()),
            "duplicates": 0,
            "uniques": sum(s["uniques"] for s in by_source.values()),
            "cross_source_duplicates": 0,
            "by_source": by_source,
        }

    near_stats = {"near_duplicates_removed": 0, "sub_document_duplicates_removed": 0, "by_source": {}, "benchmark": {}}
    if near_enabled:
        near_input_sources = [(label, exact_output_paths.get(label)) for label, _ in sources_in_precedence_order]
        near_stats = deduplicate_near_cross_source_mac(
            near_input_sources, output_paths_by_label, near_cfg, logger, progress_interval,
            sample_lines=None,  # sample cap already applied in the exact pass above; near-dup reads its full (already-capped) output
        )
        if not keep_intermediate:
            for p in exact_output_paths.values():
                if os.path.exists(p):
                    os.remove(p)

    validate_dedup_accounting(exact_stats, near_stats, near_enabled, logger)

    input_lines = exact_stats["input_lines"]
    exact_removed = exact_stats["duplicates"]
    near_removed = near_stats.get("near_duplicates_removed", 0)
    sub_doc_removed = near_stats.get("sub_document_duplicates_removed", 0)
    near_and_sub_total = near_removed + sub_doc_removed
    total_removed = exact_removed + near_and_sub_total

    per_source_report = {}
    final_lines_total = 0
    estimated_words_total = 0
    for label, _ in sources_in_precedence_order:
        out_path = output_paths_by_label[label]
        final_lines = count_lines(out_path) if os.path.exists(out_path) else 0
        words = 0
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    words += len(line.split())
        final_lines_total += final_lines
        estimated_words_total += words

        ex = exact_stats.get("by_source", {}).get(
            label, {"input_lines": 0, "duplicates": 0, "uniques": 0, "cross_source_duplicates": 0}
        )
        nr = near_stats.get("by_source", {}).get(
            label, {"near_duplicates_removed": 0, "sub_document_duplicates_removed": 0}
        )
        per_source_report[label] = {
            "input_file": input_path_by_label[label],
            "output_file": out_path,
            "input_lines": ex["input_lines"],
            "exact_duplicates_removed": ex["duplicates"],
            "exact_duplicates_removed_cross_source": ex.get("cross_source_duplicates", 0),
            "near_duplicates_removed": nr.get("near_duplicates_removed", 0),
            "sub_document_duplicates_removed": nr.get("sub_document_duplicates_removed", 0),
            "final_lines": final_lines,
        }

    removal_percentage = (total_removed / input_lines * 100) if input_lines > 0 else 0.0
    total_elapsed = time.time() - run_t0
    overall_speed = input_lines / total_elapsed if total_elapsed > 0 else 0

    report = {
        "input_lines": input_lines,
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_and_sub_total,
        "near_duplicate_breakdown": {
            "near_duplicate": near_removed,
            "sub_document_duplicate": sub_doc_removed,
        },
        "total_removed": total_removed,
        "final_lines": final_lines_total,
        "estimated_words": estimated_words_total,
        "removal_percentage": round(removal_percentage, 4),
        "algorithm": {
            "exact": {"method": "sha256", "enabled": exact_enabled},
            "near_duplicate": {**near_cfg, "enabled": near_enabled, "index": "in_memory (Mac)"},
        },
        "input_file": input_path_by_label["indiccorp"],
        "output_file": output_paths_by_label["indiccorp"],
        "cross_source": True,
        "source_precedence": list(SOURCE_PRECEDENCE_ORDER),
        "exact_duplicates_removed_cross_source": exact_stats.get("cross_source_duplicates", 0),
        "sources": per_source_report,
        "benchmark": {
            **near_stats.get("benchmark", {}),
            "sample_lines_cap": args.sample_lines,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "overall_lines_per_second": round(overall_speed, 2),
        },
    }

    meta_dir = str(repo_root / config["paths"]["metadata"])
    os.makedirs(meta_dir, exist_ok=True)
    report_name = "deduplication_report_mac_benchmark.json" if args.sample_lines else "deduplication_report_mac.json"
    report_path = os.path.join(meta_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("CROSS-SOURCE DEDUPLICATION SUMMARY (Mac)")
    logger.info("=" * 70)
    logger.info(f"  input lines (all sources):        {input_lines:,}")
    logger.info(
        f"  exact duplicates removed:         {exact_removed:,}  "
        f"(cross-source: {exact_stats.get('cross_source_duplicates', 0):,})"
    )
    logger.info(f"  near duplicates removed:          {near_removed:,}")
    logger.info(f"  sub-document duplicates removed:  {sub_doc_removed:,}")
    logger.info(f"  total removed:                    {total_removed:,}")
    logger.info(f"  final lines (all sources):        {final_lines_total:,}")
    for label in SOURCE_PRECEDENCE_ORDER:
        s = per_source_report.get(label, {})
        logger.info(f"    [{label}] input={s.get('input_lines', 0):,} -> final={s.get('final_lines', 0):,}")
    logger.info(f"  estimated words:                  {estimated_words_total:,}")
    logger.info(f"  removal percentage:               {removal_percentage:.2f}%")
    logger.info(f"  total elapsed:                     {total_elapsed:.1f}s ({overall_speed:,.0f} lines/s)")
    mem = peak_memory_mb()
    if mem is not None:
        logger.info(f"  peak RSS:                          {mem:,.1f} MB")
    logger.info(f"Deduplication report saved to {report_path}")

    # Disk-saving cleanup: quality/ is fully redundant once
    # deduplicated/ exists (deduplicated/ is the source-wise cleaned
    # corpus this project keeps permanently — see shared/run_stats.py).
    # NEVER runs for a --sample-lines benchmark run — that's a test
    # slice of the corpus, not a real consumption of quality/'s data.
    # Only cleans up a given source's quality/ files (main output +
    # rejected file) when THAT source used its default (config-resolved)
    # input path — a CLI --{source}-input override means the caller
    # pointed this run at something else, so that source's real
    # quality/ files (if any) are left untouched.
    if not args.sample_lines:
        quality_cfg = config.get("quality", {})
        for name in SOURCE_PRECEDENCE_ORDER:
            if name not in existing_inputs:
                continue
            cli_input = getattr(args, f"{name}_input", None)
            if cli_input:
                continue
            rejected_filename = quality_cfg.get(
                _QUALITY_REJECTED_FILENAME_CONFIG_KEY.get(name, ""), f"{name}_ne_quality_rejected.txt"
            )
            rejected_path = os.path.join(os.path.dirname(input_path_by_label[name]), rejected_filename)
            cleanup_prior_stage_files(
                config, repo_root, stage_to_delete="quality",
                files=[input_path_by_label[name], rejected_path], logger=logger,
            )


if __name__ == "__main__":
    main()
