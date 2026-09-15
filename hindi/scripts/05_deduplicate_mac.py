#!/usr/bin/env python3
"""
05_deduplicate_mac.py — Mac-Optimized Cross-Source Exact + Near-Duplicate +
Sub-Document Deduplication (IndicCorp + Wikipedia + News)

SEPARATE FROM hindi/scripts/05_deduplicate.py. This file exists purely as a
faster alternative for a machine with abundant RAM (developed against
Apple M4 / 24 GB) — it does NOT replace, modify, or get imported by
05_deduplicate.py, and 05_deduplicate.py is unaffected by anything in this
file. Use 05_deduplicate.py on constrained (~8 GB) machines; use this
script on the Mac.

-----------------------------------------------------------------------------
THREE SOURCES, NOT TWO
-----------------------------------------------------------------------------
Same source model as 05_deduplicate.py (see that script's module docstring
for the full rationale — reproduced here, not altered):

    wikipedia   — hindi/data/intermediate/quality/wikipedia_hi_quality.txt
    news        — hindi/data/intermediate/quality/news_hi_quality.txt
                  (already a PRE-MERGED aggregate of every ENABLED news
                  outlet, combined into ONE file by
                  manual/05_aggregate_news_corpus.py, upstream of this script —
                  NOT multiple separate dedup input streams)
    indiccorp   — hindi/data/intermediate/quality/indiccorp_hi_quality.txt

Individual news outlets are NEVER a separate top-level source here — their
aggregation is entirely manual/05_aggregate_news_corpus.py's concern, upstream
of both dedup scripts.

METHODOLOGY — IDENTICAL to 05_deduplicate.py, only the near-duplicate INDEX
STORAGE differs:
  1. EXACT duplicate detection — SHA-256 line hash, shared across all
     three sources (deduplicate_exact_cross_source() below is a verbatim
     copy of 05_deduplicate.py's function of the same name — not
     reimplemented, not altered).
  2. NEAR-DUPLICATE / SUB-DOCUMENT detection — word shingles -> MinHash
     signature -> LSH banding for candidate generation -> Jaccard estimate
     (near-duplicate) / overlap-containment estimate (sub-document
     duplicate) for confirmation. Uses the EXACT SAME primitives as
     05_deduplicate.py: shared/near_dup.py's MinHasher,
     signature_and_shingle_count, lsh_band_keys, jaccard_estimate,
     overlap_estimate — imported unchanged, never reimplemented here.

THE ONLY ALGORITHMIC-INFRASTRUCTURE CHANGE: the near-duplicate pass uses
shared/near_dup_mac.py's InMemoryMinHashDedupIndex (a plain-dict-backed LSH
index) instead of shared/near_dup.py's SQLite-backed MinHashDedupIndex.
This is the SAME shared/near_dup_mac.py module already created and
validated for the Nepali Mac pipeline (nepali/scripts/05_deduplicate_mac.py)
— NOT a new/second in-memory-index implementation. Candidate-generation and
confirmation semantics (Jaccard, overlap/containment, source precedence,
sub-document detection) are unchanged.

Cross-source deduplication and source precedence (Wikipedia > News >
IndicCorp) are identical to 05_deduplicate.py — see that script's module
docstring for the full SOURCE_PRECEDENCE_ORDER rationale, which applies
unchanged here.

PARAMETERS: this script does NOT introduce new Mac-tuned near-duplicate
parameters the way the Nepali Mac script did. Hindi's existing
`deduplication.near_duplicate` block (hindi_data.yaml) already reflects a
prior deliberate 8-GB-RAM tuning pass (8 bands x 4 rows, threshold 0.40,
32 permutations, 3-word shingles) and changing those numbers would change
DEDUPLICATION QUALITY, not just speed — out of scope for a storage-layer
optimization. This script reads the SAME `deduplication` config block
05_deduplicate.py uses by default, and optionally reads a
`deduplication_mac` override block instead if hindi_data.yaml defines one
(useful later if the Mac's extra RAM ever justifies e.g. more permutations
for higher accuracy — not done here since no such requirement was given).

Compatibility with the rest of the Hindi pipeline (downstream stages
unmodified):
  * Same input filenames as 05_deduplicate.py: quality.<source>_output_dir
    / quality.<source>_output_filename, defaulting to
    `{source}_hi_quality.txt` in quality.output_dir.
  * Same output filenames as 05_deduplicate.py: paths.deduplicated/
    `{source}_hi_deduplicated.txt` for indiccorp, wikipedia, and news —
    never a merged manual file. Downstream scripts (08/09/11/10/15) read
    these exact three filenames regardless of which dedup script produced
    them.
  * Same CLI surface: --config, --indiccorp-input/--output,
    --wikipedia-input/--output, --news-input/--output, the generic
    repeatable --input-override/--output-override SOURCE=PATH, --resume,
    --overwrite, --in-memory (exact pass), --skip-near-dup.
  * Mac-only addition: --sample-lines N (cap each source to at most N
    lines, for benchmarking on a 500K-1M-line sample before running the
    full corpus).

Usage:
    python hindi/scripts/05_deduplicate_mac.py --config hindi/configs/hindi_data.yaml

    # Benchmark on a 1M-line-per-source sample first:
    python hindi/scripts/05_deduplicate_mac.py --config hindi/configs/hindi_data.yaml \\
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
from shared.tokenizer.data_utils import iter_lines, count_lines
from shared.tokenizer.near_dup import (
    MinHasher,
    jaccard_estimate,
    lsh_band_keys,
    overlap_estimate,
    signature_and_shingle_count,
)
# THE MAC OPTIMIZATION: same shared/near_dup_mac.py module already created
# and validated for Nepali — NOT a new/duplicate in-memory index.
from shared.tokenizer.near_dup_mac import InMemoryMinHashDedupIndex
# Same accounting-reconciliation check the Linux script uses — reused, not
# reimplemented, so there is one validation implementation shared by both.
from shared.tokenizer.dedup_accounting import validate_dedup_accounting
from shared.tokenizer.run_stats import cleanup_prior_stage_files

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None


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
    unavailable — never raises."""
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
    return usage / divisor


def capped_lines(path: str, limit: Optional[int]) -> Iterator[str]:
    """Like shared.tokenizer.data_utils.iter_lines(path), but stops after *limit*
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
# not this script's optimization target, so it is reproduced unchanged
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
# THE MAC OPTIMIZATION. Uses InMemoryMinHashDedupIndex (shared/near_dup_mac.py,
# same module used by the Nepali Mac script) instead of MinHashDedupIndex
# (shared/near_dup.py, SQLite-backed). All candidate-generation and
# confirmation logic (LSH banding, Jaccard estimate, overlap/containment
# estimate, sub-document detection, source precedence) is identical to
# 05_deduplicate.py's deduplicate_near_cross_source() — including its
# per-source candidate-generation statistics (existing Hindi-specific
# reporting, not a new addition) — only the index storage call sites
# (index.find_candidates / index.add) changed target class.
# ---------------------------------------------------------------------------

def deduplicate_near_cross_source_mac(
    sources,
    output_paths: Dict[str, str],
    near_cfg: Dict[str, Any],
    logger: logging.Logger,
    progress_interval: int,
    sample_lines: Optional[int] = None,
) -> Dict[str, Any]:
    shingle_type = near_cfg.get("shingle_type", "word")
    shingle_size = near_cfg.get("shingle_size", 3)
    num_perm = near_cfg.get("num_minhash_permutations", 32)
    bands = near_cfg.get("lsh_bands", 8)
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

    def _new_source_stats():
        return {
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
            "eligible_documents": 0,
            "candidates_generated": 0,
            "documents_with_candidates": 0,
            "comparisons_performed": 0,
            "max_candidates_single_doc": 0,
        }

    stats = {label: _new_source_stats() for label, _ in sources}

    next_line_id = 0
    total = 0
    t0 = time.time()

    try:
        for label, path in sources:
            source_start_time = time.time()
            logger.info("=" * 60)
            logger.info(f"NEAR-DUPLICATE SOURCE STARTED: {label}")
            logger.info(f"Input: {path}")
            logger.info("=" * 60)

            if not path or not os.path.exists(path):
                logger.warning(f"[{label}] near-dup input not found — treating as an empty source: {path}")
                logger.info("=" * 60)
                logger.info(f"NEAR-DUPLICATE SOURCE FINISHED: {label}")
                logger.info(f"Input lines: 0")
                logger.info("=" * 60)
                continue

            out_f = out_files.get(label)
            s = stats[label]
            for line in capped_lines(path, sample_lines):
                total += 1
                s["input_lines"] += 1
                if not line:
                    s["blank_lines_skipped"] += 1
                    continue

                word_count = len(line.split())
                if word_count < min_document_words:
                    out_f.write(line + "\n")
                    s["kept"] += 1
                    s["kept_words"] += word_count
                    s["short_lines_kept"] += 1
                    continue

                s["eligible_documents"] += 1
                sig, shingle_count = signature_and_shingle_count(
                    line, minhasher, shingle_size, shingle_type
                )
                band_keys = lsh_band_keys(sig, bands, rows)
                candidates = index.find_candidates(band_keys)
                n_candidates = len(candidates)
                s["candidates_generated"] += n_candidates
                if n_candidates > 0:
                    s["documents_with_candidates"] += 1
                if n_candidates > s["max_candidates_single_doc"]:
                    s["max_candidates_single_doc"] = n_candidates

                verdict = None
                for _cid, csig, ccount in candidates:
                    s["comparisons_performed"] += 1
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
                        s["sub_document_duplicates_removed"] += 1
                    else:
                        s["near_duplicates_removed"] += 1
                else:
                    out_f.write(line + "\n")
                    s["kept"] += 1
                    s["kept_words"] += word_count
                    index.add(next_line_id, sig, shingle_count, band_keys)
                    next_line_id += 1

                if total % progress_interval == 0:
                    elapsed = time.time() - t0
                    speed = total / elapsed if elapsed > 0 else 0
                    avg_cand = s["candidates_generated"] / s["eligible_documents"] if s["eligible_documents"] else 0.0
                    mem = peak_memory_mb()
                    mem_str = f"{mem:,.0f} MB" if mem is not None else "n/a"
                    logger.info(
                        f"  [{label}] {total:,} lines scanned (all sources so far) | "
                        f"this source: scanned {s['input_lines']:,}, kept {s['kept']:,}, "
                        f"near-dup removed {s['near_duplicates_removed']:,}, "
                        f"sub-doc removed {s['sub_document_duplicates_removed']:,} | "
                        f"candidates {s['candidates_generated']:,} (avg {avg_cand:.2f}/doc, "
                        f"max {s['max_candidates_single_doc']:,}) | "
                        f"comparisons {s['comparisons_performed']:,} | "
                        f"{speed:,.0f} lines/s | peak RSS={mem_str}"
                    )

            source_elapsed = time.time() - source_start_time
            logger.info("=" * 60)
            logger.info(f"NEAR-DUPLICATE SOURCE FINISHED: {label}")
            logger.info(f"Input lines: {s['input_lines']:,}")
            logger.info(f"Kept: {s['kept']:,}")
            logger.info(f"Near duplicates removed: {s['near_duplicates_removed']:,}")
            logger.info(f"Sub-document duplicates removed: {s['sub_document_duplicates_removed']:,}")
            logger.info(
                f"Candidates generated: {s['candidates_generated']:,} | "
                f"comparisons performed: {s['comparisons_performed']:,} | "
                f"elapsed: {source_elapsed:.1f}s"
            )
            logger.info("=" * 60)
    finally:
        for f in out_files.values():
            f.close()

    elapsed_total = time.time() - t0
    combined_candidates = sum(s["candidates_generated"] for s in stats.values())
    combined_eligible = sum(s["eligible_documents"] for s in stats.values())
    combined = {
        "input_lines": sum(s["input_lines"] for s in stats.values()),
        "near_duplicates_removed": sum(s["near_duplicates_removed"] for s in stats.values()),
        "sub_document_duplicates_removed": sum(s["sub_document_duplicates_removed"] for s in stats.values()),
        "kept": sum(s["kept"] for s in stats.values()),
        "kept_words": sum(s["kept_words"] for s in stats.values()),
        "short_lines_kept": sum(s["short_lines_kept"] for s in stats.values()),
        "blank_lines_skipped": sum(s["blank_lines_skipped"] for s in stats.values()),
        "by_source": stats,
        "candidate_stats": {
            "total_scanned": sum(s["input_lines"] for s in stats.values()),
            "eligible_documents": combined_eligible,
            "short_lines_skipped": sum(s["short_lines_kept"] for s in stats.values()),
            "candidates_generated": combined_candidates,
            "documents_with_candidates": sum(s["documents_with_candidates"] for s in stats.values()),
            "comparisons_performed": sum(s["comparisons_performed"] for s in stats.values()),
            "max_candidates_single_doc": max((s["max_candidates_single_doc"] for s in stats.values()), default=0),
            "avg_candidates_per_eligible_doc": round(combined_candidates / combined_eligible, 4) if combined_eligible else 0.0,
            "throughput_lines_per_sec": round(sum(s["input_lines"] for s in stats.values()) / elapsed_total, 2) if elapsed_total > 0 else 0.0,
            "elapsed_seconds": round(elapsed_total, 2),
            "index_memory_report": index.memory_report(),
            "peak_rss_mb": peak_memory_mb(),
            "by_source": {
                label: {
                    "eligible_documents": s["eligible_documents"],
                    "candidates_generated": s["candidates_generated"],
                    "documents_with_candidates": s["documents_with_candidates"],
                    "comparisons_performed": s["comparisons_performed"],
                    "max_candidates_single_doc": s["max_candidates_single_doc"],
                    "avg_candidates_per_eligible_doc": (
                        round(s["candidates_generated"] / s["eligible_documents"], 4) if s["eligible_documents"] else 0.0
                    ),
                }
                for label, s in stats.items()
            },
        },
    }

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"NEAR DUPLICATES REMOVED (all sources): {combined['near_duplicates_removed']:,}")
    logger.info(f"SUB-DOCUMENT DUPLICATES REMOVED (all sources): {combined['sub_document_duplicates_removed']:,}")
    mem = peak_memory_mb()
    if mem is not None:
        logger.info(f"PEAK RSS: {mem:,.1f} MB")
    logger.info(
        f"CROSS-SOURCE NEAR-DUPLICATE DEDUPLICATION FINISHED: {combined['input_lines']:,} lines scanned, "
        f"{combined['kept']:,} kept in {elapsed:.1f}s"
    )
    logger.info("=" * 70)

    return combined


# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

# PRECEDENCE ORDER (highest first) — see 05_deduplicate.py's module
# docstring "Cross-Source Deduplication & Source Precedence" for the
# rationale (identical here). THREE top-level corpus sources:
# "wikipedia" (single curated encyclopedic dump), "news" (a PRE-MERGED
# aggregate of every enabled news outlet, produced upstream by
# manual/05_aggregate_news_corpus.py), and "indiccorp" (public). Individual
# news outlets are NEVER a separate source here — their aggregation is
# entirely manual/05_aggregate_news_corpus.py's concern, upstream of this
# script.
SOURCE_PRECEDENCE_ORDER = ("wikipedia", "news", "indiccorp")

PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("wikipedia", "news")

DEFAULT_QUALITY_FILENAME = {name: f"{name}_hi_quality.txt" for name in SOURCE_PRECEDENCE_ORDER}
DEFAULT_DEDUP_FILENAME = {name: f"{name}_hi_deduplicated.txt" for name in SOURCE_PRECEDENCE_ORDER}

# Mirrors 04_quality_filter.py's _REJECTED_FILENAME_CONFIG_KEY — needed here
# ONLY to compute the path of that stage's rejected-lines file for this
# source, so post-dedup cleanup can delete it alongside
# {source}_hi_quality.txt (same lifecycle — rejected files are not
# preserved independently of their stage's main output).
_QUALITY_REJECTED_FILENAME_CONFIG_KEY = {
    "indiccorp": "rejected_filename",
    "wikipedia": "wikipedia_rejected_filename",
    "news": "news_rejected_filename",
}


def parse_source_overrides(pairs, valid_sources, flag_name: str, logger: logging.Logger) -> Dict[str, str]:
    """Parse repeated `--input-override SOURCE=PATH` / `--output-override
    SOURCE=PATH` CLI arguments into {source: path} — see 05_deduplicate.py's
    identical helper for the rationale (generic override instead of one
    flag per news source)."""
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


def resolve_source_paths(
    config: Dict[str, Any],
    repo_root: Path,
    args,
    input_overrides: Dict[str, str],
    output_overrides: Dict[str, str],
) -> tuple:
    """Resolve (input_path, output_path) for every source in
    SOURCE_PRECEDENCE_ORDER ("wikipedia", "news", "indiccorp") — see
    05_deduplicate.py's identical function for the full docstring;
    behavior here is the same, config-driven with CLI-flag overrides
    (--indiccorp-input/--wikipedia-input/--news-input plus the generic
    --input-override/--output-override)."""
    quality_cfg = config.get("quality", {})
    shared_quality_dir = quality_cfg.get("output_dir", "hindi/data/intermediate/quality")
    dedup_dir = config["paths"]["deduplicated"]

    input_path_by_label: Dict[str, str] = {}
    output_path_by_label: Dict[str, str] = {}

    for name in SOURCE_PRECEDENCE_ORDER:
        cli_input = input_overrides.get(name) or getattr(args, f"{name}_input", None)
        cli_output = output_overrides.get(name) or getattr(args, f"{name}_output", None)

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
                     "IndicCorp, Wikipedia, and News (News is itself a "
                     "pre-merged aggregate of all enabled news outlets). "
                     "Separate from 05_deduplicate.py (Linux/8GB) — same "
                     "methodology, output filenames, and near-duplicate "
                     "parameters by default; RAM-backed index instead of "
                     "SQLite."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--indiccorp-input", "--input", dest="indiccorp_input", type=str,
                         help="Override IndicCorp input file path")
    parser.add_argument("--wikipedia-input", dest="wikipedia_input", type=str,
                         help="Override Wikipedia input file path")
    parser.add_argument("--news-input", dest="news_input", type=str,
                         help="Override News (pre-merged, all enabled outlets) input file path")
    parser.add_argument("--indiccorp-output", "--output", dest="indiccorp_output", type=str,
                         help="Override IndicCorp output file path")
    parser.add_argument("--wikipedia-output", dest="wikipedia_output", type=str,
                         help="Override Wikipedia output file path")
    parser.add_argument("--news-output", dest="news_output", type=str,
                         help="Override News output file path")
    parser.add_argument("--input-override", action="append", default=[], metavar="SOURCE=PATH",
                         help="Override the quality-stage input path for 'wikipedia', 'news', "
                              "or 'indiccorp' (e.g. --input-override news=/path/to/file.txt). "
                              "Equivalent to --wikipedia-input/--news-input/--indiccorp-input; "
                              "provided for symmetry/scripting.")
    parser.add_argument("--output-override", action="append", default=[], metavar="SOURCE=PATH",
                         help="Override the deduplicated-stage output path for 'wikipedia', "
                              "'news', or 'indiccorp'. Repeatable, same SOURCE=PATH syntax as "
                              "--input-override.")
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

    progress_interval = config.get("logging", {}).get("progress_interval", 100000)
    # Reuses the SAME `deduplication` block 05_deduplicate.py reads by
    # default — preserves current Hindi deduplication quality (8x4 bands/
    # rows, threshold 0.40, 32 permutations, 3-word shingles). Only reads
    # `deduplication_mac` instead if hindi_data.yaml explicitly defines
    # that override block.
    dedup_cfg = config.get("deduplication_mac", config.get("deduplication", {}))
    exact_enabled = dedup_cfg.get("exact", {}).get("enabled", True)
    near_cfg = dedup_cfg.get("near_duplicate", {})
    near_enabled = near_cfg.get("enabled", True) and not args.skip_near_dup
    keep_intermediate = config.get("storage", {}).get("keep_intermediate_files", True)

    repo_root = Path(args.config).resolve().parent.parent.parent

    input_overrides = parse_source_overrides(args.input_override, SOURCE_PRECEDENCE_ORDER, "--input-override", logger)
    output_overrides = parse_source_overrides(args.output_override, SOURCE_PRECEDENCE_ORDER, "--output-override", logger)
    input_path_by_label, output_paths_by_label = resolve_source_paths(
        config, repo_root, args, input_overrides, output_overrides
    )

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

    # SOURCE PRECEDENCE: Wikipedia > News > IndicCorp, same as 05_deduplicate.py.
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
                os.path.dirname(output_paths_by_label[name]), f"{name}_hi_exact_deduplicated.txt"
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

    near_stats = {"near_duplicates_removed": 0, "sub_document_duplicates_removed": 0, "by_source": {}, "candidate_stats": {}}
    if near_enabled:
        near_input_sources = [(label, exact_output_paths.get(label)) for label, _ in sources_in_precedence_order]
        near_stats = deduplicate_near_cross_source_mac(
            near_input_sources, output_paths_by_label, near_cfg, logger, progress_interval,
            sample_lines=None,  # sample cap already applied in the exact pass; near-dup reads its (already-capped) output
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
        "input_file": input_path_by_label[PUBLIC_SOURCE],
        "output_file": output_paths_by_label[PUBLIC_SOURCE],
        "cross_source": True,
        "source_precedence": list(SOURCE_PRECEDENCE_ORDER),
        "source_precedence_rationale": (
            "Wikipedia > News > IndicCorp. Manual sources (wikipedia/news) "
            "are deduplicated BEFORE IndicCorp (public); when identical/"
            "near-identical text exists in both a manual source and "
            "IndicCorp, the manual copy survives, protecting the project's "
            ">=20% manual-content-token requirement (config "
            "manual.min_fraction / manual.enforce). Within manual sources: "
            "Wikipedia before News because Wikipedia is a single curated, "
            "already-deduplicated-by-construction source, while News "
            "aggregates up to 10 outlets that plausibly syndicate "
            "overlapping content — see 05_deduplicate.py's module "
            "docstring for the full rationale, including the alternative "
            "ordering considered and rejected."
        ),
        "exact_duplicates_removed_cross_source": exact_stats.get("cross_source_duplicates", 0),
        "sources": per_source_report,
        "candidate_stats": near_stats.get("candidate_stats", {}),
        "benchmark": {
            "sample_lines_cap": args.sample_lines,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "overall_lines_per_second": round(input_lines / total_elapsed, 2) if total_elapsed > 0 else 0,
        },

        # ---- Manual GROUPING (wikipedia + news combined) — accounting/
        # target-fraction convenience only (config manual.min_fraction /
        # manual.enforce); NOT a separate logical/top-level source and NOT
        # a merged file — derived here purely by summing the two manual
        # sources' own per-source report entries above. ----
        "manual_totals": {
            "sources": list(MANUAL_SOURCES),
            "final_lines": sum(per_source_report.get(name, {}).get("final_lines", 0) for name in MANUAL_SOURCES),
            "input_lines": sum(per_source_report.get(name, {}).get("input_lines", 0) for name in MANUAL_SOURCES),
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
    logger.info(f"  total elapsed:                     {total_elapsed:.1f}s")
    mem = peak_memory_mb()
    if mem is not None:
        logger.info(f"  peak RSS:                          {mem:,.1f} MB")

    cand_stats = near_stats.get("candidate_stats")
    if cand_stats:
        logger.info("")
        logger.info("  --- LSH candidate-generation statistics (near-dup pass) ---")
        logger.info(f"  total documents scanned:          {cand_stats['total_scanned']:,}")
        logger.info(f"  eligible for near-dedup:           {cand_stats['eligible_documents']:,}")
        logger.info(f"  skipped (< min_document_words):    {cand_stats['short_lines_skipped']:,}")
        logger.info(f"  total LSH candidate pairs:          {cand_stats['candidates_generated']:,}")
        logger.info(f"  documents with >=1 candidate:       {cand_stats['documents_with_candidates']:,}")
        logger.info(f"  Jaccard/overlap comparisons:        {cand_stats['comparisons_performed']:,}")
        logger.info(f"  max candidates for a single doc:    {cand_stats['max_candidates_single_doc']:,}")
        logger.info(f"  avg candidates per eligible doc:    {cand_stats['avg_candidates_per_eligible_doc']:.4f}")
        logger.info(f"  throughput:                         {cand_stats['throughput_lines_per_sec']:,.0f} lines/s")
        for label in SOURCE_PRECEDENCE_ORDER:
            by_src = cand_stats.get("by_source", {}).get(label)
            if by_src:
                logger.info(
                    f"    [{label}] eligible={by_src['eligible_documents']:,} "
                    f"candidates={by_src['candidates_generated']:,} "
                    f"(avg {by_src['avg_candidates_per_eligible_doc']:.2f}/doc, "
                    f"max {by_src['max_candidates_single_doc']:,}) "
                    f"comparisons={by_src['comparisons_performed']:,}"
                )
    manual_totals = report["manual_totals"]
    logger.info(
        f"  manual (wikipedia+news) final lines: {manual_totals['final_lines']:,} "
        f"(from input {manual_totals['input_lines']:,})"
    )

    logger.info(f"Deduplication report saved to {report_path}")

    # Disk-saving cleanup: quality/ is fully redundant once
    # deduplicated/ exists (deduplicated/ is the source-wise cleaned
    # corpus this project keeps permanently — see shared/run_stats.py).
    # Only cleans up a given source's quality/ files (main output +
    # rejected file) when THAT source used its default (config-resolved)
    # input path — a CLI --{source}-input override means the caller
    # pointed this run at something else, so that source's real
    # quality/ files (if any) are left untouched.
    quality_cfg = config.get("quality", {})
    for name in SOURCE_PRECEDENCE_ORDER:
        if name not in existing_inputs:
            continue
        cli_input = getattr(args, f"{name}_input", None)
        if cli_input:
            continue
        rejected_filename = quality_cfg.get(
            _QUALITY_REJECTED_FILENAME_CONFIG_KEY.get(name, ""), f"{name}_hi_quality_rejected.txt"
        )
        rejected_path = os.path.join(os.path.dirname(input_path_by_label[name]), rejected_filename)
        cleanup_prior_stage_files(
            config, repo_root, stage_to_delete="quality",
            files=[input_path_by_label[name], rejected_path], logger=logger,
        )


if __name__ == "__main__":
    main()
