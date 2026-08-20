#!/usr/bin/env python3
"""
05_deduplicate.py — Cross-Source Exact + Near-Duplicate + Sub-Document
Deduplication (IndicCorp + Onlinekhabar + Khabarhub, together)

Implements the "Deduplication" procedure from CL3410 Lecture 3:

  1. EXACT duplicate detection — SHA-256 line hash, SQLite- or
     memory-backed. See deduplicate_in_memory / deduplicate_sqlite
     (single-source, unchanged) and deduplicate_exact_cross_source
     (multi-source — see below).

  2. NEAR-DUPLICATE and SUB-DOCUMENT-duplicate detection — on the
     exact-deduplicated output:
       a. represent each line as its set of overlapping 3-WORD shingles
       b. summarize each shingle set with a MinHash signature
       c. use LSH banding for CANDIDATE GENERATION so an incoming line is
          only ever compared against previously-kept lines sharing a band
          — never the whole corpus (no O(N^2) document-to-document
          comparison)
       d. CONFIRM each candidate with an actual similarity estimate before
          removing anything:
            - Jaccard estimate >= similarity_threshold        -> NEAR DUPLICATE
            - overlap/containment estimate >=
              sub_document_overlap_threshold (for a pair whose
              shingle-set sizes differ substantially)          -> SUB-DOCUMENT DUPLICATE
     See deduplicate_near (single-source, unchanged) and
     deduplicate_near_cross_source (multi-source — see below).

-----------------------------------------------------------------------------
THREE SOURCES, NOT TWO
-----------------------------------------------------------------------------
The Nepali corpus has exactly TWO top-level sources — indiccorp and manual
— but manual itself is now TWO subcategories: Onlinekhabar (news) and
Khabarhub (news) (see nepali/scripts/manual/collect_news.py). This script
therefore deduplicates THREE labeled input streams together, not two:

    indiccorp      — nepali/data/intermediate/quality/indiccorp_ne_quality.txt
    onlinekhabar   — nepali/data/intermediate/quality/onlinekhabar_ne_quality.txt
    khabarhub      — nepali/data/intermediate/quality/khabarhub_ne_quality.txt

(each produced by 05_quality_filter.py for its own source). Devanagari,
ZWNJ, ZWJ, punctuation, and whitespace are exactly as produced by script
03; this stage only reads lines to hash/shingle them — kept lines are
written through byte-for-byte unchanged.

SOURCES is a single ordered list resolved once in main() (see
`SOURCE_PRECEDENCE_ORDER` / `resolve_source_paths()`) — everything below
main(), including deduplicate_exact_cross_source() and
deduplicate_near_cross_source(), was ALREADY generic over an arbitrary
list of `(label, path)` pairs (this is not new code, and it is not
touched here beyond this docstring); the two-source assumption lived
entirely in main()'s path resolution, CLI argument list, and logging,
which is what this revision generalizes. Adding a fourth source in the
future means extending SOURCE_PRECEDENCE_ORDER, not touching the
dedup/candidate-generation algorithm.

-----------------------------------------------------------------------------
CROSS-SOURCE DEDUPLICATION & SOURCE PRECEDENCE
-----------------------------------------------------------------------------
Deduplication MUST see all three sources TOGETHER, in ONE pass, so a
duplicate that spans sources is actually caught — running each source
through its own independent deduplication run would never notice that the
SAME (or near-identical) line appears in more than one, since each run
only ever compares a source against itself. Concretely:

  * The EXACT pass (deduplicate_exact_cross_source) shares ONE hash index
    across all three sources' input streams.
  * The NEAR-DUPLICATE / SUB-DOCUMENT pass (deduplicate_near_cross_source)
    shares ONE MinHash/LSH index (shared/near_dup.py's MinHashDedupIndex,
    reused unmodified) across all three sources' exact-deduplicated
    streams.

Both passes process sources in a fixed PRECEDENCE ORDER —

    Onlinekhabar > Khabarhub > IndicCorp

— and keep the FIRST occurrence of a given (near-)duplicate hash/
signature, dropping later occurrences regardless of which source they
came from. Concretely: if the same (or near-identical) text exists in
Onlinekhabar and IndicCorp, the ONLINEKHABAR copy survives; if it exists
in Onlinekhabar and Khabarhub, the ONLINEKHABAR copy survives; etc.

RATIONALE (deterministic, not a silent/arbitrary choice): this project
already has an explicit source-priority signal — the manual-content-token
floor (`manual.min_fraction` / `manual.enforce` in nepali_data.yaml, which
Onlinekhabar/Khabarhub both count toward as manual sources). Manual data
is scarce and expensive to collect relative to IndicCorp's much larger
public pool, so when the same text exists in both a manual source and
IndicCorp, keeping the manual copy protects that manual-fraction
requirement instead of eroding it. Within the two manual sources,
Onlinekhabar is ranked before Khabarhub simply by planning-target size
(see nepali_data.yaml's `manual.news` block) as the same kind of explicit,
documented tie-break this project already uses elsewhere, rather than an
arbitrary order.

Every surviving line's source is preserved: kept lines are written to
SOURCE-SPECIFIC output files (`indiccorp_ne_deduplicated.txt`,
`onlinekhabar_ne_deduplicated.txt`, `khabarhub_ne_deduplicated.txt`) —
never merged into one file at this stage — and the deduplication report
records per-source input/removed/survivor counts alongside the pre-
existing combined/backward-compatible top-level keys (see main()).

Memory guarantees (hundreds of millions of lines):
  • Exact pass: SQLite-backed unique-hash index, O(1) memory (default), or
    a bounded in-memory dict (--in-memory) — same SHA-256 algorithm as the
    single-source pass, just shared across all three sources' streams.
  • Near-duplicate pass: only compact MinHash signatures + shingle counts
    for previously-KEPT lines (from ANY source) are persisted (SQLite,
    shared/near_dup.py's MinHashDedupIndex, used exactly as in the
    single-source pass) — never the corpus text itself. Duplicates are
    never indexed, which keeps large duplicate clusters from growing the
    index or candidate-set size unboundedly. Each source is streamed once.
  • No O(N^2) comparison in either pass: LSH banding still means an
    incoming line — from any source — is only ever compared against
    previously-kept lines sharing an LSH band, never the whole corpus.
  • This remains usable on an 8 GB RAM machine: nothing above holds full
    corpus text in memory (both passes stream line-by-line); the only
    process-memory growth with corpus size is the bounded --in-memory
    dict option (off by default) and shared/near_dup.py's SQLite page
    cache, which is capped (see that module's MinHashDedupIndex pragmas),
    not corpus-proportional.

Usage:
    python nepali/scripts/05_deduplicate.py --config nepali/configs/nepali_data.yaml

    # Explicit per-source overrides (defaults shown above are normally enough):
    python nepali/scripts/05_deduplicate.py --config nepali/configs/nepali_data.yaml \\
        --indiccorp-input nepali/data/intermediate/quality/indiccorp_ne_quality.txt \\
        --onlinekhabar-input nepali/data/intermediate/quality/onlinekhabar_ne_quality.txt \\
        --khabarhub-input nepali/data/intermediate/quality/khabarhub_ne_quality.txt \\
        --indiccorp-output nepali/data/intermediate/deduplicated/indiccorp_ne_deduplicated.txt \\
        --onlinekhabar-output nepali/data/intermediate/deduplicated/onlinekhabar_ne_deduplicated.txt \\
        --khabarhub-output nepali/data/intermediate/deduplicated/khabarhub_ne_deduplicated.txt

If a given source's input file does not exist (e.g. it has not been
collected/quality-filtered yet), that source is treated as an empty
source for this run — deduplication degrades gracefully to whichever
sources ARE present (a warning is logged per missing source); this keeps
the script usable at every stage of standing each source's pipeline up,
without ever performing three independent single-source runs. At least
ONE source's input must exist or the script fails loudly rather than
silently producing empty output.
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
from typing import Any, Dict, Optional, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.data_utils import iter_lines, count_lines
from shared.near_dup import (
    MinHasher,
    MinHashDedupIndex,
    jaccard_estimate,
    lsh_band_keys,
    overlap_estimate,
    signature_and_shingle_count,
)
from shared.run_stats import cleanup_prior_stage_files

# Mirrors 05_quality_filter.py's _REJECTED_FILENAME_CONFIG_KEY — needed
# here ONLY to compute each source's quality-stage rejected-file path, so
# cleanup after this run can delete it alongside {source}_ne_quality.txt
# (same lifecycle — rejected files are not preserved independently of
# their stage's main output).
_QUALITY_REJECTED_FILENAME_CONFIG_KEY = {
    "indiccorp": "rejected_filename",
    "onlinekhabar": "onlinekhabar_rejected_filename",
    "khabarhub": "khabarhub_rejected_filename",
}


def capped_lines(path: str, limit: Optional[int]):
    """Like shared.data_utils.iter_lines(path), but stops after *limit*
    lines when limit is not None. Used only for --sample-lines
    smoke-testing; the full run (limit=None, the default) streams every
    line exactly as before this flag existed — production behavior is
    unchanged unless --sample-lines is explicitly passed."""
    if limit is None:
        yield from iter_lines(path)
        return
    n = 0
    for line in iter_lines(path):
        if n >= limit:
            return
        n += 1
        yield line


def validate_dedup_accounting(
    exact_stats: Dict[str, Any],
    near_stats: Dict[str, Any],
    near_enabled: bool,
    logger: logging.Logger,
) -> None:
    """Verify every input line is accounted for by exactly one outcome
    category, per source. Fails loudly (exits non-zero) rather than
    silently producing a corpus whose own statistics don't add up.

    EXACT pass: every line is either a duplicate or a unique — no other
    category exists there, so input_lines == duplicates + uniques always.

    NEAR-DUP pass: every line is either kept (already includes
    short-line-kept, since those are written straight to `kept`), removed
    as a near-duplicate, removed as a sub-document-duplicate, or a blank
    line (read but never written/classified) — so
    input_lines == kept + near_duplicates_removed +
    sub_document_duplicates_removed + blank_lines_skipped.
    """
    ok = True
    for label, ex in exact_stats.get("by_source", {}).items():
        total = ex["input_lines"]
        accounted = ex["duplicates"] + ex["uniques"]
        if total != accounted:
            logger.error(
                f"[{label}] EXACT-PASS ACCOUNTING MISMATCH: input_lines={total:,} "
                f"but duplicates+uniques={accounted:,}"
            )
            ok = False

    if near_enabled:
        for label, nr in near_stats.get("by_source", {}).items():
            total = nr["input_lines"]
            accounted = (
                nr["kept"] + nr["near_duplicates_removed"]
                + nr["sub_document_duplicates_removed"] + nr.get("blank_lines_skipped", 0)
            )
            if total != accounted:
                logger.error(
                    f"[{label}] NEAR-DUP-PASS ACCOUNTING MISMATCH: input_lines={total:,} "
                    f"but kept+near_dup_removed+sub_doc_removed+blank_lines_skipped={accounted:,}"
                )
                ok = False

    if not ok:
        logger.error(
            "Deduplication accounting does not reconcile — refusing to treat "
            "this run as valid. This indicates a bug in the counting logic, "
            "not necessarily a corpus problem."
        )
        sys.exit(1)
    logger.info("Deduplication accounting verified: every input line is accounted for.")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging based on config."""
    logging_config = config.get("logging", {})
    level_str = logging_config.get("level", "INFO")
    fmt = logging_config.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")

    logging.basicConfig(level=getattr(logging, level_str), format=fmt)
    return logging.getLogger("05_deduplicate")


# ---------------------------------------------------------------------------
# Pass 1: EXACT duplicate removal (unchanged logic; now returns stats)
# ---------------------------------------------------------------------------

def deduplicate_in_memory(input_path: str, output_path: str, logger: logging.Logger, progress_interval: int, resume: bool) -> Dict[str, int]:
    """Deduplicate using an in-memory set of hashes."""
    seen_hashes: Set[str] = set()
    start_line = 0
    mode = "a" if resume else "w"

    if resume and os.path.exists(output_path):
        logger.info(f"Resuming: Loading existing hashes from {output_path}...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                h = hashlib.sha256(line.encode("utf-8")).hexdigest()
                seen_hashes.add(h)
                start_line += 1
        logger.info(f"Loaded {len(seen_hashes)} hashes from existing output.")
    elif not resume and os.path.exists(output_path):
        os.remove(output_path)

    total_lines = start_line
    duplicates = 0
    unique_lines = len(seen_hashes)

    # When resuming in-memory, we can't easily skip input lines unless we assume
    # we processed exactly start_line *input* lines, which is false since duplicates were removed.
    # Therefore, we must process from the beginning of input, but we shouldn't append lines we already know!
    # Wait, if we append lines we already know, we just skip them. But what if there are duplicates *after* start_line?
    # We will just write new unique lines. Since mode="a", we don't truncate.
    # Wait, if we read the existing output, we populated seen_hashes.
    # If we read the input from the beginning, we will hash each line.
    # If it's in seen_hashes, we skip (it might be a duplicate, or it might be one of the lines we already outputted).
    # But wait, we won't know which duplicates are NEW vs OLD.
    # It's better to just process the input and rewrite output if in-memory is used, or accept inaccurate duplicate counts.
    # Let's accept inaccurate duplicate counts for in-memory resume.

    logger.info("Starting in-memory deduplication...")
    with open(output_path, mode, encoding="utf-8", buffering=8192) as out_f:
        for line in iter_lines(input_path):
            total_lines += 1
            h = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(h)
                unique_lines += 1
                out_f.write(line + "\n")

            if total_lines % progress_interval == 0:
                ratio = (duplicates / total_lines) * 100 if total_lines > 0 else 0
                logger.info(f"Processed {total_lines} lines. Duplicates: {duplicates} ({ratio:.2f}%). Unique: {unique_lines}.")

    ratio = (duplicates / total_lines) * 100 if total_lines > 0 else 0
    logger.info(f"Finished. Total lines processed: {total_lines}, Duplicates: {duplicates}, Unique: {unique_lines}, Dedup ratio: {ratio:.2f}%")
    return {"input_lines": total_lines, "duplicates": duplicates, "uniques": unique_lines}


def deduplicate_sqlite(input_path: str, output_path: str, db_path: str, logger: logging.Logger, progress_interval: int, resume: bool) -> Dict[str, int]:
    """Deduplicate using a SQLite database to store hashes."""
    if not resume:
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE IF NOT EXISTS seen_hashes (hash TEXT PRIMARY KEY);")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER);")
    conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('input_lines', 0);")
    conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('duplicates', 0);")
    conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('uniques', 0);")
    conn.commit()

    cursor = conn.cursor()
    cursor.execute("SELECT value FROM meta WHERE key='input_lines'")
    input_lines = cursor.fetchone()[0]
    cursor.execute("SELECT value FROM meta WHERE key='duplicates'")
    duplicates = cursor.fetchone()[0]
    cursor.execute("SELECT value FROM meta WHERE key='uniques'")
    uniques = cursor.fetchone()[0]

    if not resume:
        input_lines = 0
        duplicates = 0
        uniques = 0
        cursor.execute("UPDATE meta SET value=0")
        conn.commit()

    mode = "a" if input_lines > 0 else "w"

    logger.info(f"Starting SQLite deduplication from line {input_lines}...")

    batch_size = 100000

    with open(output_path, mode, encoding="utf-8", buffering=8192) as out_f:
        # We need to skip lines if resuming
        input_iter = iter_lines(input_path)

        for _ in range(input_lines):
            try:
                next(input_iter)
            except StopIteration:
                break

        for line in input_iter:
            input_lines += 1
            h = hashlib.sha256(line.encode("utf-8")).hexdigest()

            cursor.execute("INSERT OR IGNORE INTO seen_hashes (hash) VALUES (?)", (h,))
            if cursor.rowcount > 0:
                # New unique
                uniques += 1
                out_f.write(line + "\n")
            else:
                duplicates += 1

            if input_lines % batch_size == 0:
                cursor.execute("UPDATE meta SET value=? WHERE key='input_lines'", (input_lines,))
                cursor.execute("UPDATE meta SET value=? WHERE key='duplicates'", (duplicates,))
                cursor.execute("UPDATE meta SET value=? WHERE key='uniques'", (uniques,))
                conn.commit()

            if input_lines % progress_interval == 0:
                ratio = (duplicates / input_lines) * 100 if input_lines > 0 else 0
                logger.info(f"Processed {input_lines} lines. Duplicates: {duplicates} ({ratio:.2f}%). Unique: {uniques}.")

        # Final commit
        cursor.execute("UPDATE meta SET value=? WHERE key='input_lines'", (input_lines,))
        cursor.execute("UPDATE meta SET value=? WHERE key='duplicates'", (duplicates,))
        cursor.execute("UPDATE meta SET value=? WHERE key='uniques'", (uniques,))
        conn.commit()

    conn.close()

    ratio = (duplicates / input_lines) * 100 if input_lines > 0 else 0
    logger.info(f"Finished. Total lines processed: {input_lines}, Duplicates: {duplicates}, Unique: {uniques}, Dedup ratio: {ratio:.2f}%")
    return {"input_lines": input_lines, "duplicates": duplicates, "uniques": uniques}


# ---------------------------------------------------------------------------
# Pass 2: NEAR-DUPLICATE / SUB-DOCUMENT duplicate removal (new)
# ---------------------------------------------------------------------------

def deduplicate_near(
    input_path: str,
    output_path: str,
    near_cfg: Dict[str, Any],
    logger: logging.Logger,
    progress_interval: int,
) -> Dict[str, int]:
    """Remove near-duplicate and sub-document-duplicate lines from the
    (already exact-deduplicated) *input_path*, streaming, using MinHash +
    LSH banding for candidate generation and Jaccard/overlap similarity
    for confirmation. See module docstring / nepali_data.yaml `deduplication`
    block for the algorithm."""
    shingle_type = near_cfg.get("shingle_type", "word")
    shingle_size = near_cfg.get("shingle_size", 3)
    num_perm = near_cfg.get("num_minhash_permutations", 32)
    bands = near_cfg.get("lsh_bands", 16)
    rows = near_cfg.get("lsh_rows", 2)
    similarity_threshold = near_cfg.get("similarity_threshold", 0.30)
    sub_doc_overlap_threshold = near_cfg.get("sub_document_overlap_threshold", 0.85)
    sub_doc_length_ratio_max = near_cfg.get("sub_document_length_ratio_max", 0.70)
    min_document_words = near_cfg.get("min_document_words", 4)
    seed = near_cfg.get("seed", 42)

    logger.info("=" * 70)
    logger.info("NEAR-DUPLICATE DEDUPLICATION STARTED")
    logger.info("=" * 70)
    logger.info(
        f"shingle_type={shingle_type} shingle_size={shingle_size} "
        f"num_perm={num_perm} lsh_bands={bands} lsh_rows={rows}"
    )
    logger.info(
        f"similarity_threshold={similarity_threshold} "
        f"sub_document_overlap_threshold={sub_doc_overlap_threshold} "
        f"min_document_words={min_document_words}"
    )

    minhasher = MinHasher(num_perm=num_perm, seed=seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    db_path = output_path + ".near_dup_index.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    index = MinHashDedupIndex(db_path, num_perm=num_perm)

    total = 0
    near_dup_removed = 0
    sub_doc_removed = 0
    kept = 0
    kept_words = 0
    short_lines_kept = 0
    next_line_id = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8", buffering=1 << 20) as out_f:
        for line in iter_lines(input_path):
            total += 1
            if not line:
                continue

            word_count = len(line.split())
            if word_count < min_document_words:
                # Too short for a reliable shingle set — always kept (exact
                # dedup already caught literal repeats) and never indexed,
                # so it can't cause spurious band collisions for others.
                out_f.write(line + "\n")
                kept += 1
                kept_words += word_count
                short_lines_kept += 1
                continue

            sig, shingle_count = signature_and_shingle_count(
                line, minhasher, shingle_size, shingle_type
            )
            band_keys = lsh_band_keys(sig, bands, rows)
            candidates = index.find_candidates(band_keys)

            verdict = None  # ("near_duplicate" | "sub_document", score)
            for _cid, csig, ccount in candidates:
                j = jaccard_estimate(sig, csig)
                shorter, longer = sorted((shingle_count, ccount))
                length_ratio = shorter / longer if longer > 0 else 1.0

                if length_ratio <= sub_doc_length_ratio_max:
                    ov = overlap_estimate(j, shingle_count, ccount)
                    if ov >= sub_doc_overlap_threshold:
                        verdict = ("sub_document", ov)
                        break  # sub-document match is decisive; stop scanning

                if j >= similarity_threshold and verdict is None:
                    verdict = ("near_duplicate", j)
                    # keep scanning remaining candidates in case a stronger
                    # sub-document match turns up

            if verdict is not None:
                kind, _score = verdict
                if kind == "sub_document":
                    sub_doc_removed += 1
                else:
                    near_dup_removed += 1
            else:
                out_f.write(line + "\n")
                kept += 1
                kept_words += word_count
                index.add(next_line_id, sig, shingle_count, band_keys)
                next_line_id += 1

            if total % progress_interval == 0:
                elapsed = time.time() - t0
                speed = total / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {total:,} lines scanned | kept {kept:,} | "
                    f"near-dup removed {near_dup_removed:,} | "
                    f"sub-document removed {sub_doc_removed:,} | "
                    f"{speed:,.0f} lines/s"
                )

    index.close()
    for ext in ("", "-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"NEAR DUPLICATES REMOVED: {near_dup_removed:,}")
    logger.info(f"SUB-DOCUMENT DUPLICATES REMOVED: {sub_doc_removed:,}")
    logger.info(
        f"NEAR-DUPLICATE DEDUPLICATION FINISHED: {total:,} lines scanned, "
        f"{kept:,} kept in {elapsed:.1f}s"
    )
    logger.info("=" * 70)

    return {
        "input_lines": total,
        "near_duplicates_removed": near_dup_removed,
        "sub_document_duplicates_removed": sub_doc_removed,
        "kept": kept,
        "kept_words": kept_words,
        "short_lines_kept": short_lines_kept,
    }


# ---------------------------------------------------------------------------
# Pass 1 (cross-source): EXACT duplicate removal across MULTIPLE labeled
# sources, sharing ONE hash index — see module docstring "Cross-Source
# Deduplication & Source Precedence". Same SHA-256 algorithm as
# deduplicate_in_memory/deduplicate_sqlite above; those two functions are
# left untouched (still used directly for single-source use) — this is a
# new orchestration layer over the same hashing logic, not a redesign of it.
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
    """EXACT-duplicate pass across multiple labeled sources.

    sources: ordered [(label, input_path), ...] — PRECEDENCE ORDER: the
        FIRST source to contain a given line's hash keeps it; a later
        source's copy of the same hash is removed as a duplicate. Callers
        pass sources in SOURCE_PRECEDENCE_ORDER (Onlinekhabar > Khabarhub >
        IndicCorp) so cross-source duplicates survive as
        whichever source ranks highest (see module docstring for
        rationale). This function itself is agnostic to how many sources
        are passed or what they're named.
    output_paths: label -> output file path for that source's surviving
        (exact-deduplicated) lines.
    Returns combined stats plus a "by_source" breakdown, including
    per-source "cross_source_duplicates" (of that source's duplicates, how
    many were duplicates of a DIFFERENT source's earlier line — i.e. would
    NOT have been caught by deduplicating each source independently).
    """
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
# Pass 2 (cross-source): NEAR-DUPLICATE / SUB-DOCUMENT duplicate removal
# across MULTIPLE labeled sources, sharing ONE MinHash/LSH index — see
# module docstring "Cross-Source Deduplication & Source Precedence". This
# reuses shared/near_dup.py's MinHasher, MinHashDedupIndex,
# signature_and_shingle_count, lsh_band_keys, jaccard_estimate, and
# overlap_estimate EXACTLY as deduplicate_near() does above (that function
# is left untouched, still used directly for single-source use) — only the
# multi-source orchestration (which output file a kept line goes to) is new.
# ---------------------------------------------------------------------------

def deduplicate_near_cross_source(
    sources,
    output_paths: Dict[str, str],
    near_cfg: Dict[str, Any],
    logger: logging.Logger,
    progress_interval: int,
    sample_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """NEAR-DUPLICATE / SUB-DOCUMENT pass across multiple labeled sources.

    sources: ordered [(label, input_path), ...] — PRECEDENCE ORDER, same
        semantics as deduplicate_exact_cross_source: the first source to
        KEEP a given (near-)unique line wins; a later source's near/sub-
        document duplicate of it is removed.
    output_paths: label -> output file path for that source's surviving
        lines.
    """
    shingle_type = near_cfg.get("shingle_type", "word")
    shingle_size = near_cfg.get("shingle_size", 3)
    num_perm = near_cfg.get("num_minhash_permutations", 32)
    bands = near_cfg.get("lsh_bands", 16)
    rows = near_cfg.get("lsh_rows", 2)
    similarity_threshold = near_cfg.get("similarity_threshold", 0.30)
    sub_doc_overlap_threshold = near_cfg.get("sub_document_overlap_threshold", 0.85)
    sub_doc_length_ratio_max = near_cfg.get("sub_document_length_ratio_max", 0.70)
    min_document_words = near_cfg.get("min_document_words", 4)
    seed = near_cfg.get("seed", 42)

    logger.info("=" * 70)
    logger.info("CROSS-SOURCE NEAR-DUPLICATE DEDUPLICATION STARTED")
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

    minhasher = MinHasher(num_perm=num_perm, seed=seed)

    output_dir = next((os.path.dirname(p) for p in output_paths.values() if p), ".")
    os.makedirs(output_dir or ".", exist_ok=True)
    db_path = os.path.join(output_dir or ".", "cross_source_near_dup_index.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    index = MinHashDedupIndex(db_path, num_perm=num_perm)

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

                verdict = None
                for _cid, csig, ccount in candidates:
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
                    logger.info(
                        f"  [{label}] {total:,} lines scanned (all sources) | "
                        f"kept so far (this source) {stats[label]['kept']:,} | {speed:,.0f} lines/s"
                    )
    finally:
        index.close()
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)
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
    }

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"NEAR DUPLICATES REMOVED (all sources): {combined['near_duplicates_removed']:,}")
    logger.info(f"SUB-DOCUMENT DUPLICATES REMOVED (all sources): {combined['sub_document_duplicates_removed']:,}")
    logger.info(
        f"CROSS-SOURCE NEAR-DUPLICATE DEDUPLICATION FINISHED: {combined['input_lines']:,} lines scanned, "
        f"{combined['kept']:,} kept in {elapsed:.1f}s"
    )
    logger.info("=" * 70)

    return combined


# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

# PRECEDENCE ORDER (highest first) — see module docstring "Cross-Source
# Deduplication & Source Precedence" for the rationale. Adding/removing a
# source means editing ONLY this tuple (plus, if truly a new source, its
# default quality-input-filename in resolve_source_paths()) — nothing
# else in this file assumes a fixed count or specific set of sources.
SOURCE_PRECEDENCE_ORDER = ("onlinekhabar", "khabarhub", "indiccorp")

# Default quality-stage input filename per source, following the existing
# `{source}_ne_quality.txt` convention (05_quality_filter.py's output).
DEFAULT_QUALITY_FILENAME = {name: f"{name}_ne_quality.txt" for name in SOURCE_PRECEDENCE_ORDER}

# Default deduplicated-stage output filename per source.
DEFAULT_DEDUP_FILENAME = {name: f"{name}_ne_deduplicated.txt" for name in SOURCE_PRECEDENCE_ORDER}


def resolve_source_paths(config: Dict[str, Any], repo_root: Path, args) -> tuple:
    """Resolve (input_path, output_path) for every source in
    SOURCE_PRECEDENCE_ORDER, config-driven with CLI-flag overrides —
    generalizes what used to be individually hardcoded indiccorp_input_path/
    onlinekhabar_input_path/etc. local variables in main(). Returns
    (input_path_by_label, output_path_by_label).

    Every source's quality-input directory/filename and dedup-output
    filename are independently configurable via nepali_data.yaml
    (quality.<source>_output_dir / quality.<source>_output_filename;
    paths.deduplicated for the shared output directory), falling back to
    the `{source}_ne_quality.txt` / `{source}_ne_deduplicated.txt`
    convention already used by indiccorp — so onlinekhabar/khabarhub
    work out of the box even before nepali_data.yaml is edited to name
    them explicitly.
    """
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
        description="Cross-source exact + near-duplicate + sub-document "
                     "deduplication across IndicCorp, Onlinekhabar, and "
                     "Khabarhub TOGETHER (one shared pass, not three "
                     "independent per-source runs — see module docstring)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    for name in SOURCE_PRECEDENCE_ORDER:
        # --indiccorp-input/--output keep their original --input/--output
        # aliases for backward compatibility with existing invocations.
        input_aliases = [f"--{name}-input"] + (["--input"] if name == "indiccorp" else [])
        output_aliases = [f"--{name}-output"] + (["--output"] if name == "indiccorp" else [])
        parser.add_argument(*input_aliases, dest=f"{name}_input", type=str,
                             help=f"Override {name} quality-stage input file path")
        parser.add_argument(*output_aliases, dest=f"{name}_output", type=str,
                             help=f"Override {name} deduplicated-stage output file path")
    parser.add_argument("--resume", action="store_true", help="Resume the EXACT pass from previous run (near-dup pass always runs fully)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output (ignores resume)")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory dict instead of SQLite for the exact pass")
    parser.add_argument("--skip-near-dup", action="store_true", help="Force-disable the near-duplicate pass regardless of config")
    parser.add_argument("--sample-lines", type=int, default=None,
                         help="Smoke-test mode: cap each source to at most N lines "
                              "(e.g. 5000) instead of processing the full corpus. "
                              "Defaults to None (unlimited / full corpus) — omit this "
                              "flag for a production run.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    progress_interval = config.get("logging", {}).get("progress_interval", 1000000)
    dedup_cfg = config.get("deduplication", {})
    exact_enabled = dedup_cfg.get("exact", {}).get("enabled", True)
    near_cfg = dedup_cfg.get("near_duplicate", {})
    near_enabled = near_cfg.get("enabled", True) and not args.skip_near_dup
    keep_intermediate = config.get("storage", {}).get("keep_intermediate_files", True)

    # Resolve paths
    # Config-relative (not script-relative) so a scratch/test config can
    # point the whole pipeline at an isolated directory tree — matches the
    # other pipeline scripts (08/09/11/15).
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

    # SOURCE PRECEDENCE: Onlinekhabar > Khabarhub > IndicCorp —
    # see module docstring "Cross-Source Deduplication & Source Precedence".
    sources_in_precedence_order = [(name, input_path_by_label[name]) for name in SOURCE_PRECEDENCE_ORDER]

    for name in SOURCE_PRECEDENCE_ORDER:
        found = "FOUND" if name in existing_inputs else "NOT FOUND — treated as empty source"
        logger.info(f"{name} input path:  {input_path_by_label[name]}  [{found}]")
        logger.info(f"{name} output path: {output_paths_by_label[name]}")
    logger.info(f"exact.enabled={exact_enabled}  near_duplicate.enabled={near_enabled}")
    logger.info(
        "Cross-source dedup: " + " + ".join(SOURCE_PRECEDENCE_ORDER) +
        " are deduplicated TOGETHER in one pass "
        "(source precedence: " + " > ".join(SOURCE_PRECEDENCE_ORDER) + " on ties)."
    )

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
        db_path = os.path.join(os.path.dirname(output_paths_by_label["indiccorp"]), "cross_source_exact_dedup.sqlite")
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

    # NOTE: when near_enabled is False, exact_output_paths == output_paths_by_label
    # above, so the exact pass wrote the final per-source outputs directly —
    # nothing further to do here in that case.
    near_stats = {"near_duplicates_removed": 0, "sub_document_duplicates_removed": 0, "by_source": {}}
    if near_enabled:
        near_input_sources = [(label, exact_output_paths.get(label)) for label, _ in sources_in_precedence_order]
        near_stats = deduplicate_near_cross_source(
            near_input_sources, output_paths_by_label, near_cfg, logger, progress_interval,
            sample_lines=None,  # sample cap already applied in the exact pass above; near-dup reads its (already-capped) output
        )
        if not keep_intermediate:
            for p in exact_output_paths.values():
                if os.path.exists(p):
                    os.remove(p)

    validate_dedup_accounting(exact_stats, near_stats, near_enabled, logger)

    # ---- Combined + per-source statistics & metadata ----------------------
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

    report = {
        # ---- Backward-compatible combined top-level keys (SAME schema as
        # the original two-source report; now aggregated over ALL FOUR
        # sources).
        # input_file/output_file specifically still point at indiccorp
        # (the original single-source anchor) for any consumer that reads
        # only those two keys; report["sources"] has the full per-source
        # breakdown for everything else. ----
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
            "near_duplicate": {**near_cfg, "enabled": near_enabled},
        },
        "input_file": input_path_by_label["indiccorp"],
        "output_file": output_paths_by_label["indiccorp"],

        # ---- Cross-source fields ----
        "cross_source": True,
        "source_precedence": list(SOURCE_PRECEDENCE_ORDER),
        "source_precedence_rationale": (
            "Onlinekhabar > Khabarhub > IndicCorp. Manual sources "
            "(onlinekhabar/khabarhub) are deduplicated BEFORE "
            "IndicCorp (public); when identical/near-identical text exists "
            "in both a manual source and IndicCorp, the manual copy "
            "survives, protecting the project's >=20% manual-content-token "
            "requirement (config manual.min_fraction / manual.enforce). "
            "Within manual sources: Onlinekhabar before Khabarhub by "
            "planning-target size (see nepali_data.yaml's manual.news "
            "block) — see module docstring for full rationale."
        ),
        "exact_duplicates_removed_cross_source": exact_stats.get("cross_source_duplicates", 0),
        "sources": per_source_report,
    }

    meta_dir = str(repo_root / config["paths"]["metadata"])
    os.makedirs(meta_dir, exist_ok=True)
    report_path = os.path.join(meta_dir, "deduplication_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("CROSS-SOURCE DEDUPLICATION SUMMARY")
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
            _QUALITY_REJECTED_FILENAME_CONFIG_KEY.get(name, ""), f"{name}_ne_quality_rejected.txt"
        )
        rejected_path = os.path.join(os.path.dirname(input_path_by_label[name]), rejected_filename)
        cleanup_prior_stage_files(
            config, repo_root, stage_to_delete="quality",
            files=[input_path_by_label[name], rejected_path], logger=logger,
        )


if __name__ == "__main__":
    main()
