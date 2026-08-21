#!/usr/bin/env python3
"""
05_deduplicate.py — Cross-Source Exact + Near-Duplicate + Sub-Document
Deduplication (IndicCorp + Wikipedia + News, together)

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
The Hindi corpus has exactly THREE top-level sources — indiccorp,
wikipedia, and news. This script therefore deduplicates THREE labeled
input streams together:

    wikipedia   — hindi/data/intermediate/quality/wikipedia_hi_quality.txt
    news        — hindi/data/intermediate/quality/news_hi_quality.txt
                  (already a PRE-MERGED aggregate of every ENABLED news
                  outlet — dainik_jagran, hindustan, abp, amar_ujala,
                  dainik_bhaskar, navbharat_times, aaj_tak, ndtv, jansatta,
                  prabhat_khabar — combined into ONE file by
                  manual/05_aggregate_news_corpus.py, immediately after
                  03_collect_news.py, long before this script ever runs; NOT
                  multiple separate dedup input streams)
    indiccorp   — hindi/data/intermediate/quality/indiccorp_hi_quality.txt

(each produced by 04_quality_filter.py for its own source). Individual
news outlets are NEVER a separate top-level source at this stage — their
aggregation is entirely manual/05_aggregate_news_corpus.py's concern, upstream
of this script. Wikipedia was likewise never merged with anything; it has
always been produced as its own single file (see
manual/01_download_wikipedia.py + manual/02_extract_wikipedia.py) and simply flows through
02_normalize.py / 03_filter_hindi.py / 04_quality_filter.py as its own
source, exactly like indiccorp.

Devanagari, ZWNJ, ZWJ, punctuation, and whitespace are exactly as produced
by script 02; this stage only reads lines to hash/shingle them — kept
lines are written through byte-for-byte unchanged.

SOURCES is a single ordered list resolved once in main() (see
`SOURCE_PRECEDENCE_ORDER` / `resolve_source_paths()`) — everything below
main(), including deduplicate_exact_cross_source() and
deduplicate_near_cross_source(), was ALREADY generic over an arbitrary
list of `(label, path)` pairs (this is not new code, and it is not
touched here beyond this docstring); the three-source model lives
entirely in SOURCE_PRECEDENCE_ORDER, main()'s path resolution, CLI
argument list, and logging.

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

    Wikipedia > News > IndicCorp

— and keep the FIRST occurrence of a given (near-)duplicate hash/
signature, dropping later occurrences regardless of which source they
came from. Concretely: if the same (or near-identical) text exists in
Wikipedia and IndicCorp, the WIKIPEDIA copy survives; if it exists in
Wikipedia and News, the WIKIPEDIA copy survives; if it exists in News and
IndicCorp, the NEWS copy survives.

RATIONALE (deterministic, not a silent/arbitrary choice): this project
already has an explicit source-priority signal — the manual-content-token
floor (`manual.min_fraction` / `manual.enforce` in hindi_data.yaml, which
Wikipedia/News both count toward as manual sources). Manual data is scarce
and expensive to collect relative to IndicCorp's much larger public pool,
so when the same text exists in both a manual source and IndicCorp,
keeping the manual copy protects that manual-fraction requirement instead
of eroding it.

Within manual: Wikipedia is ranked before News because Wikipedia is a
single curated, high-quality, already-deduplicated-by-construction
encyclopedic source (one dump, no cross-outlet redundancy), whereas News
is a pre-merged aggregate of up to 10 outlets that plausibly
republish/syndicate overlapping wire-service content — i.e. News is more
likely to contain the near-duplicate that actually needs a precedence
tie-break, so preserving the Wikipedia copy when a collision occurs
protects the more distinct, harder-to-replace content.

This is a REASONED DEFAULT, not the only defensible choice, and it is
called out here explicitly as a judgment call: an alternative ordering
(News > Wikipedia, by raw planning-token volume — News totals roughly
120-150M target tokens across enabled outlets vs. Wikipedia's ~20M target)
was considered and rejected in favor of the distinctiveness argument
above. A future maintainer who weighs raw target-token volume more
heavily than source-distinctiveness may reasonably prefer the opposite
order; if so, change ONLY the SOURCE_PRECEDENCE_ORDER tuple below (and
this rationale paragraph) — nothing else in this file assumes a
particular order among sources.

Every surviving line's source is preserved: kept lines are written to
SOURCE-SPECIFIC output files (`indiccorp_hi_deduplicated.txt`,
`wikipedia_hi_deduplicated.txt`, `news_hi_deduplicated.txt`) — never
merged into one file at this stage — and the deduplication report records
per-source input/removed/survivor counts alongside the pre-existing
combined/backward-compatible top-level keys (see main()).

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

Usage:
    python hindi/scripts/05_deduplicate.py --config hindi/configs/hindi_data.yaml

    # Explicit per-source overrides (defaults shown above are normally enough).
    python hindi/scripts/05_deduplicate.py --config hindi/configs/hindi_data.yaml \\
        --indiccorp-input hindi/data/intermediate/quality/indiccorp_hi_quality.txt \\
        --wikipedia-input hindi/data/intermediate/quality/wikipedia_hi_quality.txt \\
        --news-input hindi/data/intermediate/quality/news_hi_quality.txt

FAILURE SAFETY: a source that is REQUIRED but whose quality-filtered input
file does not exist is a hard, loud ERROR (this script exits non-zero) —
never silently treated as an empty source. indiccorp has no enable/disable
switch anywhere in this project, so it is always required. Wikipedia is
required iff it is enabled in config; News is required iff at least one
constituent news outlet is enabled in config — see source_required_map().
If a manual source is disabled, its absent quality-filtered input is
expected (not required) and is treated as an empty source with a warning
— this is what keeps the script usable at every stage of standing each
source's pipeline up without ever letting an ENABLED source silently
vanish. At least ONE source's input must exist or the script fails loudly
rather than silently producing empty output.
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


def capped_lines(path: str, limit):
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
    category exists there (no blank-line/short-line skip in the exact
    pass), so input_lines == duplicates + uniques always.

    NEAR-DUP pass: every line is either kept (this already includes
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
    for confirmation. See module docstring / hindi_data.yaml `deduplication`
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
    # ---- candidate-generation statistics (see module docstring item 3) ----
    eligible_documents = 0
    candidates_generated = 0
    documents_with_candidates = 0
    max_candidates_single_doc = 0
    comparisons_performed = 0
    t0 = time.time()

    try:
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

                eligible_documents += 1
                sig, shingle_count = signature_and_shingle_count(
                    line, minhasher, shingle_size, shingle_type
                )
                band_keys = lsh_band_keys(sig, bands, rows)
                candidates = index.find_candidates(band_keys)
                n_candidates = len(candidates)
                candidates_generated += n_candidates
                if n_candidates > 0:
                    documents_with_candidates += 1
                if n_candidates > max_candidates_single_doc:
                    max_candidates_single_doc = n_candidates

                verdict = None  # ("near_duplicate" | "sub_document", score)
                for _cid, csig, ccount in candidates:
                    comparisons_performed += 1
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
                    avg_cand = candidates_generated / eligible_documents if eligible_documents else 0.0
                    logger.info(
                        f"  {total:,} lines scanned | kept {kept:,} | "
                        f"near-dup removed {near_dup_removed:,} | "
                        f"sub-document removed {sub_doc_removed:,} | "
                        f"candidates {candidates_generated:,} (avg {avg_cand:.2f}/doc, "
                        f"max {max_candidates_single_doc:,}) | "
                        f"comparisons {comparisons_performed:,} | "
                        f"{speed:,.0f} lines/s"
                    )
    finally:
        index.close()
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)

    elapsed = time.time() - t0
    throughput = total / elapsed if elapsed > 0 else 0.0
    avg_candidates = candidates_generated / eligible_documents if eligible_documents else 0.0
    logger.info("=" * 70)
    logger.info(f"NEAR DUPLICATES REMOVED: {near_dup_removed:,}")
    logger.info(f"SUB-DOCUMENT DUPLICATES REMOVED: {sub_doc_removed:,}")
    logger.info(
        f"NEAR-DUPLICATE DEDUPLICATION FINISHED: {total:,} lines scanned, "
        f"{kept:,} kept in {elapsed:.1f}s ({throughput:,.0f} lines/s)"
    )
    logger.info("=" * 70)

    return {
        "input_lines": total,
        "near_duplicates_removed": near_dup_removed,
        "sub_document_duplicates_removed": sub_doc_removed,
        "kept": kept,
        "kept_words": kept_words,
        "short_lines_kept": short_lines_kept,
        "candidate_stats": {
            "total_scanned": total,
            "eligible_documents": eligible_documents,
            "short_lines_skipped": short_lines_kept,
            "candidates_generated": candidates_generated,
            "documents_with_candidates": documents_with_candidates,
            "comparisons_performed": comparisons_performed,
            "max_candidates_single_doc": max_candidates_single_doc,
            "avg_candidates_per_eligible_doc": round(avg_candidates, 4),
            "throughput_lines_per_sec": round(throughput, 2),
            "elapsed_seconds": round(elapsed, 2),
        },
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
        pass sources in SOURCE_PRECEDENCE_ORDER (Wikipedia > News >
        IndicCorp) so cross-source duplicates survive as whichever source
        ranks highest (see module docstring for rationale). This function
        itself is agnostic to how many sources are passed or what they're
        named.
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
            # ---- candidate-generation statistics (see module docstring item 3) ----
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
                    logger.info(
                        f"  [{label}] {total:,} lines scanned (all sources so far) | "
                        f"this source: scanned {s['input_lines']:,}, kept {s['kept']:,}, "
                        f"near-dup removed {s['near_duplicates_removed']:,}, "
                        f"sub-doc removed {s['sub_document_duplicates_removed']:,} | "
                        f"candidates {s['candidates_generated']:,} (avg {avg_cand:.2f}/doc, "
                        f"max {s['max_candidates_single_doc']:,}) | "
                        f"comparisons {s['comparisons_performed']:,} | "
                        f"{speed:,.0f} lines/s"
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
        index.close()
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)
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
# Deduplication & Source Precedence" for the rationale. THREE top-level
# corpus sources: "wikipedia" (single curated encyclopedic dump),
# "news" (a PRE-MERGED aggregate of every enabled news outlet, produced
# upstream by manual/05_aggregate_news_corpus.py — not multiple separate dedup
# input streams), and "indiccorp" (public). Individual news outlets are
# NEVER a separate source here — their aggregation is entirely
# manual/05_aggregate_news_corpus.py's concern, upstream of this script.
SOURCE_PRECEDENCE_ORDER = ("wikipedia", "news", "indiccorp")

PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("wikipedia", "news")

# Default quality-stage input filename per source, following the existing
# `{source}_hi_quality.txt` convention (04_quality_filter.py's output).
DEFAULT_QUALITY_FILENAME = {name: f"{name}_hi_quality.txt" for name in SOURCE_PRECEDENCE_ORDER}

# Default deduplicated-stage output filename per source.
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


def source_required_map(config: Dict[str, Any]) -> Dict[str, bool]:
    """indiccorp has no enable/disable switch anywhere in this project, so
    it is always required. wikipedia is required iff
    config['manual']['wikipedia']['enabled']; news is required iff at
    least one constituent news outlet is enabled in config
    (config['manual']['news']['enabled'] globally, AND at least one
    per-outlet 'enabled' flag under it) — mirrors 07_split_dataset.py's /
    08_decontaminate.py's identical per-source requiredness check. If a
    manual source is disabled, its absent quality-filtered input is
    expected, not an error."""
    manual_cfg = config.get("manual", {})
    wiki_enabled = manual_cfg.get("wikipedia", {}).get("enabled", True)
    news_cfg = manual_cfg.get("news", {})
    news_globally_enabled = news_cfg.get("enabled", True)
    any_news_enabled = news_globally_enabled and any(
        src_cfg.get("enabled", False)
        for name, src_cfg in news_cfg.items()
        if isinstance(src_cfg, dict)
    )
    return {
        PUBLIC_SOURCE: True,
        "wikipedia": bool(wiki_enabled),
        "news": bool(any_news_enabled),
    }


def parse_source_overrides(pairs, valid_sources, flag_name: str, logger: logging.Logger) -> Dict[str, str]:
    """Parse repeated `--input-override SOURCE=PATH` / `--output-override
    SOURCE=PATH` CLI arguments into {source: path}. Generic — equivalent
    to --wikipedia-input/--news-input/--indiccorp-input etc., provided for
    symmetry/scripting (see resolve_source_paths)."""
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
    SOURCE_PRECEDENCE_ORDER ("wikipedia", "news", "indiccorp"),
    config-driven with CLI-flag overrides. Returns
    (input_path_by_label, output_path_by_label).

    Each source's quality-input directory/filename and dedup-output
    filename are independently configurable via hindi_data.yaml
    (quality.<source>_output_dir / quality.<source>_output_filename;
    paths.deduplicated for the shared output directory), falling back to
    the `{source}_hi_quality.txt` / `{source}_hi_deduplicated.txt`
    convention. --indiccorp-input/--wikipedia-input/--news-input (named
    flags) and --input-override/--output-override (generic) are merged
    here, with the override dict taking precedence when both are given
    for the same source.
    """
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
        description="Cross-source exact + near-duplicate + sub-document "
                     "deduplication across IndicCorp, Wikipedia, and News "
                     "TOGETHER in one shared pass (News is itself a "
                     "pre-merged aggregate of all enabled news outlets — "
                     "see module docstring)."
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

    progress_interval = config.get("logging", {}).get("progress_interval", 100000)
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

    # FAILURE SAFETY: a source that is required (indiccorp always;
    # wikipedia/news iff enabled in config — see source_required_map())
    # but whose quality-filtered input is missing is a hard error, never a
    # silent "treated as empty source" — matches 07_split_dataset.py's /
    # 08_decontaminate.py's identical guard. Do NOT interpret a
    # missing-but-required Wikipedia/News input as an empty corpus.
    required = source_required_map(config)
    missing_required_sources = [
        name for name in SOURCE_PRECEDENCE_ORDER
        if name not in existing_inputs and required[name]
    ]
    if missing_required_sources:
        for name in missing_required_sources:
            logger.error(
                f"REQUIRED source missing: '{name}' is required (enabled in "
                f"config) but its quality-filtered input was not found: "
                f"{input_path_by_label[name]}. Refusing to silently treat "
                f"it as an empty source — run its manual/ collector script(s) "
                f"then 03/04/05 for '{name}' first."
            )
        sys.exit(1)

    # SOURCE PRECEDENCE: Wikipedia > News > IndicCorp — see module
    # docstring "Cross-Source Deduplication & Source Precedence".
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
        # the pre-cross-source report; now aggregated over ALL THREE
        # sources) ----
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
        "input_file": input_path_by_label[PUBLIC_SOURCE],
        "output_file": output_paths_by_label[PUBLIC_SOURCE],

        # ---- Cross-source fields (new) ----
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
            "overlapping content — see module docstring for the full "
            "rationale, including the alternative ordering considered and "
            "rejected."
        ),
        "exact_duplicates_removed_cross_source": exact_stats.get("cross_source_duplicates", 0),
        "sources": per_source_report,
        "candidate_stats": near_stats.get("candidate_stats", {}),

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
