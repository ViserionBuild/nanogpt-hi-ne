#!/usr/bin/env python3
"""
05_aggregate_news_corpus.py — Combine ALL enabled news outlets into ONE
real, top-level "news" corpus source for the common pipeline.

There are exactly THREE top-level (logical) corpus sources in this
project:

    indiccorp   — 01_download_indiccorp.py
    wikipedia   — hindi/scripts/manual/01_download_wikipedia.py +
                  02_extract_wikipedia.py already produce ONE flat text
                  file for the whole dump; it needs NO merge step and is
                  fed straight into 02_normalize.py --source wikipedia.
    news        — hindi/scripts/manual/03_collect_news.py collects each
                  enabled outlet (dainik_jagran, hindustan, abp,
                  amar_ujala, dainik_bhaskar, navbharat_times, aaj_tak,
                  ndtv, jansatta, prabhat_khabar) into its OWN sharded
                  JSONL files. THIS script is the single deterministic
                  place where those per-outlet shards are merged into one
                  real "news" corpus file, immediately after
                  03_collect_news.py runs and before 02_normalize.py --source
                  news ever sees the data. Individual outlets remain
                  METADATA only (source_name on each provenance record)
                  from this point on — never separate top-level/logical
                  corpus sources for accounting, target calculation, or
                  final verification.

Wikipedia is explicitly OUT OF SCOPE for this script — see
manual/01_download_wikipedia.py / 02_extract_wikipedia.py, which already produce
wikipedia_hi_raw.txt directly; that file needs no aggregation and is read
straight from manual.wikipedia.output_dir by 02_normalize.py.

Input:
    hindi/data/raw/manual/news/<source_name>_*.jsonl

Output:
    hindi/data/intermediate/news/news_raw.txt (config paths.news_intermediate)
        One paragraph/sentence per line (same convention as
        indiccorp_hi_*_candidate.txt / wikipedia_hi_raw.txt), containing
        ONLY the text consumed by the common pipeline — feed this into
        02_normalize.py via --source news (or --input/--output directly),
        e.g.:
            python hindi/scripts/02_normalize.py --config <cfg> \\
                --source news
        (02_normalize.py/03/04/05 are NOT modified by this script — this
        just produces a file in the same shape they already accept.)

    hindi/data/intermediate/news/news_provenance.jsonl
        ONE record per INCLUDED document, so every line in news_raw.txt
        can be traced back to its source. Every record carries:
            logical_source ("news"), source_type ("news"),
            source_name, source_url, document_id, title, line_start,
            line_count, word_count, character_count, sha256
        document_id = sha256(source_url)[:16] — stable/deterministic
        across re-runs (not a random UUID), so the same source always
        gets the same id. source_name is METADATA ONLY on each record —
        never a grouping key for corpus accounting downstream.

    hindi/data/raw/manual/news/news_aggregation_report.json
        The documents/words-per-outlet + TOTAL NEWS report (see
        --report below), an explicit by_source_name METADATA breakdown
        (informational only — see module docstring above), exclusion
        counts (never-silent: every excluded document is counted under a
        specific reason, never just dropped), and a reconciliation block
        proving seen == included + excluded for every outlet.

-----------------------------------------------------------------------------
ENABLED/DISABLED SOURCES (config-driven, respected here — not just at
collection time)
-----------------------------------------------------------------------------
  * manual.news.enabled == false       -> ALL news outlets contribute
    nothing, regardless of individual outlet settings.
  * manual.news.<source_name>.enabled == false -> that outlet
    contributes nothing, even if shard files happen to exist on disk
    from before it was disabled.
  * An outlet that IS enabled but has NO collected data on disk is a
    HARD, LOUD failure (logged as ERROR, and this script exits non-zero
    after printing the full report) — it is never silently treated as
    "just zero documents", which would hide a broken/never-run
    collector. A disabled source contributing zero documents is normal
    and only logged at INFO.

-----------------------------------------------------------------------------
VALIDATION / EXCLUSION (nothing is ever silently dropped)
-----------------------------------------------------------------------------
A source document is EXCLUDED from news_raw.txt (but always counted
under a specific, reported reason) when:
  * extraction_status is anything other than "success"/"partial" — no
    reliable text exists yet for that document.
  * its extracted text is empty/near-empty after stripping
    (< MIN_DOCUMENT_CHARS characters).
  * its Devanagari-character ratio is implausibly low for Hindi content
    (reusing manual_common's devanagari_ratio) — a second safety net in
    case a document was marked "success" at collection time but
    shouldn't have been.
  * it is an EXACT duplicate (by whole-document SHA-256, after stripping
    and collapsing whitespace) of a document already included from any
    outlet (cross-outlet dedup, not just within one outlet) — this also
    makes duplicate INPUT FILES (e.g. an accidental repeat shard, or the
    same shard reachable via two glob matches) a no-op on the output
    rather than a source of duplicated content.

Word/character counts reported here are WHITESPACE words and raw
characters — NOT BPE tokens. Actual token counts are measured downstream
by the trained tokenizer (scripts 09/12), same as every other source
in this pipeline; this script does not claim or estimate token counts.

Usage:
    python hindi/scripts/manual/05_aggregate_news_corpus.py \\
        --config hindi/configs/hindi_data.yaml

    # Fast, network-free (this script never touches the network anyway)
    # check of what WOULD be aggregated, without reading full document
    # content or writing output:
    python hindi/scripts/manual/05_aggregate_news_corpus.py \\
        --config hindi/configs/hindi_data.yaml --dry-run

    # Force regeneration of an existing combined corpus:
    python hindi/scripts/manual/05_aggregate_news_corpus.py \\
        --config hindi/configs/hindi_data.yaml --overwrite
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from manual_common import (  # noqa: E402
    MIN_CHARS_FOR_DEVANAGARI_CHECK, MIN_DEVANAGARI_RATIO, NEWS_SOURCE_NAMES,
    devanagari_ratio,
)

MIN_DOCUMENT_CHARS = 20  # below this (after stripping), a document is empty/invalid


# ---------------------------------------------------------------------------
# Config / logging
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("05_aggregate_news_corpus")


def document_id_for(source_url: str) -> str:
    """Deterministic short id — same source_url always yields the same id."""
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]


def iter_metadata_records(metadata_path: Path):
    if not metadata_path.exists():
        return
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Per-document reading + validation
# ---------------------------------------------------------------------------

def validate_document(lines: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns (is_valid, reason_if_invalid, stats). stats always includes
    char_count/word_count/devanagari_ratio/nonspace_count so callers can
    report even on excluded documents.
    """
    combined = "\n".join(lines)
    char_count = len(combined)
    word_count = sum(len(l.split()) for l in lines)
    ratio, nonspace_count = devanagari_ratio(combined)
    stats = {
        "char_count": char_count,
        "word_count": word_count,
        "devanagari_ratio": ratio,
        "nonspace_count": nonspace_count,
    }

    stripped_total = sum(len(l) for l in lines)
    if stripped_total < MIN_DOCUMENT_CHARS:
        return False, "empty_or_invalid", stats
    if nonspace_count >= MIN_CHARS_FOR_DEVANAGARI_CHECK and ratio < MIN_DEVANAGARI_RATIO:
        return False, "low_devanagari_ratio", stats
    return True, "", stats


def document_hash(lines: List[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def process_news(
    config: Dict[str, Any],
    repo_root: Path,
    combined_f,
    provenance_f,
    seen_hashes: Dict[str, str],
    line_no: int,
    counters: Dict[str, int],
    by_source_name: Dict[str, Dict[str, int]],
    excluded: Dict[str, int],
    hard_failures: List[str],
    reconciliation: Dict[str, int],
    logger: logging.Logger,
    dry_run: bool,
) -> int:
    """Process every enabled news outlet — sharded JSONL, one record per
    article, written by 03_collect_news.py (concurrent collection, UNCHANGED
    by this script). Each JSONL line already carries both the metadata
    fields AND the article text together, so there is no separate
    text_path to resolve/read here.

    Iterates over every source_name in NEWS_SOURCE_NAMES, deterministically
    ordered. A source_name is SKIPPED (contributes nothing) if disabled —
    either globally (manual.news.enabled) or individually
    (manual.news.<source_name>.enabled). An ENABLED source_name with no
    shard files on disk is a HARD, LOUD failure (recorded in
    hard_failures, logged as ERROR) — never silently zero.

    by_source_name accumulates a documents/words/characters breakdown per
    source_name IN ADDITION to the overall "news" counters — METADATA
    ONLY, for provenance/diagnostics, never a separate logical corpus
    source. A trained tokenizer still measures the actual BPE counts
    later — this is raw word/char counts only, same caveat as everywhere
    else in this script."""
    manual_cfg = config.get("manual", {})
    news_cfg = manual_cfg.get("news", {})
    news_globally_enabled = news_cfg.get("enabled", True)
    raw_dir = repo_root / news_cfg.get("output_dir", "hindi/data/raw/manual/news")

    if not news_globally_enabled:
        logger.info("[news] manual.news.enabled is false — contributing 0 documents "
                     "from ALL news outlets.")
        return line_no

    n_records = 0
    shard_paths_seen: set = set()

    for source_name in NEWS_SOURCE_NAMES:
        src_cfg = news_cfg.get(source_name, {})
        source_enabled = news_globally_enabled and src_cfg.get("enabled", False)

        if not source_enabled:
            logger.info(f"[news:{source_name}] disabled — contributing 0 documents.")
            continue

        shards = sorted(raw_dir.glob(f"{source_name}_*.jsonl"))
        # Duplicate input files (e.g. the same shard reachable via two glob
        # matches, or a stray copy) must not be double-processed.
        dedup_shards = []
        for shard in shards:
            resolved = shard.resolve()
            if resolved in shard_paths_seen:
                logger.warning(f"[news:{source_name}] skipping duplicate input file "
                                f"(already processed): {shard}")
                continue
            shard_paths_seen.add(resolved)
            dedup_shards.append(shard)

        if not dedup_shards:
            msg = (f"[news:{source_name}] ENABLED but no shard files found at "
                   f"{raw_dir} — collector has not been run (or produced 0 "
                   f"output) for this source. This is a hard failure, not a "
                   f"silent zero.")
            logger.error(msg)
            hard_failures.append(msg)
            continue

        source_seen = 0
        source_included = 0
        source_excluded = 0

        for shard in dedup_shards:
            with open(shard, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except json.JSONDecodeError:
                        excluded["malformed_jsonl"] = excluded.get("malformed_jsonl", 0) + 1
                        continue
                    n_records += 1
                    source_seen += 1

                    title = rec.get("title", "(untitled)")
                    source_url = rec.get("source_url", "")
                    status = rec.get("extraction_status")
                    text = rec.get("text", "")

                    if status not in ("success", "partial"):
                        reason = status or "unknown_status"
                        excluded[reason] = excluded.get(reason, 0) + 1
                        source_excluded += 1
                        continue

                    if dry_run:
                        source_included += 1
                        continue

                    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                    is_valid, reason, stats = validate_document(lines)
                    if not is_valid:
                        excluded[reason] = excluded.get(reason, 0) + 1
                        source_excluded += 1
                        logger.info(f"[news:{source_name}] excluded ({reason}): {title[:60]}")
                        continue

                    h = document_hash(lines)
                    if h in seen_hashes:
                        excluded["duplicate_documents"] = excluded.get("duplicate_documents", 0) + 1
                        source_excluded += 1
                        logger.info(
                            f"[news:{source_name}] excluded (exact duplicate of "
                            f"{seen_hashes[h]}): {title[:60]}"
                        )
                        continue
                    seen_hashes[h] = f"news:{source_name}:{title[:60]}"

                    doc_id = rec.get("doc_id") or document_id_for(source_url)
                    line_start = line_no
                    for line in lines:
                        combined_f.write(line + "\n")
                        line_no += 1

                    provenance = {
                        "logical_source": "news",
                        "source_type": "news",
                        "source_name": source_name,
                        "source_url": source_url,
                        "document_id": doc_id,
                        "title": title,
                        "language": "hi",
                        "line_start": line_start,
                        "line_count": len(lines),
                        "word_count": stats["word_count"],
                        "character_count": stats["char_count"],
                        "extraction_status": status,
                        "sha256": h,
                    }
                    provenance_f.write(json.dumps(provenance, ensure_ascii=False) + "\n")
                    source_included += 1

                    counters["documents"] = counters.get("documents", 0) + 1
                    counters["words"] = counters.get("words", 0) + stats["word_count"]
                    counters["characters"] = counters.get("characters", 0) + stats["char_count"]

                    sn = by_source_name.setdefault(source_name, {"documents": 0, "words": 0, "characters": 0})
                    sn["documents"] += 1
                    sn["words"] += stats["word_count"]
                    sn["characters"] += stats["char_count"]

        reconciliation[f"news:{source_name}:seen"] = source_seen
        reconciliation[f"news:{source_name}:included"] = source_included
        reconciliation[f"news:{source_name}:excluded"] = source_excluded
        if source_seen != source_included + source_excluded:
            msg = (f"[news:{source_name}] reconciliation mismatch: seen={source_seen} "
                   f"!= included={source_included} + excluded={source_excluded}")
            logger.error(msg)
            hard_failures.append(msg)

    if n_records == 0:
        logger.info(f"[news] no records processed under {raw_dir} (no enabled source "
                     f"had shard files).")

    return line_no


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report_table(sources_report: Dict[str, Dict[str, int]], total: Dict[str, int],
                        excluded: Dict[str, int],
                        by_source_name: Optional[Dict[str, Dict[str, int]]] = None) -> None:
    print("\n" + "=" * 60)
    print(" NEWS CORPUS AGGREGATION SUMMARY ".center(60, "="))
    print("=" * 60)
    s = sources_report.get("news", {"documents": 0, "words": 0, "characters": 0})
    print(f"{'News':16s} documents: {s['documents']:>6,}   words: {s['words']:>12,}")
    if by_source_name:
        for name in NEWS_SOURCE_NAMES:
            if name not in by_source_name:
                continue
            sn = by_source_name.get(name, {"documents": 0, "words": 0, "characters": 0})
            print(f"  - {name:18s} documents: {sn['documents']:>6,}   words: {sn['words']:>12,}")
    print("-" * 60)
    print(f"{'Total News':16s} documents: {total['documents']:>6,}   words: {total['words']:>12,}")
    print("-" * 60)
    if excluded:
        print("Excluded (never silently dropped):")
        for reason, count in sorted(excluded.items(), key=lambda x: -x[1]):
            print(f"  {reason:30s}: {count:,}")
    else:
        print("Excluded: none")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine ALL enabled news outlets into ONE real, "
                     "top-level 'news' corpus source for the common "
                     "normalization/filtering/dedup/tokenizer pipeline. "
                     "Individual news outlets remain metadata (source_name) "
                     "only, never separate logical/top-level sources. "
                     "Wikipedia is out of scope for this script — it "
                     "already produces one file naturally, see "
                     "manual/01_download_wikipedia.py / 02_extract_wikipedia.py."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to hindi_data.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be aggregated without reading full "
                             "document content or writing any output.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing combined corpus/provenance/report.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(config)
    repo_root = Path(args.config).resolve().parent.parent.parent

    news_intermediate_dir = repo_root / config.get("paths", {}).get(
        "news_intermediate", "hindi/data/intermediate/news"
    )
    combined_path = news_intermediate_dir / "news_raw.txt"
    provenance_path = news_intermediate_dir / "news_provenance.jsonl"

    # Report lives alongside the raw JSONL shards it summarizes
    # (data/raw/manual/news/), not in a separate top-level metadata
    # directory — matches the Nepali manual-collector convention (see
    # nepali/scripts/manual/collect_news.py's news_collection_summary.json).
    news_cfg = config.get("manual", {}).get("news", {})
    report_dir = repo_root / news_cfg.get("output_dir", "hindi/data/raw/manual/news")
    report_path = report_dir / "news_aggregation_report.json"

    logger.info("=" * 70)
    logger.info("Hindi NEWS Corpus Aggregation (logical source: news)")
    logger.info("=" * 70)
    logger.info(f"Input:      hindi/data/raw/manual/news/<source_name>_*.jsonl")
    logger.info(f"Output:     {combined_path}")
    logger.info(f"Provenance: {provenance_path}")
    logger.info(f"Report:     {report_path}")

    if not args.dry_run:
        if combined_path.exists() and not args.overwrite:
            logger.error(
                f"Output already exists: {combined_path}. "
                f"Use --overwrite to regenerate."
            )
            sys.exit(1)
        news_intermediate_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

    sources_report: Dict[str, Dict[str, int]] = {}
    by_source_name: Dict[str, Dict[str, int]] = {}
    excluded: Dict[str, int] = {}
    hard_failures: List[str] = []
    reconciliation: Dict[str, int] = {}
    seen_hashes: Dict[str, str] = {}
    line_no = 0

    combined_f = open(combined_path, "w", encoding="utf-8") if not args.dry_run else None
    provenance_f = open(provenance_path, "w", encoding="utf-8") if not args.dry_run else None

    try:
        counters: Dict[str, int] = {}
        line_no = process_news(
            config, repo_root, combined_f, provenance_f, seen_hashes,
            line_no, counters, by_source_name, excluded, hard_failures,
            reconciliation, logger, args.dry_run,
        )
        sources_report["news"] = {
            "documents": counters.get("documents", 0),
            "words": counters.get("words", 0),
            "characters": counters.get("characters", 0),
        }
    finally:
        if combined_f:
            combined_f.close()
        if provenance_f:
            provenance_f.close()

    total = {
        "documents": sum(s["documents"] for s in sources_report.values()),
        "words": sum(s["words"] for s in sources_report.values()),
        "characters": sum(s["characters"] for s in sources_report.values()),
    }

    print_report_table(sources_report, total, excluded, by_source_name)

    if hard_failures:
        print("HARD FAILURES (enabled source(s) with no usable data, or a "
              "reconciliation mismatch) — never silently treated as zero:")
        for msg in hard_failures:
            print(f"  - {msg}")
        print()

    if args.dry_run:
        logger.info("Dry run complete — no output written, no existing files touched.")
        if hard_failures:
            logger.error(f"{len(hard_failures)} hard failure(s) detected. Exiting non-zero.")
            sys.exit(1)
        return

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "logical_source": "news",
        "sources": sources_report,
        "by_source_name": by_source_name,
        "total_news": total,
        "excluded": excluded,
        "reconciliation": reconciliation,
        "hard_failures": hard_failures,
        "combined_corpus_path": str(combined_path),
        "provenance_path": str(provenance_path),
        "note": (
            "documents/words/characters are RAW whitespace-word and "
            "character counts of INCLUDED documents only. These are NOT "
            "BPE token counts — actual token counts are measured "
            "downstream by the trained tokenizer (scripts 09/12), same "
            "as every other source in this pipeline. by_source_name is "
            "METADATA/provenance only — individual news outlets are never "
            "separate logical/top-level corpus sources; 'news' is the "
            "logical source key used for accounting, targets, fraction "
            "(combined with 'wikipedia' under the manual: target/fraction "
            "grouping — see hindi_data.yaml's manual.min_fraction), and "
            "final verification."
        ),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Combined corpus: {combined_path} ({line_no:,} lines)")
    logger.info(f"Provenance:      {provenance_path}")

    if hard_failures:
        logger.error(f"{len(hard_failures)} hard failure(s) detected — see report and "
                      f"log above. Exiting non-zero (output files above ARE still "
                      f"written for inspection, but should not be treated as complete).")
        sys.exit(1)


if __name__ == "__main__":
    main()
