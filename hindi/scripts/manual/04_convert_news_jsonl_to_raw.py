#!/usr/bin/env python3
"""
04_convert_news_jsonl_to_raw.py — DEPRECATED, NOT part of the required pipeline

-----------------------------------------------------------------------------
DEPRECATED
-----------------------------------------------------------------------------
manual/05_aggregate_news_corpus.py (the actual news aggregator —
originally 01d_aggregate_manual_corpus.py, renamed to
01d_aggregate_news_corpus.py under the indiccorp/wikipedia/news 3-source
restructuring when it started handling news only, never wikipedia, then
moved into scripts/manual/ with its numeric prefix dropped, then given
back a numeric prefix — 05_ — once every manual/ collector script was
numbered by run order, this script itself getting 04_ ahead of it despite
being optional/skippable, purely to keep filenames sorted in pipeline
position) reads 03_collect_news.py's sharded
JSONL DIRECTLY — `raw_dir.glob(f"{source_name}_*.jsonl")`. It never
reads this script's `{source}_hi_raw.txt` output. Running this script is
therefore NOT a required step before it and is NOT part of the
documented pipeline order (see hindi/README.md /
Hindi_Corpus_Pipeline_Run_Instructions.md, which no longer list it as
"Step 1c"). It is kept only as an optional, standalone debugging aid if
you want a human-readable flattened text file for one news source
outside the normal pipeline flow — nothing downstream consumes its
output.

Original docstring follows, describing what the script itself still does:
-----------------------------------------------------------------------------

04_convert_news_jsonl_to_raw.py — Bridge 03_collect_news.py's JSONL shards into a
single per-source raw text file that 02_normalize.py can consume.

03_collect_news.py writes sharded JSONL under manual.news.output_dir:

    dainik_jagran_00000.jsonl, dainik_jagran_00001.jsonl, ...
    hindustan_00000.jsonl, ...
    abp_00000.jsonl, ...

Each line is one document record (doc_id, text, source_name, ...) — see
03_collect_news.py's module docstring for the full schema. The rest of the
pipeline (02_normalize.py onward) expects one plain-text document per line
in a single file, matching the convention already used for Wikipedia
(wikipedia_hi_raw.txt, produced by 02_extract_wikipedia.py). This script
performs the equivalent conversion for every news source:

    dainik_jagran_*.jsonl  -->  dainik_jagran_hi_raw.txt
    hindustan_*.jsonl      -->  hindustan_hi_raw.txt
    abp_*.jsonl             -->  abp_hi_raw.txt
    (etc. for any other source present in manual.news, enabled or not —
    a source with no shards yet simply produces zero documents, reported
    as a warning, never a crash)

Design (identical to nepali/scripts/manual/convert_news_jsonl_to_raw.py):
  - Streams shard files one at a time, one JSON line at a time. Never
    loads the corpus into memory.
  - Deterministic ordering: shards are processed in sorted filename order
    (matching JsonlShardWriter's own shard-index naming), and documents
    within a shard are processed in file order.
  - A document's "text" field (already a cleaned article body, but still
    possibly multi-line/multi-paragraph) is flattened to a single line by
    joining on whitespace, so the one-document-per-line invariant holds.
  - Malformed JSON lines and records missing/empty "text" are skipped and
    counted, never silently dropped without a trace.
  - Duplicate doc_id values (e.g. from overlapping shard reruns) are
    skipped and counted — each document is written at most once.
  - This script does NOT do any quality/Devanagari filtering — that is
    03_filter_hindi.py / 04_quality_filter.py's job, same as every other
    raw source in this pipeline.

Usage:
    python hindi/scripts/manual/04_convert_news_jsonl_to_raw.py \\
        --config hindi/configs/hindi_data.yaml --source dainik_jagran

    python hindi/scripts/manual/04_convert_news_jsonl_to_raw.py \\
        --config hindi/configs/hindi_data.yaml --source all --overwrite
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from manual_common import NEWS_SOURCE_NAMES  # noqa: E402

WHITESPACE_RE = re.compile(r"\s+")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("04_convert_news_jsonl_to_raw")


def flatten_text(text: str) -> str:
    """Collapse a possibly multi-line document body into one line."""
    return WHITESPACE_RE.sub(" ", text).strip()


def convert_source(
    source: str,
    shard_dir: Path,
    output_path: Path,
    logger: logging.Logger,
) -> Dict[str, int]:
    shards = sorted(shard_dir.glob(f"{source}_*.jsonl"))
    if not shards:
        logger.warning(f"[{source}] no JSONL shards found in {shard_dir}")
        return {
            "shards": 0, "records_seen": 0, "malformed_skipped": 0,
            "empty_text_skipped": 0, "duplicates_skipped": 0,
            "documents_written": 0, "words_written": 0,
        }

    logger.info(f"[{source}] found {len(shards)} shard(s) in {shard_dir}")

    seen_doc_ids = set()
    records_seen = 0
    malformed_skipped = 0
    empty_text_skipped = 0
    duplicates_skipped = 0
    documents_written = 0
    words_written = 0
    t0 = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for shard in shards:
            with open(shard, "r", encoding="utf-8") as in_f:
                for raw_line in in_f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    records_seen += 1
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        malformed_skipped += 1
                        continue

                    doc_id = record.get("doc_id")
                    text = record.get("text")
                    if not text or not isinstance(text, str):
                        empty_text_skipped += 1
                        continue

                    if doc_id is not None:
                        if doc_id in seen_doc_ids:
                            duplicates_skipped += 1
                            continue
                        seen_doc_ids.add(doc_id)

                    flat = flatten_text(text)
                    if not flat:
                        empty_text_skipped += 1
                        continue

                    out_f.write(flat + "\n")
                    documents_written += 1
                    words_written += len(flat.split())

    elapsed = time.time() - t0
    logger.info(
        f"[{source}] {records_seen:,} records seen -> {documents_written:,} "
        f"documents written ({words_written:,} words) in {elapsed:.1f}s "
        f"(malformed={malformed_skipped:,}, empty_text={empty_text_skipped:,}, "
        f"duplicates={duplicates_skipped:,})"
    )

    return {
        "shards": len(shards),
        "records_seen": records_seen,
        "malformed_skipped": malformed_skipped,
        "empty_text_skipped": empty_text_skipped,
        "duplicates_skipped": duplicates_skipped,
        "documents_written": documents_written,
        "words_written": words_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 03_collect_news.py's JSONL shards into a single "
                     "one-document-per-line raw text file per source, "
                     "matching the raw-text convention 02_normalize.py "
                     "expects (as already used by Wikipedia's "
                     "wikipedia_hi_raw.txt)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--source", type=str, choices=list(NEWS_SOURCE_NAMES) + ["all"], default="all",
                         help="Which news source to convert (default: all sources "
                              "present in manual.news).")
    parser.add_argument("--output", type=str, help="Output file path (only valid with a single --source).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()

    logger = setup_logger()
    logger.warning(
        "DEPRECATED: 04_convert_news_jsonl_to_raw.py is NOT part of the required "
        "pipeline — manual/05_aggregate_news_corpus.py reads 03_collect_news.py's "
        "JSONL shards directly and never reads this script's output. Running "
        "this is optional/standalone; nothing downstream consumes it."
    )
    config = load_config(args.config)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent

    if args.output and args.source == "all":
        logger.error("--output requires a single --source, not 'all'.")
        sys.exit(1)

    news_cfg = config["manual"]["news"]
    shard_dir = repo_root / news_cfg["output_dir"]

    if args.source == "all":
        sources = [name for name in NEWS_SOURCE_NAMES if name in news_cfg]
    else:
        sources = [args.source]

    reports = {}
    for source in sources:
        src_cfg = news_cfg.get(source, {})
        raw_filename = src_cfg.get("raw_text_filename", f"{source}_hi_raw.txt")
        output_path = Path(args.output) if args.output else shard_dir / raw_filename

        if output_path.exists() and not args.overwrite:
            logger.error(f"[{source}] output already exists: {output_path}. Use --overwrite.")
            sys.exit(1)

        reports[source] = convert_source(source, shard_dir, output_path, logger)

    total_docs = sum(r["documents_written"] for r in reports.values())
    total_words = sum(r["words_written"] for r in reports.values())
    logger.info(f"TOTAL: {total_docs:,} documents, {total_words:,} words written across {list(reports.keys())}")


if __name__ == "__main__":
    main()
