#!/usr/bin/env python3
"""
02_extract_wikipedia.py — Hindi Wikipedia `pages-articles.xml.bz2` Extractor

Converts the raw Wikimedia XML dump downloaded by 01_download_wikipedia.py
into plain Hindi article text, one paragraph per output line, so it can be
fed into the EXISTING common preprocessing pipeline (02_normalize.py
onward) via --input/--output — see the README for the exact command.

Input:
    hindi/data/raw/manual/wikipedia/hiwiki-YYYYMMDD-pages-articles.xml.bz2

Output:
    hindi/data/raw/manual/wikipedia/wikipedia_hi_raw.txt

This script does ONLY dump extraction + MediaWiki markup cleanup. It does
NOT perform (these belong to the common pipeline, scripts 02-10):
    * Unicode normalization
    * general whitespace normalization
    * Hindi-script filtering
    * quality filtering
    * safety/blocklist filtering
    * exact / near-duplicate / sub-document deduplication
    * decontamination
    * BPE tokenization
    * corpus-size selection

Streaming design:
    The .xml.bz2 file is decompressed and parsed incrementally with
    xml.etree.ElementTree.iterparse — the complete dump is NEVER held in
    memory. Each <page> element is processed and cleared as soon as it is
    consumed, so memory stays bounded regardless of dump size.

Rerun safety:
    Output is written to a temporary file and atomically renamed into
    place only once extraction finishes successfully, so a killed/failed
    run never leaves a half-written file mistaken for a complete one.
    Re-running (or --overwrite) reprocesses the dump deterministically —
    the same input dump always produces byte-identical output.

Usage:
    python hindi/scripts/manual/02_extract_wikipedia.py \\
        --config hindi/configs/hindi_data.yaml

    # Explicit paths:
    python hindi/scripts/manual/02_extract_wikipedia.py \\
        --config hindi/configs/hindi_data.yaml \\
        --input hindi/data/raw/manual/wikipedia/hiwiki-20260801-pages-articles.xml.bz2 \\
        --output hindi/data/raw/manual/wikipedia/wikipedia_hi_raw.txt
"""

import argparse
import bz2
import glob
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


SOURCE_NAME = "wikipedia"
LANGUAGE = "hi"

# Only the main/article content namespace. Everything else (Talk, User,
# File, Category, Template, Help, Portal, MediaWiki, Module, ...) is
# metadata/non-article content and is skipped.
ARTICLE_NAMESPACE = "0"

FILE_PREFIXES = ("file:", "image:", "चित्र:", "फ़ाइल:")
CATEGORY_PREFIXES = ("category:", "श्रेणी:")


# ---------------------------------------------------------------------------
# Configuration / logging
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("02_extract_wikipedia")


# ---------------------------------------------------------------------------
# Wikitext -> plain text cleaning
#
# Deliberately a best-effort, regex/bracket-depth cleaner (no new heavy
# dependency such as mwparserfromhell) — it is expected to leave some
# residual junk on edge cases, which the existing quality filter
# (04_quality_filter.py) is responsible for catching. This stage must NOT
# grow into a second quality-filtering pipeline (see module docstring).
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_REF_SELFCLOSE_RE = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)
_REF_PAIR_RE = re.compile(r"<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_REFERENCES_TAG_RE = re.compile(r"<references\b[^>]*/>|<references\b[^>]*>.*?</references>",
                                 re.IGNORECASE | re.DOTALL)
_EXTLINK_WITH_TEXT_RE = re.compile(r"\[(?:https?|ftp)://\S+\s+([^\]]+)\]", re.IGNORECASE)
_EXTLINK_BARE_RE = re.compile(r"\[(?:https?|ftp)://\S+\]", re.IGNORECASE)
_BOLD_ITALIC_RE = re.compile(r"'{2,3}")
_HEADING_RE = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)
_MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_templates(text: str) -> str:
    """Remove {{ ... }} templates, including nested ones, via depth counting."""
    out: List[str] = []
    i, n, depth = 0, len(text), 0
    while i < n:
        if text[i] == "{" and i + 1 < n and text[i + 1] == "{":
            depth += 1
            i += 2
            continue
        if text[i] == "}" and i + 1 < n and text[i + 1] == "}" and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def process_wikilinks(text: str) -> str:
    """
    Resolve [[...]] wiki-link syntax:
      * [[File:...]] / [[Image:...]] (any caption/params) -> dropped entirely
      * [[Category:...]]                                  -> dropped entirely
      * [[Target|Display text]]                            -> "Display text"
      * [[Target]]                                         -> "Target"
    Depth-aware so nested [[...]] inside File/Image captions doesn't break.
    """
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "[" and i + 1 < n and text[i + 1] == "[":
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text[j] == "[" and j + 1 < n and text[j + 1] == "[":
                    depth += 1
                    j += 2
                    continue
                if text[j] == "]" and j + 1 < n and text[j + 1] == "]":
                    depth -= 1
                    j += 2
                    continue
                j += 1
            inner = text[i + 2:max(j - 2, i + 2)]
            lower_inner = inner.lower()
            if lower_inner.startswith(FILE_PREFIXES) or lower_inner.startswith(CATEGORY_PREFIXES):
                pass  # drop entirely — not article body text
            else:
                display = process_wikilinks(inner)  # resolve any nested links first
                if "|" in display:
                    display = display.rsplit("|", 1)[-1]
                out.append(display)
            i = j
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def clean_wikitext(text: str) -> str:
    """Strip MediaWiki markup from *text*, returning plain article text."""
    if not text:
        return ""

    text = _COMMENT_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    text = _REF_SELFCLOSE_RE.sub("", text)
    text = _REF_PAIR_RE.sub("", text)
    text = _REFERENCES_TAG_RE.sub("", text)
    text = strip_templates(text)
    text = process_wikilinks(text)
    text = _EXTLINK_WITH_TEXT_RE.sub(r"\1", text)
    text = _EXTLINK_BARE_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _HEADING_RE.sub(r"\1", text)
    text = _MAGIC_WORD_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)

    return text


def wikitext_to_paragraphs(text: str) -> List[str]:
    """
    Split cleaned text into paragraphs (blank-line-separated blocks),
    collapsing single internal newlines (soft line-wraps) into spaces.
    Only structural splitting — no whitespace/quality normalization
    (that belongs to the common pipeline).
    """
    cleaned = clean_wikitext(text)
    paragraphs = []
    for block in re.split(r"\n\s*\n", cleaned):
        para = " ".join(block.split("\n")).strip()
        if para:
            paragraphs.append(para)
    return paragraphs


# ---------------------------------------------------------------------------
# Dump input resolution
# ---------------------------------------------------------------------------

def resolve_input_dump(wiki_dir: Path, logger: logging.Logger) -> Path:
    """Locate the downloaded pages-articles dump in *wiki_dir*."""
    metadata_path = wiki_dir / "download_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        candidate = wiki_dir / meta.get("filename", "")
        if meta.get("status") == "completed" and candidate.exists():
            return candidate
        logger.warning(
            f"download_metadata.json found but status={meta.get('status')!r} "
            f"or file missing; falling back to directory scan."
        )

    candidates = sorted(
        p for p in glob.glob(str(wiki_dir / "hiwiki-*-pages-articles.xml.bz2"))
        if "meta-history" not in p and "metahistory" not in p
    )
    if not candidates:
        logger.error(
            f"No hiwiki-*-pages-articles.xml.bz2 dump found in {wiki_dir}. "
            f"Run 01_download_wikipedia.py first, or pass --input explicitly."
        )
        sys.exit(1)
    return Path(candidates[-1])


# ---------------------------------------------------------------------------
# Streaming XML extraction
# ---------------------------------------------------------------------------

def local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_page(page_elem: ET.Element) -> Tuple[str, Optional[str], bool, Optional[str]]:
    """
    Return (title, ns, is_redirect, revision_text) for a <page> element.
    revision_text is None if missing/deleted.
    """
    title_el = page_elem.find("{*}title")
    ns_el = page_elem.find("{*}ns")
    redirect_el = page_elem.find("{*}redirect")
    text_el = page_elem.find("{*}revision/{*}text")

    title = title_el.text if title_el is not None and title_el.text else ""
    ns = ns_el.text if ns_el is not None else None
    is_redirect = redirect_el is not None

    revision_text: Optional[str] = None
    if text_el is not None and text_el.get("deleted") != "deleted":
        revision_text = text_el.text

    return title, ns, is_redirect, revision_text


def run_extraction(
    input_path: Path,
    output_path: Path,
    logger: logging.Logger,
    progress_interval: int = 2000,
) -> Dict[str, Any]:
    stats = {
        "pages_encountered": 0,
        "article_pages_encountered": 0,
        "non_article_namespace_skipped": 0,
        "redirects_skipped": 0,
        "empty_pages_skipped": 0,
        "deleted_or_missing_revision_skipped": 0,
        "extraction_failures": 0,
        "articles_extracted": 0,
        "output_line_count": 0,
        "approx_character_count": 0,
        "approx_word_count": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    t_start = time.time()

    with bz2.open(input_path, "rb") as bz_f, \
         open(tmp_output_path, "w", encoding="utf-8", buffering=1 << 20) as out_f:

        context = ET.iterparse(bz_f, events=("end",))
        for event, elem in context:
            if local_tag(elem.tag) != "page":
                continue

            stats["pages_encountered"] += 1

            try:
                title, ns, is_redirect, revision_text = extract_page(elem)

                if ns != ARTICLE_NAMESPACE:
                    stats["non_article_namespace_skipped"] += 1
                    continue
                stats["article_pages_encountered"] += 1

                if is_redirect:
                    stats["redirects_skipped"] += 1
                    continue

                if revision_text is None:
                    stats["deleted_or_missing_revision_skipped"] += 1
                    continue

                if not revision_text.strip():
                    stats["empty_pages_skipped"] += 1
                    continue

                paragraphs = wikitext_to_paragraphs(revision_text)
                if not paragraphs:
                    stats["empty_pages_skipped"] += 1
                    continue

                for para in paragraphs:
                    out_f.write(para + "\n")
                    stats["output_line_count"] += 1
                    stats["approx_character_count"] += len(para)
                    stats["approx_word_count"] += len(para.split())

                stats["articles_extracted"] += 1

            except Exception as exc:
                stats["extraction_failures"] += 1
                logger.warning(f"Failed to extract page #{stats['pages_encountered']}: {exc}")

            finally:
                # Bound memory: drop this page's subtree (text/attrib/children)
                # now that it has been consumed. Standard idiom for
                # xml.etree.ElementTree.iterparse over huge XML dumps.
                elem.clear()

            if stats["pages_encountered"] % progress_interval == 0:
                now = time.time()
                elapsed = now - t_start
                speed = stats["pages_encountered"] / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {stats['pages_encountered']:,} pages seen | "
                    f"{stats['articles_extracted']:,} articles extracted | "
                    f"{stats['output_line_count']:,} lines | "
                    f"{speed:.1f} pages/s"
                )

    tmp_output_path.replace(output_path)

    elapsed_total = time.time() - t_start
    logger.info(f"Extraction pass finished in {elapsed_total:.1f}s.")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract plain Hindi article text from the Wikipedia pages-articles dump."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to hindi_data.yaml")
    parser.add_argument("--input", type=str, help="Override path to the .xml.bz2 dump")
    parser.add_argument("--output", type=str, help="Override output text path")
    parser.add_argument("--report", type=str, help="Override extraction report JSON path")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-extract even if the output already exists")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    repo_root = Path(args.config).resolve().parent.parent.parent

    wiki_cfg = config.get("manual", {}).get("wikipedia", {})
    wiki_dir = repo_root / wiki_cfg.get("output_dir", "hindi/data/raw/manual/wikipedia")

    input_path = Path(args.input) if args.input else resolve_input_dump(wiki_dir, logger)
    if not input_path.exists():
        logger.error(f"Input dump not found: {input_path}")
        sys.exit(1)

    output_filename = wiki_cfg.get("raw_text_filename", "wikipedia_hi_raw.txt")
    output_path = Path(args.output) if args.output else wiki_dir / output_filename

    report_path = (
        Path(args.report) if args.report
        else repo_root / wiki_cfg.get(
            "extraction_report_path", "hindi/data/metadata/wikipedia_extraction_report.json"
        )
    )

    if output_path.exists() and not args.overwrite:
        logger.error(
            f"Output already exists: {output_path}. Use --overwrite to re-extract."
        )
        sys.exit(1)

    dump_filename = input_path.name
    dump_date_match = re.search(r"hiwiki-(\d{8})-pages-articles", dump_filename)
    dump_date = dump_date_match.group(1) if dump_date_match else "unknown"

    logger.info("=" * 70)
    logger.info("Hindi Wikipedia pages-articles Extractor")
    logger.info("=" * 70)
    logger.info(f"source:        {SOURCE_NAME}")
    logger.info(f"language:      {LANGUAGE}")
    logger.info(f"Input dump:    {input_path}")
    logger.info(f"Dump date:     {dump_date}")
    logger.info(f"Output:        {output_path}")
    logger.info(f"Report:        {report_path}")

    started_at = datetime.now(timezone.utc).isoformat()
    progress_interval = config.get("logging", {}).get("wikipedia_progress_interval", 2000)

    stats = run_extraction(input_path, output_path, logger, progress_interval=progress_interval)

    finished_at = datetime.now(timezone.utc).isoformat()

    report = {
        "source": SOURCE_NAME,
        "language": LANGUAGE,
        "dump_filename": dump_filename,
        "dump_date": dump_date,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "processing_started_utc": started_at,
        "processing_finished_utc": finished_at,
        **stats,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)
    for k, v in stats.items():
        logger.info(f"{k:35s}: {v:,}" if isinstance(v, int) else f"{k:35s}: {v}")
    logger.info(f"Report written to: {report_path}")
    logger.info(
        "NOTE: this output is Wikipedia-specific raw text only. Feed it into "
        "the common pipeline next, e.g.:\n"
        f"  python hindi/scripts/02_normalize.py --config {args.config} "
        f"--input {output_path} --output hindi/data/intermediate/normalized/wikipedia_hi_normalized.txt"
    )


if __name__ == "__main__":
    main()
