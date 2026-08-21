"""
manual_common.py — Shared interface for Hindi MANUAL-corpus collectors.

This is a LIBRARY, not a runnable script. It is imported by every
hindi/scripts/manual/collect_*.py collector so they share one metadata
schema and one set of CLI/testing conventions, instead of each
reinventing them:

  * one JSONL metadata record per collected document/source, in the
    schema below (data_source/source_type/title/source_url/language/
    collection_date/original_filename/local_filename/
    license_or_access_basis/extraction_method)
  * "never silently overwrite": a source already recorded (by
    source_url) in the metadata JSONL is skipped unless the caller asks
    to redo it
  * --dry-run (resolve config/paths/state, no network, no files written)
    and --smoke-test (connectivity probe only, never a full download) —
    the same two lightweight-testing conventions used by
    01_download_indiccorp.py
  * explicitly-selected sources only — no crawling/discovery logic lives
    here; each collect_*.py supplies its own hand-curated SOURCES list
    (or, for 03_collect_news.py, per-source sitemap URLs from config)

Not used by 01_download_wikipedia.py / 02_extract_wikipedia.py — those two are
plain standalone scripts (download the dump, then extract it) with their
own argparse/config loading, run directly one after another with no
orchestrating wrapper and no shared collector metadata schema.

Adapted from nepali/scripts/manual/manual_common.py — same architecture,
Hindi-specific constants (LANGUAGE_CODE, NEWS_SOURCE_NAMES) only.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import yaml

# The TWO MANUAL subcategories (see hindi_data.yaml `manual:` block).
# Wikipedia and news are subcategories of the single "manual" top-level
# corpus source — never separate top-level sources (the other top-level
# source is "indiccorp", handled entirely by 01_download_indiccorp.py).
SOURCE_TYPES = ("wikipedia", "news")

# Within source_type="news", which named outlet a document came from.
# Kept as a separate, explicit tuple (rather than open-ended free text)
# so downstream token-accounting-by-source stays a closed, checkable set.
# This includes sources currently DISABLED in hindi_data.yaml's
# manual.news block (see that file for why each is disabled) — the set
# is closed over every source this project has evaluated, not just the
# ones presently enabled, so enabling one later needs no code change here.
NEWS_SOURCE_NAMES = (
    "dainik_jagran", "hindustan", "abp",
    "amar_ujala", "dainik_bhaskar", "navbharat_times", "aaj_tak",
    "ndtv", "jansatta", "prabhat_khabar",
)

LANGUAGE_CODE = "hi"


# ---------------------------------------------------------------------------
# Config / logging
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Configure and return a logger, matching the numbered pipeline's format."""
    lcfg = (config or {}).get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger(name)


def collection_date_now() -> str:
    """UTC calendar date (YYYY-MM-DD) this collection run happened on."""
    return datetime.now(timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    """One record per collected manual-corpus document/source. Every
    collect_*.py collector (wikipedia, news) writes this SAME schema, so
    downstream code can read every manual/<category>/*.jsonl file
    identically regardless of source_type.

    license_or_access_basis must state EVIDENCE (a terms-of-use page, an
    explicit robots.txt check, an official publication basis) — never
    "public domain" or "openly licensed" guessed without a source backing
    it. Collectors that cannot state this should not record the document.

    extraction_status must be one of "success" / "partial" / "failed" /
    "needs_ocr" — a failed or needs-OCR extraction is still RECORDED
    (never silently dropped, and never silently reported as empty text);
    only its status and (if available) error detail differ.
    """
    data_source: str            # always "manual"
    source_type: str            # one of SOURCE_TYPES
    title: str
    source_url: str
    language: str                # "hi"
    document_type: str            # e.g. "html_article", "wikimedia_dump"
    collection_date: str          # ISO date (UTC)
    original_filename: str
    local_filename: str
    license_or_access_basis: str
    extraction_method: str
    extraction_status: str        # "success" / "partial" / "failed" / "needs_ocr"
    source_name: str = ""          # e.g. "dainik_jagran" / "hindustan" / "wikipedia" —
                                    # the specific named outlet/site, distinct from
                                    # the broader source_type ("news"/"wikipedia").
                                    # Used for per-source token accounting downstream.
    publisher: str = ""
    grade: str = ""
    subject: str = ""
    publication_year: str = ""
    session: str = ""
    date: str = ""
    house: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# extraction_status values that mean a source_url should NOT be
# retried on a normal (non --overwrite) run. Deliberately excludes
# "failed": a failed download/extraction must be retried next run, not
# skipped forever just because a metadata record exists for its URL —
# see MetadataStore.completed_urls().
COMPLETED_STATUSES = ("success", "partial", "needs_ocr")


class MetadataStore:
    """
    Append-only JSONL metadata sidecar, one line per DocumentMetadata
    record. Never silently overwrites: existing_urls()/completed_urls()
    let a caller skip a source_url that already has a record, matching
    the "never silently overwrite" requirement without needing a full
    read-modify-write of the file for every new record.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def _latest_status_by_url(self) -> Dict[str, Optional[str]]:
        """Map source_url -> its CURRENT extraction_status. upsert()
        guarantees at most one record per source_url, so this is only
        ambiguous if the file was ever written via append() or hand-
        edited into having duplicates — handled defensively by taking
        the LAST occurrence per URL (upsert's rewrite-then-append always
        places the current record last for a given URL)."""
        latest: Dict[str, Optional[str]] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = rec.get("source_url")
                    if url:
                        latest[url] = rec.get("extraction_status")
        return latest

    def existing_urls(self) -> Set[str]:
        """URLs that have ANY metadata record, regardless of status —
        including "failed". Do NOT use this to decide whether to skip a
        source on a normal run (a "failed" URL must be retried); use
        completed_urls() for that instead."""
        return set(self._latest_status_by_url().keys())

    def completed_urls(self) -> Set[str]:
        """URLs whose CURRENT metadata record has extraction_status in
        COMPLETED_STATUSES (success/partial/needs_ocr) — i.e. sources
        that should be SKIPPED on a normal (non --overwrite) run.

        A "failed" status is deliberately excluded: that source must be
        retried on the next normal run, not skipped just because a
        (failed) metadata record exists for its URL. This is what
        collectors should check before deciding to (re)download a
        source.
        """
        return {
            url for url, status in self._latest_status_by_url().items()
            if status in COMPLETED_STATUSES
        }

    def append(self, record: DocumentMetadata) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def upsert(self, record: DocumentMetadata) -> None:
        """Insert *record*, replacing any existing record for the same
        source_url. Safe to call whether or not the URL was already
        present — an --overwrite re-collection never leaves two records
        for the same source_url behind."""
        kept_lines: List[str] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        rec = json.loads(stripped)
                    except json.JSONDecodeError:
                        kept_lines.append(stripped)
                        continue
                    if rec.get("source_url") != record.source_url:
                        kept_lines.append(stripped)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# CLI scaffolding
# ---------------------------------------------------------------------------

def build_arg_parser(description: str):
    """Argparse parser shared by every collect_*.py: --config plus the
    three lightweight-testing/idempotency flags used across this project's
    downloaders (01_download_indiccorp.py)."""
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, required=True, help="Path to hindi_data.yaml")
    p.add_argument("--dry-run", action="store_true",
                    help="Resolve config/paths/state and report what a real run "
                         "would do. No network calls, no files written.")
    p.add_argument("--smoke-test", action="store_true",
                    help="Probe source(s) for reachability only. Does NOT "
                         "download the full document/dump.")
    p.add_argument("--overwrite", action="store_true",
                    help="Re-collect even if already recorded in metadata.")
    return p


# ---------------------------------------------------------------------------
# Lightweight connectivity probing (never a full download)
# ---------------------------------------------------------------------------

def probe_url(
    url: str,
    logger: logging.Logger,
    timeout: int = 20,
    headers: Optional[Dict[str, str]] = None,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    HEAD-probe *url* (falling back to a 1KB ranged GET if HEAD is refused
    or uninformative). Never downloads the full resource — used by every
    collector's --smoke-test.

    *verify* off is NEVER the default. A TLS failure here is reported via
    info["ssl_error"] rather than raising, so a smoke test can report it
    clearly instead of crashing.
    """
    info: Dict[str, Any] = {"url": url, "reachable": False}
    req_headers = headers or {}
    try:
        resp = requests.head(url, headers=req_headers, allow_redirects=True,
                              timeout=timeout, verify=verify)
        info.update({
            "final_url": resp.url,
            "status_code": resp.status_code,
            "content_length": resp.headers.get("Content-Length"),
            "content_type": resp.headers.get("Content-Type"),
            "accept_ranges": resp.headers.get("Accept-Ranges"),
            "reachable": resp.status_code < 400,
        })
        if resp.status_code >= 400 or not resp.headers.get("Content-Length"):
            # Some servers reject/under-report HEAD; a tiny ranged GET is
            # still far cheaper than a full download.
            range_headers = dict(req_headers)
            range_headers["Range"] = "bytes=0-1023"
            r2 = requests.get(url, headers=range_headers, stream=True,
                               timeout=timeout, verify=verify)
            info["status_code"] = r2.status_code
            info["final_url"] = r2.url
            info["content_type"] = r2.headers.get("Content-Type", info.get("content_type"))
            info["reachable"] = r2.status_code < 400
            r2.close()
    except requests.exceptions.SSLError as exc:
        info["error"] = str(exc)
        info["ssl_error"] = True
        logger.warning(f"TLS verification failed probing {url}: {exc}")
    except requests.RequestException as exc:
        info["error"] = str(exc)
        logger.warning(f"Probe failed for {url}: {exc}")
    return info


class DownloadError(RuntimeError):
    """Raised when a single-document download fails outright (e.g. a bad
    URL, non-2xx status, or repeated network errors). Callers must catch
    this and record the failure loudly — never let it disappear a source
    from the run's output silently."""


def download_stream(
    url: str,
    output_path: Path,
    logger: logging.Logger,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    max_retries: int = 5,
    verify: bool = True,
) -> int:
    """
    Stream *url* to *output_path*, resuming via HTTP Range from any
    partial file already on disk, retrying transient errors with
    backoff. Never holds the full response body in memory. Returns the
    final size on disk.

    Raises DownloadError if the resource cannot be fetched at all (bad
    status, exhausted retries, or a TLS verification failure) — the raw
    file is preserved as-is (never silently truncated/deleted) so a
    caller can inspect what was downloaded so far.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_headers = dict(headers or {})

    for attempt in range(1, max_retries + 1):
        resume_from = output_path.stat().st_size if output_path.exists() else 0
        req_headers = dict(base_headers)
        if resume_from:
            req_headers["Range"] = f"bytes={resume_from}-"
        try:
            with requests.get(url, headers=req_headers, stream=True,
                               timeout=timeout, verify=verify) as resp:
                if resume_from and resp.status_code == 200:
                    logger.warning(
                        f"Server ignored Range for {url}; restarting from byte 0."
                    )
                    resume_from = 0
                    mode = "wb"
                elif resume_from and resp.status_code == 206:
                    mode = "ab"
                elif resp.status_code == 200:
                    mode = "wb"
                elif resp.status_code == 416:
                    break  # already fully downloaded
                elif resp.status_code >= 400:
                    raise DownloadError(
                        f"HTTP {resp.status_code} fetching {url}"
                    )
                else:
                    mode = "wb"

                with open(output_path, mode) as out_f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            out_f.write(chunk)
            return output_path.stat().st_size

        except DownloadError:
            raise
        except requests.exceptions.SSLError as exc:
            raise DownloadError(
                f"TLS verification failed fetching {url}: {exc}. If this "
                f"is a known/documented server misconfiguration, a "
                f"collector may opt into verify=False explicitly and "
                f"visibly — never silently."
            ) from exc
        except requests.RequestException as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt}/{max_retries} downloading {url} "
                    f"failed: {exc}. Retrying in {wait}s …"
                )
                time.sleep(wait)
            else:
                raise DownloadError(f"Failed to download {url}: {exc}") from exc

    return output_path.stat().st_size if output_path.exists() else 0


MIN_CHARS_FOR_DEVANAGARI_CHECK = 200
MIN_DEVANAGARI_RATIO = 0.05


def devanagari_ratio(text: str) -> Tuple[float, int]:
    """Return (devanagari_ratio, non_whitespace_char_count) over *text*."""
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return 0.0, 0
    dev = sum(1 for c in nonspace if 0x0900 <= ord(c) <= 0x097F)
    return dev / len(nonspace), len(nonspace)
