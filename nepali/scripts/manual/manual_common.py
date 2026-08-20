"""
manual_common.py — Shared interface for Nepali MANUAL-corpus collectors.

This is a LIBRARY, not a runnable script. It is imported by every
nepali/scripts/manual/collect_*.py collector so they share one metadata
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
    01_download_indiccorp.py and 01b_download_manual.py
  * explicitly-selected sources only — no crawling/discovery logic lives
    here; each collect_*.py supplies its own hand-curated SOURCES list
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

# The manual-corpus subcategory (see nepali_data.yaml `manual:` block).
# News is a subcategory of the single "manual" top-level corpus source —
# never a separate top-level source (the other top-level source is
# "indiccorp", handled entirely by 01_download_indiccorp.py).
#
# NOTE: government/textbooks/parliament/other_official were explored and
# implemented in an earlier iteration of this project but are OUT OF
# SCOPE — their collectors were moved to
# nepali/scripts/manual/_deprecated_out_of_scope/ and are not part of
# the active pipeline. See that directory's README for why.
#
# Wikipedia (collect_wikipedia.py) and Setopati were REMOVED 2026-08-20 —
# the manual corpus is now Onlinekhabar + Khabarhub news only (both
# open, unauthenticated WordPress REST API sources — see
# collect_news.py's module docstring). Wikipedia's collector script and
# already-downloaded dump/extracted text were deleted outright (not just
# disabled) since they contributed a small fraction of manual tokens
# relative to news volume; Setopati was dropped because its discoverable-
# URL pool (homepage/category-page crawling, no article sitemap/API) was
# nearly exhausted at ~256 docs.
SOURCE_TYPES = ("news",)

# Within source_type="news", which named outlet a document came from.
# Kept as a separate, explicit tuple (rather than open-ended free text)
# so downstream token-accounting-by-source stays a closed, checkable set.
# Each is a genuinely distinct source_name with its own shard file
# sequence (onlinekhabar_*.jsonl / khabarhub_*.jsonl) — see
# collect_news.py.
NEWS_SOURCE_NAMES = ("onlinekhabar", "khabarhub")

LANGUAGE_CODE = "ne"


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
    collect_*.py collector (wikipedia, government, textbooks, parliament,
    other_official) writes this SAME schema, so downstream code can read
    every manual/<category>/*.jsonl file identically regardless of
    source_type.

    license_or_access_basis must state EVIDENCE (a terms-of-use page, an
    explicit license notice, an official publication basis) — never
    "public domain" or "openly licensed" guessed without a source backing
    it. Collectors that cannot state this should not record the document.

    extraction_status must be one of "success" / "partial" / "failed" /
    "needs_ocr" — a failed or needs-OCR extraction is still RECORDED
    (never silently dropped, and never silently reported as empty text);
    only its status and (if available) error detail differ. "needs_ocr"
    covers BOTH image-only/scanned PDFs (zero extractable text) AND PDFs
    that DO yield non-empty text but at an implausibly low Devanagari
    ratio for Nepali content — the latter is a real, observed failure
    mode: some official Nepal government PDFs (e.g. older Curriculum
    Development Centre textbooks) embed text using legacy non-Unicode
    Devanagari fonts (e.g. Preeti), so naive PDF text extraction returns
    readable-looking but WRONG characters (glyph codepoints, not actual
    Devanagari) — see extract_pdf_text().

    publisher/grade/subject/publication_year are optional — used by
    textbook records, left "" by collectors (wikipedia, government, ...)
    where they don't apply, rather than forking the schema per collector.

    session/date/house are optional — used by parliament records (session
    = e.g. "Joint Sitting Session 2079" / "House of Representatives 11th
    Session"; house = "House of Representatives" / "National Assembly" /
    "Joint Sitting"; date = the BS date/date-range printed on the
    document where available), left "" for other source_types.
    """
    data_source: str            # always "manual"
    source_type: str            # one of SOURCE_TYPES
    title: str
    source_url: str
    language: str                # "ne"
    document_type: str            # e.g. "pdf", "html", "wikimedia_dump"
    collection_date: str          # ISO date (UTC)
    original_filename: str
    local_filename: str
    license_or_access_basis: str
    extraction_method: str
    extraction_status: str        # "success" / "partial" / "failed" / "needs_ocr"
    source_name: str = ""          # "onlinekhabar" / "khabarhub" — the specific
                                    # named outlet/site, distinct from the broader
                                    # source_type ("news"). Used for per-source
                                    # token accounting downstream.
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
    downloaders (01_download_indiccorp.py, 01b_download_manual.py)."""
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, required=True, help="Path to nepali_data.yaml")
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

    *headers* lets a caller pass a site-specific User-Agent — e.g.
    Wikimedia's dumps mirror rejects the bare default
    "python-requests/x.y" UA with HTTP 403 (see 01b_download_manual.py's
    HTTP_HEADERS), so a probe against it must reuse that same header or
    it will misreport "unreachable".

    *verify* off is NEVER the default — see download_stream's docstring
    for when/how a collector may opt into it explicitly. A TLS failure
    here is reported via info["ssl_error"] rather than raising, so a
    smoke test can report it clearly instead of crashing.
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


# ---------------------------------------------------------------------------
# Single-document collection (government / textbooks / parliament /
# other_official — anything that is "download one file, maybe extract
# text, record metadata", unlike wikipedia's dump-based pipeline)
# ---------------------------------------------------------------------------

class DownloadError(RuntimeError):
    """Raised when a single-document download fails outright (e.g. a bad
    URL, non-2xx status, or repeated network errors). Callers must catch
    this and record the failure loudly — never let it disappear a source
    from the run's output silently."""


REQUIRED_SOURCE_FIELDS = ("source_url", "title", "license_or_access_basis")


def validate_source_entry(entry: Dict[str, Any], index: int) -> None:
    """Fail loudly if a curated SOURCES entry is missing a required field
    — an incomplete manifest entry must never silently produce an
    incomplete/misleading metadata record."""
    missing = [f for f in REQUIRED_SOURCE_FIELDS if not entry.get(f)]
    if missing:
        raise ValueError(
            f"SOURCES[{index}] is missing required field(s) {missing}: {entry!r}"
        )


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

    *verify* — LEAVE THIS True UNLESS you have a specific, documented
    reason not to (e.g. a government site with a confirmed-legitimate CA
    certificate but an incomplete chain — a server misconfiguration, not
    a security compromise). Never flip this on to silently work around an
    error; a caller doing so should make it an explicit, visible opt-in
    (e.g. a config flag defaulting to False) — see collect_parliament.py.
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
            # Retrying won't fix a certificate problem — fail immediately
            # with a message pointing at the explicit opt-in, instead of
            # burning through max_retries with backoff for nothing.
            raise DownloadError(
                f"TLS verification failed fetching {url}: {exc}. If this "
                f"is a known/documented server misconfiguration (e.g. an "
                f"incomplete certificate chain from a legitimate CA), a "
                f"collector may opt into verify=False explicitly and "
                f"visibly (see collect_parliament.py's "
                f"allow_insecure_ssl config flag) — never silently."
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


def extract_pdf_text(pdf_path: Path, logger: logging.Logger) -> Tuple[Optional[str], str]:
    """
    Extract text from a PDF using pypdf. Returns (text_or_None, status)
    where status is one of:

      "success" — every page yielded text, and the extracted text is
                   plausibly real Devanagari content.
      "partial" — some pages failed/came back empty but at least one
                   succeeded, and overall Devanagari content looks real.
      "needs_ocr" — the PDF opened fine but no reliable text could be
                   recovered: EITHER every page came back with zero
                   extractable text (a scanned/image-only PDF), OR text
                   WAS extracted but its Devanagari-character ratio is
                   implausibly low for Nepali content. The second case is
                   a real, observed failure mode: some official Nepal
                   government PDFs (e.g. older Curriculum Development
                   Centre textbooks, confirmed during source inspection)
                   embed text using a legacy non-Unicode Devanagari font
                   (e.g. Preeti) — pypdf then returns non-empty but
                   GARBLED text (raw glyph codepoints, not actual
                   Devanagari characters). Both cases need OCR (or a
                   font-specific remapping tool, out of scope here) to
                   recover real text — never silently accepted as good
                   extraction, and never silently treated as empty.
      "failed" — pypdf is unavailable, or the file could not be opened at
                   all (corrupt/unreadable PDF).

    A single page/document worth of PDF content is bounded and small
    relative to the corpus as a whole — pypdf necessarily parses one PDF
    into memory to extract it, same as any PDF library; this is NOT the
    "never load the whole corpus into RAM" concern, which is about
    streaming the raw bytes to disk during DOWNLOAD (see download_stream)
    and never materializing the full multi-GB corpus at once.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning(
            "pypdf not installed (pip install pypdf); cannot extract PDF "
            "text. The raw PDF is still saved; extraction_status='failed'."
        )
        return None, "failed"

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        logger.error(f"Could not open PDF {pdf_path}: {exc}")
        return None, "failed"

    pages_text: List[str] = []
    pages_failed = 0
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(f"  Failed to extract page {i} of {pdf_path.name}: {exc}")
            text = ""
        if text.strip():
            pages_text.append(text)
        else:
            pages_failed += 1

    if not pages_text:
        # Zero pages yielded any text at all — the classic scanned/
        # image-only PDF signature.
        return None, "needs_ocr"

    combined = "\n".join(pages_text)
    dev_ratio, nonspace_count = devanagari_ratio(combined)

    if nonspace_count < MIN_CHARS_FOR_DEVANAGARI_CHECK or dev_ratio < MIN_DEVANAGARI_RATIO:
        logger.warning(
            f"  {pdf_path.name}: extracted {nonspace_count:,} non-space "
            f"chars but only {dev_ratio:.1%} are Devanagari — likely a "
            f"legacy-font-encoded or non-Nepali PDF. Flagging needs_ocr "
            f"rather than accepting as real Nepali text."
        )
        return combined, "needs_ocr"

    status = "success" if pages_failed == 0 else "partial"
    return combined, status

