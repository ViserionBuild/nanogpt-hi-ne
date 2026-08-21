#!/usr/bin/env python3
"""
01_download_wikipedia.py — Hindi Wikipedia `pages-articles` Dump Downloader

Downloads the official Hindi Wikipedia article-text dump
(`hiwiki-YYYYMMDD-pages-articles.xml.bz2`) from the Wikimedia dumps
mirror, for use as an additional MANUAL corpus source (CL3410's >=20%
manual-data requirement).

This script does ONLY the download. It does NOT:
  * extract/parse the XML
  * run normalization, Hindi-script filtering, quality filtering,
    deduplication, or tokenization
  * touch the IndicCorpV2 downloader or any common preprocessing stage

Those steps belong to later manual-pipeline stages: run
02_extract_wikipedia.py next (produces one flat wikipedia_hi_raw.txt file,
no separate aggregation step needed), then 02_normalize.py --source
wikipedia onward. Run this script directly — there is no orchestrating
wrapper around it and 02_extract_wikipedia.py.

Never downloads pages-meta-history / full revision-history dumps — only
the current-revision `pages-articles` dump, which is exactly what
Wikimedia labels the "recommended for most users" article-text dump.

Dump resolution:
  * If `manual.wikipedia.dump_url` is set in config to a concrete URL,
    that URL is used verbatim (and `manual.wikipedia.dump_filename` /
    `dump_date`, if given, are used for bookkeeping).
  * Otherwise ("auto"/unset), the script queries the Wikimedia dumps
    index for hiwiki, walks dump dates newest-first, and picks the most
    recent date whose `dumpstatus.json` reports the `articlesdump` job
    as `status: "done"` — i.e. the latest *completed* dump, never a
    dump still in progress.

Resumability:
  * A partially-downloaded file is resumed via HTTP Range requests.
  * A `download_metadata.json` sidecar records status ("in_progress" /
    "completed" / "failed"); a run that finds status == "completed" and
    a size/checksum-valid file on disk exits immediately without
    re-downloading. Use --overwrite to force a clean re-download.

Usage:
    python hindi/scripts/manual/01_download_wikipedia.py \\
        --config hindi/configs/hindi_data.yaml

    # Force a clean re-download:
    python hindi/scripts/manual/01_download_wikipedia.py \\
        --config hindi/configs/hindi_data.yaml --overwrite
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import yaml


DUMPS_BASE = "https://dumps.wikimedia.org"
WIKI_DB = "hiwiki"

# Wikimedia's dumps mirror rejects requests with the bare default
# "python-requests/x.y" User-Agent (HTTP 403). A descriptive UA is also
# required by Wikimedia's bot-access policy.
HTTP_HEADERS = {
    "User-Agent": (
        "LMA-IDV-HindiCorpusPipeline/1.0 "
        "(educational research; contact: viratvikash151@gmail.com)"
    )
}

# Only ever fetch the "articles" (current-revision, no history) job —
# never pages-meta-history / metahistory / full-history jobs.
ALLOWED_JOB_NAMES = ("articlesdump", "articlesmultistreamdump")
FORBIDDEN_FILENAME_SUBSTRINGS = ("meta-history", "metahistory", "pages-meta-history")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("01_download_wikipedia")


# ---------------------------------------------------------------------------
# Dump discovery (dynamic resolution of the latest COMPLETED dump)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, logger: logging.Logger, timeout: int = 30) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.debug(f"GET {url} failed: {exc}")
        return None


def list_dump_dates(logger: logging.Logger) -> list:
    """Return hiwiki dump dates (YYYYMMDD strings), newest first."""
    index_url = f"{DUMPS_BASE}/{WIKI_DB}/"
    try:
        resp = requests.get(index_url, headers=HTTP_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Could not fetch dump index {index_url}: {exc}")
        sys.exit(1)

    dates = sorted(set(re.findall(r'href="(\d{8})/"', resp.text)), reverse=True)
    if not dates:
        logger.error(f"No dump date directories found at {index_url}")
        sys.exit(1)
    return dates


def resolve_latest_completed_dump(
    logger: logging.Logger, max_dates_to_check: int = 12
) -> Tuple[str, str, str, Optional[int], Optional[str]]:
    """
    Walk hiwiki dump dates newest-first and return the first one whose
    `pages-articles.xml.bz2` job is complete.

    Returns (dump_date, filename, url, size_bytes_or_None, sha1_or_None).
    Exits with an error if no completed dump can be found.
    """
    dates = list_dump_dates(logger)

    for dump_date in dates[:max_dates_to_check]:
        status_url = f"{DUMPS_BASE}/{WIKI_DB}/{dump_date}/dumpstatus.json"
        status = _http_get_json(status_url, logger)
        if not status:
            logger.info(f"  {dump_date}: dumpstatus.json unavailable, skipping")
            continue

        jobs = status.get("jobs", {})
        for job_name in ALLOWED_JOB_NAMES:
            job = jobs.get(job_name)
            if not job or job.get("status") != "done":
                continue

            expected_name = f"{WIKI_DB}-{dump_date}-pages-articles.xml.bz2"
            files = job.get("files", {})

            file_entry = files.get(expected_name)
            if file_entry is None:
                # Some dump layouts only expose the multistream variant;
                # fall back to any single non-history pages-articles file.
                for fname, fentry in files.items():
                    if (
                        fname.endswith("pages-articles.xml.bz2")
                        and not any(bad in fname for bad in FORBIDDEN_FILENAME_SUBSTRINGS)
                    ):
                        expected_name, file_entry = fname, fentry
                        break

            if file_entry is None:
                continue

            if any(bad in expected_name for bad in FORBIDDEN_FILENAME_SUBSTRINGS):
                logger.warning(f"  Refusing history-bearing dump file: {expected_name}")
                continue

            url = f"{DUMPS_BASE}{file_entry['url']}" if file_entry["url"].startswith("/") else file_entry["url"]
            size = file_entry.get("size")
            sha1 = file_entry.get("sha1")
            logger.info(f"  Found completed dump: {dump_date} -> {expected_name}")
            return dump_date, expected_name, url, size, sha1

        logger.info(f"  {dump_date}: articlesdump job not done, skipping")

    logger.error(
        f"No completed pages-articles dump found in the last "
        f"{max_dates_to_check} hiwiki dump dates. Aborting."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Download (resumable, retried)
# ---------------------------------------------------------------------------

def sha1_of_file(path: Path, logger: logging.Logger) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def download_with_resume(
    url: str,
    output_path: Path,
    logger: logging.Logger,
    expected_size: Optional[int],
    max_retries: int = 5,
    timeout: int = 60,
) -> None:
    """
    Download *url* to *output_path*, resuming from any existing partial
    file via HTTP Range, retrying on transient errors with backoff.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        resume_from = output_path.stat().st_size if output_path.exists() else 0

        if expected_size is not None and resume_from >= expected_size:
            break  # already fully downloaded

        headers = dict(HTTP_HEADERS)
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
                if resume_from > 0 and resp.status_code == 200:
                    # Server ignored Range and is sending the whole file again.
                    logger.warning(
                        "Server does not support resume (got 200 for a Range "
                        "request); restarting download from byte 0."
                    )
                    resume_from = 0
                    mode = "wb"
                elif resume_from > 0 and resp.status_code == 206:
                    mode = "ab"
                elif resp.status_code == 200:
                    mode = "wb"
                elif resp.status_code == 416:
                    # Range not satisfiable -> local file is already complete.
                    break
                else:
                    resp.raise_for_status()
                    mode = "wb"

                total = resp.headers.get("Content-Length")
                total = int(total) + resume_from if total else expected_size

                downloaded = resume_from
                last_log = time.time()
                with open(output_path, mode) as out_f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_log > 5:
                            pct = f"{downloaded / total * 100:.1f}%" if total else "?"
                            logger.info(f"  Downloaded {downloaded:,} bytes ({pct})")
                            last_log = time.time()

            logger.info(f"  Download stream finished: {downloaded:,} bytes")
            return

        except requests.RequestException as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {wait}s (will resume from current size) …"
                )
                time.sleep(wait)
            else:
                raise

    logger.info("File already fully downloaded on disk; skipping transfer.")


def validate_download(
    output_path: Path,
    expected_size: Optional[int],
    expected_sha1: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Validate the downloaded file's size and (if available) checksum."""
    if not output_path.exists():
        logger.error(f"Expected output file missing: {output_path}")
        sys.exit(1)

    actual_size = output_path.stat().st_size
    if actual_size == 0:
        logger.error(f"Downloaded file is empty: {output_path}")
        sys.exit(1)

    if expected_size is not None and actual_size != expected_size:
        logger.error(
            f"Size mismatch: expected {expected_size:,} bytes, "
            f"got {actual_size:,} bytes. Download is incomplete/corrupt. "
            f"Re-run to resume, or use --overwrite to restart."
        )
        sys.exit(1)

    # bz2 magic number sanity check ("BZh").
    with open(output_path, "rb") as f:
        magic = f.read(3)
    if magic != b"BZh":
        logger.error(
            f"Downloaded file does not look like a bzip2 stream "
            f"(magic={magic!r}). Refusing to mark as complete."
        )
        sys.exit(1)

    checksum = None
    if expected_sha1:
        logger.info("  Verifying SHA-1 checksum …")
        checksum = sha1_of_file(output_path, logger)
        if checksum.lower() != expected_sha1.lower():
            logger.error(
                f"SHA-1 mismatch: expected {expected_sha1}, got {checksum}. "
                f"Aborting — refusing to mark a corrupt dump as complete."
            )
            sys.exit(1)
        logger.info("  SHA-1 OK.")
    else:
        checksum = sha1_of_file(output_path, logger)
        logger.info(f"  No reference checksum provided by Wikimedia; recorded SHA-1: {checksum}")

    return {"size_bytes": actual_size, "sha1": checksum}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Hindi Wikipedia pages-articles dump (manual corpus source)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to hindi_data.yaml")
    parser.add_argument("--overwrite", action="store_true",
                         help="Discard any existing partial/complete download and re-download")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    repo_root = Path(args.config).resolve().parent.parent.parent

    wiki_cfg = config.get("manual", {}).get("wikipedia", {})
    if not wiki_cfg.get("enabled", True):
        logger.info("manual.wikipedia.enabled is false in config; nothing to do.")
        return

    resume = wiki_cfg.get("resume", True)

    output_dir_cfg = wiki_cfg.get("output_dir", "hindi/data/raw/manual/wikipedia")
    output_dir = repo_root / output_dir_cfg
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "download_metadata.json"

    logger.info("=" * 70)
    logger.info("Hindi Wikipedia pages-articles Dump Downloader")
    logger.info("=" * 70)

    # ---- Resolve dump URL/date/filename ---------------------------------
    configured_url = wiki_cfg.get("dump_url")
    expected_size = None
    expected_sha1 = None

    if configured_url and configured_url.lower() != "auto":
        dump_url = configured_url
        dump_date = wiki_cfg.get("dump_date", "unknown")
        filename = wiki_cfg.get("dump_filename") or Path(dump_url).name
        if any(bad in filename for bad in FORBIDDEN_FILENAME_SUBSTRINGS):
            logger.error(
                f"Configured dump_filename/url looks like a revision-history "
                f"dump ({filename}). This script only downloads pages-articles "
                f"dumps. Aborting."
            )
            sys.exit(1)
        logger.info(f"Using configured dump URL: {dump_url}")
    else:
        logger.info("Resolving latest COMPLETED hiwiki pages-articles dump …")
        dump_date, filename, dump_url, expected_size, expected_sha1 = resolve_latest_completed_dump(logger)
        override_filename = wiki_cfg.get("dump_filename")
        if override_filename:
            filename = override_filename

    output_path = output_dir / filename

    logger.info(f"Dump date:           {dump_date}")
    logger.info(f"Filename:            {filename}")
    logger.info(f"URL:                 {dump_url}")
    logger.info(f"Output path:         {output_path}")
    logger.info(f"Expected size:       {expected_size if expected_size else 'unknown'}")
    logger.info(f"Expected SHA-1:      {expected_sha1 if expected_sha1 else 'unknown'}")

    # ---- Overwrite handling ----------------------------------------------
    if args.overwrite:
        for p in (output_path, metadata_path):
            if p.exists():
                p.unlink()
                logger.info(f"Removed existing: {p}")

    # ---- Safe re-run: skip if already validated complete ------------------
    existing_meta = load_metadata(metadata_path)
    if (
        not args.overwrite
        and existing_meta
        and existing_meta.get("status") == "completed"
        and existing_meta.get("filename") == filename
        and output_path.exists()
    ):
        on_disk_size = output_path.stat().st_size
        if existing_meta.get("downloaded_size_bytes") == on_disk_size:
            logger.info(
                f"Dump already downloaded and validated complete: {output_path} "
                f"({on_disk_size:,} bytes). Use --overwrite to re-download."
            )
            return
        logger.warning(
            "Metadata says completed but on-disk size differs "
            "(file changed externally); re-validating."
        )

    if not resume and output_path.exists() and not args.overwrite:
        logger.error(
            f"{output_path} already exists and manual.wikipedia.resume is "
            f"false. Use --overwrite to redo, or enable resume."
        )
        sys.exit(1)

    # ---- Download -----------------------------------------------------------
    save_metadata(metadata_path, {
        "source_url": dump_url,
        "dump_date": dump_date,
        "filename": filename,
        "downloaded_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "expected_size_bytes": expected_size,
        "sha1": None,
        "expected_sha1": expected_sha1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
    })

    try:
        download_with_resume(dump_url, output_path, logger, expected_size)
    except requests.RequestException as exc:
        logger.error(f"Download failed: {exc}")
        save_metadata(metadata_path, {
            "source_url": dump_url,
            "dump_date": dump_date,
            "filename": filename,
            "downloaded_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "expected_size_bytes": expected_size,
            "sha1": None,
            "expected_sha1": expected_sha1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
        })
        sys.exit(1)

    # ---- Validate -------------------------------------------------------
    logger.info("Validating downloaded dump …")
    stats = validate_download(output_path, expected_size, expected_sha1, logger)

    metadata = {
        "source_url": dump_url,
        "dump_date": dump_date,
        "filename": filename,
        "downloaded_size_bytes": stats["size_bytes"],
        "expected_size_bytes": expected_size,
        "sha1": stats["sha1"],
        "expected_sha1": expected_sha1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
    }
    save_metadata(metadata_path, metadata)

    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output:   {output_path}")
    logger.info(f"Size:     {stats['size_bytes']:,} bytes")
    logger.info(f"SHA-1:    {stats['sha1']}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(
        "NOTE: this dump is raw compressed XML. Extraction/normalization/"
        "filtering/dedup/tokenization happen in later manual-pipeline stages, "
        "not here."
    )


if __name__ == "__main__":
    main()
