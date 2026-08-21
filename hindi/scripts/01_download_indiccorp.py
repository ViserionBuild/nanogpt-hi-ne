#!/usr/bin/env python3
"""
01_download_indiccorp.py — Partial Range-Based IndicCorpV2 Hindi Downloader

Downloads a partial candidate corpus from AI4Bharat IndicCorpV2 Hindi
files (hi-1.txt, hi-2.txt, hi-3.txt, ...) using HTTP Range requests.
Only enough data is downloaded to reach the configured word target;
additional files are used only if earlier files are exhausted.

Stopping criterion (whichever comes first):
  • the configured word target (default 500,000,000 whitespace words ±
    tolerance), OR
  • all available source files are exhausted.

Key design decisions:
  • Downloads only enough data to reach the target; chunks are fetched in
    a deterministic shuffled order per file so the sample is representative
    and reproducible.
  • Preserves complete sentence boundaries (one sentence per line); the
    output is a *candidate* corpus — NO preprocessing happens here.
  • Resumable via a JSON state file; never restarts from zero.
  • Strictly verifies HTTP 206 Partial Content + Content-Range so a server
    that ignores Range requests can never silently corrupt the local file.
  • Counts *whitespace words* (text.split()), NOT BPE tokens.  The actual
    token count is measured later, after the tokenizer is trained.

Usage:
    python hindi/scripts/01_download_indiccorp.py \\
        --config hindi/configs/hindi_data.yaml

    # Test with small target:
    python hindi/scripts/01_download_indiccorp.py \\
        --config hindi/configs/hindi_data.yaml \\
        --target-words 1000000 --chunk-size-bytes 16777216

    # Resume interrupted download:
    python hindi/scripts/01_download_indiccorp.py \\
        --config hindi/configs/hindi_data.yaml --resume

    # Force a clean restart:
    python hindi/scripts/01_download_indiccorp.py \\
        --config hindi/configs/hindi_data.yaml --overwrite
"""

import argparse
import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configure and return a logger."""
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("01_download_indiccorp")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def probe_remote_file(
    url: str, logger: logging.Logger
) -> Tuple[int, bool, str]:
    """
    Determine remote file size and whether the server supports byte ranges.

    Returns (remote_size_bytes, supports_ranges, final_url).
    Falls back to a small Range GET if HEAD is insufficient or forbidden.

    Fails safely (sys.exit) if the size or range support cannot be
    determined — partial downloading must never silently become a full
    download of the entire file.
    """
    final_url = url

    # --- Attempt HEAD first ---------------------------------------------------
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30)
        final_url = resp.url
        logger.info(f"HEAD status:        {resp.status_code}")
        logger.info(f"Original URL:       {url}")
        logger.info(f"Final URL:          {final_url}")
        logger.info(f"Content-Length:     {resp.headers.get('Content-Length', 'n/a')}")
        logger.info(f"Accept-Ranges:      {resp.headers.get('Accept-Ranges', 'n/a')}")

        if resp.status_code == 200:
            cl = resp.headers.get("Content-Length")
            ar = resp.headers.get("Accept-Ranges", "").lower()
            if cl and ar == "bytes":
                return int(cl), True, final_url
            if cl and int(cl) > 0:
                # Accept-Ranges not explicit — verify with a test range request
                pass
    except requests.RequestException as exc:
        logger.warning(f"HEAD request failed ({exc}); trying Range GET probe …")

    # --- Probe with a tiny Range GET ------------------------------------------
    try:
        probe_headers = {"Range": "bytes=0-1023"}
        resp = requests.get(
            url, headers=probe_headers, stream=True,
            allow_redirects=True, timeout=30,
        )
        final_url = resp.url
        logger.info(f"Range probe status: {resp.status_code}")

        if resp.status_code == 206:
            cr = resp.headers.get("Content-Range", "")
            logger.info(f"Content-Range:      {cr}")
            # Content-Range: bytes 0-1023/79983290646
            if "/" in cr:
                total = cr.split("/")[-1].strip()
                if total != "*":
                    return int(total), True, final_url
        elif resp.status_code == 200:
            # Server ignored the Range header — no partial support
            cl = resp.headers.get("Content-Length")
            if cl:
                return int(cl), False, final_url
        elif resp.status_code in (403, 401, 404, 405):
            logger.error(
                f"Server returned HTTP {resp.status_code} "
                f"for {final_url} — access is denied or the object is missing.  "
                f"Cannot determine file size / range support.  Aborting safely."
            )
            sys.exit(1)
    except requests.RequestException as exc:
        logger.error(f"Range probe also failed: {exc}")

    logger.error(
        "Could not determine remote file size or Range support.  "
        "Aborting — a partial download is not possible without a "
        "range-capable server, and the downloader will not silently "
        "fetch the entire file."
    )
    sys.exit(1)


def _parse_content_range(cr: str) -> Optional[Tuple[int, int, int]]:
    """Parse 'bytes START-END/TOTAL' -> (start, end, total). None if invalid."""
    if not cr or not cr.lower().startswith("bytes "):
        return None
    try:
        range_part, _, total_part = cr.partition("/")
        start_str, _, end_str = range_part[6:].partition("-")
        return int(start_str), int(end_str), int(total_part)
    except (ValueError, IndexError):
        return None


def download_range(
    url: str,
    start: int,
    end: int,
    logger: logging.Logger,
    timeout: int = 120,
    max_retries: int = 3,
) -> bytes:
    """
    Download bytes [start, end] from *url*, streaming with a progress bar.

    Critical safety:
      • Raises if the server returns 200 OK instead of 206 Partial Content.
      • Verifies Content-Range and raises if the returned range does not
        match the requested range — prevents appending the whole remote
        file onto a partial local download.

    Returns the raw chunk bytes.
    """
    headers = {"Range": f"bytes={start}-{end}"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url, headers=headers, stream=True,
                allow_redirects=True, timeout=timeout,
            )

            if resp.status_code == 200:
                raise RuntimeError(
                    f"Server returned 200 OK instead of 206 Partial Content "
                    f"for Range bytes={start}-{end}.  The server does not "
                    f"honor byte-range requests.  Aborting to prevent "
                    f"appending the entire remote file to the partial "
                    f"local download."
                )

            if resp.status_code != 206:
                raise RuntimeError(
                    f"Unexpected status {resp.status_code} for Range "
                    f"bytes={start}-{end}"
                )

            # Strict Content-Range verification
            cr = resp.headers.get("Content-Range", "")
            parsed = _parse_content_range(cr)
            if parsed is None:
                raise RuntimeError(
                    f"Missing/invalid Content-Range header for Range "
                    f"bytes={start}-{end}: got {cr!r}.  Refusing to use "
                    f"unverifiable data."
                )
            cr_start, cr_end, cr_total = parsed
            expected_len = end - start + 1
            if cr_start != start:
                raise RuntimeError(
                    f"Content-Range start mismatch: requested start={start}, "
                    f"server returned start={cr_start} (header {cr!r}). "
                    f"Aborting to avoid corrupting the local file."
                )
            if cr_end != end:
                raise RuntimeError(
                    f"Content-Range end mismatch: requested end={end}, "
                    f"server returned end={cr_end} (header {cr!r}). "
                    f"Aborting to avoid corrupting the local file."
                )
            if cr_total <= 0 or cr_total < expected_len:
                raise RuntimeError(
                    f"Content-Range total looks wrong: {cr!r}. Aborting."
                )

            # Stream the body with a progress bar
            total_bytes = end - start + 1
            buf = bytearray()
            with tqdm(
                total=total_bytes, unit="B", unit_scale=True,
                desc=f"  chunk {start:,}-{end:,}", leave=False,
                ncols=100,
            ) as bar:
                for data in resp.iter_content(chunk_size=1 << 20):
                    buf.extend(data)
                    bar.update(len(data))

            if len(buf) != expected_len:
                raise RuntimeError(
                    f"Downloaded {len(buf)} bytes but expected {expected_len} "
                    f"for Range bytes={start}-{end}. Aborting."
                )
            return bytes(buf)

        except (requests.RequestException, RuntimeError) as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed: {exc}  "
                    f"Retrying in {wait}s …"
                )
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Chunk / sentence extraction
# ---------------------------------------------------------------------------

def extract_complete_sentences(
    raw_bytes: bytes,
    chunk_start: int,
    chunk_end: int,
    remote_size: int,
) -> List[str]:
    """
    Decode a byte range into complete sentences (one per line).

    Discards the first partial line (unless chunk starts at byte 0)
    and the last partial line (unless chunk ends at the last byte).
    Every sentence returned is complete — chunks never cut a sentence.
    """
    # errors="replace": a range boundary may split a multi-byte UTF-8
    # character; the partial line containing it is discarded anyway.
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.split("\n")

    # Drop first partial line if not at file start
    if chunk_start > 0 and lines:
        lines = lines[1:]

    # Drop last partial line if not at file end
    if chunk_end < remote_size - 1 and lines and lines[-1] != "":
        lines = lines[:-1]

    # Filter empty strings
    return [l for l in lines if l.strip()]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state(path: str) -> Optional[Dict[str, Any]]:
    """Load selection state from JSON, or return None if not found."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(path: str, state: Dict[str, Any]) -> None:
    """Atomically save state to JSON (write-then-rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Streaming validation
# ---------------------------------------------------------------------------

def validate_output(path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Stream through the output file and compute summary statistics.

    Runs strict UTF-8 validation line-by-line (never loads the whole file
    into RAM) and reports malformed lines explicitly.
    """
    sentences = 0
    words = 0
    chars = 0
    min_wc = float("inf")
    max_wc = 0
    malformed = 0
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for raw in f:
            sha.update(raw)
            try:
                line = raw.decode("utf-8", errors="strict").rstrip("\n")
            except UnicodeDecodeError:
                malformed += 1
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line.strip():
                continue
            sentences += 1
            wc = len(line.split())
            words += wc
            chars += len(line)
            if wc < min_wc:
                min_wc = wc
            if wc > max_wc:
                max_wc = wc

    if min_wc == float("inf"):
        min_wc = 0

    return {
        "sentences": sentences,
        "whitespace_words": words,
        "characters": chars,
        "avg_words_per_sentence": round(words / max(sentences, 1), 2),
        "min_sentence_words": min_wc,
        "max_sentence_words": max_wc,
        "malformed_lines": malformed,
        "sha256": sha.hexdigest(),
        "file_size_bytes": os.path.getsize(path),
    }


# ---------------------------------------------------------------------------
# State save helpers
# ---------------------------------------------------------------------------

def _build_state(
    source_urls: List[str],
    current_file_index: int,
    current_file_ranges_completed: List[int],
    target_words: int,
    tolerance: int,
    chunk_size: int,
    seed: int,
    words_collected: int,
    sentences_collected: int,
    bytes_downloaded: int,
    output_bytes: int,
    status: str,
) -> Dict[str, Any]:
    return {
        "source_urls": source_urls,
        "current_file_index": current_file_index,
        "current_file_ranges_completed": current_file_ranges_completed,
        "target_words": target_words,
        "tolerance_words": tolerance,
        "chunk_size_bytes": chunk_size,
        "random_seed": seed,
        "words_collected": words_collected,
        "sentences_collected": sentences_collected,
        "bytes_downloaded": bytes_downloaded,
        "output_bytes": output_bytes,
        "status": status,
        # Provenance (CL3410 requirement 10): wall-clock time of this save.
        # When status == "completed" this is the download completion time.
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_state_inline(
    path: str,
    source_urls: List[str],
    current_file_index: int,
    current_file_ranges_completed: List[int],
    target_words: int,
    tolerance: int,
    chunk_size: int,
    seed: int,
    words_collected: int,
    sentences_collected: int,
    bytes_downloaded: int,
    output_bytes: int,
    status: str = "in_progress",
) -> None:
    """Save download state to JSON."""
    state = _build_state(
        source_urls, current_file_index, current_file_ranges_completed,
        target_words, tolerance, chunk_size, seed,
        words_collected, sentences_collected, bytes_downloaded,
        output_bytes, status,
    )
    save_state(path, state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a ~500M-word candidate subset of IndicCorpV2 Hindi "
            "using HTTP Range requests."
        )
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output and state")
    parser.add_argument("--target-words", type=int,
                        help="Override target whitespace-word count")
    parser.add_argument("--chunk-size-bytes", type=int,
                        help="Override range chunk size in bytes")
    args = parser.parse_args()

    # Ctrl-C must interrupt the download cleanly.  Without an explicit
    # handler, a background job (e.g. under a non-interactive shell) can
    # inherit SIG_IGN for SIGINT and silently swallow user interrupts.
    def _raise_interrupt(signum: int, frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _raise_interrupt)

    config = load_config(args.config)
    logger = setup_logging(config)
    sel_cfg = config.get("selection", {})

    repo_root = Path(args.config).resolve().parent.parent.parent

    # ---- Resolve paths -------------------------------------------------------
    raw_dir = repo_root / config["paths"]["raw_indiccorp"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(raw_dir / config["indiccorp"]["candidate_filename"])
    state_path = str(raw_dir / "indiccorp_selection_state.json")
    report_path = str(raw_dir / "indiccorp_selection_report.json")

    # ---- Load source URLs ----------------------------------------------------
    indiccorp_cfg = config.get("indiccorp", {})
    if "source_urls" in indiccorp_cfg:
        source_urls = list(indiccorp_cfg["source_urls"])
    elif "source_url" in indiccorp_cfg:
        source_urls = [indiccorp_cfg["source_url"]]
    else:
        logger.error(
            "No source URLs found in config under "
            "indiccorp.source_urls or indiccorp.source_url."
        )
        sys.exit(1)

    target_words = (
        args.target_words
        if args.target_words is not None
        else sel_cfg.get("target_words", 500_000_000)
    )
    # Tolerance is configurable but is capped so small test targets
    # (e.g. --target-words 1000000) still have a sensible acceptance band.
    tolerance = min(
        sel_cfg.get("tolerance_words", 5_000_000),
        max(1, target_words // 10),
    )
    chunk_size = (
        args.chunk_size_bytes
        if args.chunk_size_bytes is not None
        else sel_cfg.get("chunk_size_bytes", 1 << 30)  # 1 GB default
    )
    seed = sel_cfg.get("random_seed", 42)

    logger.info("=" * 70)
    logger.info("IndicCorpV2 Hindi — Partial Range Downloader")
    logger.info("=" * 70)
    logger.info(f"Source files:        {len(source_urls)}")
    for i, url in enumerate(source_urls):
        logger.info(f"  [{i}] {url}")
    logger.info(f"Target words:        {target_words:,}")
    logger.info(f"Tolerance:           ±{tolerance:,}")
    logger.info(f"Chunk size:          {chunk_size / (1 << 20):.0f} MB")
    logger.info(f"Random seed:         {seed}")
    logger.info(f"Output:              {output_path}")
    logger.info(f"State:               {state_path}")

    # ---- Handle overwrite / resume -------------------------------------------
    if args.overwrite:
        for p in (output_path, state_path, report_path):
            if os.path.exists(p):
                os.remove(p)
                logger.info(f"Removed existing: {p}")

    existing_state = load_state(state_path) if args.resume else None

    if existing_state and existing_state.get("status") == "completed":
        logger.info(
            f"Previous download already completed "
            f"({existing_state['words_collected']:,} words). "
            f"Use --overwrite to redo."
        )
        return

    if args.resume and not existing_state:
        if os.path.exists(output_path):
            logger.error(
                f"--resume requested but no state file was found at "
                f"{state_path} while an output file exists.  The output "
                f"cannot be safely resumed without its state.  "
                f"Use --overwrite to start fresh."
            )
            sys.exit(1)

    if existing_state and args.resume and not os.path.exists(output_path):
        logger.error(
            f"State file exists ({state_path}) but the output file "
            f"{output_path} is missing.  The state cannot be resumed "
            f"against a deleted corpus.  Use --overwrite to start fresh."
        )
        sys.exit(1)

    if not args.resume and not args.overwrite:
        if os.path.exists(output_path):
            logger.error(
                f"Output file exists: {output_path}. "
                f"Use --resume to continue or --overwrite to redo."
            )
            sys.exit(1)

    # ---- Restore state if resuming -------------------------------------------
    words_collected: int = 0
    sentences_collected: int = 0
    bytes_downloaded: int = 0
    output_bytes: int = 0
    current_file_index: int = 0
    current_file_ranges_completed: List[int] = []

    if existing_state:
        # Reject old single-file state format
        if "source_urls" not in existing_state:
            logger.error(
                "Saved state uses the old single-file format "
                "(missing 'source_urls').  Use --overwrite to restart "
                "with the current IndicCorpV2 multi-file format."
            )
            sys.exit(1)

        saved_urls = existing_state["source_urls"]
        if saved_urls != source_urls:
            logger.error(
                "Saved state was created with different source URLs.  "
                "Refusing to resume.  Use --overwrite to restart."
            )
            sys.exit(1)

        words_collected = existing_state.get("words_collected", 0)
        sentences_collected = existing_state.get("sentences_collected", 0)
        bytes_downloaded = existing_state.get("bytes_downloaded", 0)
        output_bytes = existing_state.get("output_bytes", 0)
        current_file_index = existing_state.get("current_file_index", 0)
        current_file_ranges_completed = existing_state.get(
            "current_file_ranges_completed", []
        )

        state_target = existing_state.get("target_words")
        if state_target is not None and state_target != target_words:
            logger.warning(
                f"Saved state was created with target_words={state_target:,} "
                f"but the current target is {target_words:,}.  Continuing "
                f"with {target_words:,}."
            )

        logger.info(
            f"Resuming from file {current_file_index}: "
            f"{words_collected:,} words, "
            f"{sentences_collected:,} sentences, "
            f"{len(current_file_ranges_completed)} chunks done in current file"
        )

    # ---- Download loop over files --------------------------------------------
    t_start = time.time()
    target_upper = target_words + tolerance
    lower_bound = target_words - tolerance
    target_reached = False
    final_url: str = source_urls[0]

    # On resume, truncate the output file to the last state-recorded byte
    # offset.  This removes any orphaned partial-chunk data written just
    # before an interruption, guaranteeing resume never duplicates chunks.
    if existing_state:
        actual_size = os.path.getsize(output_path)
        truncate_to = min(output_bytes, actual_size) if output_bytes > 0 else actual_size
        if truncate_to < actual_size:
            with open(output_path, "r+b") as f:
                f.truncate(truncate_to)
            logger.info(
                f"Truncated output file to {truncate_to:,} bytes "
                f"(removed {actual_size - truncate_to:,} orphaned bytes)"
            )
        output_bytes = truncate_to

    # Open output in append mode (safe for resume)
    out_mode = "a" if existing_state else "w"

    try:
        with open(output_path, out_mode, encoding="utf-8", buffering=1 << 20) as out_f:
            for file_idx, source_url in enumerate(source_urls):
                if file_idx < current_file_index:
                    continue

                # ---- Probe remote file ---------------------------------------
                logger.info(f"\nProbing file {file_idx}/{len(source_urls)}: {source_url}")
                remote_size, supports_ranges, final_url = probe_remote_file(
                    source_url, logger
                )
                logger.info(f"Remote file size:    {remote_size:,} bytes "
                            f"({remote_size / (1 << 30):.2f} GB)")

                if not supports_ranges:
                    logger.error(
                        "The server does NOT support HTTP byte-range "
                        "requests (Accept-Ranges != bytes).  Partial "
                        "downloading is unavailable.  Aborting safely."
                    )
                    sys.exit(1)

                # ---- Build chunk plan ----------------------------------------
                n_chunks = (remote_size + chunk_size - 1) // chunk_size
                logger.info(f"Chunk plan:          {n_chunks} chunks of "
                            f"~{chunk_size // (1 << 20)} MB")

                chunks: List[Tuple[int, int, int]] = []
                for i in range(n_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size - 1, remote_size - 1)
                    chunks.append((i, start, end))

                # Deterministic shuffled order (seed varies per file)
                rng = random.Random(seed + file_idx)
                chunk_order = list(range(n_chunks))
                rng.shuffle(chunk_order)

                # Completed chunks for this file
                if file_idx == current_file_index:
                    completed_set = set(current_file_ranges_completed)
                else:
                    completed_set = set()

                # ---- Chunk download loop -------------------------------------
                for order_idx, chunk_idx in enumerate(chunk_order):
                    if words_collected >= lower_bound:
                        target_reached = True
                        break

                    if chunk_idx in completed_set:
                        continue

                    _, byte_start, byte_end = chunks[chunk_idx]
                    chunk_bytes = byte_end - byte_start + 1

                    logger.info(
                        f"  Chunk {order_idx + 1}/{n_chunks} "
                        f"(idx={chunk_idx}, bytes {byte_start:,}–{byte_end:,}, "
                        f"{chunk_bytes / (1 << 20):.0f} MB) …"
                    )

                    # Download
                    try:
                        raw = download_range(
                            final_url, byte_start, byte_end, logger
                        )
                    except Exception as exc:
                        logger.error(
                            f"Failed to download chunk {chunk_idx}: {exc}"
                        )
                        logger.info(
                            "Saving state and exiting. "
                            "Use --resume to continue."
                        )
                        _save_state_inline(
                            state_path, source_urls, file_idx,
                            list(completed_set),
                            target_words, tolerance, chunk_size, seed,
                            words_collected, sentences_collected,
                            bytes_downloaded, output_bytes,
                            status="interrupted",
                        )
                        sys.exit(1)

                    bytes_downloaded += len(raw)

                    # Extract complete sentences
                    sentences = extract_complete_sentences(
                        raw, byte_start, byte_end, remote_size
                    )

                    # Write sentences, counting words; stop within chunk
                    chunk_words = 0
                    chunk_sents = 0
                    for sent in sentences:
                        wc = len(sent.split())
                        if words_collected + chunk_words + wc > target_upper:
                            # Writing this sentence would push past the upper
                            # bound.  Only write it if we are still below the
                            # lower bound (sentence integrity takes priority).
                            if words_collected + chunk_words < lower_bound:
                                out_f.write(sent + "\n")
                                chunk_words += wc
                                chunk_sents += 1
                            break
                        out_f.write(sent + "\n")
                        chunk_words += wc
                        chunk_sents += 1

                    words_collected += chunk_words
                    sentences_collected += chunk_sents
                    completed_set.add(chunk_idx)

                    elapsed = time.time() - t_start
                    speed_mbps = bytes_downloaded / (1 << 20) / max(elapsed, 0.1)
                    pct = words_collected / target_words * 100

                    logger.info(
                        f"  +{chunk_sents:,} sentences, +{chunk_words:,} words  │  "
                        f"Total: {sentences_collected:,} sents, "
                        f"{words_collected:,} words ({pct:.1f}%)  │  "
                        f"{speed_mbps:.1f} MB/s"
                    )

                    # Flush so that output_bytes reflects durable file content,
                    # then save state (atomic write-then-rename).
                    out_f.flush()
                    output_bytes = os.path.getsize(output_path)

                    _save_state_inline(
                        state_path, source_urls, file_idx,
                        list(completed_set),
                        target_words, tolerance, chunk_size, seed,
                        words_collected, sentences_collected,
                        bytes_downloaded, output_bytes,
                        status="in_progress",
                    )

                    if words_collected >= lower_bound:
                        target_reached = True
                        break

                # Update current file tracking for state
                current_file_index = file_idx
                current_file_ranges_completed = list(completed_set)

                if target_reached:
                    break

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user (SIGINT). Saving progress …")
        _save_state_inline(
            state_path, source_urls, current_file_index,
            current_file_ranges_completed,
            target_words, tolerance, chunk_size, seed,
            words_collected, sentences_collected, bytes_downloaded,
            output_bytes, status="in_progress",
        )
        logger.info(
            f"Saved progress: file {current_file_index}, "
            f"{words_collected:,} words.  Resume later with --resume."
        )
        sys.exit(130)

    # ---- Mark completed ------------------------------------------------------
    _save_state_inline(
        state_path, source_urls, current_file_index,
        current_file_ranges_completed,
        target_words, tolerance, chunk_size, seed,
        words_collected, sentences_collected, bytes_downloaded,
        output_bytes, status="completed",
    )

    elapsed_total = time.time() - t_start

    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Bytes downloaded:          {bytes_downloaded:,} bytes")
    logger.info(f"Files used:                {current_file_index + 1}/{len(source_urls)}")
    logger.info(f"Sentences collected:       {sentences_collected:,}")
    logger.info(f"Whitespace words:          {words_collected:,}")
    logger.info(f"Time elapsed:              {elapsed_total:.0f}s")

    # ---- Validation ----------------------------------------------------------
    logger.info("\nRunning streaming validation …")
    stats = validate_output(output_path, logger)

    logger.info(f"Local file size:           {stats['file_size_bytes']:,} bytes")
    logger.info(f"Sentences (validation):    {stats['sentences']:,}")
    logger.info(f"Whitespace words (valid.): {stats['whitespace_words']:,}")
    logger.info(f"Avg words/sentence:        {stats['avg_words_per_sentence']}")
    logger.info(f"Min sentence words:        {stats['min_sentence_words']}")
    logger.info(f"Max sentence words:        {stats['max_sentence_words']}")
    logger.info(f"Malformed UTF-8 lines:     {stats['malformed_lines']}")
    logger.info(f"SHA-256:                   {stats['sha256']}")
    logger.info("=" * 70)

    # ---- Persist validation report -------------------------------------------
    report = {
        "source_urls": source_urls,
        "final_url": final_url,
        "local_size_bytes": stats["file_size_bytes"],
        "bytes_downloaded": bytes_downloaded,
        "num_files_used": current_file_index + 1,
        "num_files_total": len(source_urls),
        "sentences_collected": stats["sentences"],
        "whitespace_words_collected": stats["whitespace_words"],
        "avg_words_per_sentence": stats["avg_words_per_sentence"],
        "min_sentence_words": stats["min_sentence_words"],
        "max_sentence_words": stats["max_sentence_words"],
        "malformed_lines": stats["malformed_lines"],
        "sha256": stats["sha256"],
        "target_words": target_words,
        "tolerance_words": tolerance,
        "state_file": state_path,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Validation report saved to {report_path}")


if __name__ == "__main__":
    main()
