"""Corpus cleaning, normalization, deduplication, and splitting utilities
shared by both language pipelines. Each language script calls these with
its own paths/config — no data itself is shared between languages.

All heavy operations are streaming / memory-bounded: files are processed
line-by-line, dedup is backed by SQLite on disk, and train/val/test splits
use a deterministic per-line hash instead of loading the corpus into RAM.
"""
import hashlib
import json
import os
import re
import sqlite3
import unicodedata


# ---------------------------------------------------------------------------
# Line I/O
# ---------------------------------------------------------------------------

def count_lines(path: str) -> int:
    """Count newlines in *path* without decoding (byte-level, streaming)."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def iter_lines(path: str):
    """Yield each line of *path* WITHOUT its trailing newline.

    Downstream scripts consistently write ``line + "\\n"`` back out, so the
    output stays one sentence per line with no stray blank lines.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\r\n")


def line_hash(seed: int, index: int) -> float:
    """Deterministic, reproducible hash of (seed, line_index) in [0, 1).

    Used for streaming corpus selection (script 11) and train/val/test
    splitting (script 08) so no permutation arrays or Python sets are ever
    materialized in memory. Identical inputs → identical decisions.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b":")
    h.update(str(index).encode("utf-8"))
    return int.from_bytes(h.digest(), "big") / 2**64


# ---------------------------------------------------------------------------
# Normalization / filtering
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """NFC-normalize and strip control characters / stray whitespace.

    ZWJ (U+200D) and ZWNJ (U+200C) are **preserved** — they are meaningful
    in Devanagari (e.g. conjuncts, half forms). Only the zero-width space
    (U+200B) and BOM (U+FEFF) are removed.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)  # ZWSP/BOM only
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def basic_filters(text: str, min_chars: int = 20, max_url_ratio: float = 0.3) -> bool:
    """Return False for lines/docs that look like junk (too short, mostly URLs, etc.)."""
    if len(text) < min_chars:
        return False
    url_chars = sum(len(m) for m in re.findall(r"https?://\S+", text))
    if len(text) > 0 and (url_chars / len(text)) > max_url_ratio:
        return False
    return True


# ---------------------------------------------------------------------------
# Deduplication (disk-backed SQLite — memory-bounded at any corpus size)
# ---------------------------------------------------------------------------

def dedup_stream(input_path: str, output_path: str, db_path: str = None) -> int:
    """Exact-match dedup of *input_path* into *output_path*.

    Hashes are stored in a SQLite table (WAL mode) so memory stays bounded
    regardless of corpus size. Returns the number of unique lines written.
    """
    if db_path is None:
        db_path = output_path + ".sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY);")
    conn.execute("BEGIN;")

    unique = 0
    try:
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8", buffering=1 << 20) as fout:
            for line in fin:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                h = hashlib.sha256(line.encode("utf-8")).hexdigest()
                cur = conn.execute("SELECT 1 FROM seen WHERE hash=?", (h,))
                if cur.fetchone() is None:
                    conn.execute("INSERT INTO seen (hash) VALUES (?)", (h,))
                    fout.write(line + "\n")
                    unique += 1
                    if unique % 200_000 == 0:
                        conn.commit()
    finally:
        conn.commit()
        conn.close()

    if os.path.exists(db_path):
        os.remove(db_path)
    return unique


def dedup_lines(lines, use_hash=True):
    """In-memory dedup for *small* line collections (convenience helper).

    For large corpora prefer :func:`dedup_stream` (SQLite-backed) or
    script 06, which never materialize the whole corpus in RAM.
    """
    seen = set()
    out = []
    for line in lines:
        key = hashlib.md5(line.encode("utf-8")).hexdigest() if use_hash else line
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


def clean_corpus(input_paths, output_path, min_chars=20):
    """Stream raw text file(s), normalize + filter + SQLite-dedup, write one
    cleaned line-per-sentence file. Memory-bounded. Returns line count."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".clean_tmp"

    with open(tmp_path, "w", encoding="utf-8", buffering=1 << 20) as tmpf:
        for p in input_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = normalize_unicode(raw_line)
                    if basic_filters(line, min_chars=min_chars):
                        tmpf.write(line + "\n")

    try:
        n = dedup_stream(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return n


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def train_val_test_split(input_path, out_dir, val_frac=0.05, test_frac=0.05, seed=42):
    """Streaming train/val/test split using a deterministic per-line hash.

    Each line is bucketed by ``line_hash(seed, line_index)`` into
    train/val/test proportionally, so the whole corpus never needs to be
    shuffled or held in memory. Deterministic for a given input + seed.
    """
    train_frac = 1.0 - val_frac - test_frac
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.txt")
    val_path = os.path.join(out_dir, "val.txt")
    test_path = os.path.join(out_dir, "test.txt")

    counts = {"train": 0, "val": 0, "test": 0}
    with open(train_path, "w", encoding="utf-8") as ft, \
         open(val_path, "w", encoding="utf-8") as fv, \
         open(test_path, "w", encoding="utf-8") as fte:
        for idx, line in enumerate(iter_lines(input_path)):
            if not line:
                continue
            h = line_hash(seed, idx)
            if h < train_frac:
                ft.write(line + "\n")
                counts["train"] += 1
            elif h < train_frac + val_frac:
                fv.write(line + "\n")
                counts["val"] += 1
            else:
                fte.write(line + "\n")
                counts["test"] += 1
    return counts


# ---------------------------------------------------------------------------
# Token counting (streaming)
# ---------------------------------------------------------------------------

def approx_token_count(path, tokenizer=None):
    """Streaming token count: real BPE tokens if a tokenizer is given,
    otherwise a whitespace-based approximation. Never loads the file."""
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if tokenizer is not None:
                total += len(tokenizer.encode(line, out_type=int))
            else:
                total += len(line.split())
    return total


def write_corpus_stats(stats_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    print(f"[stats] wrote {out_path}")
