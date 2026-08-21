"""MinHash + LSH-banding near-duplicate detection.

Used by:
  * the DEDUPLICATION stage (``hindi/scripts/06_deduplicate.py`` and
    ``nepali/scripts/05_deduplicate.py``, plus their Mac counterparts) —
    full corpus-wide near-duplicate / sub-document-duplicate removal,
    following CL3410 Lecture 3: represent documents as overlapping
    K-WORD shingles, summarize with MinHash, generate candidates via LSH
    banding (so no document is ever compared against every other
    document), and confirm with Jaccard/overlap similarity before
    removing anything.
  * the DECONTAMINATION stage (``hindi/scripts/09_decontaminate.py`` and
    the equivalent Nepali script) — catches paraphrased / near-identical
    held-out validation & test lines that survive exact-hash matching,
    using character shingles (see ``char_shingles`` docstring).

Both stages, both languages, and both platform variants (SQLite-backed
``MinHashDedupIndex`` here / in-memory ``InMemoryMinHashDedupIndex`` in
``shared/near_dup_mac.py``) share the same MinHash/LSH primitives in this
one module — this is what makes the implementation reusable everywhere
instead of each stage/language/platform growing its own copy.

MinHash engine
---------------
Signatures are produced by ``datasketch.MinHash`` (``update_batch()`` over
all shingles in one call, not one ``update()`` per shingle — the batched
C-accelerated path is materially faster at corpus scale) using
``xxhash.xxh32`` as the hash function instead of a cryptographic hash —
xxhash is a non-cryptographic hash designed for exactly this kind of
high-throughput, large-volume hashing, and no cryptographic property
(collision resistance against an adversary) is needed here: shingles are
project-internal text, not attacker-controlled input. The resulting
signature is converted to a ``LeanMinHash`` (drops MinHash's permutation-
generator state, keeping only the compact hash-value array) before being
read out as a plain tuple of ints for storage — the same plain-tuple
contract every caller in this codebase (``MinHashDedupIndex``,
``InMemoryMinHashDedupIndex``, both ``06_deduplicate*.py`` scripts, both
``09_decontaminate.py`` scripts) already expects, so nothing downstream of
``MinHasher.signature()`` needed to change.

Everything here is deterministic given a seed: identical input text and
identical config always produce identical signatures/band keys, which is
required for every stage to be reproducible.
"""
import hashlib
import os
import re
import sqlite3
import sys

import xxhash
from datasketch import LeanMinHash, MinHash

_WHITESPACE_RE = re.compile(r"\s+")


def _fast_hash32(data: bytes) -> int:
    """xxhash32 — the hash function datasketch.MinHash uses per shingle.
    Non-cryptographic and fast; shingles are project-internal text, not
    adversarial input, so no cryptographic collision-resistance property
    is needed here (unlike, say, hashing for security purposes)."""
    return xxhash.xxh32(data).intdigest()


def char_shingles(text: str, k: int = 5) -> set:
    """Character k-gram shingle set of *text* (whitespace-collapsed).

    Character shingles (rather than word shingles) are used for
    decontamination because they are robust to small edits,
    spacing/punctuation differences, and morphological variation — exactly
    the kind of paraphrase-level drift that can leak a test item into
    training data.
    """
    text = _WHITESPACE_RE.sub(" ", text.strip())
    if len(text) <= k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def word_shingles(text: str, k: int = 3) -> set:
    """Overlapping k-WORD shingle set of *text* — the representation
    CL3410 Lecture 3 specifies for (near-)duplicate detection: a document
    becomes the set of its overlapping k-word windows, e.g. for k=3::

        "पुराने शहर के पास ऐतिहासिक बाजार"
        -> {"पुराने शहर के", "शहर के पास", "के पास ऐतिहासिक", "पास ऐतिहासिक बाजार"}

    Words are split on whitespace only — punctuation stays attached to its
    word (matching how the rest of this pipeline tokenizes, e.g. quality
    filtering's word counts) and Devanagari/ZWNJ/ZWJ content is passed
    through unchanged since it is never re-normalized here.
    """
    words = text.split()
    if len(words) <= k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def shingles(text: str, k: int, mode: str = "char") -> set:
    """Dispatch to :func:`char_shingles` or :func:`word_shingles` by *mode*."""
    if mode == "word":
        return word_shingles(text, k=k)
    if mode == "char":
        return char_shingles(text, k=k)
    raise ValueError(f"Unknown shingle mode: {mode!r} (expected 'char' or 'word')")


class MinHasher:
    """Deterministic MinHash signature generator, seeded and reusable.

    Backed by datasketch.MinHash + xxhash32 + update_batch() + LeanMinHash
    (see module docstring). Public API (``signature(shingle_set) -> tuple``
    of ``num_perm`` ints) is unchanged from the prior hand-rolled
    implementation, so every caller (MinHashDedupIndex,
    InMemoryMinHashDedupIndex, 06_deduplicate*.py, 09_decontaminate.py, in
    both languages) works unmodified."""

    def __init__(self, num_perm: int, seed: int):
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        self.seed = seed

    def signature(self, shingle_set) -> tuple:
        """Return a length-``num_perm`` MinHash signature for *shingle_set*."""
        m = MinHash(num_perm=self.num_perm, seed=self.seed, hashfunc=_fast_hash32)
        if shingle_set:
            m.update_batch([s.encode("utf-8") for s in shingle_set])
        lean = LeanMinHash(m)
        return tuple(int(x) for x in lean.hashvalues)


def lsh_band_keys(signature: tuple, bands: int, rows: int) -> list:
    """Split a MinHash *signature* into ``bands`` LSH band keys.

    Two documents sharing ANY band key are near-duplicate candidates.
    Probability two documents with true Jaccard similarity ``s`` share at
    least one band is approximately ``1 - (1 - s**rows)**bands`` — more
    bands / fewer rows = more sensitive (catches more near-dups, more
    false positives); fewer bands / more rows = stricter.
    """
    needed = bands * rows
    if needed > len(signature):
        raise ValueError(
            f"bands*rows ({needed}) exceeds signature length ({len(signature)})"
        )
    keys = []
    for band_idx in range(bands):
        chunk = signature[band_idx * rows: band_idx * rows + rows]
        key = hashlib.blake2b(
            f"{band_idx}:{chunk}".encode("utf-8"), digest_size=8
        ).hexdigest()
        keys.append(key)
    return keys


def line_band_keys(
    text: str, minhasher: MinHasher, bands: int, rows: int, shingle_size: int,
    shingle_type: str = "char",
) -> list:
    """Convenience: text -> shingles -> MinHash signature -> LSH band keys.

    ``shingle_type`` defaults to ``"char"`` to preserve decontamination's
    existing behavior; pass ``"word"`` for the professor's 3-word-shingle
    deduplication scheme.
    """
    shingle_set = shingles(text, k=shingle_size, mode=shingle_type)
    sig = minhasher.signature(shingle_set)
    return lsh_band_keys(sig, bands, rows)


def signature_and_shingle_count(
    text: str, minhasher: MinHasher, shingle_size: int, shingle_type: str = "word",
):
    """Return (MinHash signature, |shingle set|) for *text* in one pass.

    The shingle count is kept alongside the signature so similarity can
    later be estimated WITHOUT re-reading the original text — this is what
    lets the deduplication index (:class:`MinHashDedupIndex`) stay
    streaming: only compact per-line signatures are persisted, never the
    corpus itself.
    """
    shingle_set = shingles(text, k=shingle_size, mode=shingle_type)
    sig = minhasher.signature(shingle_set)
    return sig, len(shingle_set)


def jaccard_estimate(sig_a: tuple, sig_b: tuple) -> float:
    """MinHash estimate of Jaccard(A, B): fraction of signature positions
    that agree. This is the standard unbiased MinHash similarity estimator
    — no O(N^2) comparison of the underlying shingle sets is ever needed."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def overlap_estimate(jaccard: float, count_a: int, count_b: int) -> float:
    """Estimate the OVERLAP (containment) coefficient
    ``|A ∩ B| / min(|A|, |B|)`` from a Jaccard estimate and the two
    documents' shingle-set sizes, via inclusion-exclusion:
    ``|A ∩ B| = J * (|A| + |B|) / (1 + J)``.

    Overlap/containment is what CL3410 Lecture 3 calls SUB-DOCUMENT
    duplicate detection: a short document whose shingles are almost
    entirely CONTAINED in a much longer one (e.g. a repeated boilerplate
    sentence embedded in otherwise-different longer text) can have low
    Jaccard similarity (because the union is dominated by the longer
    document) while still being an near-total duplicate of part of it —
    Jaccard alone would miss this, overlap catches it.
    """
    if count_a <= 0 or count_b <= 0:
        return 0.0
    if jaccard <= 0:
        return 0.0
    intersection = jaccard * (count_a + count_b) / (1.0 + jaccard)
    return min(1.0, intersection / min(count_a, count_b))


class MinHashDedupIndex:
    """Streaming, disk-backed MinHash/LSH index optimized for large corpora.

    API is intentionally compatible with the original implementation:
      - find_candidates(band_keys) -> [(line_id, signature, shingle_count), ...]
      - add(line_id, signature, shingle_count, band_keys)
      - close()

    Optimizations (all serialization/query/transaction-level — LSH
    candidate-generation and Jaccard/overlap confirmation semantics are
    byte-for-byte unchanged):
      * MinHash signatures are stored as compact binary BLOBs instead of
        comma-separated text, avoiding large string serialization/parsing.
      * Band keys are stored as raw 8-byte BLOBs instead of 16-character hex
        TEXT — half the bytes per (band_key, line_id) row and a smaller,
        cheaper-to-scan index, which is what most of this index's disk/
        memory footprint is spent on (bands rows outnumber signature rows
        by a factor of `lsh_bands`).
      * Disposable index uses SQLite journal_mode=OFF + synchronous=OFF.
        The index is temporary and is deleted by 06_deduplicate.py after the
        run, so crash recovery is not needed for this derived index.
      * locking_mode=EXCLUSIVE: this database is only ever opened by a
        single process/connection for its whole lifetime (created fresh,
        used, deleted at the end of one dedup run), so paying for shared-
        lock bookkeeping on every statement is pure overhead.
      * A single cursor is reused for the object's lifetime instead of
        allocating a new one per call.
      * find_candidates() keeps the existing single-JOIN-query retrieval
        (bands JOIN signatures, GROUP BY line_id) from the prior
        optimization pass — that already replaced an earlier two-round-trip
        implementation and is left untouched here, not re-litigated.
      * Larger SQLite cache and in-memory temp storage reduce filesystem
        work (still small relative to an 8 GB machine's budget — this
        index, not corpus text, is the only thing held in SQLite; corpus
        lines themselves are still streamed, never loaded into memory).
      * Larger inter-commit batch (fewer, cheaper transactions) — safe
        because journal_mode=OFF already means a crash loses the whole
        (disposable, regenerable) index regardless of commit frequency;
        this only affects how often SQLite flushes, not how much data is
        held in Python-side memory (still O(1) per line).
      * Candidate lookup still has exactly the same LSH semantics: every
        previously-kept line sharing ANY requested band is returned once.
      * All candidates are still returned; there is no sampling/cap, so recall
        is not intentionally reduced.
    """

    def __init__(self, db_path: str, num_perm: int):
        self.num_perm = num_perm
        # datasketch's MinHash hashvalues are uint32 (xxhash32-derived), so
        # 4 bytes/value is sufficient — half the footprint of the prior
        # hand-rolled 64-bit signatures, for free.
        self._sig_bytes = 4 * num_perm
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=OFF;")
        self._conn.execute("PRAGMA synchronous=OFF;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA locking_mode=EXCLUSIVE;")
        self._conn.execute("PRAGMA cache_size=-131072;")  # ~128 MiB
        self._conn.execute("PRAGMA mmap_size=536870912;")  # up to 512 MiB
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS signatures ("
            " line_id INTEGER PRIMARY KEY,"
            " signature BLOB NOT NULL,"
            " shingle_count INTEGER NOT NULL);"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS bands ("
            " band_key BLOB NOT NULL,"
            " line_id INTEGER NOT NULL);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bands_key_line "
            "ON bands (band_key, line_id);"
        )
        self._conn.commit()
        self._cur = self._conn.cursor()
        self._pending = 0
        self._commit_every = 200_000

    @staticmethod
    def _pack(signature: tuple) -> bytes:
        import array
        a = array.array("I", signature)
        if sys.byteorder != "little":
            a.byteswap()
        return a.tobytes()

    def _unpack(self, packed: bytes) -> tuple:
        import array
        a = array.array("I")
        a.frombytes(packed)
        if sys.byteorder != "little":
            a.byteswap()
        if len(a) != self.num_perm:
            raise ValueError(
                f"Invalid MinHash signature length: expected {self.num_perm}, got {len(a)}"
            )
        return tuple(a)

    @staticmethod
    def _blob_keys(band_keys: list) -> list:
        """Convert lsh_band_keys()'s hex-string band keys to compact 8-byte
        BLOBs for storage/lookup. lsh_band_keys() itself is left returning
        hex strings unchanged (it's shared with every decontaminate.py
        script's own hand-rolled TEXT-keyed tables) — this conversion is
        purely internal to this index."""
        return [bytes.fromhex(bk) for bk in band_keys]

    def find_candidates(self, band_keys: list) -> list:
        """Return every previously-kept line sharing at least one band key.

        Candidate semantics are unchanged from the original implementation.
        """
        if not band_keys:
            return []

        blob_keys = self._blob_keys(band_keys)
        placeholders = ",".join("?" for _ in blob_keys)
        self._cur.execute(
            f"SELECT s.line_id, s.signature, s.shingle_count "
            f"FROM bands b JOIN signatures s ON s.line_id = b.line_id "
            f"WHERE b.band_key IN ({placeholders}) "
            f"GROUP BY s.line_id",
            blob_keys,
        )
        rows = self._cur.fetchall()
        return [(lid, self._unpack(sig), cnt) for lid, sig, cnt in rows]

    def add(
        self,
        line_id: int,
        signature: tuple,
        shingle_count: int,
        band_keys: list,
    ) -> None:
        """Index a KEPT (non-duplicate) line."""
        self._cur.execute(
            "INSERT INTO signatures (line_id, signature, shingle_count) "
            "VALUES (?, ?, ?)",
            (line_id, self._pack(signature), shingle_count),
        )
        self._cur.executemany(
            "INSERT INTO bands (band_key, line_id) VALUES (?, ?)",
            ((bk, line_id) for bk in self._blob_keys(band_keys)),
        )

        self._pending += 1
        if self._pending >= self._commit_every:
            self._conn.commit()
            self._pending = 0

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
