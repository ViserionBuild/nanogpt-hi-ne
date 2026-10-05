"""In-memory MinHash + LSH-banding near-duplicate index — a RAM-backed
alternative to shared/near_dup.py's SQLite-backed MinHashDedupIndex, for
machines with abundant RAM relative to corpus size (e.g. Apple M4 / 24 GB)
where avoiding repeated SQLite disk/database round-trips for the LSH index
is worth the memory.

This module is READ-ONLY with respect to shared/near_dup.py: it imports
(never modifies) MinHasher, the shingle functions, LSH banding, and the
Jaccard/overlap estimators from there, so the actual deduplication
ALGORITHM — shingle representation, MinHash signatures, LSH candidate
generation, Jaccard/overlap confirmation — is byte-for-byte identical to
the Linux/SQLite implementation. Only the storage layer for "which
previously-kept lines does this LSH band key point at" differs.

API-compatible with shared/near_dup.py's MinHashDedupIndex, so
nepali/scripts/05_deduplicate_mac.py can use it as a drop-in replacement:

    find_candidates(band_keys) -> [(line_id, signature, shingle_count), ...]
    add(line_id, signature, shingle_count, band_keys)
    close()

Memory guarantee: the CORPUS TEXT is never held here (or anywhere in this
module) — callers stream lines from disk and only call add() for lines
they have already decided to KEEP (duplicates are never indexed, same as
the SQLite version). What IS held in RAM, per kept line:

  * one MinHash signature: a tuple of `num_perm` Python ints
    (~num_perm * 28 bytes as boxed ints in a tuple, e.g. ~1.8 KB at
    num_perm=64 — small relative to 24 GB)
  * one shingle_count int
  * `lsh_bands` postings-list entries (band_key -> [line_id, ...])

For tens of millions of kept lines this is a few GB, comfortably within a
24 GB machine's budget, while completely avoiding SQLite's per-row
serialization/transaction/disk-IO overhead that the Linux implementation
pays for to stay safe on 8 GB RAM.
"""
from collections import defaultdict
from typing import Dict, List, Tuple


class InMemoryMinHashDedupIndex:
    """RAM-backed MinHash/LSH index. See module docstring for the memory
    model and API-compatibility contract with
    shared/near_dup.py:MinHashDedupIndex."""

    def __init__(self, num_perm: int):
        self.num_perm = num_perm
        self._signatures: Dict[int, Tuple[int, ...]] = {}
        self._shingle_counts: Dict[int, int] = {}
        # band_key (8-byte, same as the SQLite version's BLOB storage) ->
        # list of line_ids sharing that band. A plain dict-of-lists is the
        # RAM equivalent of the SQLite version's (band_key, line_id)
        # indexed table.
        self._bands: Dict[bytes, List[int]] = defaultdict(list)

    @staticmethod
    def _blob_keys(band_keys: List[str]) -> List[bytes]:
        """Convert lsh_band_keys()'s hex-string band keys to 8-byte bytes
        objects for dict-key storage — mirrors
        shared/near_dup.py:MinHashDedupIndex._blob_keys() so both indexes
        use the same compact key representation."""
        return [bytes.fromhex(bk) for bk in band_keys]

    def find_candidates(self, band_keys: List[str]) -> List[Tuple[int, Tuple[int, ...], int]]:
        """Return every previously-kept line sharing at least one band key,
        each returned exactly once (matching the SQLite version's
        `GROUP BY line_id` dedup-of-candidates semantics)."""
        if not band_keys:
            return []
        seen_ids = set()
        results = []
        for bk in self._blob_keys(band_keys):
            for line_id in self._bands.get(bk, ()):
                if line_id in seen_ids:
                    continue
                seen_ids.add(line_id)
                results.append((line_id, self._signatures[line_id], self._shingle_counts[line_id]))
        return results

    def add(self, line_id: int, signature: tuple, shingle_count: int, band_keys: List[str]) -> None:
        """Index a KEPT (non-duplicate) line."""
        self._signatures[line_id] = signature
        self._shingle_counts[line_id] = shingle_count
        for bk in self._blob_keys(band_keys):
            self._bands[bk].append(line_id)

    def close(self) -> None:
        """No-op flush (nothing to persist for an in-memory structure) —
        present so callers written against MinHashDedupIndex's API (which
        always calls .close() at the end of a run) work unchanged. Frees
        the index's memory explicitly rather than waiting for GC, since a
        single process may run this index once per corpus-sized job."""
        self._signatures.clear()
        self._shingle_counts.clear()
        self._bands.clear()

    def __len__(self) -> int:
        return len(self._signatures)

    def memory_report(self) -> Dict[str, int]:
        """Cheap, approximate memory accounting for benchmarking/logging —
        NOT exact (Python object overhead varies), but enough to sanity-
        check that the index is staying within the expected order of
        magnitude for a given corpus size."""
        num_kept = len(self._signatures)
        num_band_entries = sum(len(v) for v in self._bands.values())
        return {
            "kept_lines_indexed": num_kept,
            "unique_band_keys": len(self._bands),
            "band_postings_entries": num_band_entries,
        }
