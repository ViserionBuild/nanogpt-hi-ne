"""Shared runtime accounting-reconciliation check for the deduplication
stage (exact + near-duplicate passes), used by all four dedup entry
points: hindi/scripts/06_deduplicate.py, nepali/scripts/05_deduplicate.py,
and their Mac counterparts (06_deduplicate_mac.py / 05_deduplicate_mac.py
in Hindi / Nepali respectively).

Extracted verbatim (not reimplemented) from the validation originally
added to the two Linux scripts, so there is exactly ONE accounting-check
implementation shared across all four scripts rather than four copies.
This module has no dependency on shared/near_dup.py — it only reconciles
the stats dicts each dedup pass already returns, regardless of which
near-duplicate index backend (SQLite or in-memory) produced them.
"""
import logging
import sys
from typing import Any, Dict


def validate_dedup_accounting(
    exact_stats: Dict[str, Any],
    near_stats: Dict[str, Any],
    near_enabled: bool,
    logger: logging.Logger,
) -> None:
    """Verify every input line is accounted for by exactly one outcome
    category, per source. Fails loudly (exits non-zero) rather than
    silently producing a corpus whose own statistics don't add up.

    EXACT pass: every line is either a duplicate or a unique — no other
    category exists there, so input_lines == duplicates + uniques always.

    NEAR-DUP pass: every line is either kept (already includes
    short-line-kept, since those are written straight to `kept`), removed
    as a near-duplicate, removed as a sub-document-duplicate, or a blank
    line (read but never written/classified) — so
    input_lines == kept + near_duplicates_removed +
    sub_document_duplicates_removed + blank_lines_skipped.
    """
    ok = True
    for label, ex in exact_stats.get("by_source", {}).items():
        total = ex["input_lines"]
        accounted = ex["duplicates"] + ex["uniques"]
        if total != accounted:
            logger.error(
                f"[{label}] EXACT-PASS ACCOUNTING MISMATCH: input_lines={total:,} "
                f"but duplicates+uniques={accounted:,}"
            )
            ok = False

    if near_enabled:
        for label, nr in near_stats.get("by_source", {}).items():
            total = nr["input_lines"]
            accounted = (
                nr["kept"] + nr["near_duplicates_removed"]
                + nr["sub_document_duplicates_removed"] + nr.get("blank_lines_skipped", 0)
            )
            if total != accounted:
                logger.error(
                    f"[{label}] NEAR-DUP-PASS ACCOUNTING MISMATCH: input_lines={total:,} "
                    f"but kept+near_dup_removed+sub_doc_removed+blank_lines_skipped={accounted:,}"
                )
                ok = False

    if not ok:
        logger.error(
            "Deduplication accounting does not reconcile — refusing to treat "
            "this run as valid. This indicates a bug in the counting logic, "
            "not necessarily a corpus problem."
        )
        sys.exit(1)
    logger.info("Deduplication accounting verified: every input line is accounted for.")
