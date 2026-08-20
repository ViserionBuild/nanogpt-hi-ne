"""run_stats.py — Per-pipeline-stage run-summary logging + prior-stage
cleanup, shared by every numbered nepali/scripts/0N_*.py stage script.

Disk is the limiting resource on a single laptop-scale pipeline run, not
CPU/RAM, and each stage (02_normalize, 03_filter_nepali, 04_quality_filter,
...) writes a FULL COPY of the corpus. Keeping every stage's output on
disk simultaneously does not fit. Two things address that:

  1. record_run() — writes one JSON summary per (stage, source, run) to
     paths.run_stats_dir (default nepali/data/run_stats/), so the
     numbers a stage produced are never lost even after its output files
     are deleted. One file per invocation, named
     "{stage_number}_{source}_{timestamp}.json" — never overwritten by a
     later run of the same stage, so re-running a stage to compare
     thresholds keeps every prior run's numbers instead of clobbering
     them.

  2. cleanup_prior_stage_files() — deletes SPECIFIC, NAMED files that
     belonged to a prior stage (never a whole directory) once the
     CURRENT stage has finished writing its own output for that same
     source. Each pipeline stage's output directory
     (normalized/filtered/quality) holds ONE FILE PER SOURCE
     (indiccorp_ne_normalized.txt, onlinekhabar_ne_normalized.txt, ...),
     and 03/04/05 are invoked separately per source — so cleanup must
     delete only the file(s) for the source that was JUST consumed,
     never every file in the directory, or finishing source A's run
     would destroy source B's still-needed prior-stage file. Only
     stage NAMES in DELETABLE_STAGES may be deleted this way — raw/,
     deduplicated/, split/, decontaminated/, and final/ are refused
     regardless of what's asked (see PROTECTED_STAGES) since those must
     survive permanently (source of truth, source-wise cleaned corpus,
     train/val/test split, and the merged/tokenized final corpus).

Both are best-effort additions to a stage script's existing success
path — call them AFTER a stage's normal output is fully written and
verified non-empty, never before, so a crash mid-stage never deletes the
one copy of the previous stage's data without a working replacement.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Stage names this module is ALLOWED to delete files from, once their
# consumer stage has finished with a given source. Anything not in this
# set is refused even if explicitly requested — see PROTECTED_STAGES
# below for the permanent set this guards.
DELETABLE_STAGES = ("normalized", "filtered", "quality")

# Never deleted by this module, no matter what stage name is passed to
# cleanup_prior_stage_files() — raw is the only copy of collected data,
# deduplicated is the final source-wise-cleaned corpus, split is the
# train/val/test partition, decontaminated/final are downstream of that
# split. All must survive for provenance, re-splitting, and grading
# purposes.
PROTECTED_STAGES = ("raw", "raw_indiccorp", "deduplicated", "split", "decontaminated", "final")


def _run_stats_dir(config: Dict[str, Any], repo_root: Path) -> Path:
    d = repo_root / config.get("paths", {}).get("run_stats", "nepali/data/run_stats")
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_run(
    config: Dict[str, Any],
    repo_root: Path,
    stage_number: str,
    stage_name: str,
    source: str,
    summary: Dict[str, Any],
    logger=None,
) -> Path:
    """Write one JSON run-summary file for this invocation.

    *stage_number*/*stage_name* identify the pipeline stage (e.g. "05",
    "quality_filter"); *source* is the corpus source this run processed
    (e.g. "indiccorp", "onlinekhabar", "khabarhub", "manual"). *summary*
    is a flat, JSON-serializable dict of whatever counts/paths/timings
    that stage wants recorded — this function does not interpret it,
    only stamps it with run metadata and writes it out.

    Returns the path written. Never overwrites a prior run's file for
    the same stage/source — the timestamp in the filename makes every
    invocation distinct, so re-running a stage (e.g. to compare
    thresholds) keeps every previous run's numbers on disk.
    """
    out_dir = _run_stats_dir(config, repo_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{stage_number}_{source}_{ts}.json"
    path = out_dir / filename

    record = {
        "stage_number": stage_number,
        "stage_name": stage_name,
        "source": source,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    if logger:
        logger.info(f"Run stats saved to {path}")
    return path


def cleanup_prior_stage_files(
    config: Dict[str, Any],
    repo_root: Path,
    stage_to_delete: str,
    files: List[Union[str, Path]],
    logger=None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Delete exactly the given *files* (specific paths belonging to
    *stage_to_delete* for the ONE source just consumed) — ONLY if
    stage_to_delete is in DELETABLE_STAGES. Refuses (logs an error,
    deletes nothing, returns skipped=True) for anything in
    PROTECTED_STAGES or any unrecognized name — this is a safety
    backstop, not just documentation, since a typo'd stage name must
    never silently delete protected data.

    Deliberately takes explicit file paths rather than a directory to
    sweep — each stage directory holds one file per SOURCE
    (indiccorp_ne_normalized.txt, onlinekhabar_ne_normalized.txt, ...)
    and stages are run once per source, so deleting "everything in the
    directory" after just ONE source's run would destroy every OTHER
    source's still-needed file.

    A missing file (already deleted, or never existed for this source)
    is not an error — it's logged and skipped in the per-file result,
    since the caller's job (freeing this source's prior-stage disk
    usage) is already satisfied.

    Returns a small report dict (deleted_files, missing_files,
    freed_bytes, skipped, reason) — callers fold this into their own
    record_run() summary rather than this function writing its own
    separate run-stats file, since a cleanup is always a side effect of
    some stage's run, not an independent stage in its own right.
    """
    if stage_to_delete in PROTECTED_STAGES:
        msg = (
            f"cleanup_prior_stage_files: REFUSING to delete stage '{stage_to_delete}' "
            f"— it is in PROTECTED_STAGES (never auto-deleted)."
        )
        if logger:
            logger.error(msg)
        return {"skipped": True, "reason": "protected_stage", "stage": stage_to_delete}

    if stage_to_delete not in DELETABLE_STAGES:
        msg = (
            f"cleanup_prior_stage_files: REFUSING to delete unrecognized stage "
            f"'{stage_to_delete}' — not in DELETABLE_STAGES {DELETABLE_STAGES}."
        )
        if logger:
            logger.error(msg)
        return {"skipped": True, "reason": "unrecognized_stage", "stage": stage_to_delete}

    deleted_files = []
    missing_files = []
    freed_bytes = 0
    for f in files:
        p = Path(f)
        if not p.exists():
            missing_files.append(str(p))
            continue
        size = p.stat().st_size
        if dry_run:
            if logger:
                logger.info(f"[dry-run] would delete {p} ({size:,} bytes)")
        else:
            p.unlink()
            if logger:
                logger.info(f"Deleted {p} ({size:,} bytes) — stage '{stage_to_delete}' no longer needed.")
        deleted_files.append(str(p))
        freed_bytes += size

    return {
        "skipped": False,
        "stage": stage_to_delete,
        "deleted_files": deleted_files,
        "missing_files": missing_files,
        "freed_bytes": freed_bytes,
        "dry_run": dry_run,
    }
