"""
tokenizer_corpus_analysis.py — reusable, STATELESS internal utilities for the
post-tokenization "final tokenizer + corpus analysis" stage (script 16 in
Hindi's scripts/ directory, script 13 in Nepali's — the two pipelines'
script numbering has diverged, but this is the same final analysis stage
in both).

IMPORTANT: this module holds no per-language state whatsoever — every
function takes its inputs (paths, config, an already-loaded tokenizer,
counters) as parameters and returns a plain dict. hindi/scripts/16_*.py and
nepali/scripts/13_*.py each call these functions with their OWN paths/
config/tokenizer and write to their OWN output files; nothing here ever
combines data from two calls, and no function retains data between calls.
This is what keeps Hindi and Nepali completely independent while avoiding
two copies of ~600 lines of identical streaming/reporting logic.

MEMORY: every streaming function here processes its input file exactly
once, line by line (or reuses shared/tokenizer_utils.count_tokens_streaming,
which already does this). Token-frequency counting keeps only a
{token_id: count} dict bounded by the tokenizer's vocabulary size (tens of
thousands of entries) — never a per-occurrence list and never the corpus
itself in memory.
"""

import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shared.data_utils import iter_lines
from shared.line_provenance import has_sidecar, iter_sidecar_rows, sidecar_path
from shared.tokenizer_utils import count_tokens_streaming


# ---------------------------------------------------------------------------
# 1. Tokenizer information
# ---------------------------------------------------------------------------

def load_tokenizer_info(sp, model_path: str, tok_cfg: Dict[str, Any],
                         tokenizer_config_json_path: Optional[str]) -> Dict[str, Any]:
    """Assemble tokenizer metadata. `actual_vocab_size` is read from the
    ALREADY-LOADED SentencePieceProcessor *sp* — never assumed from config.
    `training_metadata` is read from the tokenizer's own `_config.json`
    (written by script 10) if it exists, otherwise omitted (never
    fabricated)."""
    special_ids = tok_cfg.get("special_token_ids", {})
    special_tokens = tok_cfg.get("special_tokens", {})

    training_metadata = None
    if tokenizer_config_json_path and os.path.exists(tokenizer_config_json_path):
        with open(tokenizer_config_json_path, "r", encoding="utf-8") as f:
            training_metadata = json.load(f)

    model_size_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else None

    return {
        "model_path": model_path,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": round(model_size_bytes / (1024 * 1024), 4) if model_size_bytes else None,
        "algorithm": tok_cfg.get("algorithm", "bpe"),
        "requested_vocab_size": tok_cfg.get("default_vocab_size"),
        "actual_vocab_size": sp.GetPieceSize(),
        "character_coverage": tok_cfg.get("character_coverage"),
        "byte_fallback": tok_cfg.get("byte_fallback"),
        "special_token_ids": {
            "unk": special_ids.get("unk"),
            "bos": special_ids.get("bos"),
            "eos": special_ids.get("eos"),
            "pad": special_ids.get("pad"),
        },
        "special_tokens": {
            "unk": special_tokens.get("unk"),
            "bos": special_tokens.get("bos"),
            "eos": special_tokens.get("eos"),
            "pad": special_tokens.get("pad"),
        },
        "training_metadata": training_metadata,
        "training_metadata_available": training_metadata is not None,
    }


# ---------------------------------------------------------------------------
# 2. Corpus statistics (per split)
# ---------------------------------------------------------------------------

def split_stats(sp, path: str, logger, progress_interval: int) -> Dict[str, Any]:
    """Stream *path* through the tokenizer once (reusing
    shared.tokenizer_utils.count_tokens_streaming — the project's existing
    streaming line/word/char/content-token counter) and derive the
    requested per-split ratios. Fertility is CONTENT tokens / whitespace
    words (BOS/EOS are never added by EncodeAsIds, so they can never leak
    in here)."""
    base = count_tokens_streaming(sp, path, logger, progress_interval)
    lines = base["total_lines"]
    words = base["total_words"]
    chars = base["total_chars"]
    tokens = base["total_tokens"]

    return {
        "file": path,
        "present": os.path.exists(path),
        "lines": lines,
        "words": words,
        "characters": chars,
        "content_tokens": tokens,
        "avg_words_per_line": round(words / lines, 6) if lines else 0.0,
        "avg_characters_per_line": round(chars / lines, 6) if lines else 0.0,
        "avg_characters_per_word": round(chars / words, 6) if words else 0.0,
        "tokens_per_word": round(tokens / words, 6) if words else 0.0,
        "fertility": round(tokens / words, 6) if words else 0.0,
        "characters_per_token": round(chars / tokens, 6) if tokens else 0.0,
        "tokens_per_character": round(tokens / chars, 6) if chars else 0.0,
    }


def combine_split_stats(*stats: Dict[str, Any]) -> Dict[str, Any]:
    """Sum several split_stats() dicts (e.g. train+validation+test) into one
    COMBINED entry with the same derived ratios recomputed from the sums —
    never averaging pre-computed ratios."""
    lines = sum(s["lines"] for s in stats)
    words = sum(s["words"] for s in stats)
    chars = sum(s["characters"] for s in stats)
    tokens = sum(s["content_tokens"] for s in stats)
    return {
        "lines": lines,
        "words": words,
        "characters": chars,
        "content_tokens": tokens,
        "avg_words_per_line": round(words / lines, 6) if lines else 0.0,
        "avg_characters_per_line": round(chars / lines, 6) if lines else 0.0,
        "avg_characters_per_word": round(chars / words, 6) if words else 0.0,
        "tokens_per_word": round(tokens / words, 6) if words else 0.0,
        "fertility": round(tokens / words, 6) if words else 0.0,
        "characters_per_token": round(chars / tokens, 6) if tokens else 0.0,
        "tokens_per_character": round(tokens / chars, 6) if chars else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Logical source accounting
# ---------------------------------------------------------------------------

def manual_breakdown_from_sidecar(sp, manual_path: str, logger, progress_interval: int) -> Dict[str, Any]:
    """Manual-internal Wikipedia/News(+per-outlet) breakdown for a language
    whose manual corpus is ALREADY one combined file with a line-provenance
    sidecar (Hindi's current architecture). Diagnostic/reporting only —
    never a separate top-level source. Returns {'available': False, ...}
    if manual_path has no sidecar (nothing fabricated)."""
    if not os.path.exists(manual_path) or not has_sidecar(manual_path):
        return {"available": False, "by_source_type": {}, "by_source_name": {}}

    by_type: Dict[str, Dict[str, int]] = {}
    by_name: Dict[str, Dict[str, int]] = {}
    from itertools import zip_longest
    _MISSING = object()
    n = 0
    with open(manual_path, "r", encoding="utf-8") as fin:
        text_gen = (raw.rstrip("\r\n") for raw in fin)
        prov_gen = iter_sidecar_rows(sidecar_path(manual_path))
        for line, prov in zip_longest(text_gen, prov_gen, fillvalue=_MISSING):
            if line is _MISSING or prov is _MISSING:
                logger.error(
                    f"Line-provenance sidecar row count does not match "
                    f"{manual_path}'s line count — refusing to silently "
                    f"misattribute provenance."
                )
                raise SystemExit(1)
            source_type, source_name, _doc_id = prov
            if line:
                ids = sp.EncodeAsIds(line)
                n_tokens = len(ids)
                n_words = len(line.split())
                st = by_type.setdefault(source_type, {"documents": 0, "words": 0, "content_tokens": 0})
                st["documents"] += 1
                st["words"] += n_words
                st["content_tokens"] += n_tokens
                sn = by_name.setdefault(source_name, {"documents": 0, "words": 0, "content_tokens": 0})
                sn["documents"] += 1
                sn["words"] += n_words
                sn["content_tokens"] += n_tokens
            n += 1
            if logger and n % progress_interval == 0:
                logger.info(f"  [manual breakdown] {n:,} lines scanned")

    return {"available": True, "by_source_type": by_type, "by_source_name": by_name}


def manual_breakdown_from_files(sp, files_by_label: Dict[str, str], logger, progress_interval: int) -> Dict[str, Any]:
    """Manual-internal breakdown for a language whose manual subcategories
    are still SEPARATE final files (e.g. Nepali's
    final/{wikipedia,onlinekhabar,setopati}_train.txt) — each file is
    independently re-tokenized with the SAME final tokenizer. Diagnostic/
    reporting only — never a separate top-level source."""
    by_name: Dict[str, Dict[str, Any]] = {}
    any_present = False
    for label, path in files_by_label.items():
        if not os.path.exists(path):
            by_name[label] = {"documents": None, "words": 0, "content_tokens": 0, "present": False}
            continue
        any_present = True
        stats = split_stats(sp, path, logger, progress_interval)
        by_name[label] = {
            "words": stats["words"],
            "content_tokens": stats["content_tokens"],
            "lines": stats["lines"],
            "present": True,
        }
    return {"available": any_present, "by_source_name": by_name}


def build_sources_section(
    indiccorp_stats: Dict[str, Any],
    manual_stats: Dict[str, Any],
    total_content_tokens: int,
    manual_breakdown: Dict[str, Any],
    breakdown_kind: str,
) -> Dict[str, Any]:
    """Assemble the JSON report's "sources" block: exactly indiccorp +
    manual as top-level sources, with an optional diagnostic breakdown of
    manual's internal composition (Wikipedia/News for Hindi, Wikipedia/
    Onlinekhabar/Setopati for Nepali) — never promoted to top-level."""
    indiccorp_tokens = indiccorp_stats["content_tokens"]
    manual_tokens = manual_stats["content_tokens"]
    manual_fraction = (manual_tokens / total_content_tokens) if total_content_tokens else 0.0

    return {
        "indiccorp": {
            "content_tokens": indiccorp_tokens,
            "words": indiccorp_stats["words"],
            "lines": indiccorp_stats["lines"],
        },
        "manual": {
            "content_tokens": manual_tokens,
            "words": manual_stats["words"],
            "lines": manual_stats["lines"],
            "breakdown_kind": breakdown_kind,
            "breakdown": manual_breakdown,
        },
        "total": {
            "content_tokens": total_content_tokens,
        },
        "manual_fraction": round(manual_fraction, 6),
        "invariant_indiccorp_plus_manual_equals_total": (indiccorp_tokens + manual_tokens) == total_content_tokens,
    }


# ---------------------------------------------------------------------------
# 4 & 5. Streaming token-ID histogram (shared by tokenizer-quality on
# validation, and frequency-analysis on train) + derived statistics
# ---------------------------------------------------------------------------

def streaming_token_id_histogram(sp, path: str, logger, progress_interval: int) -> Dict[str, Any]:
    """ONE streaming pass over *path*: tokenize each line, tally a
    {token_id: count} Counter (bounded by vocab size, never by corpus
    size — this is the 'exact streaming' approach, not an approximation)
    plus total occurrences/lines/words. Never builds a list of every token
    occurrence and never loads the corpus into memory."""
    counts: Counter = Counter()
    total_tokens = 0
    total_lines = 0
    total_words = 0
    t0 = time.time()
    for line in iter_lines(path) if os.path.exists(path) else []:
        if line:
            ids = sp.EncodeAsIds(line)
            counts.update(ids)
            total_tokens += len(ids)
            total_words += len(line.split())
        total_lines += 1
        if logger and total_lines % progress_interval == 0:
            elapsed = time.time() - t0
            speed = total_lines / elapsed if elapsed > 0 else 0
            logger.info(f"  [token histogram] {total_lines:,} lines | {total_tokens:,} tokens | {speed:,.0f} lines/s")
    return {
        "counts": counts,
        "total_tokens": total_tokens,
        "total_lines": total_lines,
        "total_words": total_words,
    }


def tokenizer_quality_from_histogram(hist: Dict[str, Any], unk_id: Optional[int], actual_vocab_size: int) -> Dict[str, Any]:
    """Derive UNK rate, fertility, chars/token (N/A without char count —
    caller supplies characters separately if available), tokens/word,
    unique-ID usage, and vocabulary utilization from a
    streaming_token_id_histogram() result on the VALIDATION corpus."""
    counts = hist["counts"]
    total_tokens = hist["total_tokens"]
    total_words = hist["total_words"]
    unk_count = counts.get(unk_id, 0) if unk_id is not None else 0
    unique_ids_used = len(counts)

    freqs = sorted(counts.values())
    singleton = sum(1 for c in freqs if c == 1)
    le5 = sum(1 for c in freqs if c <= 5)
    le10 = sum(1 for c in freqs if c <= 10)

    return {
        "validation_lines": hist["total_lines"],
        "validation_tokens": total_tokens,
        "unk_count": unk_count,
        "unk_rate": round(unk_count / total_tokens, 8) if total_tokens else 0.0,
        "fertility": round(total_tokens / total_words, 6) if total_words else 0.0,
        "tokens_per_word": round(total_tokens / total_words, 6) if total_words else 0.0,
        "unique_token_ids_used": unique_ids_used,
        "vocabulary_utilization_percent": round(unique_ids_used / actual_vocab_size * 100, 6) if actual_vocab_size else 0.0,
        "unused_vocabulary_ids": max(0, actual_vocab_size - unique_ids_used),
        "singleton_token_ids": singleton,
        "token_ids_le_5_occurrences": le5,
        "token_ids_le_10_occurrences": le10,
    }


def frequency_analysis_from_histogram(
    sp, hist: Dict[str, Any], bos_id: Optional[int], eos_id: Optional[int],
    top_ns: Sequence[int] = (20, 50, 100),
) -> Dict[str, Any]:
    """Top-K / coverage / singleton statistics for the TRAIN corpus's
    token-ID histogram. BOS/EOS ids are excluded from the counter before
    computing anything (EncodeAsIds never emits them, so in practice this
    is a no-op safety net, not a real filter — see module docstring)."""
    counts = Counter(hist["counts"])
    for special_id in (bos_id, eos_id):
        if special_id is not None and special_id in counts:
            del counts[special_id]

    total_occurrences = sum(counts.values())
    unique_ids = len(counts)
    ranked = counts.most_common()

    def _top(n: int) -> List[Dict[str, Any]]:
        rows = []
        for token_id, count in ranked[:n]:
            piece = sp.IdToPiece(token_id)
            rows.append({
                "token_id": token_id,
                "piece": piece,
                "count": count,
                "percentage": round(count / total_occurrences * 100, 6) if total_occurrences else 0.0,
            })
        return rows

    def _coverage(n: int) -> float:
        top_sum = sum(c for _tid, c in ranked[:n])
        return round(top_sum / total_occurrences * 100, 6) if total_occurrences else 0.0

    freqs = sorted(counts.values())
    median_freq = 0.0
    if freqs:
        mid = len(freqs) // 2
        median_freq = float(freqs[mid]) if len(freqs) % 2 == 1 else (freqs[mid - 1] + freqs[mid]) / 2.0

    singleton = sum(1 for c in freqs if c == 1)
    le5 = sum(1 for c in freqs if c <= 5)
    le10 = sum(1 for c in freqs if c <= 10)

    return {
        "total_token_occurrences": total_occurrences,
        "unique_token_ids_used": unique_ids,
        "top_20": _top(20) if 20 in top_ns else _top(min(20, unique_ids)),
        "top_50": _top(50) if 50 in top_ns else _top(min(50, unique_ids)),
        "top_100": _top(100) if 100 in top_ns else _top(min(100, unique_ids)),
        "coverage_top_20_percent": _coverage(20),
        "coverage_top_50_percent": _coverage(50),
        "coverage_top_100_percent": _coverage(100),
        "median_frequency_among_used_ids": median_freq,
        "singleton_token_count": singleton,
        "token_ids_le_5_occurrences": le5,
        "token_ids_le_10_occurrences": le10,
        # Full sorted frequency list (id, count) for the Zipf plot — bounded
        # by vocab size, not corpus size, so safe to keep in memory/JSON.
        "zipf_distribution": [{"rank": i + 1, "token_id": tid, "count": c} for i, (tid, c) in enumerate(ranked)],
    }


# ---------------------------------------------------------------------------
# 6. Tokenization examples
# ---------------------------------------------------------------------------

def build_examples(sp, examples: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """*examples* is a list of (category, text) pairs supplied by the
    caller (each top-level script owns its own language-specific example
    sentences — nothing here is language-specific)."""
    rows = []
    for category, text in examples:
        pieces = sp.EncodeAsPieces(text)
        ids = sp.EncodeAsIds(text)
        words = len(text.split())
        chars = len(text)
        tokens = len(ids)
        rows.append({
            "category": category,
            "text": text,
            "pieces": pieces,
            "token_ids": ids,
            "word_count": words,
            "token_count": tokens,
            "character_count": chars,
            "fertility": round(tokens / words, 6) if words else 0.0,
            "characters_per_token": round(chars / tokens, 6) if tokens else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# 7. Binary token corpus analysis (reads existing outputs — never
# regenerates the binary corpus)
# ---------------------------------------------------------------------------

def binary_corpus_analysis(tokenized_dir: str, splits: Sequence[str], logger) -> Dict[str, Any]:
    """Read tokenization_metadata.json (script 14's output, already on
    disk) plus the .bin files' actual sizes. Verifies
    stored_tokens == content_tokens + bos_tokens + eos_tokens and that the
    .bin file size matches stored_tokens * dtype_size — never regenerates
    or re-tokenizes anything."""
    meta_path = os.path.join(tokenized_dir, "tokenization_metadata.json")
    result: Dict[str, Any] = {"metadata_file": meta_path, "available": os.path.exists(meta_path), "splits": {}}
    if not result["available"]:
        logger.warning(f"tokenization_metadata.json not found at {meta_path} — binary corpus section will be empty.")
        return result

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dtype = meta.get("dtype", "uint32")
    dtype_bytes = {"uint32": 4, "int32": 4, "uint16": 2, "int16": 2}.get(dtype, 4)
    token_counts = meta.get("token_counts", {})

    for split in splits:
        entry = token_counts.get(split, {})
        content = entry.get("content_tokens")
        bos = entry.get("bos_tokens")
        eos = entry.get("eos_tokens")
        stored = entry.get("stored_tokens")
        bin_path = os.path.join(tokenized_dir, f"{split}.bin")
        actual_bytes = os.path.getsize(bin_path) if os.path.exists(bin_path) else None
        expected_bytes = stored * dtype_bytes if stored is not None else None

        reconciles = None
        if content is not None and bos is not None and eos is not None and stored is not None:
            reconciles = (content + bos + eos) == stored
        size_consistent = None
        if actual_bytes is not None and expected_bytes is not None:
            size_consistent = actual_bytes == expected_bytes

        result["splits"][split] = {
            "content_tokens": content,
            "bos_tokens": bos,
            "eos_tokens": eos,
            "stored_tokens": stored,
            "stored_equals_content_plus_bos_plus_eos": reconciles,
            "binary_file": bin_path,
            "binary_file_exists": os.path.exists(bin_path),
            "binary_file_size_bytes": actual_bytes,
            "binary_file_size_mb": round(actual_bytes / (1024 * 1024), 4) if actual_bytes else None,
            "dtype": dtype,
            "bytes_per_token": dtype_bytes,
            "expected_size_bytes": expected_bytes,
            "size_consistent_with_dtype": size_consistent,
        }

    result["target"] = {
        "target_content_tokens": meta.get("target_content_tokens"),
        "indiccorp_target_tokens": meta.get("indiccorp_target_tokens"),
        "manual_target_tokens": meta.get("manual_target_tokens"),
        "total_target_tokens": meta.get("total_target_tokens"),
        "difference_from_target": meta.get("difference_from_target"),
        "percentage_from_target": meta.get("percentage_from_target"),
    }
    return result


def tokenization_performance(tokenized_dir: str) -> Dict[str, Any]:
    """script 14 does not persist elapsed-time/throughput to
    tokenization_metadata.json (only logs it) — so this always reports
    'available: False' with no invented numbers, UNLESS a future version of
    script 14 starts recording e.g. 'elapsed_seconds'/'*_per_second' keys,
    in which case they are surfaced here verbatim."""
    meta_path = os.path.join(tokenized_dir, "tokenization_metadata.json")
    if not os.path.exists(meta_path):
        return {"available": False, "note": "tokenization_metadata.json not found."}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    perf_keys = [k for k in meta.keys() if "elapsed" in k or "_per_second" in k or "_per_sec" in k]
    if not perf_keys:
        return {
            "available": False,
            "note": "script 14 (tokenize_corpus.py) does not record timing/throughput "
                    "in tokenization_metadata.json — not invented here.",
        }
    return {"available": True, **{k: meta[k] for k in perf_keys}}


# ---------------------------------------------------------------------------
# 9. Data integrity
# ---------------------------------------------------------------------------

def read_verification_report(metadata_dir: str) -> Dict[str, Any]:
    """Read script 15's verification_report.json if present — reused, not
    re-derived (leakage/decontamination re-verification is expensive and
    already authoritative there). Returns {'available': False} if absent."""
    path = os.path.join(metadata_dir, "verification_report.json")
    if not os.path.exists(path):
        return {"available": False, "file": path}
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    decon = report.get("decontamination_verification", {})
    return {
        "available": True,
        "file": path,
        "all_passed": report.get("all_passed"),
        "exact_matches": decon.get("exact_matches"),
        "near_dup_matches": decon.get("near_dup_matches"),
    }


def data_integrity_checks(
    tokenizer_model_exists: bool,
    tokenizer_loadable: bool,
    train_path: str,
    validation_path: str,
    test_path: str,
    indiccorp_path: str,
    manual_path: str,
    train_tokens: int,
    validation_tokens: int,
    test_tokens: int,
    invariant_indiccorp_plus_manual_equals_total: Optional[bool],
    binary_reconciliation: Dict[str, Any],
    manual_fraction: float,
    manual_min_fraction: float,
    indiccorp_target: int,
    manual_target: int,
    total_target: int,
    indiccorp_tokens: int,
    manual_tokens: int,
    total_tokens: int,
    leakage: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the JSON report's 'integrity' section — pure aggregation of
    booleans already computed elsewhere in this module or passed in by the
    caller; performs no I/O of its own beyond the existence checks below."""
    checks: Dict[str, Any] = {
        "tokenizer_model_exists": tokenizer_model_exists,
        "tokenizer_vocabulary_loadable": tokenizer_loadable,
        "train_input_exists": os.path.exists(train_path),
        "validation_input_exists": os.path.exists(validation_path),
        "test_input_exists": os.path.exists(test_path),
        "indiccorp_input_exists": os.path.exists(indiccorp_path),
        "manual_input_exists": os.path.exists(manual_path),
        "train_tokens_nonzero": train_tokens > 0,
        "validation_tokens_nonzero": validation_tokens > 0,
        "test_tokens_nonzero": test_tokens > 0,
        "indiccorp_plus_manual_equals_total": invariant_indiccorp_plus_manual_equals_total,
        "meets_indiccorp_target": indiccorp_tokens >= indiccorp_target if indiccorp_target else None,
        "meets_manual_target": manual_tokens >= manual_target if manual_target else None,
        "meets_total_target": total_tokens >= total_target if total_target else None,
        "meets_manual_fraction_requirement": manual_fraction >= manual_min_fraction,
        "binary_stored_equals_content_plus_bos_plus_eos": all(
            v.get("stored_equals_content_plus_bos_plus_eos") in (True, None)
            for v in binary_reconciliation.get("splits", {}).values()
        ) if binary_reconciliation.get("available") else None,
        "binary_size_consistent_with_dtype": all(
            v.get("size_consistent_with_dtype") in (True, None)
            for v in binary_reconciliation.get("splits", {}).values()
        ) if binary_reconciliation.get("available") else None,
        "leakage_available": leakage.get("available", False),
        "leakage_all_passed": leakage.get("all_passed"),
    }
    # Overall status: every boolean check that is not None must be True.
    relevant = [v for v in checks.values() if isinstance(v, bool)]
    checks["overall_status"] = "PASS" if all(relevant) else "FAIL"
    return checks


# ---------------------------------------------------------------------------
# 11. Human-readable TXT report
# ---------------------------------------------------------------------------

def render_txt_report(language_label: str, report: Dict[str, Any]) -> str:
    """Render the TXT report for ONE language from its own already-built
    JSON-shaped `report` dict. Pure string formatting — no I/O, no
    cross-language state."""
    tok = report["tokenizer"]
    corpus = report["corpus"]
    sources = report["sources"]
    quality = report["tokenizer_quality"]
    freq = report["frequency_analysis"]
    binc = report["binary_corpus"]
    targets = report["targets"]
    integrity = report["integrity"]

    manual_bd = sources["manual"]["breakdown"]
    if sources["manual"]["breakdown_kind"] == "sidecar":
        wiki_tok = manual_bd.get("by_source_type", {}).get("wikipedia", {}).get("content_tokens", 0)
        news_tok = manual_bd.get("by_source_type", {}).get("news", {}).get("content_tokens", 0)
        manual_lines = f"    Wikipedia: {wiki_tok:,}\n    News:      {news_tok:,}\n"
    elif manual_bd.get("available"):
        by_name = manual_bd.get("by_source_name", {})
        manual_lines = "".join(
            f"    {name.capitalize()}: {info.get('content_tokens', 0):,}\n"
            for name, info in by_name.items()
        )
    else:
        manual_lines = "    (breakdown not available)\n"

    lines = []
    lines.append("=" * 60)
    lines.append("FINAL TOKENIZER + CORPUS ANALYSIS")
    lines.append(language_label.upper())
    lines.append("=" * 60)
    lines.append("")
    lines.append("TOKENIZER")
    lines.append(f"  Model:              {tok['model_path']}")
    lines.append(f"  Vocabulary:         {tok['requested_vocab_size']}")
    lines.append(f"  Actual vocabulary:  {tok['actual_vocab_size']}")
    lines.append(f"  Character coverage: {tok['character_coverage']}")
    lines.append(f"  Byte fallback:      {tok['byte_fallback']}")
    lines.append("")
    lines.append("CORPUS")
    lines.append(f"  Train lines:            {corpus['train']['lines']:,}")
    lines.append(f"  Train words:            {corpus['train']['words']:,}")
    lines.append(f"  Train characters:       {corpus['train']['characters']:,}")
    lines.append(f"  Train BPE tokens:       {corpus['train']['content_tokens']:,}")
    lines.append(f"  Validation BPE tokens:  {corpus['validation']['content_tokens']:,}")
    lines.append(f"  Test BPE tokens:        {corpus['test']['content_tokens']:,}")
    lines.append(f"  Total BPE tokens:       {corpus['combined']['content_tokens']:,}")
    lines.append("")
    lines.append("SOURCES")
    lines.append(f"  IndicCorp: {sources['indiccorp']['content_tokens']:,}")
    lines.append(f"  Manual:    {sources['manual']['content_tokens']:,}")
    lines.append(manual_lines.rstrip("\n"))
    lines.append(f"  Manual fraction: {sources['manual_fraction'] * 100:.2f}%")
    lines.append(f"  Total: {sources['total']['content_tokens']:,}")
    lines.append("")
    lines.append("TOKENIZER QUALITY (measured on validation)")
    lines.append(f"  Fertility:              {quality['fertility']:.4f}")
    lines.append(f"  Tokens/word:            {quality['tokens_per_word']:.4f}")
    lines.append(f"  UNK rate:               {quality['unk_rate']:.6%}")
    lines.append(f"  Unique IDs used:        {quality['unique_token_ids_used']:,}")
    lines.append(f"  Vocabulary utilization: {quality['vocabulary_utilization_percent']:.2f}%")
    lines.append("")
    lines.append("TOKEN FREQUENCY (measured on train)")
    lines.append(f"  Unique IDs:    {freq['unique_token_ids_used']:,}")
    lines.append(f"  Top 20 coverage:  {freq['coverage_top_20_percent']:.2f}%")
    lines.append(f"  Top 50 coverage:  {freq['coverage_top_50_percent']:.2f}%")
    lines.append(f"  Top 100 coverage: {freq['coverage_top_100_percent']:.2f}%")
    lines.append(f"  Singleton IDs: {freq['singleton_token_count']:,}")
    lines.append(f"  <=5 frequency:  {freq['token_ids_le_5_occurrences']:,}")
    lines.append(f"  <=10 frequency: {freq['token_ids_le_10_occurrences']:,}")
    lines.append("")
    lines.append("BINARY CORPUS")
    train_bin = binc.get("splits", {}).get("train", {})
    lines.append(f"  Content tokens: {train_bin.get('content_tokens')}")
    lines.append(f"  BOS:            {train_bin.get('bos_tokens')}")
    lines.append(f"  EOS:            {train_bin.get('eos_tokens')}")
    lines.append(f"  Stored tokens:  {train_bin.get('stored_tokens')}")
    lines.append(f"  Binary size:    {train_bin.get('binary_file_size_mb')} MB")
    lines.append(f"  Bytes/token:    {train_bin.get('bytes_per_token')}")
    lines.append("")
    lines.append("TARGETS")
    lines.append(f"  IndicCorp: {sources['indiccorp']['content_tokens']:,} / {targets['indiccorp_target_tokens']:,} "
                 f"[{'PASS' if integrity.get('meets_indiccorp_target') else 'FAIL'}]")
    lines.append(f"  Manual:    {sources['manual']['content_tokens']:,} / {targets['manual_target_tokens']:,} "
                 f"[{'PASS' if integrity.get('meets_manual_target') else 'FAIL'}]")
    lines.append(f"  Total:     {sources['total']['content_tokens']:,} / {targets['total_target_tokens']:,} "
                 f"[{'PASS' if integrity.get('meets_total_target') else 'FAIL'}]")
    lines.append(f"  Manual fraction: {sources['manual_fraction'] * 100:.2f}% / "
                 f">= {targets['manual_min_fraction'] * 100:.0f}% "
                 f"[{'PASS' if integrity.get('meets_manual_fraction_requirement') else 'FAIL'}]")
    lines.append(f"  Overall: {integrity['overall_status']}")
    lines.append("")
    lines.append("INTEGRITY")
    lines.append(f"  Token count reconciliation: {'PASS' if integrity.get('indiccorp_plus_manual_equals_total') else 'FAIL'}")
    lines.append(f"  BOS/EOS reconciliation:     {integrity.get('binary_stored_equals_content_plus_bos_plus_eos')}")
    leak = report.get("leakage", {})
    if leak.get("available"):
        lines.append(f"  Leakage:                    all_passed={leak.get('all_passed')}")
    else:
        lines.append("  Leakage:                    not available (run script 15)")
    lines.append(f"  Binary validation:          {integrity.get('binary_size_consistent_with_dtype')}")
    lines.append(f"  Overall status:             {integrity['overall_status']}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("TOKENIZATION EXAMPLES")
    lines.append("=" * 60)
    for ex in report.get("examples", []):
        lines.append("")
        lines.append(f"[{ex['category']}] {ex['text']}")
        lines.append(f"  pieces: {ex['pieces']}")
        lines.append(f"  words={ex['word_count']} tokens={ex['token_count']} "
                     f"fertility={ex['fertility']:.4f} chars/token={ex['characters_per_token']:.4f}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("END")
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Optional figures
# ---------------------------------------------------------------------------

def maybe_plot_figures(report: Dict[str, Any], figures_dir: str, language_label: str, logger) -> List[str]:
    """Best-effort figure generation — never fatal. Returns the list of
    files actually written (empty list if matplotlib is unavailable or
    plotting fails for any reason)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping optional figures.")
        return []

    written: List[str] = []
    os.makedirs(figures_dir, exist_ok=True)

    try:
        zipf = report["frequency_analysis"]["zipf_distribution"]
        if zipf:
            ranks = [r["rank"] for r in zipf]
            freqs = [r["count"] for r in zipf]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.loglog(ranks, freqs, marker=".", linestyle="none", markersize=2)
            ax.set_xlabel("Token rank (log)")
            ax.set_ylabel("Frequency (log)")
            ax.set_title(f"{language_label} — Token Frequency Distribution (Zipf)")
            path = os.path.join(figures_dir, "token_frequency_distribution.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    except Exception as e:  # pragma: no cover - best-effort only
        logger.warning(f"Could not render token_frequency_distribution.png: {e}")

    try:
        q = report["tokenizer_quality"]
        fig, ax = plt.subplots(figsize=(6, 5))
        metrics = ["Fertility", "Tokens/word", "UNK rate (%)"]
        values = [q["fertility"], q["tokens_per_word"], q["unk_rate"] * 100]
        ax.bar(metrics, values, color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_title(f"{language_label} — Tokenizer Quality")
        path = os.path.join(figures_dir, "tokenizer_quality_comparison.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not render tokenizer_quality_comparison.png: {e}")

    try:
        used = q["unique_token_ids_used"]
        vocab = report["tokenizer"]["actual_vocab_size"]
        unused = max(0, vocab - used)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie([used, unused], labels=["Used", "Unused"], autopct="%1.1f%%",
               colors=["#4C72B0", "#DDDDDD"])
        ax.set_title(f"{language_label} — Vocabulary Utilization")
        path = os.path.join(figures_dir, "vocabulary_utilization.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not render vocabulary_utilization.png: {e}")

    return written
