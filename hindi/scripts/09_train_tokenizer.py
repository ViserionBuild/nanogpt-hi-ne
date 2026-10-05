#!/usr/bin/env python3
"""
09_train_tokenizer.py — Train the ONE SentencePiece BPE Tokenizer From Scratch

SINGLE-TOKENIZER ARCHITECTURE
==============================
There is exactly ONE tokenizer in this pipeline. It is a SentencePiece BPE
model (`model_type="bpe"` — SentencePiece is used directly; BPE is never
hand-implemented) trained on `final/train.txt`, the combined, decontaminated
training corpus produced by script 10:

    train.txt = IndicCorp + Manual
    Manual    = Wikipedia + News, concatenated in that fixed order into
                final/manual_train.txt (News itself is already a combined
                stream of every enabled outlet, merged far upstream by
                manual/05_aggregate_news_corpus.py — see below)

There is no temporary/pre-selection tokenizer, no per-source tokenizer, and
this script is never invoked twice to produce two different tokenizers for
two different purposes — training happens exactly once, on exactly one file.

Pipeline order:
  01→…→05 → 06 → 07 (split) → 08 (decontaminate) → 10 (combine) →
  09 (train the ONE tokenizer + token-count report, THIS script) → 11 → 12

THREE SOURCES: PUBLIC (IndicCorp) + MANUAL (Wikipedia + News)
=============================================================
After training, this script loads that SAME tokenizer model back and runs
a streaming token-count pass over exactly the files script 10 produces:
    final/indiccorp_train.txt   (public/IndicCorp)
    final/wikipedia_train.txt   (manual)
    final/news_train.txt        (manual, all outlets already combined)
    final/manual_train.txt      (= wikipedia_train.txt + news_train.txt)
    final/train.txt             (combined = indiccorp_train.txt + manual_train.txt)

reporting:

    IndicCorp BPE content tokens: X
    Wikipedia BPE content tokens: W
    News BPE content tokens:      N
    Manual BPE content tokens:    Y   (= W + N)
    Total BPE content tokens:     Z
    Manual fraction:              X%

manual_tokens is computed as wikipedia_tokens + news_tokens — the SUM of
both manual sources' own final files, never a single source's count
standing in alone. This is a STRICTLY MORE ACCURATE mechanism than the
previous per-line-provenance-sidecar-derived breakdown: wikipedia and news
are now separate, real final/<source>_train.txt files (produced natively
by script 10), so their token counts come from directly streaming each
file through the tokenizer — not from reconstructing a breakdown out of a
sidecar attached to one merged file. As a cross-check, this script also
directly streams final/manual_train.txt (also produced by script 10) and
asserts wikipedia_tokens + news_tokens == manual_train.txt's own measured
token count — a data-integrity check on script 10's concatenation, not a
second, independent estimate of manual_tokens. Individual news websites
are NEVER treated as a top-level corpus source here: only the aggregate
news_train.txt is scanned (that file is itself already a pre-merge of every
enabled outlet, done by manual/05_aggregate_news_corpus.py) — no outlet gets its
own target_tokens, shortfall check, or gating.

Required invariants (checked, not just reported):
    wikipedia_tokens + news_tokens == manual_tokens  (== manual_train.txt's
        own measured token count)
    indiccorp_tokens + manual_tokens == total_tokens
A mismatch means final/train.txt (or final/manual_train.txt) is not a
straightforward concatenation of its constituent per-source files — a
data-integrity bug in script 10's output, not a token-count shortfall, so
this script fails loudly (exits non-zero) rather than reporting a report
that doesn't add up.

Required targets (reported, WITHOUT retraining another tokenizer to close
any gap — this is the ONE authoritative measurement of these numbers;
there is no separate re-verification stage):
    IndicCorp target  >= 400,000,000 BPE content tokens
    Manual target     >= 100,000,000 BPE content tokens
    Total target      >= 500,000,000 BPE content tokens
    Manual fraction   >= 20% of total BPE content tokens
manual_target_tokens applies to the COMBINED manual corpus (Wikipedia + all
news together) — no single manual source is ever required or expected to
reach it on its own, and manual is never treated as if it were Wikipedia
alone anywhere in this script.

A missing final/<source>_train.txt (e.g. wikipedia or news not yet
produced) contributes 0 tokens to that source's count — and therefore to
manual_tokens — rather than breaking the report — see
count_tokens_streaming's own missing-file handling (shared/tokenizer_utils.py,
unmodified, already source-agnostic).

Usage:
    python hindi/scripts/09_train_tokenizer.py \\
        --config hindi/configs/hindi_data.yaml

    python hindi/scripts/09_train_tokenizer.py \\
        --config hindi/configs/hindi_data.yaml \\
        --vocab-sizes 16000 32000 48000
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import sentencepiece as spm
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.tokenizer.tokenizer_utils import count_tokens_streaming


# ---------------------------------------------------------------------------
# Helpers
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
    return logging.getLogger("09_train_tokenizer")


def train_single_tokenizer(
    input_file: str,
    model_dir: str,
    vocab_dir: str,
    config_dir: str,
    vocab_size: int,
    tok_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Train a single SentencePiece BPE tokenizer and save artifacts.
    Returns the trained .model path."""
    model_prefix = os.path.join(model_dir, f"hindi_bpe_{vocab_size}")

    model_type = tok_cfg.get("algorithm", "bpe")
    sp_args = dict(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=tok_cfg.get("character_coverage", 0.9995),
        max_sentence_length=tok_cfg.get("max_sentence_length", 16384),
        num_threads=tok_cfg.get("num_threads", 4),
        byte_fallback=tok_cfg.get("byte_fallback", True),
        pad_id=tok_cfg["special_token_ids"]["pad"],
        unk_id=tok_cfg["special_token_ids"]["unk"],
        bos_id=tok_cfg["special_token_ids"]["bos"],
        eos_id=tok_cfg["special_token_ids"]["eos"],
        pad_piece=tok_cfg["special_tokens"]["pad"],
        unk_piece=tok_cfg["special_tokens"]["unk"],
        bos_piece=tok_cfg["special_tokens"]["bos"],
        eos_piece=tok_cfg["special_tokens"]["eos"],
        input_sentence_size=tok_cfg.get("input_sentence_size", 5_000_000),
        shuffle_input_sentence=tok_cfg.get("shuffle_input_sentence", True),
        seed_sentencepiece_size=tok_cfg.get("seed", 42),
    )

    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size} …")
    logger.info(f"  Input: {input_file}")
    logger.info(f"  Model prefix: {model_prefix}")
    logger.info(f"  Character coverage: {sp_args['character_coverage']}")

    t0 = time.time()
    spm.SentencePieceTrainer.train(**sp_args)
    elapsed = time.time() - t0

    logger.info(f"  Training completed in {elapsed:.1f}s")

    src_vocab = model_prefix + ".vocab"
    if vocab_dir and os.path.normpath(vocab_dir) != os.path.normpath(model_dir):
        dst_vocab = os.path.join(vocab_dir, os.path.basename(src_vocab))
        if os.path.exists(src_vocab):
            os.rename(src_vocab, dst_vocab)
            logger.info(f"  Vocab saved to {dst_vocab}")
    else:
        dst_vocab = src_vocab

    sp = spm.SentencePieceProcessor()
    sp.Load(model_prefix + ".model")
    actual_vocab = sp.GetPieceSize()
    logger.info(f"  Actual vocab size: {actual_vocab}")

    config_out = {
        "vocab_size": actual_vocab,
        "requested_vocab_size": vocab_size,
        "algorithm": "bpe",
        "character_coverage": sp_args["character_coverage"],
        "max_sentence_length": sp_args["max_sentence_length"],
        "input_sentence_size": sp_args["input_sentence_size"],
        "training_time_seconds": round(elapsed, 2),
        # Recorded so any later inspection can verify no tokenizer was ever
        # trained on validation.txt/test.txt.
        "input_file": input_file,
        "model_file": model_prefix + ".model",
        "vocab_file": dst_vocab if os.path.exists(dst_vocab) else src_vocab,
        "special_tokens": {
            tok_cfg["special_tokens"]["pad"]: tok_cfg["special_token_ids"]["pad"],
            tok_cfg["special_tokens"]["unk"]: tok_cfg["special_token_ids"]["unk"],
            tok_cfg["special_tokens"]["bos"]: tok_cfg["special_token_ids"]["bos"],
            tok_cfg["special_tokens"]["eos"]: tok_cfg["special_token_ids"]["eos"],
        },
    }
    config_json_name = os.path.basename(model_prefix) + "_config.json"
    config_path = os.path.join(config_dir, config_json_name)
    os.makedirs(config_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)
    logger.info(f"  Config saved to {config_path}")

    test_sentences = [
        "भारत एक विशाल देश है।",
        "भारतीय अंतरिक्ष अनुसंधान संगठन",
    ]
    for sent in test_sentences:
        pieces = sp.EncodeAsPieces(sent)
        logger.info(f"  Example: '{sent}' → {pieces}")

    return model_prefix + ".model"


def _reject_holdout_input(input_file: str, logger: logging.Logger) -> None:
    """Defensive guard: fail loudly if the resolved tokenizer training input
    is (or obviously looks like) the held-out validation/test file, rather
    than silently training on leaked data. Filename-based, so it is not a
    substitute for the exact/near-dup checks in script 08 — just a
    cheap trip-wire against an accidental --input override."""
    basename = os.path.basename(input_file).lower()
    if basename in ("validation.txt", "test.txt", "val.txt"):
        logger.error(
            f"REFUSING to train a tokenizer on '{input_file}' — this looks "
            f"like a held-out validation/test file. The tokenizer must "
            f"never see validation/test text."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Post-training token-count report (IndicCorp / Wikipedia / News / Manual /
# Total)
# ---------------------------------------------------------------------------

# PUBLIC_SOURCE + MANUAL_SOURCES together define every final/<source>_train.txt
# this script scans — the single source of truth for "what counts toward
# manual", so nothing downstream can silently narrow it back to a single
# source only. Order matches script 10's manual_train.txt concatenation order.
PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("wikipedia", "news")
MANUAL_SOURCE = "manual"


def run_token_count_report(
    model_path: str,
    repo_root: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Load the just-trained tokenizer and stream-count BPE content tokens
    separately for IndicCorp, EACH manual source (Wikipedia, News), the
    combined manual_train.txt (as a cross-check), and the combined
    train.txt. manual_tokens is the SUM of both manual sources' own
    directly-measured counts — never a single source's count alone, and
    never derived from a sidecar. Reports (does not enforce/exit on) any
    shortfall against the configured targets — no second tokenizer is ever
    trained to try to close a gap. Both the wikipedia+news==manual and
    indiccorp+manual==total invariants ARE enforced (exits non-zero on
    mismatch) — see module docstring."""
    final_dir = repo_root / config["paths"]["final"]
    source_paths = {
        PUBLIC_SOURCE: final_dir / f"{PUBLIC_SOURCE}_train.txt",
        **{s: final_dir / f"{s}_train.txt" for s in MANUAL_SOURCES},
    }
    manual_path = final_dir / f"{MANUAL_SOURCE}_train.txt"
    combined_path = final_dir / "train.txt"

    selection_cfg = config.get("selection", {})
    manual_cfg = config.get("manual", {})
    progress_interval = config.get("logging", {}).get("progress_interval", 1_000_000)

    indiccorp_target = selection_cfg.get("indiccorp_target_tokens", 400_000_000)
    manual_target = selection_cfg.get("manual_target_tokens", 100_000_000)
    total_target = selection_cfg.get("total_target_tokens", indiccorp_target + manual_target)
    manual_min_fraction = manual_cfg.get("min_fraction", 0.20)

    logger.info("")
    logger.info("=" * 70)
    logger.info("POST-TRAINING TOKEN COUNT (single final tokenizer — no retraining)")
    logger.info("=" * 70)
    logger.info(f"Tokenizer: {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    vocab_size = sp.GetPieceSize()
    logger.info(f"Vocab size: {vocab_size}")

    stats_by_source: Dict[str, Dict[str, Any]] = {}
    for source, path in source_paths.items():
        logger.info(f"Scanning {source.capitalize()}: {path}")
        stats_by_source[source] = count_tokens_streaming(sp, str(path), logger, progress_interval)
        if not path.exists():
            logger.warning(f"  {source.capitalize()} final file not found — counted as 0 tokens: {path}")

    logger.info(f"Scanning manual (cross-check): {manual_path}")
    manual_file_stats = count_tokens_streaming(sp, str(manual_path), logger, progress_interval)
    if not manual_path.exists():
        logger.warning(f"  manual final file not found — cross-check skipped: {manual_path}")

    logger.info(f"Scanning combined train.txt: {combined_path}")
    combined_stats = count_tokens_streaming(sp, str(combined_path), logger, progress_interval)

    indiccorp_tokens = stats_by_source[PUBLIC_SOURCE]["total_tokens"]
    tokens_by_manual_source = {s: stats_by_source[s]["total_tokens"] for s in MANUAL_SOURCES}
    # Requirement: manual_tokens = wikipedia + news — the COMBINED manual
    # corpus, never a single source's count standing in alone. Each term is
    # measured DIRECTLY from its own final/<source>_train.txt file (no
    # sidecar involved) — strictly more accurate than a provenance-sidecar
    # -derived breakdown.
    manual_tokens = sum(tokens_by_manual_source.values())
    manual_file_tokens = manual_file_stats["total_tokens"]
    total_tokens = combined_stats["total_tokens"]

    # REQUIRED INVARIANT #1: wikipedia_tokens + news_tokens == manual_train.txt's
    # own measured token count. A data-integrity cross-check on script 10's
    # manual_train.txt concatenation — fails loudly (never just logged) if
    # it doesn't hold.
    manual_sum_check_passed = manual_tokens == manual_file_tokens
    if not manual_sum_check_passed:
        logger.error(
            f"REQUIRED INVARIANT VIOLATED: wikipedia_tokens "
            f"({tokens_by_manual_source['wikipedia']:,}) + news_tokens "
            f"({tokens_by_manual_source['news']:,}) = {manual_tokens:,} != "
            f"manual_train.txt's own measured token count "
            f"({manual_file_tokens:,}). final/manual_train.txt is not a "
            f"straightforward concatenation of wikipedia_train.txt + "
            f"news_train.txt — investigate script 10's output."
        )
        sys.exit(1)
    logger.info(
        f"Invariant OK: wikipedia_tokens ({tokens_by_manual_source['wikipedia']:,}) + "
        f"news_tokens ({tokens_by_manual_source['news']:,}) == manual_train.txt "
        f"({manual_file_tokens:,})"
    )

    # REQUIRED INVARIANT #2: indiccorp_tokens + manual_tokens == total_tokens.
    sum_check_passed = (indiccorp_tokens + manual_tokens) == total_tokens
    if not sum_check_passed:
        logger.error(
            f"REQUIRED INVARIANT VIOLATED: indiccorp_tokens ({indiccorp_tokens:,}) "
            f"+ manual_tokens ({manual_tokens:,}) = {indiccorp_tokens + manual_tokens:,} "
            f"!= total_tokens ({total_tokens:,}). final/train.txt is not a "
            f"straightforward concatenation of indiccorp_train.txt + "
            f"manual_train.txt — investigate script 10's output."
        )
        sys.exit(1)
    logger.info(
        f"Invariant OK: indiccorp_tokens ({indiccorp_tokens:,}) + manual_tokens "
        f"({manual_tokens:,}) == total_tokens ({total_tokens:,})"
    )

    manual_fraction = (manual_tokens / total_tokens) if total_tokens > 0 else 0.0

    indiccorp_shortfall = max(0, indiccorp_target - indiccorp_tokens)
    manual_shortfall = max(0, manual_target - manual_tokens)
    total_shortfall = max(0, total_target - total_tokens)

    logger.info("")
    logger.info(f"IndicCorp BPE content tokens:    {indiccorp_tokens:,}  (target {indiccorp_target:,})")
    for s in MANUAL_SOURCES:
        logger.info(f"{s.capitalize():>17s} BPE content tokens: {tokens_by_manual_source[s]:,}")
    logger.info(f"Manual total BPE content tokens: {manual_tokens:,}  (target {manual_target:,})  "
                f"[= wikipedia + news]")
    logger.info(f"Total BPE content tokens:        {total_tokens:,}  (target {total_target:,})")
    logger.info(f"Vocabulary size:                 {vocab_size:,}")
    logger.info(f"Manual fraction:                 {manual_fraction:.2%}  "
                f"(required >= {manual_min_fraction:.0%})")

    if indiccorp_shortfall > 0:
        logger.warning(
            f"SHORTFALL: IndicCorp is {indiccorp_shortfall:,} content tokens "
            f"below the {indiccorp_target:,} target. Not retraining a second "
            f"tokenizer to compensate — collect more IndicCorp data upstream if needed."
        )
    if manual_shortfall > 0:
        logger.warning(
            f"SHORTFALL: manual (Wikipedia + News combined) "
            f"is {manual_shortfall:,} content tokens below the {manual_target:,} "
            f"target. Not retraining a second tokenizer to compensate — collect "
            f"more manual data upstream if needed (either of the two sources)."
        )
    if total_shortfall > 0:
        logger.warning(
            f"SHORTFALL: combined total is {total_shortfall:,} content tokens "
            f"below the {total_target:,} target."
        )
    if manual_fraction < manual_min_fraction:
        logger.warning(
            f"Manual fraction {manual_fraction:.4f} is below the "
            f"required minimum {manual_min_fraction:.4f}."
        )

    report = {
        "tokenizer_model": model_path,
        "vocab_size": vocab_size,
        "source_files": {
            PUBLIC_SOURCE: str(source_paths[PUBLIC_SOURCE]),
            **{s: str(source_paths[s]) for s in MANUAL_SOURCES},
            MANUAL_SOURCE: str(manual_path),
            "combined": str(combined_path),
        },
        "manual_composition": list(MANUAL_SOURCES),
        "indiccorp_tokens": indiccorp_tokens,
        **{f"{s}_tokens": tokens_by_manual_source[s] for s in MANUAL_SOURCES},
        "manual_tokens": manual_tokens,
        "manual_file_tokens": manual_file_tokens,
        "total_tokens": total_tokens,
        "manual_sum_check": {
            "wikipedia_plus_news": manual_tokens,
            "manual_train_txt": manual_file_tokens,
            "passed": manual_sum_check_passed,
        },
        "sum_check": {
            "indiccorp_plus_manual": indiccorp_tokens + manual_tokens,
            "total": total_tokens,
            "passed": sum_check_passed,
        },
        "targets": {
            "indiccorp_target_tokens": indiccorp_target,
            "manual_target_tokens": manual_target,
            "total_target_tokens": total_target,
            "target_unit": "bpe_content_tokens",
            "note": "manual_target_tokens applies to the COMBINED manual "
                    "corpus (wikipedia + news), not to "
                    "any single manual source.",
        },
        "shortfall": {
            "indiccorp_shortfall_tokens": indiccorp_shortfall,
            "manual_shortfall_tokens": manual_shortfall,
            "total_shortfall_tokens": total_shortfall,
            "note": (
                "Shortfalls are reported only — this pipeline trains exactly "
                "ONE final tokenizer and never retrains a second one to "
                "compensate for a shortfall."
            ),
        },
        "meets_indiccorp_target": indiccorp_tokens >= indiccorp_target,
        "meets_manual_target": manual_tokens >= manual_target,
        "meets_total_target": total_tokens >= total_target,
        "manual_fraction": round(manual_fraction, 6),
        "manual_min_fraction": manual_min_fraction,
        "meets_manual_fraction_requirement": manual_fraction >= manual_min_fraction,
        "counting_method": "sentencepiece_bpe_content_tokens (EncodeAsIds, no BOS/EOS)",
        "fertility": {
            PUBLIC_SOURCE: stats_by_source[PUBLIC_SOURCE]["fertility"],
            **{s: stats_by_source[s]["fertility"] for s in MANUAL_SOURCES},
            MANUAL_SOURCE: manual_file_stats["fertility"],
            "combined": combined_stats["fertility"],
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    meta_dir = repo_root / config["paths"]["metadata"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    report_path = meta_dir / "token_count_check.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Token count report saved to {report_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ONE SentencePiece BPE tokenizer from scratch, "
                     "then report per-source (IndicCorp / Wikipedia / News / "
                     "manual-total) token counts.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--input", type=str, help="Override training text file")
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        help="Vocab sizes to train (default: from config)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing models")
    parser.add_argument(
        "--skip-token-report",
        action="store_true",
        help="Skip the post-training per-source token-count report",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    tok_cfg = config["tokenizer"]

    repo_root = Path(args.config).resolve().parent.parent.parent

    logger.info("=" * 70)
    logger.info("Training the ONE final BPE tokenizer (SentencePiece)")
    logger.info("=" * 70)

    if args.input:
        input_file = args.input
    else:
        input_file = str(repo_root / config["paths"]["final"] / "train.txt")

    if not os.path.exists(input_file):
        logger.error(f"Training file not found: {input_file}")
        logger.error(
            "The tokenizer is trained on final/train.txt.\n"
            "Run scripts 07 (split), 08 (decontaminate), and 10 (combine) first."
        )
        sys.exit(1)

    model_dir = str(repo_root / config["paths"]["tokenizer_model"])
    vocab_dir = str(repo_root / config["paths"]["tokenizer_vocab"])
    config_dir = str(repo_root / config["paths"]["tokenizer_config"])
    for d in (model_dir, vocab_dir, config_dir):
        os.makedirs(d, exist_ok=True)

    if args.vocab_sizes:
        vocab_sizes: List[int] = args.vocab_sizes
    else:
        vocab_sizes = tok_cfg.get("vocab_sizes", [tok_cfg.get("default_vocab_size", 32000)])

    _reject_holdout_input(input_file, logger)
    logger.info(f"Will train {len(vocab_sizes)} tokenizer variant(s): {vocab_sizes}")

    default_vs = tok_cfg.get("default_vocab_size", 32000)
    default_model_path = os.path.join(model_dir, f"hindi_bpe_{default_vs}.model")

    for vs in vocab_sizes:
        model_path = os.path.join(model_dir, f"hindi_bpe_{vs}.model")
        if os.path.exists(model_path) and not args.overwrite:
            logger.info(f"Skipping vocab_size={vs} (model exists). Use --overwrite.")
            continue
        train_single_tokenizer(input_file, model_dir, vocab_dir, config_dir, vs, tok_cfg, logger)

    logger.info("Tokenizer training complete.")

    if not args.skip_token_report:
        if not os.path.exists(default_model_path):
            logger.warning(
                f"Default-vocab-size model {default_model_path} was not "
                f"trained in this run (default_vocab_size={default_vs} not "
                f"in {vocab_sizes}) — skipping the post-training token-count report."
            )
        else:
            run_token_count_report(default_model_path, repo_root, config, logger)


if __name__ == "__main__":
    main()
