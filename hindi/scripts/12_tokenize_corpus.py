#!/usr/bin/env python3
"""
12_tokenize_corpus.py — Tokenize Final Corpus into Binary Token IDs

Reads the train / validation / test text splits and converts them into
compact uint32 binary files using the chosen SentencePiece tokenizer.

Memory guarantees:
  • The text is tokenized line-by-line; token IDs are accumulated in a
    bounded buffer and flushed to the output file as uint32 chunks — the
    whole corpus is NEVER held in memory as a Python list or numpy array.
  • Metadata distinguishes CONTENT tokens (ids from the text, BOS/EOS
    excluded) from STORED tokens (content + added BOS/EOS), and the
    combined target (total_target_tokens = indiccorp_target_tokens +
    manual_target_tokens, 500M by default) is measured on train.txt's
    CONTENT tokens only — train.txt is IndicCorp + Wikipedia + News
    together (Wikipedia + News combined = "Manual" for target/fraction
    accounting purposes only), so it is compared against the combined
    total, never against indiccorp_target_tokens (400M) alone.
  • An optional streaming verification pass re-reads each .bin and checks
    the stored-token count and file size.

Usage:
    python hindi/scripts/12_tokenize_corpus.py \
        --config hindi/configs/hindi_data.yaml \
        --tokenizer-model hindi/tokenizer/model/hindi_bpe_32000.model
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import sentencepiece as spm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.tokenizer.tokenizer_utils import count_tokens_streaming

# PUBLIC_SOURCE + MANUAL_SOURCES together define every final/<source>_train.txt
# this script scans for the source breakdown — matches script 10's
# indiccorp/wikipedia/news constants and its final/{source}_train.txt outputs
# (wikipedia_train.txt + news_train.txt concatenated, in that order, also
# produce manual_train.txt for backward-compat reporting — see script 10).
PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("wikipedia", "news")


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
    return logging.getLogger("12_tokenize_corpus")


def tokenize_file(
    sp: spm.SentencePieceProcessor,
    input_path: str,
    output_path: str,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    logger: logging.Logger,
    progress_interval: int = 1_000_000,
    chunk_tokens: int = 1_000_000,
) -> Dict[str, int]:
    """
    Tokenize a text file line-by-line and write a flat uint32 binary.

    IDs are flushed to disk in bounded chunks, so memory stays flat even
    for corpora with hundreds of millions of tokens.

    Returns counts for content / bos / eos / stored tokens.
    """
    content_tokens = 0
    bos_tokens = 0
    eos_tokens = 0
    line_count = 0

    chunk: list = []
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as f, \
         open(output_path, "wb", buffering=1 << 20) as out_f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            ids = sp.EncodeAsIds(line)
            content_tokens += len(ids)
            if add_bos:
                chunk.append(bos_id)
                bos_tokens += 1
            chunk.extend(ids)
            if add_eos:
                chunk.append(eos_id)
                eos_tokens += 1
            line_count += 1

            if len(chunk) >= chunk_tokens:
                np.asarray(chunk, dtype=np.uint32).tofile(out_f)
                chunk = []

            if line_count % progress_interval == 0:
                elapsed = time.time() - t0
                speed = line_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {line_count:>12,} lines | "
                    f"{content_tokens:>14,} content tokens | "
                    f"{speed:,.0f} lines/s"
                )

        if chunk:
            np.asarray(chunk, dtype=np.uint32).tofile(out_f)

    elapsed = time.time() - t0
    stored_tokens = content_tokens + bos_tokens + eos_tokens
    logger.info(
        f"  Tokenization complete: {line_count:,} lines → "
        f"{stored_tokens:,} stored tokens "
        f"(content {content_tokens:,} + bos {bos_tokens:,} + eos {eos_tokens:,}) "
        f"in {elapsed:.1f}s"
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")

    return {
        "lines": line_count,
        "content_tokens": content_tokens,
        "bos_tokens": bos_tokens,
        "eos_tokens": eos_tokens,
        "stored_tokens": stored_tokens,
    }


def verify_bin(path: str, expected_stored: int, logger: logging.Logger) -> bool:
    """Streaming re-read of a .bin to confirm stored-token count and size."""
    expected_bytes = expected_stored * 4
    actual_bytes = os.path.getsize(path)
    ok = True
    if actual_bytes != expected_bytes:
        logger.error(
            f"  File size mismatch for {path}: expected {expected_bytes:,} bytes, "
            f"got {actual_bytes:,}"
        )
        ok = False
    count = 0
    with open(path, "rb") as f:
        while True:
            arr = np.fromfile(f, dtype=np.uint32, count=1_000_000)
            if arr.size == 0:
                break
            count += arr.size
    if count != expected_stored:
        logger.error(
            f"  Token count mismatch for {path}: expected {expected_stored:,}, "
            f"read {count:,}"
        )
        ok = False
    if ok:
        logger.info(f"  Verified {path}: {count:,} tokens, size ok")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize final corpus splits into binary token-ID files."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--tokenizer-model", type=str, help="Path to .model file")
    parser.add_argument("--input-dir", type=str, help="Override input directory")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bins")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the streaming post-write verification pass")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    tok_cfg = config.get("tokenizer", {})
    tokenization_cfg = config.get("tokenization", {})
    progress_interval = config.get("logging", {}).get("progress_interval", 1_000_000)
    chunk_tokens = tokenization_cfg.get("chunk_tokens", 1_000_000)
    do_verify = tokenization_cfg.get("verify", True) and not args.no_verify

    repo_root = Path(args.config).resolve().parent.parent.parent

    # Resolve tokenizer model
    if args.tokenizer_model:
        model_path = args.tokenizer_model
    else:
        default_vs = tok_cfg.get("default_vocab_size", 32000)
        model_path = str(
            repo_root / config["paths"]["tokenizer_model"]
            / f"hindi_bpe_{default_vs}.model"
        )

    if not os.path.exists(model_path):
        logger.error(f"Tokenizer model not found: {model_path}")
        sys.exit(1)

    # Resolve dirs
    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = str(repo_root / config["paths"]["final"])

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(repo_root / config["paths"]["tokenized"])

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_path}")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    vocab_size = sp.GetPieceSize()
    logger.info(f"Vocab size: {vocab_size}")

    # BOS / EOS configuration
    add_bos = tokenization_cfg.get("add_bos", True)
    add_eos = tokenization_cfg.get("add_eos", True)
    special_ids = tok_cfg.get("special_token_ids", {})
    bos_id = special_ids.get("bos", 2)
    eos_id = special_ids.get("eos", 3)

    logger.info(f"add_bos={add_bos} (id={bos_id}), add_eos={add_eos} (id={eos_id})")

    # Splits to tokenize
    splits = [
        ("train", "train.txt", "train.bin"),
        ("validation", "validation.txt", "validation.bin"),
        ("test", "test.txt", "test.bin"),
    ]

    token_counts: Dict[str, Any] = {}

    for split_name, txt_name, bin_name in splits:
        input_path = os.path.join(input_dir, txt_name)
        output_path = os.path.join(output_dir, bin_name)

        if not os.path.exists(input_path):
            logger.warning(f"Split file not found, skipping: {input_path}")
            continue

        if os.path.exists(output_path) and not args.overwrite:
            logger.info(f"Skipping {split_name} (exists). Use --overwrite.")
            # Stored-token count from file size without loading it into RAM
            stored = os.path.getsize(output_path) // 4
            token_counts[split_name] = {
                "lines": None,
                "content_tokens": None,
                "bos_tokens": None,
                "eos_tokens": None,
                "stored_tokens": stored,
                "from_existing_bin": True,
            }
            continue

        logger.info(f"\nTokenizing {split_name}: {input_path}")
        result = tokenize_file(
            sp, input_path, output_path,
            add_bos, add_eos, bos_id, eos_id,
            logger, progress_interval, chunk_tokens,
        )
        token_counts[split_name] = result

        if do_verify:
            if not verify_bin(output_path, result["stored_tokens"], logger):
                logger.error(f"Verification failed for {split_name}. Aborting.")
                sys.exit(1)

    # ---- Summary -------------------------------------------------------------
    total_content = sum(
        v.get("content_tokens") or 0 for v in token_counts.values()
    )
    total_bos = sum(v.get("bos_tokens") or 0 for v in token_counts.values())
    total_eos = sum(v.get("eos_tokens") or 0 for v in token_counts.values())
    total_stored = sum(v.get("stored_tokens") or 0 for v in token_counts.values())
    total_lines = sum(
        v.get("lines") or 0 for v in token_counts.values()
    )

    selection_cfg = config.get("selection", {})
    # The TRAIN split is IndicCorp + Wikipedia + News combined (see
    # 10_concatenate_final_corpus.py), so it must be compared against the COMBINED
    # total target (total_target_tokens = indiccorp_target_tokens +
    # manual_target_tokens, where manual = wikipedia + news for accounting
    # purposes), never against indiccorp_target_tokens alone — that would
    # compare a three-source sum against a single-source target.
    indiccorp_target = selection_cfg.get("indiccorp_target_tokens", 400_000_000)
    manual_target = selection_cfg.get("manual_target_tokens", 100_000_000)
    total_target = selection_cfg.get(
        "total_target_tokens",
        indiccorp_target + manual_target,
    )
    target_unit = selection_cfg.get("target_unit")
    if target_unit != "bpe_content_tokens":
        logger.error(
            f"selection.target_unit must be 'bpe_content_tokens', got {target_unit!r}. "
            f"Refusing to compare the 500M target against anything else."
        )
        sys.exit(1)

    # The target applies to the TRAIN split specifically — validation.txt /
    # test.txt were split off BEFORE selection (script 07) and are sized
    # independently (5%/5% of the pre-selection pool), NOT as a slice of
    # the 500M combined target. Comparing the target against train+val+test
    # summed would overstate how close the TRAINING corpus is to 500M.
    train_content = token_counts.get("train", {}).get("content_tokens") or 0
    train_stored = token_counts.get("train", {}).get("stored_tokens") or 0

    logger.info(f"\n{'='*60}")
    logger.info("TOKENIZATION SUMMARY")
    logger.info(f"{'='*60}")
    for name, v in token_counts.items():
        ct = v.get("content_tokens")
        if ct is None:
            logger.info(f"  {name:<12}: {v.get('stored_tokens', 0):>14,} stored tokens "
                        f"(from existing bin)")
        else:
            logger.info(
                f"  {name:<12}: {ct:>14,} content | {v['stored_tokens']:>14,} stored"
            )
    logger.info(f"  {'content total':<12}: {total_content:>14,} tokens (BOS/EOS excluded, train+val+test)")
    logger.info(f"  {'stored total':<12}: {total_stored:>14,} tokens (content+BOS+EOS, train+val+test)")
    logger.info(f"  {'BOS total':<12}: {total_bos:>14,}")
    logger.info(f"  {'EOS total':<12}: {total_eos:>14,}")
    logger.info(f"  {'lines total':<12}: {total_lines:>14,}")

    # ---- Per-source (IndicCorp / Wikipedia / News) breakdown -----------------
    # train.txt = indiccorp_train.txt + wikipedia_train.txt + news_train.txt,
    # file-level concatenation (see 10_concatenate_final_corpus.py; manual_train.txt =
    # wikipedia_train.txt + news_train.txt is also produced there for
    # backward-compat reporting). Reported here too — not just by script 09
    # right after training — so the breakdown reflects the SAME tokenizer
    # actually used for this binary-tokenization run, independent of
    # whichever tokenizer script 09 last reported against. Wikipedia/news are
    # now separate real files (source == file, no line-provenance sidecar
    # needed) — each is streamed natively.
    source_breakdown = None
    source_paths = {
        PUBLIC_SOURCE: os.path.join(input_dir, f"{PUBLIC_SOURCE}_train.txt"),
        **{s: os.path.join(input_dir, f"{s}_train.txt") for s in MANUAL_SOURCES},
    }
    if any(os.path.exists(p) for p in source_paths.values()):
        logger.info("")
        logger.info("=" * 60)
        logger.info("SOURCE BREAKDOWN (IndicCorp / Wikipedia / News)")
        logger.info("=" * 60)

        tokens_by_source: Dict[str, int] = {}
        for source, path in source_paths.items():
            logger.info(f"Scanning {source.capitalize()}: {path}")
            stats = count_tokens_streaming(sp, path, logger, progress_interval)
            tokens_by_source[source] = stats["total_tokens"]
            if not os.path.exists(path):
                logger.warning(f"  {source.capitalize()} final file not found — counted as 0 tokens: {path}")

        indiccorp_tokens = tokens_by_source[PUBLIC_SOURCE]
        manual_tokens = sum(tokens_by_source[s] for s in MANUAL_SOURCES)
        manual_min_fraction = config.get("manual", {}).get("min_fraction", 0.20)

        if train_content == 0 and (indiccorp_tokens or manual_tokens):
            # train.bin was skipped this run (already existed, no --overwrite),
            # so token_counts["train"]["content_tokens"] is None/0 — this pass
            # over the three source files just recomputed the real
            # content-token count directly from the text, so use THAT as
            # train_content for the rest of this report instead of silently
            # reporting 0 / a false invariant mismatch.
            train_content = indiccorp_tokens + manual_tokens
            logger.info(
                f"  (train.bin was skipped this run — train_content recomputed "
                f"from the three source files: {train_content:,})"
            )

        manual_fraction = (manual_tokens / train_content) if train_content > 0 else 0.0

        logger.info(f"  IndicCorp BPE content tokens:    {indiccorp_tokens:>14,}  (target {indiccorp_target:,})")
        for s in MANUAL_SOURCES:
            logger.info(f"{s.capitalize():>19s} BPE content tokens: {tokens_by_source[s]:>14,}")
        logger.info(f"  Manual total BPE content tokens: {manual_tokens:>14,}  (target {manual_target:,})  "
                    f"[= wikipedia + news]")
        logger.info(f"  Manual fraction:                 {manual_fraction:.2%}  (required >= {manual_min_fraction:.0%})")

        invariant_ok = (indiccorp_tokens + manual_tokens) == train_content
        if invariant_ok:
            logger.info("  Invariant OK: indiccorp_tokens + manual_tokens == train.txt content tokens")
        else:
            logger.warning(
                f"NOTE: indiccorp_tokens ({indiccorp_tokens:,}) + manual_tokens "
                f"({manual_tokens:,}) = {indiccorp_tokens + manual_tokens:,} does not "
                f"equal train.txt's content tokens ({train_content:,}) recorded when "
                f"train.bin was written — final/train.txt may no longer be a "
                f"straightforward concatenation of indiccorp_train.txt + "
                f"wikipedia_train.txt + news_train.txt. Investigate script 10's output."
            )

        source_breakdown = {
            "indiccorp_tokens": indiccorp_tokens,
            **{f"{s}_tokens": tokens_by_source[s] for s in MANUAL_SOURCES},
            "manual_tokens": manual_tokens,
            "manual_fraction": round(manual_fraction, 4),
            "manual_min_fraction": manual_min_fraction,
            "invariant_ok": invariant_ok,
        }

    diff = train_content - total_target
    pct_from_target = (diff / total_target * 100) if total_target > 0 else 0.0
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL TRAINING CORPUS TOKEN COUNT (train.txt = IndicCorp + Wikipedia + News, vs. combined 500M target)")
    logger.info("=" * 60)
    logger.info(f"FINAL CONTENT BPE TOKENS: {train_content:,}")
    logger.info(f"COMBINED TARGET CONTENT BPE TOKENS: {total_target:,} "
                f"(= indiccorp {indiccorp_target:,} + manual {manual_target:,})")
    logger.info(f"DIFFERENCE: {diff:+,}")
    logger.info(f"PERCENTAGE FROM TARGET: {pct_from_target:+.2f}%")

    # CL3410 assertion: BOS/EOS must never be counted toward the target.
    if add_bos or add_eos:
        assert train_content < train_stored, (
            f"train content tokens ({train_content:,}) should be strictly less than "
            f"stored tokens ({train_stored:,}) when BOS/EOS are added — "
            f"otherwise BOS/EOS may have leaked into the content count."
        )

    # ---- Save metadata -------------------------------------------------------
    metadata = {
        "tokenizer_model": model_path,
        "vocab_size": vocab_size,
        "dtype": "uint32",
        "add_bos": add_bos,
        "add_eos": add_eos,
        "special_token_ids": {
            "pad": special_ids.get("pad", 0),
            "unk": special_ids.get("unk", 1),
            "bos": bos_id,
            "eos": eos_id,
        },
        "token_counts": token_counts,
        "source_breakdown": source_breakdown,
        "total_content_tokens": total_content,
        "total_bos_tokens": total_bos,
        "total_eos_tokens": total_eos,
        "total_stored_tokens": total_stored,
        "train_content_tokens": train_content,
        "target_content_tokens": total_target,
        "indiccorp_target_tokens": indiccorp_target,
        "manual_target_tokens": manual_target,
        "total_target_tokens": total_target,
        "target_unit": target_unit,
        "difference_from_target": diff,
        "percentage_from_target": round(pct_from_target, 4),
        "note": (
            "target_content_tokens/difference_from_target/percentage_from_target "
            "compare TRAIN-split CONTENT tokens (BOS/EOS excluded, "
            "validation/test excluded) against the COMBINED total_target_tokens "
            "(indiccorp_target_tokens + manual_target_tokens) — train.txt is "
            "IndicCorp + Wikipedia + News together, never IndicCorp alone."
        ),
    }

    meta_path = os.path.join(output_dir, "tokenization_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {meta_path}")
    logger.info("Tokenization complete.")


if __name__ == "__main__":
    main()
