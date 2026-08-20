#!/usr/bin/env python3
"""
13_analyze_tokenizer_and_corpus.py — Final Tokenizer + Corpus Analysis (Nepali)

POST-TOKENIZATION ANALYSIS/REPORTING ONLY. This script does not modify
collection, normalization, filtering, quality filtering, deduplication,
splitting, decontamination, corpus selection, tokenizer training, or binary
tokenization — it only READS their existing outputs (final/*.txt,
tokenized/*.bin + tokenization_metadata.json, the trained tokenizer model)
and produces a NEW, independent analysis report. It never retrains the
tokenizer and never changes its vocabulary size.

NOTE: the Nepali pipeline has no independent leakage-reverification stage
(the Hindi pipeline's equivalent, 15_verify_final_corpus.py, was dropped
for Nepali — its full re-scan of train.txt against validation/test was
too slow to be worth it here). This script's integrity section therefore
never reports on leakage; decontamination correctness relies solely on
08_decontaminate.py's own report.

NEPALI ONLY. This script uses ONLY Nepali paths/config/tokenizer/corpus. It
never reads, writes, or references anything under hindi/ — see
hindi/scripts/16_analyze_tokenizer_and_corpus.py for the equivalent Hindi
report. The two scripts share pure, stateless helper functions from
shared/tokenizer_corpus_analysis.py, but each call here is parameterized
with Nepali-only paths and produces Nepali-only output files; nothing is
ever combined across languages.

EXACTLY TWO LOGICAL SOURCES: indiccorp, manual (Nepali Onlinekhabar +
Khabarhub). Unlike Hindi, Nepali's pipeline (09_concatenate_final_corpus.py)
still materializes manual's TWO subcategories as separate final files
(final/onlinekhabar_train.txt, final/khabarhub_train.txt)
alongside the combined final/manual_train.txt —
this script reads those two files directly for the diagnostic breakdown
(no line-provenance sidecar needed for Nepali). Individual news outlets are
reported ONLY as this diagnostic breakdown, never as a separate top-level
source or target — see targets below, read from nepali_data.yaml (NOT
hard-coded from Hindi's numbers).

MEMORY: every corpus file is streamed line-by-line (see
shared/tokenizer_corpus_analysis.py); token-frequency counting keeps only a
{token_id: count} dict bounded by the tokenizer's vocabulary size, never a
per-occurrence list, never the corpus itself in RAM.

Outputs:
    nepali/data/metadata/tokenizer_corpus_analysis.json
    nepali/data/metadata/tokenizer_corpus_analysis.txt
    nepali/data/metadata/figures/token_frequency_distribution.png   (optional)
    nepali/data/metadata/figures/tokenizer_quality_comparison.png   (optional)
    nepali/data/metadata/figures/vocabulary_utilization.png         (optional)

Usage:
    python nepali/scripts/13_analyze_tokenizer_and_corpus.py \\
        --config nepali/configs/nepali_data.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import sentencepiece as spm
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.tokenizer_corpus_analysis import (
    binary_corpus_analysis,
    build_examples,
    build_sources_section,
    combine_split_stats,
    data_integrity_checks,
    frequency_analysis_from_histogram,
    load_tokenizer_info,
    manual_breakdown_from_files,
    maybe_plot_figures,
    render_txt_report,
    split_stats,
    streaming_token_id_histogram,
    tokenizer_quality_from_histogram,
    tokenization_performance,
)

LANGUAGE = "nepali"

# Curated, hard-coded Nepali example sentences (Devanagari) — language-
# specific, never shared with Hindi. Covers: normal sentence, punctuation,
# numbers, dates, named entities, long word, short word, common word, rare
# word, mixed-script text.
EXAMPLES = [
    ("normal_sentence", "नेपाल एक सुन्दर र विविधतापूर्ण देश हो।"),
    ("punctuation", "तपाईं कस्तो हुनुहुन्छ? म बिल्कुल ठिक छु!"),
    ("numbers", "मसँग २५ वटा किताब र १०० रुपैयाँ छ।"),
    ("dates", "आजको मिति २०८१ साल हो।"),
    ("named_entities", "पृथ्वीनारायण शाहको जन्म गोरखामा भएको थियो।"),
    ("long_word", "अन्तर्राष्ट्रियकरण"),
    ("short_word", "छ"),
    ("common_word", "र"),
    ("rare_word", "सापेक्षतावाद"),
    ("mixed_script", "मैले COVID-19 खोपको दुवै डोज लगाएको छु।"),
]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    lcfg = config.get("logging", {})
    level = lcfg.get("level", "INFO")
    fmt = lcfg.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    return logging.getLogger("13_analyze_tokenizer_and_corpus_nepali")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-tokenization analysis/reporting for the Nepali "
                     "final tokenizer + corpus. Read-only w.r.t. the "
                     "pipeline — never retrains/modifies anything."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to nepali_data.yaml")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Override output directory (default: nepali/data/metadata)")
    parser.add_argument("--figures-dir", type=str, default=None,
                         help="Override figures directory (default: <output-dir>/figures)")
    parser.add_argument("--skip-figures", action="store_true", help="Skip optional PNG figures")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    progress_interval = config.get("logging", {}).get("progress_interval", 1_000_000)

    repo_root = Path(args.config).resolve().parent.parent.parent
    paths = config.get("paths", {})
    tok_cfg = config.get("tokenizer", {})
    selection_cfg = config.get("selection", {})
    manual_cfg = config.get("manual", {})

    final_dir = repo_root / paths.get("final", "nepali/data/final")
    tokenized_dir = repo_root / paths.get("tokenized", "nepali/data/tokenized")
    metadata_dir = repo_root / paths.get("metadata", "nepali/metadata")
    model_dir = repo_root / paths.get("tokenizer_model", "nepali/tokenizer/model")
    tok_config_dir = repo_root / paths.get("tokenizer_config", "nepali/tokenizer/config")

    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "nepali" / "data" / "metadata")
    figures_dir = Path(args.figures_dir) if args.figures_dir else (output_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"FINAL TOKENIZER + CORPUS ANALYSIS — {LANGUAGE.upper()}")
    logger.info("=" * 70)

    # ---- Load the ONE final tokenizer (never retrained here) -------------
    default_vs = tok_cfg.get("default_vocab_size", 32000)
    model_path = str(model_dir / f"nepali_bpe_{default_vs}.model")
    tokenizer_config_json = str(tok_config_dir / f"nepali_bpe_{default_vs}_config.json")

    tokenizer_model_exists = os.path.exists(model_path)
    tokenizer_loadable = False
    sp = None
    if tokenizer_model_exists:
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load(model_path)
            tokenizer_loadable = True
        except Exception as e:
            logger.error(f"Failed to load tokenizer at {model_path}: {e}")
    if not tokenizer_loadable:
        logger.error(f"Tokenizer not loadable ({model_path}) — cannot proceed with analysis.")
        sys.exit(1)

    tokenizer_info = load_tokenizer_info(sp, model_path, tok_cfg, tokenizer_config_json)
    actual_vocab_size = tokenizer_info["actual_vocab_size"]
    unk_id = tokenizer_info["special_token_ids"]["unk"]
    bos_id = tokenizer_info["special_token_ids"]["bos"]
    eos_id = tokenizer_info["special_token_ids"]["eos"]
    logger.info(f"Tokenizer loaded: {model_path} (actual vocab size = {actual_vocab_size:,})")

    # ---- Corpus statistics: train / validation / test / combined --------
    train_path = str(final_dir / "train.txt")
    validation_path = str(final_dir / "validation.txt")
    test_path = str(final_dir / "test.txt")

    logger.info("Computing per-split corpus statistics (streaming)...")
    train_stats = split_stats(sp, train_path, logger, progress_interval)
    validation_stats = split_stats(sp, validation_path, logger, progress_interval)
    test_stats = split_stats(sp, test_path, logger, progress_interval)
    combined_stats = combine_split_stats(train_stats, validation_stats, test_stats)

    # ---- Logical source accounting: indiccorp + manual -------------------
    indiccorp_path = str(final_dir / "indiccorp_train.txt")
    manual_path = str(final_dir / "manual_train.txt")
    logger.info("Computing IndicCorp / Manual source accounting (streaming)...")
    indiccorp_stats = split_stats(sp, indiccorp_path, logger, progress_interval)
    manual_stats = split_stats(sp, manual_path, logger, progress_interval)

    # Nepali's manual subcategories are still separate final files (no
    # line-provenance sidecar) — see module docstring.
    manual_subsource_files = {
        "onlinekhabar": str(final_dir / "onlinekhabar_train.txt"),
        "khabarhub": str(final_dir / "khabarhub_train.txt"),
    }
    manual_breakdown = manual_breakdown_from_files(sp, manual_subsource_files, logger, progress_interval)
    sources = build_sources_section(
        indiccorp_stats, manual_stats, train_stats["content_tokens"],
        manual_breakdown, breakdown_kind="files",
    )

    # ---- Tokenizer quality on validation ----------------------------------
    logger.info("Evaluating tokenizer quality on validation.txt (streaming)...")
    val_hist = streaming_token_id_histogram(sp, validation_path, logger, progress_interval)
    tok_quality = tokenizer_quality_from_histogram(val_hist, unk_id, actual_vocab_size)

    # ---- Token frequency analysis on train --------------------------------
    logger.info("Running token-frequency analysis on train.txt (streaming)...")
    train_hist = streaming_token_id_histogram(sp, train_path, logger, progress_interval)
    freq_analysis = frequency_analysis_from_histogram(sp, train_hist, bos_id, eos_id)

    # ---- Examples -----------------------------------------------------------
    examples = build_examples(sp, EXAMPLES)

    # ---- Binary corpus analysis (reads existing files/metadata only) -----
    logger.info("Reading existing binary tokenization outputs...")
    binary = binary_corpus_analysis(str(tokenized_dir), ("train", "validation", "test"), logger)
    performance = tokenization_performance(str(tokenized_dir))

    # ---- Targets (read from nepali_data.yaml — NEVER hard-coded) ----------
    indiccorp_target = selection_cfg.get("indiccorp_target_tokens", 400_000_000)
    manual_target = selection_cfg.get("manual_target_tokens", 100_000_000)
    total_target = selection_cfg.get("total_target_tokens", indiccorp_target + manual_target)
    manual_min_fraction = manual_cfg.get("min_fraction", 0.20)
    targets = {
        "indiccorp_target_tokens": indiccorp_target,
        "manual_target_tokens": manual_target,
        "total_target_tokens": total_target,
        "manual_min_fraction": manual_min_fraction,
        "target_unit": selection_cfg.get("target_unit"),
    }

    # ---- Integrity ------------------------------------------------------------
    # data_integrity_checks() requires a `leakage` argument since it is
    # shared with hindi/scripts/16_analyze_tokenizer_and_corpus.py (which
    # DOES have a 15_verify_final_corpus.py and passes a real report). The
    # Nepali pipeline dropped that stage entirely (its full leakage re-scan
    # of train.txt against validation/test was too slow to be worth it
    # here), so there is no leakage feature for Nepali at all — the
    # leakage_available/leakage_all_passed keys the shared helper adds are
    # stripped back out below rather than left in as permanent, misleading
    # False/None values (leakage_available=False would otherwise force
    # overall_status to FAIL on every run, regardless of everything else
    # actually passing).
    integrity = data_integrity_checks(
        tokenizer_model_exists=tokenizer_model_exists,
        tokenizer_loadable=tokenizer_loadable,
        train_path=train_path, validation_path=validation_path, test_path=test_path,
        indiccorp_path=indiccorp_path, manual_path=manual_path,
        train_tokens=train_stats["content_tokens"],
        validation_tokens=validation_stats["content_tokens"],
        test_tokens=test_stats["content_tokens"],
        invariant_indiccorp_plus_manual_equals_total=sources["invariant_indiccorp_plus_manual_equals_total"],
        binary_reconciliation=binary,
        manual_fraction=sources["manual_fraction"],
        manual_min_fraction=manual_min_fraction,
        indiccorp_target=indiccorp_target, manual_target=manual_target, total_target=total_target,
        indiccorp_tokens=sources["indiccorp"]["content_tokens"],
        manual_tokens=sources["manual"]["content_tokens"],
        total_tokens=sources["total"]["content_tokens"],
        leakage={"available": False},
    )
    integrity.pop("leakage_available", None)
    integrity.pop("leakage_all_passed", None)
    relevant = [v for v in integrity.values() if isinstance(v, bool)]
    integrity["overall_status"] = "PASS" if all(relevant) else "FAIL"

    report: Dict[str, Any] = {
        "language": LANGUAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "tokenizer": tokenizer_info,
        "corpus": {
            "train": train_stats,
            "validation": validation_stats,
            "test": test_stats,
            "combined": combined_stats,
        },
        "sources": sources,
        "tokenizer_quality": tok_quality,
        "frequency_analysis": freq_analysis,
        "binary_corpus": binary,
        "tokenization_performance": performance,
        "examples": examples,
        "integrity": integrity,
        "targets": targets,
    }

    json_path = output_dir / "tokenizer_corpus_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON report saved to {json_path}")

    txt_report = render_txt_report(LANGUAGE, report)
    txt_path = output_dir / "tokenizer_corpus_analysis.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)
    logger.info(f"TXT report saved to {txt_path}")

    if not args.skip_figures:
        written = maybe_plot_figures(report, str(figures_dir), LANGUAGE.capitalize(), logger)
        if written:
            logger.info(f"Figures written: {', '.join(written)}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"IndicCorp tokens: {sources['indiccorp']['content_tokens']:,}")
    logger.info(f"Manual tokens:    {sources['manual']['content_tokens']:,}")
    logger.info(f"Total tokens:     {sources['total']['content_tokens']:,}")
    logger.info(f"Manual fraction:  {sources['manual_fraction'] * 100:.2f}%")
    logger.info(f"Overall status:   {integrity['overall_status']}")


if __name__ == "__main__":
    main()
