import os
import sys
import argparse
import logging
import time
import yaml
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from shared.tokenizer.data_utils import count_lines, iter_lines
from shared.tokenizer.run_stats import cleanup_prior_stage_files, record_run

# Exactly three logical/top-level corpus sources — see 05_deduplicate.py's
# module docstring. "news" already contains every enabled news outlet
# (combined by manual/05_aggregate_news_corpus.py); individual outlets are
# metadata (source_name, in news_provenance.jsonl) only, never a separate
# --source value. "wikipedia" is its own real per-source file (no
# aggregation needed).
SOURCE_CHOICES = ("indiccorp", "wikipedia", "news")

# review_filename config-key per source, and its fallback default when
# that key isn't set in hindi_data.yaml. indiccorp keeps the original
# `review_filename` key name (not `indiccorp_review_filename`) for
# backward compatibility with existing configs/invocations.
_REVIEW_FILENAME_CONFIG_KEY = {
    "indiccorp": "review_filename",
    "wikipedia": "wikipedia_review_filename",
    "news": "news_review_filename",
}

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("03_filter_hindi")

def classify_chars(line: str):
    """
    Classify characters in a line.
    Returns counts for: devanagari, latin, digits, punctuation, other, whitespace, non_whitespace
    """
    counts = {
        'devanagari': 0,
        'latin': 0,
        'digits': 0,
        'punctuation': 0,
        'other': 0,
        'whitespace': 0,
        'non_whitespace': 0
    }
    
    for char in line:
        if char.isspace():
            counts['whitespace'] += 1
            continue
            
        counts['non_whitespace'] += 1
        cp = ord(char)
        
        # Devanagari digits: U+0966-U+096F, ASCII digits: U+0030-U+0039
        if (0x0966 <= cp <= 0x096F) or (0x0030 <= cp <= 0x0039):
            counts['digits'] += 1
        # Devanagari blocks
        elif (0x0900 <= cp <= 0x097F) or (0xA8E0 <= cp <= 0xA8FF) or (0x1CD0 <= cp <= 0x1CFF):
            # Exclude danda (0x0964) and double danda (0x0965)
            if cp == 0x0964 or cp == 0x0965:
                counts['punctuation'] += 1
            else:
                counts['devanagari'] += 1
        # Latin basic + extended (excluding punctuation, just checking category)
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            if char.isalpha():
                counts['latin'] += 1
            else:
                counts['other'] += 1
        else:
            cat = unicodedata.category(char)
            if cat.startswith('P'):
                counts['punctuation'] += 1
            else:
                counts['other'] += 1
                
    return counts

def default_paths(source: str, config: dict, repo_root: str) -> tuple:
    """Resolve the default (input_path, output_path, review_filename) for a
    given source — config-driven via the `{source}_hi_normalized.txt` /
    `{source}_hi_filtered.txt` naming convention already used by
    02_normalize.py. Every source is filtered with the exact same
    classify_chars()/devanagari-ratio logic (see main()); only the file
    locations differ."""
    input_path = os.path.join(repo_root, config['paths']['normalized'], f'{source}_hi_normalized.txt')
    output_path = os.path.join(repo_root, config['paths']['filtered'], f'{source}_hi_filtered.txt')
    review_filename = config['filtering'].get(
        _REVIEW_FILENAME_CONFIG_KEY[source], f'{source}_hi_review.txt'
    )
    return input_path, output_path, review_filename

def main():
    parser = argparse.ArgumentParser(description="Hindi script-based filtering.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--source", type=str, choices=list(SOURCE_CHOICES), default="indiccorp",
                         help="Which logical source corpus to filter (default: indiccorp). "
                              "'wikipedia' and 'news' are the two manual-data sources — "
                              "'news' already combines every enabled news outlet.")
    parser.add_argument("--input", type=str, help="Input file path (overrides --source default).")
    parser.add_argument("--output", type=str, help="Output file path (overrides --source default).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")

    args = parser.parse_args()
    logger = setup_logger()

    config = load_config(args.config)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    default_input_path, default_output_path, review_filename = default_paths(args.source, config, repo_root)
    input_path = args.input if args.input else default_input_path
    output_path = args.output if args.output else default_output_path
    review_path = os.path.join(os.path.dirname(output_path), review_filename)

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    skip_lines = 0
    if os.path.exists(output_path):
        if args.resume:
            skip_lines = count_lines(output_path)
            logger.info(f"Resuming... skipping {skip_lines} lines.")
        elif not args.overwrite:
            logger.error(f"Output file exists: {output_path}. Use --overwrite or --resume.")
            sys.exit(1)
        else:
            logger.info("Overwriting existing output.")
            open(output_path, 'w').close()
            open(review_path, 'w').close()
            
    mode = 'a' if args.resume else 'w'
    
    min_dev_ratio = config['filtering'].get('min_devanagari_ratio', 0.70)
    review_dev_ratio = config['filtering'].get('review_devanagari_ratio', 0.40)
    save_review = config['filtering'].get('save_review_lines', True)
    
    lines_processed = 0
    lines_kept = 0
    lines_review = 0
    lines_removed = 0
    
    global_stats = {
        'devanagari': 0, 'latin': 0, 'digits': 0, 'punctuation': 0, 'other': 0
    }
    
    start_time = time.time()
    log_interval = config.get('logging', {}).get('progress_interval', 1000000)

    logger.info(f"Starting script filtering. Input: {input_path}")

    with open(output_path, mode, encoding='utf-8') as out_f, \
         open(review_path, mode, encoding='utf-8') as rev_f:

        for line in iter_lines(input_path):
            if lines_processed < skip_lines:
                lines_processed += 1
                continue

            stripped = line
            counts = classify_chars(stripped)

            for k in global_stats.keys():
                global_stats[k] += counts[k]

            nws = counts['non_whitespace']
            if nws > 0:
                dev_ratio = counts['devanagari'] / nws
            else:
                dev_ratio = 0.0

            if dev_ratio >= min_dev_ratio:
                out_f.write(line + "\n")
                lines_kept += 1
            elif dev_ratio >= review_dev_ratio:
                if save_review:
                    rev_f.write(line + "\n")
                lines_review += 1
            else:
                lines_removed += 1

            lines_processed += 1

            if lines_processed % log_interval == 0:
                elapsed = time.time() - start_time
                speed = (lines_processed - skip_lines) / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {lines_processed} lines... Kept: {lines_kept}, Review: {lines_review}, Removed: {lines_removed}, Speed: {speed:.2f} l/s")

    total_effective = lines_kept + lines_review + lines_removed
    elapsed_total = time.time() - start_time

    if total_effective > 0:
        logger.info("Filtering complete.")
        logger.info(f"Total lines evaluated: {total_effective}")
        logger.info(f"Kept: {lines_kept} ({(lines_kept/total_effective)*100:.2f}%)")
        logger.info(f"Review: {lines_review} ({(lines_review/total_effective)*100:.2f}%)")
        logger.info(f"Removed: {lines_removed} ({(lines_removed/total_effective)*100:.2f}%)")

        total_chars = sum(global_stats.values())
        if total_chars > 0:
            logger.info("Character distribution stats:")
            for k, v in global_stats.items():
                logger.info(f"  {k}: {v} ({(v/total_chars)*100:.2f}%)")

    cleanup_report = None
    # Only auto-delete this source's normalized/ file when this run
    # actually consumed its default (config-resolved) input path — an
    # explicit --input override means the caller pointed this run at
    # something else entirely, so nothing in normalized/ should be
    # touched. Deletes ONLY {source}_hi_normalized.txt — never the whole
    # directory — since normalized/ holds one file PER SOURCE and this
    # invocation only finished consuming this one source's file.
    if args.input is None:
        cleanup_report = cleanup_prior_stage_files(
            config, Path(repo_root), stage_to_delete="normalized",
            files=[input_path], logger=logger,
        )

    record_run(
        config, Path(repo_root), stage_number="03", stage_name="filter_hindi",
        source=args.source,
        summary={
            "input_path": input_path,
            "output_path": output_path,
            "review_path": review_path,
            "lines_processed": lines_processed,
            "lines_kept": lines_kept,
            "lines_review": lines_review,
            "lines_removed": lines_removed,
            "character_distribution": global_stats,
            "elapsed_seconds": round(elapsed_total, 2),
            "cleanup_of_normalized_stage": cleanup_report,
        },
        logger=logger,
    )

if __name__ == "__main__":
    main()
