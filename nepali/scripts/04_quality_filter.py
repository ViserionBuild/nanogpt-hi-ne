import os
import sys
import argparse
import logging
import time
import yaml
import unicodedata
import re
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from shared.data_utils import count_lines, iter_lines
from shared.run_stats import cleanup_prior_stage_files, record_run

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("04_quality_filter")

# Compiled Regexes
URL_RE = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_RE = re.compile(r'<[^>]+>|&[a-zA-Z0-9#]+;')

# Punctuation stripped from word edges before blocklist matching, so that
# e.g. "शब्द।" (with a trailing danda) still matches a blocklist entry of
# "शब्द".
_BLOCKLIST_STRIP_CHARS = ".,;:!?\"'()[]{}।॥…​"


def load_blocklist(path: str, logger=None) -> set:
    """
    Load a whole-word safety/profanity blocklist: one term per line, blank
    lines and lines starting with '#' ignored. Returns an empty set (filter
    disabled) if the path is falsy or the file does not exist, so a missing
    blocklist file never breaks the pipeline.
    """
    terms = set()
    if not path or not os.path.exists(path):
        if logger is not None:
            logger.warning(f"Safety blocklist not found at {path!r} — blocklist filter disabled.")
        return terms
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            term = raw.strip()
            if term and not term.startswith('#'):
                terms.add(term)
    return terms


def _blocklist_word(w: str) -> str:
    return w.strip(_BLOCKLIST_STRIP_CHARS)

def get_repetition_stats(line: str, max_repeat_len_thresh: int):
    """
    Returns (fails_max_length_check, ratio_in_repeats)
    A character is considered part of a repeated run if it appears consecutively.
    For ratio, count all chars in runs of length >= 3? Wait, the requirement says:
    "max_repeated_char_ratio (0.30): ratio of chars that are in repeated runs"
    Let's assume "repeated runs" means length >= 2 or length >= 3. Typically length > 1 (so >= 2).
    Let's use length >= 2.
    """
    if not line:
        return False, 0.0

    chars_in_repeats = 0
    current_run_len = 1
    max_run_len = 1

    for i in range(1, len(line)):
        if line[i] == line[i-1]:
            current_run_len += 1
        else:
            if current_run_len > 1:
                chars_in_repeats += current_run_len
            if current_run_len > max_run_len:
                max_run_len = current_run_len
            current_run_len = 1

    # Handle end of string
    if current_run_len > 1:
        chars_in_repeats += current_run_len
    if current_run_len > max_run_len:
        max_run_len = current_run_len

    fails_max_len = max_run_len >= max_repeat_len_thresh
    ratio = chars_in_repeats / len(line) if line else 0.0

    return fails_max_len, ratio

def check_quality(line: str, config: dict, blocklist: set = None):
    """
    Apply quality checks. Returns reason for failure (str) or None if it passes.
    `blocklist`, if non-empty, enables the safety/profanity term filter
    (CL3410 requirement 6).
    """
    line = line.strip('\n')
    stripped = line.strip()
    if not stripped:
        return "empty_or_whitespace"

    q_config = config['quality']

    if len(stripped) < q_config['min_characters']:
        return "too_short"
    if len(stripped) > q_config['max_characters']:
        return "too_long"

    words = stripped.split()
    if len(words) < q_config['min_word_count']:
        return "too_few_words"

    urls = URL_RE.findall(stripped)
    url_chars = sum(len(u) for u in urls)
    if url_chars / len(stripped) > q_config['max_url_ratio']:
        return "high_url_ratio"

    html_matches = HTML_TAG_RE.findall(stripped)
    if len(html_matches) >= q_config['html_pattern_threshold']:
        return "too_many_html_tags"

    fails_len, repeat_ratio = get_repetition_stats(stripped, q_config['max_repeated_char_length'])
    if fails_len:
        return "max_repeated_char_length"
    if repeat_ratio > q_config['max_repeated_char_ratio']:
        return "high_repeated_char_ratio"

    nws = 0
    punct = 0
    digits = 0

    for char in stripped:
        if not char.isspace():
            nws += 1
            cp = ord(char)
            # Check digits
            if (0x0966 <= cp <= 0x096F) or (0x0030 <= cp <= 0x0039):
                digits += 1
            else:
                cat = unicodedata.category(char)
                if cat.startswith('P') or cp in (0x0964, 0x0965):
                    punct += 1

    if nws > 0:
        if punct / nws > q_config['max_punctuation_ratio']:
            return "high_punctuation_ratio"
        if digits / nws > q_config['max_digit_ratio']:
            return "high_digit_ratio"

    if blocklist:
        matches = sum(1 for w in words if _blocklist_word(w) in blocklist)
        if matches > 0:
            ratio = matches / len(words)
            if ratio > q_config.get('max_blocklist_term_ratio', 0.0):
                return "blocklisted_term"

    return None

# Every source this script can quality-filter by name: the public
# IndicCorp corpus, or either manual news source (onlinekhabar,
# khabarhub) — matching 03_filter_devanagari_script.py's source model.
SOURCE_CHOICES = ("indiccorp", "onlinekhabar", "khabarhub")

# quality.<source>_output_dir / quality.<source>_output_filename /
# quality.<source>_rejected_filename config-key prefixes, and their
# fallback defaults. indiccorp keeps the original `rejected_filename` key
# name (not `indiccorp_rejected_filename`) for backward compatibility.
_REJECTED_FILENAME_CONFIG_KEY = {
    "indiccorp": "rejected_filename",
    "onlinekhabar": "onlinekhabar_rejected_filename",
    "khabarhub": "khabarhub_rejected_filename",
}


def default_paths(source: str, config: dict, repo_root: str) -> tuple:
    """Resolve the default (input_path, output_path) for a given source.
    Requires config['paths'] — only called when --input/--output are not
    both given explicitly, so a minimal config (e.g. explicit-paths-only
    CLI usage) never needs a 'paths' key. EVERY source is quality-filtered
    with the exact same check_quality() logic, thresholds, and blocklist
    rules (see main()); only the file locations differ.

    INPUT is read from config['paths']['filtered'] (03_filter_devanagari_script.py's
    `{source}_ne_filtered.txt` output convention) for every source. OUTPUT
    for every source defaults to the SAME shared quality.output_dir (see
    nepali_data.yaml) — not paths['filtered'] — so 05_deduplicate.py's
    cross-source pass finds all four sources' quality-filtered output
    side by side."""
    q_cfg = config['quality']
    shared_output_dir = q_cfg.get('output_dir', 'nepali/data/intermediate/quality')
    input_path = os.path.join(repo_root, config['paths']['filtered'], f'{source}_ne_filtered.txt')
    output_dir = q_cfg.get(f'{source}_output_dir', shared_output_dir)
    output_filename = q_cfg.get(f'{source}_output_filename', f'{source}_ne_quality.txt')
    output_path = os.path.join(repo_root, output_dir, output_filename)
    return input_path, output_path


def default_rejected_filename(source: str, config: dict) -> str:
    """Resolve the default rejected-lines filename for a given source. Only
    reads config['quality'] — never config['paths'] — matching the
    pre-existing rejected_path resolution, which always ran regardless of
    whether --input/--output were given."""
    q_cfg = config['quality']
    return q_cfg.get(
        _REJECTED_FILENAME_CONFIG_KEY[source], f'{source}_ne_quality_rejected.txt'
    )


# Mirrors 03_filter_devanagari_script.py's _REVIEW_FILENAME_CONFIG_KEY — needed here
# ONLY to compute the path of that stage's review file for this source, so
# cleanup after this run can delete it alongside {source}_ne_filtered.txt
# (same lifecycle, per project decision — review/rejected files are not
# preserved independently of their stage's main output).
_FILTERED_REVIEW_FILENAME_CONFIG_KEY = {
    "indiccorp": "review_filename",
    "onlinekhabar": "onlinekhabar_review_filename",
    "khabarhub": "khabarhub_review_filename",
}


def default_filtered_review_filename(source: str, config: dict) -> str:
    return config.get('filtering', {}).get(
        _FILTERED_REVIEW_FILENAME_CONFIG_KEY.get(source, ""), f'{source}_ne_review.txt'
    )

def main():
    parser = argparse.ArgumentParser(
        description="Quality filter Nepali text. Works on any of this "
                     "project's sources: the public IndicCorp corpus, or "
                     "either manual news source (Onlinekhabar, Khabarhub)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--source", type=str, choices=list(SOURCE_CHOICES), default="indiccorp",
                         help="Which source corpus to quality-filter: 'indiccorp' (public), "
                              "'onlinekhabar', or 'khabarhub'. Default: indiccorp. "
                              "Ignored if --input/--output are given.")
    parser.add_argument("--input", type=str, help="Input file path (overrides --source default).")
    parser.add_argument("--output", type=str, help="Output file path (overrides --source default).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    parser.add_argument("--save-rejected", action="store_true",
                         help="Save rejected lines + their rejection reason to a review "
                              "file (default taken from config quality.save_rejected_lines; "
                              "this flag can only enable it, matching --overwrite/--resume style).")

    args = parser.parse_args()
    logger = setup_logger()

    config = load_config(args.config)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    if args.input and args.output:
        input_path = args.input
        output_path = args.output
    else:
        default_input_path, default_output_path = default_paths(args.source, config, repo_root)
        input_path = args.input if args.input else default_input_path
        output_path = args.output if args.output else default_output_path
    rejected_path = os.path.join(os.path.dirname(output_path), default_rejected_filename(args.source, config))

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # CL3410 low-resource-language caution: "inspect what is removed" rather
    # than silently deleting — mirrors 03_filter_devanagari_script.py's
    # save_review_lines pattern for its own (Devanagari-ratio) rejections.
    save_rejected = args.save_rejected or config['quality'].get('save_rejected_lines', False)

    blocklist_path = config['quality'].get('safety_blocklist_path')
    if blocklist_path and not os.path.isabs(blocklist_path):
        blocklist_path = os.path.join(repo_root, blocklist_path)
    blocklist = load_blocklist(blocklist_path, logger=logger)
    if blocklist:
        logger.info(f"Safety blocklist loaded: {len(blocklist)} term(s) from {blocklist_path}")

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
            if save_rejected:
                open(rejected_path, 'w').close()

    mode = 'a' if args.resume else 'w'

    lines_processed = 0
    lines_kept = 0
    reason_counts = {}

    start_time = time.time()
    log_interval = config.get('logging', {}).get('progress_interval', 1000000)

    logger.info(f"Starting quality filtering. Input: {input_path}")
    if save_rejected:
        logger.info(f"Rejected lines will be saved to: {rejected_path}")

    rejected_f = open(rejected_path, mode, encoding='utf-8') if save_rejected else None
    try:
        with open(output_path, mode, encoding='utf-8') as out_f:
            for line in iter_lines(input_path):
                if lines_processed < skip_lines:
                    lines_processed += 1
                    continue

                reason = check_quality(line, config, blocklist=blocklist)
                if reason is None:
                    out_f.write(line + "\n")
                    lines_kept += 1
                else:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    if rejected_f is not None:
                        rejected_f.write(f"{reason}\t{line}\n")

                lines_processed += 1

                if lines_processed % log_interval == 0:
                    elapsed = time.time() - start_time
                    speed = (lines_processed - skip_lines) / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {lines_processed} lines... Kept: {lines_kept}, Speed: {speed:.2f} l/s")
    finally:
        if rejected_f is not None:
            rejected_f.close()

    total_effective = lines_processed - skip_lines
    if total_effective > 0:
        logger.info("Quality filtering complete.")
        logger.info(f"Total lines evaluated: {total_effective}")
        logger.info(f"Kept: {lines_kept} ({(lines_kept/total_effective)*100:.2f}%)")
        logger.info("Removal reasons:")
        for r, c in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {r}: {c} ({(c/total_effective)*100:.2f}%)")
        if save_rejected:
            logger.info(f"Rejected lines (with reasons) saved to: {rejected_path}")
        else:
            logger.info(
                "Rejected lines were NOT saved for inspection (quality.save_rejected_lines "
                "is false and --save-rejected was not passed) — only aggregate counts above."
            )

    elapsed_total = time.time() - start_time

    cleanup_report = None
    # Only auto-delete this source's filtered/ files when this run
    # actually consumed its default (config-resolved) input path — an
    # explicit --input/--output override means the caller pointed this
    # run at something else, so filtered/ should not be touched. Deletes
    # BOTH {source}_ne_filtered.txt and 04's review file for this source
    # (same lifecycle — review files are not preserved independently of
    # their stage's main output) — never anything else in filtered/,
    # since that directory holds one file per SOURCE and other sources
    # may not have reached quality-filtering yet.
    if not (args.input and args.output):
        filtered_review_path = os.path.join(
            os.path.dirname(input_path), default_filtered_review_filename(args.source, config)
        )
        cleanup_report = cleanup_prior_stage_files(
            config, Path(repo_root), stage_to_delete="filtered",
            files=[input_path, filtered_review_path], logger=logger,
        )

    record_run(
        config, Path(repo_root), stage_number="04", stage_name="quality_filter",
        source=args.source,
        summary={
            "input_path": input_path,
            "output_path": output_path,
            "rejected_path": rejected_path if save_rejected else None,
            "lines_processed": lines_processed,
            "lines_kept": lines_kept,
            "removal_reason_counts": reason_counts,
            "elapsed_seconds": round(elapsed_total, 2),
            "cleanup_of_filtered_stage": cleanup_report,
        },
        logger=logger,
    )

if __name__ == "__main__":
    main()
