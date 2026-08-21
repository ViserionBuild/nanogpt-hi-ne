import os
import sys
import argparse
import logging
import unicodedata
import re
import time
import yaml
from pathlib import Path

# Add shared to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from shared.data_utils import count_lines, iter_lines
from shared.run_stats import record_run

# Exactly three logical/top-level corpus sources — see 05_deduplicate.py's
# module docstring for the full rationale. "wikipedia" is read directly
# from manual.wikipedia.output_dir (manual/01_download_wikipedia.py +
# manual/02_extract_wikipedia.py already produce one flat file, no aggregation
# needed). "news" is read from manual/05_aggregate_news_corpus.py's combined
# output (news_intermediate/news_raw.txt), which pre-merges every enabled
# news outlet into one real per-source file — individual outlets are
# never a separate --source value here.
SOURCE_CHOICES = ("indiccorp", "wikipedia", "news")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("02_normalize")

# Regex to collapse multiple spaces
SPACE_RE = re.compile(r' {2,}')

def is_unsafe_control_char(char: str) -> bool:
    """
    Check if a character is an unsafe control character.
    Removes C0 (0x00-0x1F) except \n and \t, and C1 (0x80-0x9F).
    Preserves ZWNJ (0x200C) and ZWJ (0x200D).
    """
    cp = ord(char)
    if cp == 0x200C or cp == 0x200D:
        return False
    if cp == 0x09 or cp == 0x0A or cp == 0x0D:
        return False
    if (0x00 <= cp <= 0x1F) or (0x80 <= cp <= 0x9F):
        return True
    
    # Check unicode category for 'Cc' or 'Cf' except preserved ones
    cat = unicodedata.category(char)
    if cat in ('Cc', 'Cf'):
        return True
    return False

def remove_unsafe_chars(text: str) -> str:
    return "".join(c for c in text if not is_unsafe_control_char(c))

def normalize_line(line: str) -> str:
    """
    Normalize a single line of text.
    - \r\n -> \n, tab -> space
    - NFC normalization
    - Strip leading/trailing whitespaces
    - Collapse spaces
    - Remove unsafe control chars
    """
    # Replace tabs and newlines
    line = line.replace('\r\n', '\n').replace('\t', ' ')
    # NFC normalize
    line = unicodedata.normalize('NFC', line)
    # Remove control chars
    line = remove_unsafe_chars(line)
    # Collapse spaces
    line = SPACE_RE.sub(' ', line)
    # Strip
    return line.strip()

def default_paths(source: str, config: dict, repo_root: str) -> tuple:
    """Resolve the default (input_path, output_path) for a given source.
    Every source is normalized with the exact same logic (normalize_line);
    only the file locations differ.

    - indiccorp: raw_indiccorp/{indiccorp.filename}
    - wikipedia: read directly from manual.wikipedia.output_dir/
      {raw_text_filename} (manual/01_download_wikipedia.py +
      manual/02_extract_wikipedia.py output) — NO aggregation step, this is
      already one real file.
    - news: news_intermediate/news_raw.txt (manual/05_aggregate_news_corpus.py's
      combined output — pre-merge of every enabled news outlet).

    All three write to paths['normalized']/{source}_hi_normalized.txt."""
    if source == "wikipedia":
        wiki_cfg = config['manual']['wikipedia']
        raw_filename = wiki_cfg.get('raw_text_filename', 'wikipedia_hi_raw.txt')
        input_path = os.path.join(repo_root, wiki_cfg['output_dir'], raw_filename)
    elif source == "news":
        news_dir = os.path.join(repo_root, config['paths'].get('news_intermediate', 'hindi/data/intermediate/news'))
        input_path = os.path.join(news_dir, 'news_raw.txt')
    else:
        input_path = os.path.join(repo_root, config['paths']['raw_indiccorp'], config['indiccorp']['filename'])
    output_path = os.path.join(repo_root, config['paths']['normalized'], f'{source}_hi_normalized.txt')
    return input_path, output_path

def main():
    parser = argparse.ArgumentParser(description="Normalize Hindi text.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--source", type=str, choices=list(SOURCE_CHOICES), default="indiccorp",
                         help="Which logical source corpus to normalize (default: indiccorp). "
                              "'wikipedia' reads directly from the Wikipedia dump extraction "
                              "(no aggregation needed). 'news' reads the combined output of "
                              "manual/05_aggregate_news_corpus.py, which pre-merges every enabled "
                              "news outlet — individual outlets are never a separate --source "
                              "value.")
    parser.add_argument("--input", type=str, help="Input file path (overrides --source default).")
    parser.add_argument("--output", type=str, help="Output file path (overrides --source default).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")

    args = parser.parse_args()
    logger = setup_logger()

    config = load_config(args.config)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    default_input_path, default_output_path = default_paths(args.source, config, repo_root)
    input_path = args.input if args.input else default_input_path
    output_path = args.output if args.output else default_output_path

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
            
    mode = 'a' if args.resume else 'w'
    
    lines_processed = 0
    lines_modified = 0
    lines_unchanged = 0
    start_time = time.time()
    
    log_interval = config.get('logging', {}).get('progress_interval', 1000000)
    
    logger.info(f"Starting normalization. Input: {input_path}, Output: {output_path}")
    
    with open(output_path, mode, encoding='utf-8') as out_f:
        for line in iter_lines(input_path):
            if lines_processed < skip_lines:
                lines_processed += 1
                continue
                
            stripped = line.strip('\r\n')
            normalized = normalize_line(stripped)
            
            if stripped == normalized:
                lines_unchanged += 1
            else:
                lines_modified += 1
                
            out_f.write(normalized + '\n')
            lines_processed += 1
            
            if lines_processed % log_interval == 0:
                elapsed = time.time() - start_time
                speed = (lines_processed - skip_lines) / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {lines_processed} lines... Speed: {speed:.2f} lines/sec")
                
    elapsed_total = time.time() - start_time
    logger.info(f"Finished normalization.")
    logger.info(f"Total lines processed: {lines_processed}")
    logger.info(f"Lines modified: {lines_modified}")
    logger.info(f"Lines unchanged: {lines_unchanged}")

    record_run(
        config, Path(repo_root), stage_number="02", stage_name="normalize",
        source=args.source,
        summary={
            "input_path": input_path,
            "output_path": output_path,
            "lines_processed": lines_processed,
            "lines_modified": lines_modified,
            "lines_unchanged": lines_unchanged,
            "elapsed_seconds": round(elapsed_total, 2),
        },
        logger=logger,
    )

if __name__ == "__main__":
    main()
