import argparse
import json
import logging
import os
import sys
import yaml
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from shared.tokenizer.data_utils import iter_lines

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging based on config."""
    logging_config = config.get("logging", {})
    level_str = logging_config.get("level", "INFO")
    fmt = logging_config.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")

    logging.basicConfig(level=getattr(logging, level_str), format=fmt)
    return logging.getLogger("06_statistics")

def compute_file_stats(file_path: str, logger: logging.Logger, progress_interval: int, is_dedup: bool = False) -> Dict[str, Any]:
    """Compute statistics for a single file in the pipeline."""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return {}

    sentences = 0
    total_chars = 0
    min_len = float('inf')
    max_len = 0
    est_words = 0

    # Character categories
    cat_counts = {
        "devanagari": 0,
        "latin": 0,
        "digit": 0,
        "punctuation": 0,
        "other": 0
    }

    word_counter = Counter() if is_dedup else None

    logger.info(f"Computing stats for {file_path}")
    for line in iter_lines(file_path):
        sentences += 1
        length = len(line)
        total_chars += length

        if length < min_len: min_len = length
        if length > max_len: max_len = length

        words = line.split()
        est_words += len(words)

        if is_dedup:
            word_counter.update(words)

        for char in line:
            code = ord(char)
            # Basic character categorization
            if 0x0900 <= code <= 0x097F or 0xA8E0 <= code <= 0xA8FF or 0x1CD0 <= code <= 0x1CFF:
                cat_counts["devanagari"] += 1
            elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
                cat_counts["latin"] += 1
            elif char.isdigit():
                cat_counts["digit"] += 1
            elif char in ".,;:!?'\"()[]{}<>-/\\_`~।॥":
                cat_counts["punctuation"] += 1
            elif not char.isspace():
                cat_counts["other"] += 1

        if sentences % progress_interval == 0:
            logger.info(f"Processed {sentences} lines from {file_path}")

    if sentences == 0:
        min_len = 0

    stats = {
        "lines": sentences,  # alias of "sentences" — CL3410 report naming
        "sentences": sentences,
        "total_characters": total_chars,
        "min_length": min_len if min_len != float('inf') else 0,
        "max_length": max_len,
        "mean_length": total_chars / sentences if sentences > 0 else 0,
        "estimated_words": est_words,
        "character_categories": {}
    }

    total_nonspace = sum(cat_counts.values())
    if total_nonspace > 0:
        for k, v in cat_counts.items():
            stats["character_categories"][f"{k}_percentage"] = (v / total_nonspace) * 100

    if is_dedup and word_counter:
        stats["unique_words"] = len(word_counter)
        stats["top_100_words"] = word_counter.most_common(100)

    return stats

PUBLIC_SOURCE = "indiccorp"
MANUAL_SOURCES = ("wikipedia", "news")
ALL_SOURCES = (PUBLIC_SOURCE,) + MANUAL_SOURCES

# Only "deduplicated" is checked — raw/normalized/filtered/quality are
# intentionally NOT reported here: normalized/filtered/quality are
# auto-deleted by 04/05/06 (see shared/run_stats.py's DELETABLE_STAGES)
# as soon as the next stage finishes consuming them, so by the time this
# script runs those files are already gone; raw survives but its own
# per-source counts are already recorded by 02_normalize.py's run_stats
# entry, so re-deriving them here would just duplicate that.
STAGES = ("deduplicated",)


def files_to_check_for_source(source: str, config: Dict[str, Any], repo_root: Path) -> Dict[str, str]:
    """Mirrors 05_deduplicate.py's/05_deduplicate_mac.py's default output
    filename convention — the only stage this script reports on."""
    return {
        "deduplicated": str(repo_root / config["paths"]["deduplicated"] / f"{source}_hi_deduplicated.txt"),
    }


def sum_stats(stage_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum the numeric (non-empty) per-source stats for one stage into a
    combined total. Character-category percentages and top-100-words are
    per-source only (summing percentages/Counters across sources isn't
    meaningful) — only absolute counts are combined here."""
    present = [s for s in stage_stats_list if s]
    if not present:
        return {}
    return {
        "sentences": sum(s.get("sentences", 0) for s in present),
        "total_characters": sum(s.get("total_characters", 0) for s in present),
        "estimated_words": sum(s.get("estimated_words", 0) for s in present),
        "sources_present": len(present),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive corpus statistics for all three Hindi "
                     "sources (IndicCorp, Wikipedia, News), plus "
                     "combined manual/all-sources summaries."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output", type=str, help="Override output JSON file path")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)
    progress_interval = config.get("logging", {}).get("progress_interval", 1000000)
    repo_root = Path(__file__).resolve().parent.parent.parent

    by_source: Dict[str, Dict[str, Any]] = {}
    for source in ALL_SOURCES:
        files_to_check = files_to_check_for_source(source, config, repo_root)
        source_stats = {}
        for stage in STAGES:
            is_dedup = (stage == "deduplicated")
            source_stats[stage] = compute_file_stats(
                files_to_check[stage], logger, progress_interval, is_dedup=is_dedup
            )
        by_source[source] = source_stats

    manual_summary = {
        stage: sum_stats([by_source[s][stage] for s in MANUAL_SOURCES]) for stage in STAGES
    }
    all_sources_summary = {
        stage: sum_stats([by_source[s][stage] for s in ALL_SOURCES]) for stage in STAGES
    }

    # Top-level keys stay IndicCorp's own stats, unchanged in shape, so
    # existing downstream readers keep working without modification.
    all_stats: Dict[str, Any] = dict(by_source[PUBLIC_SOURCE])
    all_stats["by_source"] = by_source
    all_stats["manual_summary"] = manual_summary
    all_stats["all_sources_summary"] = all_sources_summary

    if args.output:
        out_path = args.output
    else:
        out_path = str(repo_root / config["paths"]["metadata"] / "corpus_statistics.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=4, ensure_ascii=False)

    logger.info(f"Statistics saved to {out_path}")

    # Print summary table — one block per source, plus manual/all-sources totals
    for source in ALL_SOURCES:
        source_stats = by_source[source]
        print("\n" + "=" * 80)
        print(f"SOURCE: {source}")
        print(f"{'Stage':<15} | {'Sentences':<12} | {'Characters':<15} | {'Est. Words':<15}")
        print("-" * 80)

        for stage in STAGES:
            st = source_stats.get(stage, {})
            if not st:
                continue
            sents = st.get("sentences", 0)
            chars = st.get("total_characters", 0)
            words = st.get("estimated_words", 0)
            print(f"{stage:<15} | {sents:<12,d} | {chars:<15,d} | {words:<15,d}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("MANUAL SUMMARY (Wikipedia + News)")
    for stage in STAGES:
        st = manual_summary.get(stage, {})
        if not st:
            continue
        print(f"  {stage:<15}: {st.get('sentences', 0):,} sentences, "
              f"{st.get('estimated_words', 0):,} words "
              f"({st.get('sources_present', 0)}/{len(MANUAL_SOURCES)} sources present)")
    print("\nALL SOURCES (IndicCorp + Manual)")
    for stage in STAGES:
        st = all_sources_summary.get(stage, {})
        if not st:
            continue
        print(f"  {stage:<15}: {st.get('sentences', 0):,} sentences, "
              f"{st.get('estimated_words', 0):,} words "
              f"({st.get('sources_present', 0)}/{len(ALL_SOURCES)} sources present)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
