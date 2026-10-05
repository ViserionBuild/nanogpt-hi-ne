"""Generate the Nepali synthetic comparative-reasoning dataset — Phase 3 §3.1.

Mirrors hindi/reasoning_data/generate_dataset.py exactly — same split
structure, same per-combination sizes — reading Nepali's own lexicon so
the two languages' datasets never share entities, templates, or files.

    python nepali/reasoning_data/generate_dataset.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.reasoning import build_language_dataset, save_language_dataset  # noqa: E402
from nepali.reasoning_data.lexicon import (  # noqa: E402
    DOMAINS, PAIRWISE_TEMPLATES, TRANSITIVE_TEMPLATES,
)

OUT_DIR = REPO / "nepali" / "reasoning_data"

if __name__ == "__main__":
    built = build_language_dataset(
        DOMAINS, PAIRWISE_TEMPLATES, TRANSITIVE_TEMPLATES,
        n_train_per_combo=300,
        n_validation_per_combo=40,
        n_test_seen_per_combo=40,
        n_test_unseen_entity_per_combo=40,
        n_test_unseen_wording_per_combo=40,
        test_entity_fraction=0.25,
        seed=42,
    )
    save_language_dataset(built, str(OUT_DIR))

    stats = built["stats"]
    print(f"[nepali] wrote reasoning dataset -> {OUT_DIR}")
    for split, n in stats["n_examples_per_split"].items():
        print(f"  {split:<20} {n:>6,} examples")
    print(f"  {len(stats['combinations'])} (task_type, domain) combinations, "
          f"4 templates each (3 trained, 1 held out)")
