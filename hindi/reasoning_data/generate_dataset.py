"""Generate the Hindi synthetic comparative-reasoning dataset — Phase 3 §3.1.

Builds every split (train / validation / test_seen / test_unseen_entity /
test_unseen_wording) across all 4 domains x 2 task types x 4 surface-form
templates, and writes them plus dataset_stats.json under
hindi/reasoning_data/.

    python hindi/reasoning_data/generate_dataset.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shared.reasoning import build_language_dataset, save_language_dataset  # noqa: E402
from hindi.reasoning_data.lexicon import (  # noqa: E402
    DOMAINS, PAIRWISE_TEMPLATES, TRANSITIVE_TEMPLATES,
)

OUT_DIR = REPO / "hindi" / "reasoning_data"

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
    print(f"[hindi] wrote reasoning dataset -> {OUT_DIR}")
    for split, n in stats["n_examples_per_split"].items():
        print(f"  {split:<20} {n:>6,} examples")
    print(f"  {len(stats['combinations'])} (task_type, domain) combinations, "
          f"4 templates each (3 trained, 1 held out)")
