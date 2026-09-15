"""Assemble one language's full synthetic reasoning dataset and write it out.

Combines every (task_type, domain) combination a language's lexicon
defines into the five course-prescribed splits (§2.2 of
utility/phase3_plan.md), then writes:

  * one JSON file per split, with full provenance per example (so the
    finetuning dataset loader and the 3-way accuracy evaluator can both
    read the same files without re-deriving anything), and
  * a stats JSON summarizing size, template variety, and the entity-pool
    split, which is the "report size, template variety, and how you avoid
    train-test leakage" deliverable the spec asks for directly (§3.1).

Language-agnostic: takes the lexicon's DOMAINS/PAIRWISE_TEMPLATES/
TRANSITIVE_TEMPLATES dicts as arguments rather than importing a language
package itself.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

from .entities import EntityPool, split_entity_pool
from .generate import GeneratedExample, generate_dataset
from .templates import TemplateSet

SPLIT_NAMES = (
    "train", "validation", "test_seen",
    "test_unseen_entity", "test_unseen_wording",
)


def build_language_dataset(
    domains: Dict[str, Dict],
    pairwise_templates: Dict[str, TemplateSet],
    transitive_templates: Dict[str, TemplateSet],
    n_train_per_combo: int = 300,
    n_validation_per_combo: int = 40,
    n_test_seen_per_combo: int = 40,
    n_test_unseen_entity_per_combo: int = 40,
    n_test_unseen_wording_per_combo: int = 40,
    test_entity_fraction: float = 0.25,
    seed: int = 42,
) -> Dict[str, object]:
    """Build every split across all (task_type, domain) combinations.

    Each domain gets its own entity-pool split (age/height share the
    person pool, so both are split identically — this is fine, since the
    split is a property of the *pool*, not the task, and the same person
    pool must not leak between age and height tasks either).
    """
    pool_splits: Dict[str, tuple] = {}
    for pool in {id(v["pool"]): v["pool"] for v in domains.values()}.values():
        pool_splits[pool.category] = split_entity_pool(
            pool, test_fraction=test_entity_fraction, seed=seed
        )

    all_splits: Dict[str, List[GeneratedExample]] = {name: [] for name in SPLIT_NAMES}
    combo_stats = []

    combo_seed = seed
    for domain, spec in domains.items():
        pool = spec["pool"]
        train_pool, test_pool = pool_splits[pool.category]
        value_range = (spec["unit_low"], spec["unit_high"])

        for task_type, templates in (
            ("pairwise", pairwise_templates.get(domain)),
            ("transitive3", transitive_templates.get(domain)),
        ):
            if templates is None:
                continue
            combo_seed += 1
            ds = generate_dataset(
                train_pool, test_pool, templates, domain, value_range,
                n_train=n_train_per_combo,
                n_validation=n_validation_per_combo,
                n_test_seen_wording=n_test_seen_per_combo,
                n_test_unseen_entity=n_test_unseen_entity_per_combo,
                n_test_unseen_wording=n_test_unseen_wording_per_combo,
                task_type=task_type, seed=combo_seed,
            )
            for split_name, examples in ds.items():
                all_splits[split_name].extend(examples)
            combo_stats.append({
                "domain": domain, "task_type": task_type,
                "train_pool_size": len(train_pool), "test_pool_size": len(test_pool),
                **{f"n_{k}": len(v) for k, v in ds.items()},
            })

    stats = {
        "n_examples_per_split": {name: len(exs) for name, exs in all_splits.items()},
        "combinations": combo_stats,
        "n_domains": len(domains),
        "n_task_types": 2,
        "n_templates_per_combo": 4,
        "n_templates_trained": 3,
        "n_templates_held_out": 1,
        "test_entity_fraction": test_entity_fraction,
        "seed": seed,
        "leakage_avoidance": (
            "Each attribute domain's entity pool (person/object) is split "
            "once into disjoint train/test name sets before any example is "
            "generated (shared/reasoning/entities.py:split_entity_pool); "
            "'train', 'validation' and 'test_seen' draw only from the train "
            "names, 'test_unseen_entity' and 'test_unseen_wording' draw "
            "only from the test names. Separately, one of each task's four "
            "surface-form templates is held out of training entirely and "
            "used only in 'test_unseen_wording'. The two axes (entity "
            "identity, surface wording) are varied independently so a model "
            "can fail one without automatically failing the other."
        ),
    }
    return {"splits": all_splits, "stats": stats}


def _example_to_dict(ex: GeneratedExample) -> Dict:
    d = asdict(ex)
    return d


def save_language_dataset(built: Dict[str, object], out_dir: str) -> None:
    """Write one JSON file per split plus dataset_stats.json."""
    os.makedirs(out_dir, exist_ok=True)
    for split_name, examples in built["splits"].items():
        path = os.path.join(out_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([_example_to_dict(e) for e in examples], f,
                     indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(built["stats"], f, indent=2, ensure_ascii=False)


def load_split(out_dir: str, split_name: str) -> List[Dict]:
    """Read one split's JSON back as plain dicts (finetune dataset loader)."""
    path = os.path.join(out_dir, f"{split_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
