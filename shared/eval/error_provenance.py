"""Classify wrong reasoning predictions by where the predicted name came from.

Phase 3's qualitative examples (a handful of "gold vs. predicted" pairs)
illustrate failure modes but don't quantify them. This module answers, over
every wrong prediction in a split: did the model hallucinate a name from the
TRAINING entity pool (memorization, not reasoning), attempt a malformed or
truncated version of the actual gold name (engaged with the prompt but
failed mechanically), name a different entity that legitimately appears in
the prompt (picked the wrong side of the comparison), or something else?

Reuses the same train/test pool split (`shared/reasoning/entities.py:
split_entity_pool`) the dataset generator itself used, so "train-pool name"
here means exactly the pool the finetuning data drew from.
"""
from __future__ import annotations

from typing import Dict, List

from shared.reasoning.entities import EntityPool, split_entity_pool

TRAIN_POOL_HALLUCINATION = "train_pool_hallucination"
MALFORMED_REAL_NAME = "malformed_or_truncated_real_name"
LEGAL_WRONG_ENTITY = "legal_but_wrong_entity"
OTHER = "empty_or_other"

CATEGORY_LABELS = {
    TRAIN_POOL_HALLUCINATION: "Train-pool hallucination",
    MALFORMED_REAL_NAME: "Malformed/truncated real name",
    LEGAL_WRONG_ENTITY: "Legal but wrong entity",
    OTHER: "Empty / other",
}


def _all_names(pools: List[EntityPool]) -> set:
    names = set()
    for pool in pools:
        names.update(pool.names)
    return names


def _train_test_name_sets(*pools: EntityPool) -> tuple:
    train_names, test_names = set(), set()
    for pool in pools:
        train_pool, test_pool = split_entity_pool(pool)
        train_names.update(train_pool.names)
        test_names.update(test_pool.names)
    return train_names, test_names


def _is_malformed_attempt(predicted: str, gold: str) -> bool:
    """A short, non-empty string that is a strict substring/prefix/suffix
    fragment of the gold name (a truncated generation) or vice-versa/shares
    a long common run with it (a garbled generation), but is not itself any
    known entity name. Names in these lexicons are short (2-6 characters),
    so a shared run of 1 character is not meaningful evidence on its own —
    require at least half of the shorter string's length to overlap.
    """
    if not predicted or predicted == gold:
        return False
    if predicted in gold or gold in predicted:
        return True
    shorter, longer = sorted([predicted, gold], key=len)
    if len(shorter) < 1:
        return False
    overlap = sum(1 for ch in shorter if ch in longer)
    return overlap >= max(1, len(shorter) // 2)


def classify_prediction(record: Dict, train_names: set, test_names: set,
                        all_names: set) -> str:
    """Classify one wrong qualitative-example record into one bucket.

    ``record`` must carry ``predicted``, ``gold``, and ``entities`` (the
    real entity names that appear in that specific prompt — used to detect
    "legal but wrong entity", i.e. the model named the *other* entity in
    the comparison instead of the correct one).
    """
    predicted = record["predicted"]
    gold = record["gold"]
    entities = set(record.get("entities") or [])

    if not predicted:
        return OTHER
    if predicted in entities and predicted != gold:
        return LEGAL_WRONG_ENTITY
    if predicted in train_names and predicted not in entities:
        return TRAIN_POOL_HALLUCINATION
    if predicted not in all_names and _is_malformed_attempt(predicted, gold):
        return MALFORMED_REAL_NAME
    if predicted in test_names and predicted not in entities:
        # A real test-pool name, but not one named in this specific prompt
        # and not the gold answer either — same "picked a name it knows
        # exists" failure as train-pool hallucination, just from the wrong
        # pool half; group with hallucination since the mechanism (naming
        # a familiar entity instead of reasoning about this prompt's own
        # entities) is identical.
        return TRAIN_POOL_HALLUCINATION
    return OTHER


def error_provenance_table(qualitative_examples: List[Dict], *pools: EntityPool) -> Dict:
    """Aggregate error-provenance counts/percentages over wrong predictions.

    ``qualitative_examples`` is a flat list of per-example records (as
    produced by ``shared.eval.reasoning_metrics.evaluate_split_accuracy``,
    with ``correct=False`` entries only contributing); ``pools`` are the
    language's ``EntityPool`` objects (e.g. ``PERSON_POOL``, ``OBJECT_POOL``
    from that language's ``reasoning_data/lexicon.py``).
    """
    train_names, test_names = _train_test_name_sets(*pools)
    all_names = _all_names(list(pools))

    wrong = [r for r in qualitative_examples if not r["correct"]]
    counts = {k: 0 for k in CATEGORY_LABELS}
    for r in wrong:
        counts[classify_prediction(r, train_names, test_names, all_names)] += 1

    n_wrong = len(wrong)
    return {
        "n_wrong": n_wrong,
        "counts": counts,
        "percentages": {
            k: (v / n_wrong if n_wrong else 0.0) for k, v in counts.items()
        },
    }


def error_provenance_by_task_family(qualitative_examples: List[Dict],
                                    *pools: EntityPool) -> Dict[str, Dict]:
    """Same as ``error_provenance_table`` but bucketed by ``task_type``."""
    by_task: Dict[str, List[Dict]] = {}
    for r in qualitative_examples:
        by_task.setdefault(r.get("task_type", "unknown"), []).append(r)
    return {task: error_provenance_table(recs, *pools) for task, recs in by_task.items()}
