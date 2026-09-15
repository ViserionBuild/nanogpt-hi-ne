"""Chain sampling and label derivation for comparative-reasoning examples.

Spec 3.1's required task shapes:
  * pairwise:    "Given A > B, ... " (which is smaller/larger)
  * comparison over 2-3 entities' heights/ages/prices/quantities
  * multi-hop transitive chains ("A > B, B > C ⇒ relation between A and C")

Two task shapes are generated here, matching the course's own worked
example scope (L9 pp. 3-4 uses a 3-entity chain):

  ``pairwise``    — one relation between two entities; answer is which one
                    is smaller/larger.
  ``transitive3`` — a chain over three entities (A>B, B>C); answer is
                    which is smallest / largest / (optionally) the middle.

Every example's label is derived by plain arithmetic on the sampled
attribute values, never guessed — this is what the course's "correctness"
best-practice criterion (L9/L10) requires ("is every reply actually
right?"): the generator is the ground truth, so there is no separate
labeling step that could disagree with it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .entities import EntityPool, sample_distinct_names, sample_distinct_values
from .templates import TemplateSet


@dataclass(frozen=True)
class GeneratedExample:
    """One finetuning example: rendered text plus full provenance.

    Provenance fields (``task_type``, ``template_id``, ``entities``,
    ``values``) are kept alongside the text so the eval harness can bucket
    accuracy by exactly the axes the course rubric asks for (seen/unseen
    entity, seen/unseen wording) without re-parsing the rendered string.
    """
    prompt: str
    answer: str
    task_type: str            # "pairwise" | "transitive3"
    domain: str                # "age" | "height" | "price" | "quantity"
    template_id: int
    entities: Tuple[str, ...]
    values: Tuple[int, ...]
    split: str                # "train" | "validation" | "test"
    entity_split: str          # "train" | "test" — which entity pool this drew from


def _pairwise_label(names: List[str], values: List[int], want: str) -> str:
    """``want`` is 'smaller' or 'larger'."""
    pair = list(zip(names, values))
    pair.sort(key=lambda nv: nv[1])
    return pair[0][0] if want == "smaller" else pair[-1][0]


def _transitive3_label(names: List[str], values: List[int], want: str) -> str:
    """``want`` is 'smallest', 'largest', or 'middle'."""
    triple = list(zip(names, values))
    triple.sort(key=lambda nv: nv[1])
    return {"smallest": triple[0][0], "middle": triple[1][0],
            "largest": triple[-1][0]}[want]


_MAX_REJECTION_ATTEMPTS = 200


def generate_pairwise(
    pool: EntityPool,
    template_set: TemplateSet,
    domain: str,
    value_range: Tuple[int, int],
    n: int,
    split: str,
    entity_split: str,
    template_ids: List[int],
    rng: random.Random,
    ask: Tuple[str, ...] = ("smaller", "larger"),
    seen_prompts: Optional[set] = None,
) -> List[GeneratedExample]:
    """Sample ``n`` two-entity comparisons.

    ``template_ids`` restricts which of the four surface forms may be
    drawn (the caller passes only the three training templates for the
    train/validation split, and only the held-out template for the
    template-generalization test split).

    ``seen_prompts``, if given, is a set of prompt strings to avoid and is
    updated in place with every prompt this call emits — passing the SAME
    set across every split of a dataset build is what keeps ``test_seen``
    disjoint from the literal examples ``train`` already used (course
    intent: "did it fit the distribution" should not degenerate into
    "does it recall this exact sentence"), and keeps every split
    duplicate-free internally. Rejection sampling rather than a rescaled
    population, because the point is exactness at a small pool size, not
    throughput.
    """
    out = []
    low, high = value_range
    for _ in range(n):
        for _attempt in range(_MAX_REJECTION_ATTEMPTS):
            names = sample_distinct_names(pool, 2, rng)
            values = sample_distinct_values(low, high, 2, rng)
            want = rng.choice(ask)
            tid = rng.choice(template_ids)
            label = _pairwise_label(names, values, want)
            prompt, answer_template = template_set.render(
                tid, name_a=names[0], name_b=names[1],
                value_a=values[0], value_b=values[1], want=want,
                gender_a=pool.gender(names[0]), gender_b=pool.gender(names[1]),
            )
            if seen_prompts is None or prompt not in seen_prompts:
                break
        if seen_prompts is not None:
            seen_prompts.add(prompt)
        # answer_template is a format string with {answer} left for the label
        answer = answer_template.format(answer=label)
        out.append(GeneratedExample(
            prompt=prompt, answer=answer, task_type="pairwise", domain=domain,
            template_id=tid, entities=tuple(names), values=tuple(values),
            split=split, entity_split=entity_split,
        ))
    return out


def generate_transitive3(
    pool: EntityPool,
    template_set: TemplateSet,
    domain: str,
    value_range: Tuple[int, int],
    n: int,
    split: str,
    entity_split: str,
    template_ids: List[int],
    rng: random.Random,
    ask: Tuple[str, ...] = ("smallest", "largest", "middle"),
    seen_prompts: Optional[set] = None,
) -> List[GeneratedExample]:
    """Sample ``n`` three-entity transitive-chain comparisons.

    The chain is always presented as two pairwise facts (A vs B, B vs C)
    in the prompt; the question asks for the smallest/largest/middle
    across all three, which requires chaining the two stated facts rather
    than reading off a single one (spec 3.1: "short multi-hop comparisons
    that require chaining inequalities"). See ``generate_pairwise`` for
    what ``seen_prompts`` does.
    """
    out = []
    low, high = value_range
    for _ in range(n):
        for _attempt in range(_MAX_REJECTION_ATTEMPTS):
            names = sample_distinct_names(pool, 3, rng)
            values = sample_distinct_values(low, high, 3, rng)
            want = rng.choice(ask)
            tid = rng.choice(template_ids)
            label = _transitive3_label(names, values, want)
            # Present the chain in sampled entity order (A, B, C) as two
            # pairwise facts: A-vs-B and B-vs-C — the "chain" the model
            # must transit rather than a single already-stated fact.
            a, b, c = names
            va, vb, vc = values
            prompt, answer_template = template_set.render(
                tid, name_a=a, name_b=b, name_c=c,
                value_a=va, value_b=vb, value_c=vc, want=want,
                gender_a=pool.gender(a), gender_b=pool.gender(b),
                gender_c=pool.gender(c),
            )
            if seen_prompts is None or prompt not in seen_prompts:
                break
        if seen_prompts is not None:
            seen_prompts.add(prompt)
        answer = answer_template.format(answer=label)
        out.append(GeneratedExample(
            prompt=prompt, answer=answer, task_type="transitive3", domain=domain,
            template_id=tid, entities=tuple(names), values=tuple(values),
            split=split, entity_split=entity_split,
        ))
    return out


def generate_dataset(
    train_pool: EntityPool,
    test_pool: EntityPool,
    template_set: TemplateSet,
    domain: str,
    value_range: Tuple[int, int],
    n_train: int,
    n_validation: int,
    n_test_seen_wording: int,
    n_test_unseen_entity: int,
    n_test_unseen_wording: int,
    task_type: str,
    seed: int = 42,
) -> Dict[str, List[GeneratedExample]]:
    """Build all splits for one (task_type, domain) combination.

    Implements the course's three-way held-out structure (L9 p. 27 / p. 34)
    as four named splits:

      ``train``               — train entities, one of the 3 training templates
      ``validation``           — train entities, one of the 3 training templates
                                (disjoint examples from train, same distribution)
      ``test_seen``            — train entities, one of the 3 training templates
                                 -> "did it fit the distribution at all"
      ``test_unseen_entity``   — TEST entities, one of the 3 training templates
                                 -> "does it generalize past the trained names"
      ``test_unseen_wording``  — TEST entities, the HELD-OUT 4th template
                                 -> "does it survive a change of wording"

    ``test_unseen_wording`` deliberately also uses unseen entities: the
    course's point is that this number should reflect wording
    generalization specifically, but combining it with unseen entities
    makes it the strictest, most conservative of the three test numbers
    (a model cannot pass by having merely memorized the held-out phrasing
    applied to familiar names, because it never saw that either).
    """
    rng = random.Random(seed)
    train_ids = [t.template_id for t in template_set.train_templates]
    held_out_id = template_set.held_out_template.template_id
    gen = generate_pairwise if task_type == "pairwise" else generate_transitive3

    # One shared "already used" set across every split drawn from the TRAIN
    # entity pool: this is what keeps test_seen disjoint from the literal
    # sentences train already trained on (not just avoiding duplicates
    # within a single split) — otherwise "did it fit the distribution"
    # could quietly become "does it recall this exact example".
    train_side_seen: set = set()
    # A second, separate set for the TEST entity pool's two splits — kept
    # apart from train_side_seen because those prompts can never collide
    # with train-pool prompts anyway (disjoint names), so sharing one set
    # would just do extra rejection-sampling work for no purpose.
    test_side_seen: set = set()

    return {
        "train": gen(train_pool, template_set, domain, value_range, n_train,
                     "train", "train", train_ids, rng,
                     seen_prompts=train_side_seen),
        "validation": gen(train_pool, template_set, domain, value_range,
                          n_validation, "validation", "train", train_ids, rng,
                          seen_prompts=train_side_seen),
        "test_seen": gen(train_pool, template_set, domain, value_range,
                         n_test_seen_wording, "test", "train", train_ids, rng,
                         seen_prompts=train_side_seen),
        "test_unseen_entity": gen(test_pool, template_set, domain, value_range,
                                  n_test_unseen_entity, "test", "test",
                                  train_ids, rng, seen_prompts=test_side_seen),
        "test_unseen_wording": gen(test_pool, template_set, domain, value_range,
                                   n_test_unseen_wording, "test", "test",
                                   [held_out_id], rng, seen_prompts=test_side_seen),
    }
