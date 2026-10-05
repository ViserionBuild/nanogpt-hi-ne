"""Entity pools and the train/test split that keeps them disjoint.

Spec 3.1 requires "held-out entity names" so train/test leakage can be
ruled out; course guidance (L9 p. 34, the memorization case study on p. 22)
makes the same point concretely — a model scoring 0.27 training loss but
0.000 exact-match on unseen names has memorized 20 names, not learned the
comparison. The fix is structural: split the entity pool once, up front,
and never let a training example and a test example draw from the same
side of that split.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class EntityPool:
    """A fixed, ordered list of surface names for one entity category.

    ``names`` order does not matter semantically (attribute values are
    assigned separately per example) — it is just the pool generation
    draws from.

    ``genders``, if given, maps each name to a grammatical-gender tag
    ("m"/"f") for languages (Hindi, Nepali) whose comparative adjectives
    inflect for the subject's gender — e.g. Hindi बड़ा/बड़ी, अच्छा/अच्छी.
    Templates that need agreement look the tag up per name; a pool with no
    genders (e.g. inanimate objects rendered as a single fixed-gender noun
    class) simply omits it.
    """
    category: str          # e.g. "person", "object" — for bookkeeping only
    names: Tuple[str, ...]
    genders: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.names)

    def gender(self, name: str) -> str:
        return self.genders.get(name, "m")


def split_entity_pool(
    pool: EntityPool,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[EntityPool, EntityPool]:
    """Partition a pool into disjoint train/test name sets.

    A fixed seed makes the split reproducible from the pool alone, which
    is what the spec's "release the generation scripts" requirement needs
    — anyone re-running this gets the identical split, not a new random one.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    names = list(pool.names)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = max(1, round(len(names) * test_fraction))
    test_names = tuple(sorted(names[:n_test]))
    train_names = tuple(sorted(names[n_test:]))
    if not train_names or not test_names:
        raise ValueError(
            f"pool {pool.category!r} too small ({len(pool)}) to split "
            f"at test_fraction={test_fraction}"
        )
    overlap = set(train_names) & set(test_names)
    assert not overlap, f"train/test entity overlap: {overlap}"
    return (
        EntityPool(category=pool.category, names=train_names, genders=pool.genders),
        EntityPool(category=pool.category, names=test_names, genders=pool.genders),
    )


def sample_distinct_names(
    pool: EntityPool, k: int, rng: random.Random
) -> List[str]:
    """Draw ``k`` distinct names from ``pool`` without replacement."""
    if k > len(pool):
        raise ValueError(
            f"cannot draw {k} distinct names from a pool of {len(pool)}"
        )
    return rng.sample(list(pool.names), k)


def sample_distinct_values(
    low: int, high: int, k: int, rng: random.Random
) -> List[int]:
    """Draw ``k`` distinct integers in ``[low, high]``.

    Distinctness matters here for a different reason than for names: a
    comparison chain built on tied values (two entities with the same
    age) has no well-defined "smallest" answer, which would inject label
    noise the course's "correctness" criterion (L9/L10 best-practices
    slide) explicitly warns against.
    """
    if high - low + 1 < k:
        raise ValueError(f"range [{low}, {high}] too narrow for {k} distinct values")
    return rng.sample(range(low, high + 1), k)
