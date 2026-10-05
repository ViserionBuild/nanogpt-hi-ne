"""Synthetic comparative-reasoning data generation for Phase 3.

Language-agnostic by construction, same rule as every other module under
``shared/`` (enforced by
``tests/test_phase3.py::test_shared_reasoning_never_names_a_language``):
no file here references Hindi or Nepali by name. Each language supplies its
own entity names and template strings from
``<language>/reasoning_data/lexicon.py``; this package only supplies the
combinatorics (which entities, which relation chain, which template slot)
and the label arithmetic.
"""
from .build import build_language_dataset, load_split, save_language_dataset
from .entities import EntityPool, split_entity_pool
from .generate import GeneratedExample, generate_dataset
from .templates import Template, TemplateSet

__all__ = [
    "EntityPool",
    "split_entity_pool",
    "GeneratedExample",
    "generate_dataset",
    "Template",
    "TemplateSet",
    "build_language_dataset",
    "save_language_dataset",
    "load_split",
]
