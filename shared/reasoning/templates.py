"""Template rendering: turn a sampled relation chain into prompt/answer text.

Course rationale (L9 p. 34, p. 27): write four paraphrases of the same
underlying task ("surface forms"), train on three, hold the fourth out
completely. A model that only does well on the trained three has learned
the wording; a model that also does the fourth has learned the underlying
comparison. This module is the mechanism for that: a ``TemplateSet`` holds
exactly four ``Template`` renderers per (domain, relation-arity) combination,
and the caller (``generate.py``) is responsible for keeping template index 3
out of the training split.

A ``Template`` is a pure string-formatting function, not a class hierarchy
per language — the language-specific strings themselves (the actual Hindi
or Nepali sentences) live in ``<language>/reasoning_data/lexicon.py`` and
are passed in here as plain Python callables/format strings, so this module
stays language-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Template:
    """One surface form for one (task_type) combination.

    ``render`` takes the task's structured fields (entity names, an
    attribute word, a comparison direction) as keyword arguments and
    returns ``(prompt_text, answer_text)``. Splitting prompt from answer
    here — rather than returning one string — is what lets the finetuning
    dataset mask the loss onto only the answer tokens (course: L9 pp. 6-11,
    "the loss is summed only over the answer positions").
    """
    template_id: int          # 0-3; index 3 is the course's held-out surface form
    render: Callable[..., Tuple[str, str]]


@dataclass(frozen=True)
class TemplateSet:
    """All four surface forms for one task type, plus the train/held-out split.

    ``held_out_index`` fixes *which* of the four templates is never seen
    during training (spec/course: always exactly one). Kept configurable
    rather than hardcoded to index 3 so a test can perturb it, but every
    language config in this project uses the same index for consistency
    across Model H and Model L.
    """
    task_type: str
    templates: Tuple[Template, Template, Template, Template]
    held_out_index: int = 3

    def __post_init__(self) -> None:
        if len(self.templates) != 4:
            raise ValueError(
                f"{self.task_type}: TemplateSet requires exactly 4 surface "
                f"forms (course spec L9 p.34), got {len(self.templates)}"
            )
        if not 0 <= self.held_out_index < 4:
            raise ValueError(f"held_out_index must be in [0, 3]")

    @property
    def train_templates(self) -> List[Template]:
        return [t for t in self.templates if t.template_id != self.held_out_index]

    @property
    def held_out_template(self) -> Template:
        return next(t for t in self.templates if t.template_id == self.held_out_index)

    def render(self, template_id: int, **fields) -> Tuple[str, str]:
        for t in self.templates:
            if t.template_id == template_id:
                return t.render(**fields)
        raise KeyError(f"no template_id={template_id} in {self.task_type}")
