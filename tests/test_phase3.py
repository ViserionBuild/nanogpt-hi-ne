"""Phase 3 correctness gates.

Run with:  python -m pytest tests/test_phase3.py -v

Mirrors tests/test_phase2.py's role: checks that must pass before spending
any real training time, plus structural guarantees the report leans on
(entity/template leakage avoidance, loss-mask correctness) that would be
expensive to notice by eye in a 2,400-example JSON file.
"""
from __future__ import annotations

import ast
import random
import sys
import tokenize
import io
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.eval.reasoning_metrics import extract_predicted_answer  # noqa: E402
from shared.finetune.dataset import ReasoningDataset, encode_example  # noqa: E402
from shared.model import DecoderLM, GPTConfig  # noqa: E402
from shared.reasoning.entities import (  # noqa: E402
    EntityPool, sample_distinct_names, sample_distinct_values, split_entity_pool,
)
from shared.reasoning.generate import generate_dataset  # noqa: E402
from shared.reasoning.templates import Template, TemplateSet  # noqa: E402


def tiny_config(**kw) -> GPTConfig:
    base = dict(vocab_size=500, n_layer=2, n_head=4, d_model=64,
               d_ff=256, block_size=64)
    base.update(kw)
    return GPTConfig(**base)


# ------------------------------------------------------------------ entities

def test_split_entity_pool_is_disjoint_and_covers_the_pool():
    pool = EntityPool(category="test", names=tuple(f"n{i}" for i in range(20)))
    train, test = split_entity_pool(pool, test_fraction=0.25, seed=1)
    assert set(train.names) & set(test.names) == set()
    assert set(train.names) | set(test.names) == set(pool.names)
    assert len(test.names) == 5   # round(20 * 0.25)


def test_split_entity_pool_is_reproducible():
    pool = EntityPool(category="test", names=tuple(f"n{i}" for i in range(20)))
    a_train, a_test = split_entity_pool(pool, seed=7)
    b_train, b_test = split_entity_pool(pool, seed=7)
    assert a_train.names == b_train.names
    assert a_test.names == b_test.names


def test_split_entity_pool_rejects_too_small_a_pool():
    """A single-name pool cannot produce two non-empty sides at any
    fraction — round(1 * f) is always 0, so train would get everything
    and test would get nothing."""
    pool = EntityPool(category="test", names=("solo",))
    with pytest.raises(ValueError):
        split_entity_pool(pool, test_fraction=0.5, seed=1)


def test_gender_lookup_defaults_to_masculine_when_untagged():
    pool = EntityPool(category="test", names=("a", "b"), genders={"a": "f"})
    assert pool.gender("a") == "f"
    assert pool.gender("b") == "m"


def test_sample_distinct_values_are_actually_distinct():
    rng = random.Random(0)
    for _ in range(50):
        vals = sample_distinct_values(1, 10, 3, rng)
        assert len(set(vals)) == 3


def test_sample_distinct_names_rejects_oversized_k():
    pool = EntityPool(category="test", names=("a", "b"))
    with pytest.raises(ValueError):
        sample_distinct_names(pool, 3, random.Random(0))


# ----------------------------------------------------------------- templates

def _toy_template_set() -> TemplateSet:
    def make(i):
        return Template(i, lambda **f: (f"prompt-{i}-{f['x']}", "उत्तर: {answer}"))
    return TemplateSet(task_type="toy", templates=tuple(make(i) for i in range(4)))


def test_template_set_requires_exactly_four():
    with pytest.raises(ValueError):
        TemplateSet(task_type="toy", templates=(Template(0, lambda **f: ("", "")),))


def test_held_out_template_is_excluded_from_train_templates():
    ts = _toy_template_set()
    train_ids = {t.template_id for t in ts.train_templates}
    assert train_ids == {0, 1, 2}
    assert ts.held_out_template.template_id == 3


# -------------------------------------------------------- dataset generation

def _small_pools():
    pool = EntityPool(category="person", names=tuple(f"E{i}" for i in range(16)))
    return split_entity_pool(pool, test_fraction=0.25, seed=1)


def _pairwise_template_set():
    def render(**f):
        rel = "more" if f["value_a"] > f["value_b"] else "less"
        q = "smaller?" if f["want"] == "smaller" else "larger?"
        return (f"{f['name_a']} vs {f['name_b']}: {rel}. {q}", "उत्तर: {answer}")
    return TemplateSet(
        task_type="pairwise",
        templates=tuple(Template(i, render) for i in range(4)),
    )


def test_generated_dataset_keeps_entity_splits_disjoint():
    train_pool, test_pool = _small_pools()
    ts = _pairwise_template_set()
    ds = generate_dataset(
        train_pool, test_pool, ts, "toy_domain", (1, 100),
        n_train=30, n_validation=10, n_test_seen_wording=10,
        n_test_unseen_entity=10, n_test_unseen_wording=10,
        task_type="pairwise", seed=3,
    )
    train_entities = set()
    for ex in ds["train"] + ds["validation"] + ds["test_seen"]:
        train_entities.update(ex.entities)
    test_entities = set()
    for ex in ds["test_unseen_entity"] + ds["test_unseen_wording"]:
        test_entities.update(ex.entities)
    assert train_entities & test_entities == set()
    assert train_entities <= set(train_pool.names)
    assert test_entities <= set(test_pool.names)


def test_generated_dataset_never_trains_on_the_held_out_template():
    train_pool, test_pool = _small_pools()
    ts = _pairwise_template_set()
    ds = generate_dataset(
        train_pool, test_pool, ts, "toy_domain", (1, 100),
        n_train=30, n_validation=10, n_test_seen_wording=10,
        n_test_unseen_entity=10, n_test_unseen_wording=10,
        task_type="pairwise", seed=3,
    )
    held_out_id = ts.held_out_template.template_id
    for split_name in ("train", "validation", "test_seen", "test_unseen_entity"):
        ids = {ex.template_id for ex in ds[split_name]}
        assert held_out_id not in ids, f"{split_name} leaked the held-out template"
    assert all(ex.template_id == held_out_id for ex in ds["test_unseen_wording"])


def test_generated_dataset_test_seen_excludes_train_prompts():
    """test_seen must not literally repeat a train example (course L9 p.22:
    a model must not be able to pass 'did it fit the distribution' by
    reciting a memorized sentence)."""
    train_pool, test_pool = _small_pools()
    ts = _pairwise_template_set()
    ds = generate_dataset(
        train_pool, test_pool, ts, "toy_domain", (1, 100),
        n_train=30, n_validation=10, n_test_seen_wording=10,
        n_test_unseen_entity=10, n_test_unseen_wording=10,
        task_type="pairwise", seed=3,
    )
    train_prompts = {ex.prompt for ex in ds["train"]}
    test_seen_prompts = {ex.prompt for ex in ds["test_seen"]}
    assert not (train_prompts & test_seen_prompts)


def test_generated_labels_are_arithmetically_correct():
    """Every label must be derivable from the sampled values alone (course
    'correctness' criterion, L9/L10 best-practices slide)."""
    train_pool, test_pool = _small_pools()
    ts = _pairwise_template_set()
    ds = generate_dataset(
        train_pool, test_pool, ts, "toy_domain", (1, 100),
        n_train=50, n_validation=0, n_test_seen_wording=0,
        n_test_unseen_entity=0, n_test_unseen_wording=0,
        task_type="pairwise", seed=5,
    )
    for ex in ds["train"]:
        names, values = list(ex.entities), list(ex.values)
        smaller = names[values.index(min(values))]
        larger = names[values.index(max(values))]
        gold = extract_predicted_answer(ex.answer)
        assert gold in (smaller, larger)


# ---------------------------------------------------------------- loss mask

def test_encode_example_masks_every_prompt_position():
    class FakeTokenizer:
        def encode(self, text, out_type=int):
            # deterministic fake ids: one id per whitespace-split token
            return [hash(tok) % 400 + 10 for tok in text.split()]

    tok = FakeTokenizer()
    enc = encode_example(tok, "a b c question?", "उत्तर: name",
                         bos_id=2, eos_id=3, pad_id=0, max_length=64)
    assert enc is not None
    # every position before prompt_len-1 (i.e. every target that predicts
    # a PROMPT token) must be masked
    for t in range(enc.prompt_len - 1):
        assert enc.labels[t] == 0, f"position {t} should be masked (predicts prompt)"
    # every position from prompt_len-1 onward predicts an answer token or
    # eos, and must carry a real (non-pad) label
    for t in range(enc.prompt_len - 1, len(enc.labels) - 1):
        assert enc.labels[t] != 0, f"position {t} should carry loss (predicts answer)"


def test_encode_example_drops_examples_that_exceed_max_length():
    class FakeTokenizer:
        def encode(self, text, out_type=int):
            return list(range(100))   # always too long

    enc = encode_example(FakeTokenizer(), "prompt", "answer",
                         bos_id=2, eos_id=3, pad_id=0, max_length=10)
    assert enc is None


def test_reasoning_dataset_loss_is_finite_and_backprops():
    """End-to-end: a masked batch through the real model produces a finite
    loss with gradients on every trainable parameter — the integration
    point most likely to silently break if masking or padding disagreed
    with the model's ignore_index convention."""
    class FakeTokenizer:
        def encode(self, text, out_type=int):
            return [abs(hash(tok)) % 400 + 10 for tok in text.split()]

    examples = [
        {"prompt": "a b c question", "answer": "उत्तर: name1"},
        {"prompt": "d e question", "answer": "उत्तर: name2"},
    ]
    ds = ReasoningDataset(examples, FakeTokenizer(), bos_id=2, eos_id=3,
                         pad_id=0, max_length=64)
    assert len(ds) == 2

    cfg = tiny_config(pad_id=0, bos_id=2, eos_id=3)
    model = DecoderLM(cfg)
    rng = np.random.default_rng(0)
    x, y = ds.get_batch(2, "cpu", rng)
    out = model(x, y)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_masked_positions_contribute_zero_and_do_not_shift_the_mean():
    """The mechanical proof that ignore_index excludes masked positions:
    per-position loss at every pad_id-labeled position is exactly 0, and
    the reported (mean) loss equals the plain average of ONLY the
    unmasked per-position losses — not an average over all positions
    (which would silently dilute the answer-only loss with zeros from the
    prompt span, still numerically "working" but not what the course's
    "loss summed only over answer positions" formula (L9 pp.6-11) means).

    (Note: masking here means label == pad_id. Changing a label AWAY from
    pad_id necessarily un-masks that position — there is no separate mask
    channel — so that is not a probe of anything; the correct probe is
    reduction='none' against the model's own logits, done directly here.)
    """
    cfg = tiny_config(pad_id=0, bos_id=2, eos_id=3, vocab_size=50, block_size=16)
    model = DecoderLM(cfg)
    model.eval()

    input_ids = torch.tensor([[2, 10, 11, 12, 20, 21, 3]])
    labels = torch.tensor([[0, 0, 0, 0, 21, 3, 0]])   # only positions 4,5 unmasked

    with torch.no_grad():
        logits = model(input_ids)["logits"]
    per_position = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), labels.reshape(-1),
        ignore_index=cfg.pad_id, reduction="none",
    )
    masked_positions = [0, 1, 2, 3, 6]
    unmasked_positions = [4, 5]
    assert all(per_position[i].item() == 0.0 for i in masked_positions)
    assert all(per_position[i].item() > 0.0 for i in unmasked_positions)

    reported_loss = model(input_ids, labels)["loss"].item()
    expected_mean = per_position[unmasked_positions].mean().item()
    assert reported_loss == pytest.approx(expected_mean, abs=1e-5)


# ----------------------------------------------------------- answer parsing

@pytest.mark.parametrize("text,expected", [
    ("उत्तर: राम", "राम"),
    ("उत्तर: राम।", "राम"),
    ("उत्तर : सीता", "सीता"),
    ("उत्तर: राम है और यह सही जवाब है", "राम"),
    ("no marker here", ""),
    ("", ""),
])
def test_extract_predicted_answer(text, expected):
    assert extract_predicted_answer(text) == expected


# -------------------------------------------------- language-agnostic check

def _code_without_comments_or_docstrings(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                       tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
            continue
        out.append(tok.string)
    code = " ".join(out)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
    return ast.unparse(tree).lower()


def test_shared_reasoning_and_finetune_never_name_a_language():
    """Same guarantee as test_phase2's shared/ check, scoped explicitly to
    the two new Phase 3 packages so this test still communicates intent
    even though test_phase2.py's existing scan already covers them (it
    walks all of shared/)."""
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    for pkg in ("shared/reasoning", "shared/finetune"):
        root = repo_root / pkg
        for p in sorted(root.rglob("*.py")):
            code = _code_without_comments_or_docstrings(p)
            for lang in ("hindi", "nepali"):
                if lang in code:
                    offenders.append((str(p.relative_to(repo_root)), lang))
    assert not offenders, f"language-specific code in shared/: {offenders}"
