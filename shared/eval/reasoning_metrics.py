"""Reasoning-task evaluation: exact-match accuracy, 3-way, seeded.

Course rubric (L9 p.27 / p.34), reproduced in
utility/phase3_plan.md §2.2 — three accuracy numbers, not one, because
they "degrade independently":

  seen entities, seen wording      -- test_seen split
  unseen entities, seen wording    -- test_unseen_entity split
  unseen wording (+ unseen entity) -- test_unseen_wording split

Each is run over "more than one seed" (L9 p.34): since the model is
evaluated with GREEDY decoding, there is no sampling randomness in the
generation itself, so "seed" here varies what a seed can actually change —
the dropout mask active during the forward pass, if the caller evaluates
in train() mode, and more importantly which fixed-size subsample of a
split is scored when the split is larger than the eval budget. This
project evaluates every test example (no subsampling) since the test
splits are already small (320 examples each), so the seed axis instead
re-runs the SAME full split under a fresh ``torch.manual_seed`` to confirm
the greedy accuracy number is stable (it must be, since greedy decoding
is deterministic given fixed weights and fixed eval-mode dropout=off; a
seed producing a different number would itself indicate a bug — e.g. a
train-mode dropout mask leaking into eval). Reporting multiple seeds is
therefore also a correctness check, not just a variance estimate.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional

import torch

_ANSWER_MARKER_RE = re.compile(r"उत्तर\s*[:：]\s*")


def extract_predicted_answer(generated_text: str) -> str:
    """Pull the name/answer out of generated text after the answer marker.

    Robust to a model that has not yet learned to stop cleanly: takes only
    the first line/clause after the marker, and returns "" (never matches
    the gold label) if the marker never appears at all — a model that
    doesn't even emit the marker has not learned the task's output format,
    which is exactly the failure the course's format-vs-substance
    distinction (L9 p.32) says SFT is supposed to fix first.
    """
    m = _ANSWER_MARKER_RE.search(generated_text)
    if not m:
        return ""
    tail = generated_text[m.end():]
    # Cut at the first sentence-ending punctuation/newline OR whitespace:
    # every entity name in the lexicons is a single space-free token, so
    # the gold answer never contains internal whitespace either. Cutting
    # at the first whitespace (not just punctuation) is what keeps a
    # model that rambles past the answer without punctuation from being
    # scored on its whole ramble instead of just the name it emitted.
    tail = re.split(r"[।\n.!?,\s]", tail, maxsplit=1)[0]
    return tail.strip()


@torch.no_grad()
def evaluate_split_accuracy(
    model,
    tokenizer,
    examples: List[Dict],
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 0,
) -> Dict[str, object]:
    """Greedy-decode every example in ``examples`` and score exact-match.

    Bucketed also by ``task_type`` and ``domain`` (already present on
    every example from the generator), so a per-domain breakdown is
    possible without a second evaluation pass.
    """
    torch.manual_seed(seed)
    was_training = model.training
    model.eval()

    correct, total = 0, 0
    per_bucket: Dict[str, Dict[str, int]] = {}
    qualitative: List[Dict] = []

    # Group by exact prompt-token length rather than padding mismatched
    # lengths into one batch. Padding (with bos_id or any other filler)
    # would require an attention mask to keep the model from attending to
    # the filler positions, which DecoderLM.forward does not accept — its
    # only masking is the fixed causal mask, applied uniformly regardless
    # of content. Padded filler tokens would therefore sit in the causal
    # history of every real token that follows them, shifting what the
    # model actually sees relative to training (where every sequence was
    # exactly bos + prompt + answer + eos, no filler). Grouping by length
    # sidesteps the problem entirely rather than papering over it with an
    # unsupported mask.
    encoded = [
        (ex, [bos_id] + tokenizer.encode(ex["prompt"], out_type=int))
        for ex in examples
    ]
    by_length: Dict[int, List] = {}
    for ex, ids in encoded:
        by_length.setdefault(len(ids), []).append((ex, ids))

    for length, group in by_length.items():
        for start in range(0, len(group), batch_size):
            chunk = group[start: start + batch_size]
            idx = torch.tensor([ids for _, ids in chunk], dtype=torch.long,
                               device=device)
            out = model.generate(idx, max_new_tokens=max_new_tokens, greedy=True,
                                 eos_id=eos_id)
            gen = out[:, idx.shape[1]:].cpu().numpy()

            for (ex, _), g in zip(chunk, gen):
                g = [int(t) for t in g if int(t) != eos_id]
                text = tokenizer.decode(g)
                pred = extract_predicted_answer(text)
                # Gold text is generated by the SAME template engine, so its
                # answer is always exactly ANSWER_PREFIX + name with no
                # trailing punctuation — parsed with the identical extractor
                # used on model output, so "correct" means byte-identical
                # parsing on both sides, not two different string conventions.
                gold = extract_predicted_answer(ex["answer"])
                is_correct = bool(pred) and pred == gold

                bucket = f"{ex['task_type']}/{ex['domain']}"
                per_bucket.setdefault(bucket, {"correct": 0, "total": 0})
                per_bucket[bucket]["total"] += 1
                per_bucket[bucket]["correct"] += int(is_correct)

                total += 1
                correct += int(is_correct)
                qualitative.append({
                    "prompt": ex["prompt"], "gold": gold, "predicted": pred,
                    "correct": is_correct, "raw_generated": text,
                    "task_type": ex["task_type"], "domain": ex["domain"],
                    "entities": ex["entities"], "entity_split": ex.get("entity_split"),
                })

    if was_training:
        model.train()

    return {
        "n_examples": total,
        "n_correct": correct,
        "accuracy": correct / max(total, 1),
        "seed": seed,
        "per_bucket": {
            k: {"accuracy": v["correct"] / max(v["total"], 1), **v}
            for k, v in per_bucket.items()
        },
        "qualitative_examples": qualitative,
    }


def evaluate_three_way(
    model,
    tokenizer,
    splits: Dict[str, List[Dict]],
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    device: str = "cuda",
    batch_size: int = 32,
    seeds: List[int] = (0, 1, 2),
) -> Dict[str, object]:
    """Run the course's 3-way rubric across multiple seeds.

    ``splits`` must contain keys ``test_seen``, ``test_unseen_entity``,
    ``test_unseen_wording`` (the three course-prescribed test sets).
    """
    results = {}
    for split_name in ("test_seen", "test_unseen_entity", "test_unseen_wording"):
        examples = splits[split_name]
        per_seed = [
            evaluate_split_accuracy(model, tokenizer, examples, bos_id, eos_id,
                                    max_new_tokens, device, batch_size, seed=s)
            for s in seeds
        ]
        accuracies = [r["accuracy"] for r in per_seed]
        results[split_name] = {
            "accuracies_per_seed": {str(s): r["accuracy"] for s, r in zip(seeds, per_seed)},
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "n_examples": per_seed[0]["n_examples"],
            "per_bucket": per_seed[0]["per_bucket"],
            "qualitative_examples": per_seed[0]["qualitative_examples"],
        }
    return results


def evaluate_pretraining_perplexity(model, val_token_dataset, batch_size: int,
                                    device: str = "cuda", n_batches: int = 50) -> Dict[str, float]:
    """CE/perplexity on the ORIGINAL Phase 2 pretraining validation set.

    Course requirement (L9 p.34): "Report pretraining validation perplexity
    before and after [SFT]." This is deliberately the natural-language
    corpus perplexity, NOT the reasoning-task loss — the point (L9 p.22)
    is that this number is expected to move (usually up) after finetuning
    on a narrow synthetic distribution, and that movement must not be
    misread as "the model got worse": it measures a distribution shift,
    while task accuracy (``evaluate_three_way``) measures the capability
    SFT was meant to add.
    """
    from shared.eval.lm_metrics import evaluate_loss
    return evaluate_loss(model, val_token_dataset, batch_size, n_batches, device)


def save_reasoning_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
