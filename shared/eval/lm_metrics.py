"""Intrinsic language-modeling metrics: cross-entropy, perplexity, BPB.

Spec 2.3 "Intrinsic language-modeling metrics":
  * validation cross-entropy loss and perplexity (PPL = e^CE),
  * bits-per-byte on the same held-out text "to make the two languages
    more comparable when tokenizers differ".

**Why BPB matters here.** Perplexity is per-TOKEN, so it is denominated in
each model's own tokenizer. Phase 1 measured fertility 1.3042 for Hindi
and 1.4394 for Nepali: Nepali spends ~10% more tokens on the same amount
of text, so a raw PPL comparison between Model H and Model L is not a
like-for-like comparison of how well each models its language. BPB
renormalizes onto UTF-8 bytes, which is a tokenizer-independent unit:

    BPB = (CE_nats * n_tokens) / (ln 2 * n_bytes)

Devanagari codepoints are 3 bytes in UTF-8, so n_bytes is roughly 3x the
character count for both languages -- the measure is fair between them.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import torch


@torch.no_grad()
def evaluate_loss(
    model,
    dataset,
    batch_size: int,
    n_batches: Optional[int] = None,
    device: str = "cuda",
    seed: int = 1234,
    amp: bool = False,
) -> Dict[str, float]:
    """Mean token-level cross-entropy over a fixed batch set.

    ``n_batches=None`` evaluates the whole split in order (used for the
    final reported numbers); an integer draws that many fixed random
    batches (used for the periodic validation during training).
    """
    model.eval()
    total_loss, total_tokens, n = 0.0, 0, 0

    if n_batches is None:
        batches = _sequential_batches(dataset, batch_size, device)
    else:
        batches = dataset.get_deterministic_batches(
            batch_size, n_batches, device, seed=seed
        )

    for x, y in batches:
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            loss = model(x, y)["loss"]
        # weight by token count so partial final batches do not skew the mean
        ntok = y.numel()
        total_loss += loss.item() * ntok
        total_tokens += ntok
        n += 1

    mean_ce = total_loss / max(total_tokens, 1)
    return {
        "cross_entropy_nats": mean_ce,
        "perplexity": float(math.exp(min(mean_ce, 20.0))),
        "n_batches": n,
        "n_tokens_evaluated": total_tokens,
    }


def _sequential_batches(dataset, batch_size: int, device: str):
    """Non-overlapping windows covering the split in order.

    Used for final reported numbers so every held-out token is counted
    exactly once, rather than sampled with replacement.
    """
    import numpy as np
    import torch as _t

    b = dataset.block_size
    stride = b * batch_size
    n = dataset.n_tokens - 1
    for start in range(0, n - stride, stride):
        offs = [start + i * b for i in range(batch_size)]
        x = np.stack([dataset.data[o: o + b].astype(np.int64) for o in offs])
        y = np.stack([dataset.data[o + 1: o + 1 + b].astype(np.int64) for o in offs])
        yield _t.from_numpy(x).to(device), _t.from_numpy(y).to(device)


def bits_per_byte(cross_entropy_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Convert token-level CE (nats) to bits per UTF-8 byte.

    ``n_tokens`` and ``n_bytes`` must describe the SAME text. n_tokens is
    the token count the model was scored on; n_bytes is that text's UTF-8
    length on disk.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return (cross_entropy_nats * n_tokens) / (math.log(2) * n_bytes)


def bits_per_token(cross_entropy_nats: float) -> float:
    """CE in nats -> bits per token (a pure unit change)."""
    return cross_entropy_nats / math.log(2)


def count_file_bytes(path: str) -> int:
    """UTF-8 byte length of a text file -- the denominator for BPB."""
    return os.path.getsize(path)


@torch.no_grad()
def full_evaluation(
    model,
    dataset,
    text_path: str,
    batch_size: int,
    device: str = "cuda",
    split_name: str = "validation",
    n_batches: Optional[int] = None,
    amp: bool = False,
) -> Dict[str, object]:
    """CE / PPL / BPB for one split, ready to drop into the report's Table T1.

    Args:
        dataset: ``TokenDataset`` over that split's ``.bin``.
        text_path: the matching ``.txt``, for the UTF-8 byte count. May be
            absent (the corpora live on Drive) -- BPB is then reported as
            None rather than guessed.
    """
    res = evaluate_loss(model, dataset, batch_size, n_batches, device, amp=amp)
    ce = res["cross_entropy_nats"]

    out = {
        "split": split_name,
        "cross_entropy_nats": round(ce, 6),
        "bits_per_token": round(bits_per_token(ce), 6),
        "perplexity": round(res["perplexity"], 4),
        "n_tokens_evaluated": res["n_tokens_evaluated"],
        "n_tokens_in_split": dataset.n_tokens,
        "n_batches": res["n_batches"],
    }

    if text_path and os.path.exists(text_path):
        n_bytes = count_file_bytes(text_path)
        # Scale the evaluated-token count to the whole split: BPB must
        # compare the SAME text on both sides of the ratio, and the .txt
        # holds the entire split while evaluation may cover only part.
        frac = res["n_tokens_evaluated"] / max(dataset.n_tokens, 1)
        out["n_bytes"] = n_bytes
        out["bits_per_byte"] = round(
            bits_per_byte(ce, res["n_tokens_evaluated"], int(n_bytes * frac)), 6
        )
        out["bytes_per_token"] = round(n_bytes / max(dataset.n_tokens, 1), 4)
    else:
        out["n_bytes"] = None
        out["bits_per_byte"] = None
        out["note"] = (
            f"BPB not computed: {text_path!r} not found. The .txt splits are "
            f"gitignored and live on Drive; mount them or pass --bytes."
        )
    return out


def save_metrics(metrics: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
