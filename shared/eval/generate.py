"""Generation harness: held-out prefixes -> continuations at each setting.

Spec 2.3: "For each model, generate continuations from held-out prefixes
using greedy decoding and temperatures 0.5, 1.0, and 1.5. Against
reference continuations from the held-out set ... report BLEU, chrF,
ROUGE-L."

Protocol used here (documented so it is reproducible):
  * N documents sampled from the held-out split, at a fixed seed,
  * the first ``prefix_tokens`` tokens are the prompt,
  * the next ``max_new_tokens`` are generated,
  * the TRUE next ``max_new_tokens`` tokens are the reference.

Prompts are drawn at BOS boundaries where possible, so a prompt is the
start of a real document rather than a mid-sentence fragment.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

DECODING_SETTINGS = [
    {"name": "greedy", "greedy": True, "temperature": None},
    {"name": "temp_0.5", "greedy": False, "temperature": 0.5},
    {"name": "temp_1.0", "greedy": False, "temperature": 1.0},
    {"name": "temp_1.5", "greedy": False, "temperature": 1.5},
]


def sample_prefixes(
    dataset,
    n_samples: int,
    prefix_tokens: int,
    continuation_tokens: int,
    bos_id: int = 2,
    seed: int = 42,
    prefer_bos: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Draw (prefix, reference-continuation) pairs from a held-out split.

    When ``prefer_bos`` is set, offsets are snapped to the nearest BOS
    within a short window so prompts begin at a document boundary. Falls
    back to the raw offset when no BOS is nearby.
    """
    rng = np.random.default_rng(seed)
    need = prefix_tokens + continuation_tokens
    hi = dataset.n_tokens - need - 1
    out = []

    for _ in range(n_samples):
        off = int(rng.integers(0, hi))
        if prefer_bos:
            window = dataset.data[off: off + 256]
            hits = np.nonzero(window == bos_id)[0]
            if len(hits):
                cand = off + int(hits[0])
                if cand < hi:
                    off = cand
        chunk = dataset.data[off: off + need].astype(np.int64)
        out.append({
            "offset": off,
            "prefix": chunk[:prefix_tokens],
            "reference": chunk[prefix_tokens:],
        })
    return out


@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    samples: List[Dict],
    setting: Dict,
    max_new_tokens: int = 128,
    device: str = "cuda",
    batch_size: int = 16,
    eos_id: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Generate one continuation per sample under a single decoding setting.

    Returns decoded prompt / hypothesis / reference text, plus the raw
    hypothesis token ids (used for diversity metrics without a
    decode-reencode round trip).
    """
    model.eval()
    prompts, hyps, refs, hyp_ids = [], [], [], []

    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]
        idx = torch.tensor(np.stack([s["prefix"] for s in batch]),
                           dtype=torch.long, device=device)

        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=setting.get("temperature") or 1.0,
            greedy=setting.get("greedy", False),
            eos_id=eos_id,
        )
        gen = out[:, idx.shape[1]:].cpu().numpy()

        for s, g in zip(batch, gen):
            g = [int(t) for t in g]
            hyp_ids.append(g)
            prompts.append(_decode(tokenizer, s["prefix"]))
            hyps.append(_decode(tokenizer, g))
            refs.append(_decode(tokenizer, s["reference"]))

    return {"prompts": prompts, "hypotheses": hyps,
            "references": refs, "hypothesis_ids": hyp_ids}


def _decode(tokenizer, ids) -> str:
    """Decode ids to text, dropping specials so metrics see only content."""
    ids = [int(i) for i in ids]
    # 0-3 are pad/unk/bos/eos in every Phase 1 tokenizer
    ids = [i for i in ids if i > 3]
    try:
        return tokenizer.decode(ids)
    except Exception:
        return tokenizer.DecodeIds(ids)


def run_all_settings(
    model,
    tokenizer,
    dataset,
    n_samples: int = 500,
    prefix_tokens: int = 32,
    max_new_tokens: int = 128,
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 16,
    eos_id: Optional[int] = None,
) -> Dict[str, Dict]:
    """Run every required decoding setting over the SAME prompts.

    Using identical prompts across settings is what makes the greedy vs.
    temperature comparison meaningful -- any difference is then the
    decoding strategy, not the prompt sample.
    """
    samples = sample_prefixes(dataset, n_samples, prefix_tokens,
                              max_new_tokens, seed=seed)
    results = {}
    for setting in DECODING_SETTINGS:
        results[setting["name"]] = generate_continuations(
            model, tokenizer, samples, setting, max_new_tokens,
            device, batch_size, eos_id,
        )
    return results


def save_samples(results: Dict, path: str, n_examples: int = 10) -> None:
    """Write qualitative examples for the report's Table T5."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {}
    for name, r in results.items():
        payload[name] = [
            {"prompt": r["prompts"][i],
             "generated": r["hypotheses"][i],
             "reference": r["references"][i]}
            for i in range(min(n_examples, len(r["prompts"])))
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
