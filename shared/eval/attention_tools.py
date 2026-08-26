"""Attention analysis: entropy, mean distance, head taxonomy.

Spec 2.3 "Attention analysis": because multi-head attention was
implemented from scratch, evaluate what it learned -- heatmaps for at
least one early and one late layer across multiple heads, attention
entropy per head/layer, mean attention distance, and a discussion of
which heads look local/positional vs. content-based.

Two summary statistics carry the analysis:

**Entropy** ``H = -sum_j p_ij log p_ij``, averaged over queries i. A head
attending to exactly one key has H=0; a head spreading uniformly over t
keys has H=log(t). Low entropy means sharp, selective attention.

**Mean attention distance** ``sum_j p_ij * (i - j)``, averaged over
queries. Small values mean a head looks at the immediately preceding
tokens (local/positional); large values mean it reaches far back
(content-based, plausibly tracking topic or long-range agreement).

Both are computed only over the causally-valid region (j <= i), and both
correct for the fact that early positions simply have fewer keys
available: position 0 can only ever attend to itself, so including it
would drag every head's mean distance toward zero regardless of behavior.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch


@torch.no_grad()
def capture_attention(
    model,
    input_ids: torch.Tensor,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Run a forward pass and return per-layer attention ``(B, h, T, T)``."""
    model.eval()
    out = model(input_ids.to(device), return_attn=True)
    return out["attentions"]


def attention_entropy(
    attn: torch.Tensor,
    min_position: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Mean attention entropy per head, in nats.

    Args:
        attn: ``(B, h, T, T)`` post-softmax weights.
        min_position: skip the first positions, which have a trivially
            small key set and would bias the mean downward.

    Returns:
        ``(h,)`` mean entropy per head.
    """
    a = attn[:, :, min_position:, :].clamp_min(eps)
    # -sum p log p over the key axis, then mean over queries and batch
    ent = -(a * a.log()).sum(dim=-1)
    return ent.mean(dim=(0, 2)).cpu().numpy()


def mean_attention_distance(
    attn: torch.Tensor,
    min_position: int = 1,
) -> np.ndarray:
    """Mean distance each query attends back, in tokens.

    ``sum_j p_ij * (i - j)``, averaged over queries and batch. Only the
    causal region contributes, since future weights are exactly zero.
    """
    B, h, T, _ = attn.shape
    i = torch.arange(T, device=attn.device).view(1, 1, T, 1)
    j = torch.arange(T, device=attn.device).view(1, 1, 1, T)
    dist = (i - j).clamp_min(0).float()          # (1, 1, T, T)
    weighted = (attn * dist).sum(dim=-1)         # (B, h, T)
    return weighted[:, :, min_position:].mean(dim=(0, 2)).cpu().numpy()


def normalized_attention_distance(
    attn: torch.Tensor,
    min_position: int = 1,
) -> np.ndarray:
    """Mean distance as a fraction of the distance available to each query.

    Query i can look back at most i tokens, so raw distance is bounded by
    position. Dividing by i makes heads comparable across positions and
    across sequence lengths: 0 = always attends to self, 1 = always
    attends to the very first token.
    """
    B, h, T, _ = attn.shape
    i = torch.arange(T, device=attn.device).view(1, 1, T, 1)
    j = torch.arange(T, device=attn.device).view(1, 1, 1, T)
    dist = (i - j).clamp_min(0).float()
    weighted = (attn * dist).sum(dim=-1)                       # (B, h, T)
    avail = torch.arange(T, device=attn.device).clamp_min(1).float().view(1, 1, T)
    return (weighted / avail)[:, :, min_position:].mean(dim=(0, 2)).cpu().numpy()


def analyze_all_layers(
    model,
    input_ids: torch.Tensor,
    device: str = "cpu",
    min_position: int = 1,
) -> Dict[str, object]:
    """Entropy and distance for every (layer, head), plus a head taxonomy.

    Returns a JSON-serializable dict, which is the report's Table T6 and
    the data behind figures F16-F20.
    """
    attns = capture_attention(model, input_ids, device)
    n_layer = len(attns)
    n_head = attns[0].shape[1]
    T = attns[0].shape[-1]

    ent = np.zeros((n_layer, n_head))
    dist = np.zeros((n_layer, n_head))
    ndist = np.zeros((n_layer, n_head))

    for li, a in enumerate(attns):
        ent[li] = attention_entropy(a, min_position)
        dist[li] = mean_attention_distance(a, min_position)
        ndist[li] = normalized_attention_distance(a, min_position)

    # Classify each head against the median of its own model, so the
    # taxonomy is relative and does not depend on an arbitrary threshold.
    d_med, e_med = float(np.median(dist)), float(np.median(ent))
    heads = []
    for li in range(n_layer):
        for hi in range(n_head):
            d, e = float(dist[li, hi]), float(ent[li, hi])
            if d <= d_med and e <= e_med:
                kind = "local_sharp"        # positional: near, selective
            elif d <= d_med:
                kind = "local_diffuse"
            elif e <= e_med:
                kind = "distant_sharp"      # content-based: far, selective
            else:
                kind = "distant_diffuse"
            heads.append({
                "layer": li, "head": hi,
                "entropy_nats": round(e, 4),
                "mean_distance_tokens": round(d, 4),
                "normalized_distance": round(float(ndist[li, hi]), 4),
                "max_possible_entropy_nats": round(float(np.log(T)), 4),
                "type": kind,
            })

    return {
        "n_layer": n_layer,
        "n_head": n_head,
        "sequence_length": T,
        "entropy_by_layer_head": ent.round(4).tolist(),
        "distance_by_layer_head": dist.round(4).tolist(),
        "normalized_distance_by_layer_head": ndist.round(4).tolist(),
        "mean_entropy_per_layer": ent.mean(axis=1).round(4).tolist(),
        "mean_distance_per_layer": dist.mean(axis=1).round(4).tolist(),
        "overall_mean_entropy": round(float(ent.mean()), 4),
        "overall_mean_distance": round(float(dist.mean()), 4),
        "median_entropy": round(e_med, 4),
        "median_distance": round(d_med, 4),
        "heads": heads,
        "head_type_counts": {
            k: sum(1 for h in heads if h["type"] == k)
            for k in ("local_sharp", "local_diffuse",
                      "distant_sharp", "distant_diffuse")
        },
        "note": (
            "Entropy in nats; max is log(T) for a uniform head. Distance in "
            f"tokens; positions < {min_position} excluded because they have "
            "too few keys to be informative. Head types are relative to this "
            "model's own median entropy/distance."
        ),
    }


def get_token_pieces(tokenizer, ids) -> List[str]:
    """Token ids -> printable pieces, for heatmap tick labels."""
    pieces = []
    for i in [int(x) for x in ids]:
        try:
            p = tokenizer.id_to_piece(i)
        except Exception:
            p = str(i)
        # SentencePiece marks word starts with U+2581; a leading space reads
        # better than the raw glyph on an axis label
        pieces.append(p.replace("▁", " ").strip() or p)
    return pieces


def save_analysis(analysis: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
