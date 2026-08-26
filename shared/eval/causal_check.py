"""Empirical verification that the causal mask actually blocks the future.

Spec 2.1 stage 2 requires this explicitly: "verify empirically that the
model cannot see the future -- for example, show that changing token t+1
does not change the logits at position t."

A correct implementation gives BIT-IDENTICAL logits at positions <= t when
any token after t is changed. This is a stronger check than "approximately
equal": if the mask is applied after softmax, or is off by one, or uses a
large finite negative instead of -inf, the difference shows up as a small
but non-zero delta rather than exact equality.

The model is put in eval() mode so dropout is disabled -- otherwise the
comparison would fail for reasons unrelated to masking.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


@torch.no_grad()
def verify_causal_masking(
    model,
    seq_len: int = 64,
    vocab_size: Optional[int] = None,
    positions: Optional[List[int]] = None,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, object]:
    """Sweep positions, perturb the next token, assert the past is unchanged.

    For each position ``t``, token ``t+1`` is replaced with a different id
    and the whole sequence is re-run. Every logit at positions ``0..t``
    must be exactly unchanged.

    Args:
        model: a ``DecoderLM`` (or anything with the same forward signature).
        seq_len: sequence length to test; capped at the model's block_size.
        vocab_size: defaults to the model's configured vocabulary size.
        positions: which ``t`` values to test. Default: every position.
        device: where to run.
        seed: RNG seed for the base sequence and perturbations.

    Returns:
        A JSON-serializable report with per-position deltas and a verdict.
    """
    model = model.to(device)
    was_training = model.training
    model.eval()   # dropout OFF -- otherwise the comparison is meaningless

    cfg = model.config
    vocab_size = vocab_size or cfg.vocab_size
    seq_len = min(seq_len, cfg.block_size)
    if positions is None:
        # every position that has a successor to perturb
        positions = list(range(seq_len - 1))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randint(0, vocab_size, (1, seq_len), generator=gen).to(device)
    base_logits = model(base)["logits"]          # (1, T, V)

    results = []
    max_delta = 0.0
    for t in positions:
        perturbed = base.clone()
        # choose a genuinely different token id for position t+1
        old = int(perturbed[0, t + 1])
        new = (old + 1 + int(torch.randint(0, vocab_size - 1, (1,),
                                           generator=gen))) % vocab_size
        if new == old:
            new = (old + 1) % vocab_size
        perturbed[0, t + 1] = new

        logits = model(perturbed)["logits"]

        # positions 0..t must be untouched by a change at t+1
        delta = (logits[:, : t + 1, :] - base_logits[:, : t + 1, :]).abs().max()
        delta = float(delta)
        max_delta = max(max_delta, delta)

        # sanity check in the other direction: the change MUST affect
        # position t+1 itself, otherwise the model is ignoring its input
        # and the test above would pass vacuously.
        fwd_delta = float(
            (logits[:, t + 1, :] - base_logits[:, t + 1, :]).abs().max()
        )

        results.append({
            "position_t": t,
            "perturbed_position": t + 1,
            "max_abs_delta_at_positions_le_t": delta,
            "max_abs_delta_at_position_t_plus_1": fwd_delta,
            "past_unchanged": delta == 0.0,
            "future_changed": fwd_delta > 0.0,
        })

    if was_training:
        model.train()

    all_past_ok = all(r["past_unchanged"] for r in results)
    all_future_ok = all(r["future_changed"] for r in results)

    return {
        "test": "causal_masking_verification",
        "description": (
            "For each position t, token t+1 was replaced with a different "
            "id and the sequence re-run. Logits at positions <= t must be "
            "bit-identical (delta exactly 0.0), and the logits at t+1 must "
            "change (proving the perturbation was actually applied)."
        ),
        "sequence_length": seq_len,
        "vocab_size": vocab_size,
        "positions_tested": len(results),
        "max_abs_delta_past": max_delta,
        "all_past_positions_unchanged": all_past_ok,
        "all_perturbed_positions_changed": all_future_ok,
        "passed": bool(all_past_ok and all_future_ok),
        "per_position": results,
    }


@torch.no_grad()
def verify_mask_structure(model, seq_len: int = 16, device: str = "cpu") -> Dict:
    """Second, independent check: the attention matrix is lower-triangular.

    Complements the perturbation sweep -- that one tests behavior, this one
    inspects the weights directly. Every attention row must place exactly
    zero probability on strictly-future keys.
    """
    model = model.to(device).eval()
    idx = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    attns = model(idx, return_attn=True)["attentions"]

    upper_mass, layers = 0.0, []
    for li, a in enumerate(attns):                # a: (B, h, T, T)
        upper = torch.triu(torch.ones(seq_len, seq_len, device=device),
                           diagonal=1).bool()
        mass = float(a[0, :, upper].abs().max())
        upper_mass = max(upper_mass, mass)
        rows = float((a[0].sum(dim=-1) - 1.0).abs().max())
        layers.append({"layer": li,
                       "max_future_attention_weight": mass,
                       "max_row_sum_deviation_from_1": rows})

    return {
        "test": "attention_matrix_is_lower_triangular",
        "sequence_length": seq_len,
        "max_future_attention_weight": upper_mass,
        "passed": upper_mass == 0.0,
        "per_layer": layers,
    }
