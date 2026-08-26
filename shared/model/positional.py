"""Positional encoding schemes.

Self-attention is permutation invariant: ``Attention(QP, KP, VP)`` for a
permutation ``P`` returns the permuted output, so a transformer with no
positional signal cannot distinguish "A greater than B" from "B greater
than A". Position must be injected explicitly (spec 2.1, stage 1).

Three schemes are implemented so the choice is demonstrable rather than
asserted, and so the optional no-positional ablation is a config flag
rather than a separate codebase:

``learned``
    A trainable ``nn.Embedding(block_size, d_model)``. **This is the one
    used for both Model H and Model L.** It hard-caps sequence length at
    ``block_size`` — position ``block_size`` has no row in the table, so
    the model cannot process a longer sequence at all. That constraint is
    the point the spec asks to be explained.

``sinusoidal``
    Fixed sin/cos of geometrically-spaced frequencies (Vaswani et al.).
    Deterministic, adds no parameters, and *can* be evaluated beyond
    ``block_size`` — included to make the contrast above concrete.

``none``
    Returns zeros. Used by the bonus ablation; the model then has no way
    to represent order.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedPositional(nn.Module):
    """Trainable absolute position embeddings, one row per position.

    Parameters: ``block_size * d_model``. For the Phase 2 configuration
    that is 512 * 512 = 262,144.
    """

    def __init__(self, block_size: int, d_model: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(block_size, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions ``(T,)`` int64 -> ``(1, T, d_model)``.

        Leading singleton batch dim so the result broadcasts across the
        batch when added to token embeddings.
        """
        if positions.numel() > 0 and int(positions.max()) >= self.block_size:
            raise ValueError(
                f"position {int(positions.max())} >= block_size "
                f"{self.block_size}: learned absolute embeddings have no row "
                f"for it. This is the hard sequence-length cap of this scheme."
            )
        return self.embedding(positions).unsqueeze(0)


class SinusoidalPositional(nn.Module):
    """Fixed sinusoidal encodings — no trainable parameters.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d))`` and
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d))``. Geometrically-spaced
    wavelengths give the model a multi-scale positional signal, and because
    the table is computed rather than learned it extends past
    ``block_size`` on demand.
    """

    def __init__(self, block_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build(block_size, d_model),
                             persistent=False)

    @staticmethod
    def _build(length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        # exp(-log(10000) * 2i/d) computed in log space for numerical stability
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions ``(T,)`` -> ``(1, T, d_model)``, extending if needed."""
        needed = int(positions.max()) + 1 if positions.numel() else 0
        if needed > self.pe.shape[0]:
            self.pe = self._build(needed, self.d_model).to(self.pe.device)
        return self.pe[positions].unsqueeze(0)


class NoPositional(nn.Module):
    """Zero positional signal — the bonus ablation.

    Makes the model permutation invariant: it can only learn a
    bag-of-tokens distribution, because every ordering of the same
    multiset of tokens produces identical attention logits.
    """

    def __init__(self, block_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, positions.shape[0], self.d_model,
                           device=positions.device)


def build_positional(kind: str, block_size: int, d_model: int) -> nn.Module:
    """Factory: ``'learned' | 'sinusoidal' | 'none'`` -> module."""
    table = {
        "learned": LearnedPositional,
        "sinusoidal": SinusoidalPositional,
        "none": NoPositional,
    }
    if kind not in table:
        raise ValueError(f"unknown positional scheme {kind!r}; "
                         f"expected one of {sorted(table)}")
    return table[kind](block_size, d_model)
