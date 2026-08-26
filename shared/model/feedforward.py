"""Position-wise feed-forward network.

Spec 2.1 stage 3: "a position-wise feed-forward network (two linear layers
with a non-linearity such as GELU or SwiGLU, with an inner dimension larger
than the model dimension)".

"Position-wise" means the same MLP is applied independently to every
sequence position -- it mixes information across *channels*, never across
*positions*. Attention is the only sublayer that moves information between
positions; removing the FFN would leave a model that can route information
but not transform it non-linearly, collapsing each block toward a linear
map. The 4x inner expansion is the conventional width and is where most of
the model's parameters live (2,099,712 of the 3,152,384 per block here).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP with GELU: ``d_model -> d_ff -> d_model``.

    GELU rather than ReLU: it is smooth and non-monotonic near zero, which
    empirically trains slightly better in transformers and avoids ReLU's
    dead-unit failure mode.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 bias: bool = True) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` -> ``(B, T, d_model)``."""
        x = self.fc_in(x)      # (B, T, d_ff)
        x = self.act(x)
        x = self.fc_out(x)     # (B, T, d_model)
        return self.dropout(x)


class SwiGLU(nn.Module):
    """SwiGLU variant: ``(Swish(xW) * xV) W_out``.

    Uses three projections instead of two, so ``d_ff`` is conventionally
    scaled by 2/3 to keep the parameter count comparable. Provided as the
    spec's named alternative; the Phase 2 models use :class:`FeedForward`.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 bias: bool = True) -> None:
        super().__init__()
        hidden = int(2 * d_ff / 3)
        self.gate = nn.Linear(d_model, hidden, bias=bias)
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.gate(x)) * self.up(x)))


def build_feedforward(activation: str, d_model: int, d_ff: int,
                      dropout: float, bias: bool) -> nn.Module:
    """Factory: ``'gelu' | 'swiglu'`` -> module."""
    if activation == "gelu":
        return FeedForward(d_model, d_ff, dropout, bias)
    if activation == "swiglu":
        return SwiGLU(d_model, d_ff, dropout, bias)
    raise ValueError(f"unknown activation {activation!r}")
