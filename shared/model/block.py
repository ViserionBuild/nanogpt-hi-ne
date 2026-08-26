"""Transformer decoder block.

Spec 2.1 stage 3 requires each block to contain a multi-head causal
self-attention sublayer, a position-wise feed-forward network, residual
connections around *both* sublayers, layer normalization, and dropout.

**Pre-norm** is used (``x + Sublayer(LN(x))``) and is the default here.
Why pre-norm is more stable for deep stacks: the residual path from input
to output is an unbroken identity, so a gradient reaching layer N flows
back to layer 0 without passing through any LayerNorm Jacobian. In
post-norm (``LN(x + Sublayer(x))``) every layer's gradient is rescaled by
a LayerNorm derivative on the way back, and those factors compound
multiplicatively with depth -- which is why post-norm architectures need a
careful warmup schedule to avoid diverging in the first few hundred steps.
Pre-norm trains stably from step 0, which matters here because a Kaggle
session can die and be resumed at an arbitrary point.

The trade-off: pre-norm leaves the residual stream un-normalized as it
grows across layers, so a final LayerNorm before the output head is
required (applied in ``transformer.py``, not here).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadCausalSelfAttention
from .config import GPTConfig
from .feedforward import build_feedforward


class TransformerBlock(nn.Module):
    """One decoder block: causal self-attention + feed-forward, both residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_position = config.norm_position

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadCausalSelfAttention(
            d_model=config.d_model,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            bias=config.bias,
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = build_feedforward(
            config.activation, config.d_model, config.d_ff,
            config.dropout, config.bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``(B, T, d)`` -> ``(B, T, d)``, plus attention weights if requested.

        Dropout is applied inside each sublayer (on attention probabilities
        and on each sublayer's output projection), so the residual additions
        here are plain sums.
        """
        if self.norm_position == "pre":
            # x + Sublayer(LN(x)) -- identity residual path
            attn_out, attn_w = self.attn(self.ln_1(x), return_attn=return_attn)
            x = x + attn_out
            x = x + self.ffn(self.ln_2(x))
        else:
            # LN(x + Sublayer(x)) -- normalization on the residual path
            attn_out, attn_w = self.attn(x, return_attn=return_attn)
            x = self.ln_1(x + attn_out)
            x = self.ln_2(x + self.ffn(x))

        return x, attn_w
