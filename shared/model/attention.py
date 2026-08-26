"""Multi-head causal self-attention, implemented from first principles.

Spec 2.1 stage 2 forbids ``nn.Transformer*``, ``nn.MultiheadAttention``,
HuggingFace classes, and "any pre-built attention block". Only
``nn.Linear`` and ``nn.Dropout`` are used here; the softmax, the scaling,
the reshapes, and the causal mask are all explicit. ``F.scaled_dot_product_
attention`` is deliberately NOT used — it is a pre-built fused attention
kernel, and hiding the mask and softmax inside it would defeat the point.

Tensor-shape walkthrough for one forward pass (B=batch, T=sequence,
d=d_model, h=n_head, d_k=d/h):

    x                     (B, T, d)
    qkv = x @ W_qkv       (B, T, 3d)          one fused projection
    split -> q, k, v      (B, T, d)  each
    view  -> (B, T, h, d_k)                   split the channel dim per head
    transpose(1, 2)       (B, h, T, d_k)      heads become a batch dimension
    q @ k^T               (B, h, T, T)        every query vs. every key
    / sqrt(d_k)                               variance control, see below
    + causal mask                             -inf on future positions
    softmax(dim=-1)       (B, h, T, T)        rows sum to 1
    @ v                   (B, h, T, d_k)      weighted value average
    transpose(1, 2)       (B, T, h, d_k)      heads back beside each other
    reshape               (B, T, d)           concatenate heads
    @ W_o                 (B, T, d)           output projection
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultiHeadCausalSelfAttention(nn.Module):
    """Causal (masked) multi-head self-attention.

    Each of ``n_head`` heads attends in its own ``d_model / n_head``
    dimensional subspace, so different heads can specialise on different
    relations (positional/local vs. content-based) without competing for
    the same channels. The causal mask makes position ``t`` attend only to
    positions ``<= t``, which is what makes next-token prediction a valid
    training signal at every position simultaneously.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})"
            )

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size
        # Precomputed because it multiplies every score matrix; see forward().
        self.scale = self.head_dim ** -0.5

        # Q, K, V in one matmul. Mathematically identical to three separate
        # nn.Linear(d, d) layers -- the weight is just their concatenation --
        # but one large GEMM is materially faster on GPU than three small ones.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular causal mask, (1, 1, block, block) so it broadcasts
        # over batch and head. Registered as a non-persistent buffer: it moves
        # with .to(device) but is not written into checkpoints (it is a
        # constant, fully determined by block_size).
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size),
                             persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply causal self-attention.

        Args:
            x: ``(B, T, d_model)`` input activations.
            return_attn: if True, also return the post-softmax attention
                weights ``(B, n_head, T, T)``. Used by the Phase 2/3
                attention analysis (entropy, mean distance, heatmaps).
                Off by default so training never materializes them.

        Returns:
            ``(output (B, T, d_model), attn_weights or None)``
        """
        B, T, d = x.shape
        if d != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {d}")
        if T > self.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.block_size}: "
                f"the causal mask buffer has no entry for it."
            )

        # --- project to queries, keys, values -------------------------------
        # (B, T, d) -> (B, T, 3d) -> three tensors of (B, T, d)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # --- split into heads ------------------------------------------------
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        # The transpose puts the head axis next to the batch axis so the
        # subsequent matmuls treat (B, h) as a combined batch and each head
        # attends independently in its own d_k-dimensional subspace.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- scaled dot-product scores ---------------------------------------
        # (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        # scores[b, i, s, t] = <query s, key t> for head i of sequence b.
        #
        # Why 1/sqrt(d_k): for q, k with independent unit-variance components,
        # their dot product sums d_k such products and so has variance d_k
        # (std 8 at d_k=64). Feeding logits of that magnitude to softmax
        # saturates it -- the largest entry goes to ~1, the rest to ~0, and the
        # gradient through softmax vanishes, so attention stops learning.
        # Dividing by sqrt(d_k) restores unit variance and keeps softmax in the
        # responsive part of its range.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # --- causal masking ---------------------------------------------------
        # Set every future position to -inf BEFORE softmax, so exp(-inf) = 0
        # and future positions receive exactly zero probability. Masking after
        # softmax would leave a renormalization bug and leak future information.
        # Using -inf (not a large finite number) guarantees exact zeros.
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :T, :T], float("-inf")
        )

        # --- attention weights ------------------------------------------------
        # softmax over the KEY axis (-1): each query's weights sum to 1.
        attn = torch.softmax(scores, dim=-1)
        attn_out_weights = attn if return_attn else None
        attn = self.attn_dropout(attn)

        # --- weighted sum of values -------------------------------------------
        # (B, h, T, T) @ (B, h, T, d_k) -> (B, h, T, d_k)
        y = torch.matmul(attn, v)

        # --- concatenate heads and project ------------------------------------
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d)
        # .contiguous() is required: transpose only changes strides, and view()
        # needs a contiguous buffer to reinterpret the memory layout.
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # W_o lets the model mix information across heads; without it each
        # head's output would occupy a fixed, non-interacting channel slice.
        y = self.resid_dropout(self.out_proj(y))

        return y, attn_out_weights
