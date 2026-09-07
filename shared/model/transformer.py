"""Decoder-only (GPT-style) Transformer language model.

Assembles the four stages required by spec 2.1:

1. Input representation  -- token embeddings + positional embeddings,
                            summed, with embedding dropout.
2. Multi-head causal self-attention (``attention.py``).
3. N transformer blocks (``block.py``), then a final LayerNorm.
4. Output head + causal-LM objective -- a linear projection to vocabulary
   size, weight-tied to the input embedding, trained with cross-entropy
   between the logits at position t and the token at position t+1.

Language-agnostic: nothing here knows whether it is modeling Hindi or
Nepali. The two models differ only in their weights, their tokenizer, and
their config's ``language``/``tokenizer_model`` provenance fields.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import GPTConfig
from .positional import build_positional


class DecoderLM(nn.Module):
    """Decoder-only Transformer for causal language modeling."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # --- 1. input representation ---------------------------------------
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional = build_positional(
            config.positional, config.block_size, config.d_model
        )
        self.embd_dropout = nn.Dropout(config.embd_dropout)

        # --- 2/3. transformer stack -----------------------------------------
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        # Required by pre-norm: the residual stream is never normalized on its
        # way through the stack, so it must be normalized once before the head.
        self.ln_f = nn.LayerNorm(config.d_model)

        # --- 4. output head --------------------------------------------------
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            # Weight tying: the output head reuses the token-embedding matrix.
            # Saves vocab_size * d_model = 12,000 * 512 = 6,144,000 parameters
            # (19.5% of the untied 31,465,472 total; 24.3% of the tied
            # 25,321,472 total). Justified because both matrices map between
            # the same vocabulary and the same representation space -- the
            # embedding answers "what vector represents token i", the head
            # answers "how much does this vector look like token i".
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Applied after the generic init so it overrides it for these specific
        # projections.
        self._scale_residual_projections()

    # ------------------------------------------------------------------ init

    def _init_weights(self, module: nn.Module) -> None:
        """Normal(0, init_std) for Linear/Embedding weights; zeros for bias."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        """Scale the projections that write into the residual stream.

        Every block adds two sublayer outputs to the residual stream, so
        after N blocks the stream is a sum of 2N contributions. With
        identical init the variance grows linearly in depth and deep stacks
        start out badly conditioned. Scaling the *output* projections by
        1/sqrt(2 * n_layer) keeps the post-stack variance roughly constant
        with depth (the GPT-2 initialization scheme).
        """
        std = self.config.init_std / math.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            ffn = block.ffn
            out = getattr(ffn, "fc_out", None) or getattr(ffn, "down", None)
            nn.init.normal_(out.weight, mean=0.0, std=std)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Dict[str, object]:
        """Run the causal LM forward pass.

        Args:
            idx: ``(B, T)`` int64 token ids.
            targets: optional ``(B, T)`` int64 next-token targets. When
                given, the returned dict includes the cross-entropy loss.
            return_attn: collect per-layer attention weights
                ``(B, n_head, T, T)`` for the attention analysis.

        Returns:
            dict with ``logits`` ``(B, T, vocab_size)``, ``loss`` (or None),
            and ``attentions`` (list of per-layer weights, or None).
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size "
                f"{self.config.block_size}"
            )

        # --- 1. input representation -----------------------------------------
        # (B, T) -> (B, T, d)
        tok = self.token_embedding(idx)
        pos_ids = torch.arange(T, device=idx.device, dtype=torch.long)
        # (1, T, d), broadcast over the batch
        pos = self.positional(pos_ids)
        x = self.embd_dropout(tok + pos)

        # --- 2/3. transformer blocks ------------------------------------------
        attentions: List[torch.Tensor] = [] if return_attn else None
        for block in self.blocks:
            x, attn_w = block(x, return_attn=return_attn)
            if return_attn:
                attentions.append(attn_w)

        x = self.ln_f(x)

        # --- 4. output head ---------------------------------------------------
        logits = self.lm_head(x)                      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Causal LM objective: the logits at position t predict the token
            # at position t+1. The caller supplies targets already shifted
            # (see TokenDataset), so this is a straight element-wise
            # cross-entropy, averaged over every position in the batch.
            # ignore_index skips padded positions; pretraining has none, but
            # reasoning finetuning in Phase 3 does.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_id,
            )

        return {"logits": logits, "loss": loss, "attentions": attentions}

    # ------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` by ``max_new_tokens`` tokens.

        Args:
            idx: ``(B, T)`` prompt token ids.
            temperature: softmax temperature. Below 1 sharpens the
                distribution (more repetitive, higher overlap metrics);
                above 1 flattens it (more diverse, less coherent).
            greedy: take the argmax instead of sampling. Equivalent to the
                temperature -> 0 limit, and reported separately by the spec.
            eos_id: stop early once every sequence has emitted this token.

        Returns:
            ``(B, T + max_new_tokens)`` -- prompt followed by continuation.
        """
        was_training = self.training
        self.eval()

        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens: learned absolute positional
            # embeddings have no row beyond that, so the context window is a
            # hard cap rather than a soft preference.
            idx_cond = idx[:, -self.config.block_size:]
            logits = self.forward(idx_cond)["logits"][:, -1, :]  # (B, vocab)

            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            if eos_id is not None:
                # Once a sequence has finished, keep emitting eos so the
                # tensor stays rectangular; trimming happens at decode time.
                next_id = torch.where(finished.unsqueeze(1),
                                      torch.full_like(next_id, eos_id), next_id)
                finished |= next_id.squeeze(1) == eos_id

            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool(finished.all()):
                break

        if was_training:
            self.train()
        return idx

    # ------------------------------------------------------------ bookkeeping

    def count_parameters(self, non_embedding: bool = False) -> int:
        """Total trainable parameters, counting a tied weight exactly once."""
        seen, total = set(), 0
        for p in self.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.token_embedding.weight.numel()
            if self.config.positional == "learned":
                total -= self.positional.embedding.weight.numel()
        return total

    def parameter_breakdown(self) -> Dict[str, object]:
        """Per-component parameter counts for the report's Table T2."""
        cfg = self.config
        d, ff, L, V = cfg.d_model, cfg.d_ff, cfg.n_layer, cfg.vocab_size
        b = 1 if cfg.bias else 0

        tok = V * d
        pos = (cfg.block_size * d) if cfg.positional == "learned" else 0
        attn = (3 * d * d + 3 * d * b) + (d * d + d * b)
        mlp = (d * ff + ff * b) + (ff * d + d * b)
        lns = 2 * (2 * d)
        per_block = attn + mlp + lns

        return {
            "token_embedding": tok,
            "positional_embedding": pos,
            "per_block_attention": attn,
            "per_block_feedforward": mlp,
            "per_block_layernorms": lns,
            "per_block_total": per_block,
            "n_layer": L,
            "blocks_total": per_block * L,
            "final_layernorm": 2 * d,
            "lm_head": 0 if cfg.tie_weights else V * d,
            "total": self.count_parameters(),
            "total_non_embedding": self.count_parameters(non_embedding=True),
            "weight_tying_saved": V * d if cfg.tie_weights else 0,
        }

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str = "cuda",
    ) -> torch.optim.AdamW:
        """Build AdamW with decay applied only to 2-D weights.

        Weight decay on LayerNorm gains, biases, and embeddings hurts:
        those parameters set scale and offset rather than defining a
        function to be regularized, and shrinking them toward zero
        systematically distorts the representation. Only matrices that
        participate in matmuls are decayed.
        """
        decay, no_decay, seen = [], [], set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue        # tied weight: only decay it once
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # Fused AdamW is a large speedup on CUDA and is numerically identical.
        use_fused = device_type == "cuda" and "fused" in \
            torch.optim.AdamW.__init__.__code__.co_varnames
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
