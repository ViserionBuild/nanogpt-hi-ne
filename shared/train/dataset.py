"""Memory-mapped token dataset over the Phase 1 ``.bin`` files.

Phase 1 wrote each split as a flat ``uint32`` array of token ids with
documents concatenated as ``[BOS ... EOS][BOS ... EOS]...``. There is no
document index and no padding, so a training example is just a
``block_size``-token window sampled at a random offset. Windows cross
document boundaries, which is standard for LM pretraining -- BOS/EOS are
ordinary tokens in the stream and the model learns them as boundaries.

``np.memmap`` means the 2.6 GB train file is never loaded into RAM; the OS
pages in only the windows actually touched.

Note on dtype: the files are ``uint32``, which torch has no tensor type
for. Every batch is cast to ``int64`` (the required index type for
``nn.Embedding``). All Phase 1 ids are < 12,000 so the cast is lossless.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class TokenDataset:
    """Random-window sampler over a flat ``uint32`` token file.

    Not a ``torch.utils.data.Dataset``: at ~650M tokens the notion of an
    "epoch over indices" is unhelpful and a DataLoader's worker machinery
    adds overhead for what is a single memmap read. Batches are drawn
    directly.
    """

    def __init__(self, path: str, block_size: int, dtype: str = "uint32") -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"token file not found: {path}")
        self.path = path
        self.block_size = block_size
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(path, dtype=self.dtype, mode="r")
        self.n_tokens = int(self.data.shape[0])

        if self.n_tokens <= block_size + 1:
            raise ValueError(
                f"{path} has {self.n_tokens} tokens, too few for "
                f"block_size {block_size}"
            )

    def __len__(self) -> int:
        """Number of distinct window start positions."""
        return self.n_tokens - self.block_size - 1

    def get_batch(
        self,
        batch_size: int,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (input, target) windows.

        Targets are the inputs shifted one position left, which is the
        causal-LM objective: the logits at position t predict token t+1.

        Returns ``(x, y)`` each ``(batch_size, block_size)`` int64.
        """
        ix = torch.randint(len(self), (batch_size,), generator=generator)
        return self._gather(ix, device)

    def get_deterministic_batches(
        self,
        batch_size: int,
        n_batches: int,
        device: str = "cpu",
        seed: int = 1234,
    ):
        """Yield a FIXED set of batches, identical on every call.

        Validation must measure the same tokens at every eval step,
        otherwise the loss curve mixes real progress with sampling noise
        and is not comparable across steps or across a resume.
        """
        g = torch.Generator().manual_seed(seed)
        for _ in range(n_batches):
            ix = torch.randint(len(self), (batch_size,), generator=g)
            yield self._gather(ix, device)

    def _gather(self, ix: torch.Tensor, device: str):
        """Materialize windows at the given start offsets."""
        b = self.block_size
        # int64 copy out of the memmap; np.stack forces the read now rather
        # than holding a view into a mapped file that torch would alias.
        x = np.stack([self.data[i: i + b].astype(np.int64) for i in ix.tolist()])
        y = np.stack([self.data[i + 1: i + 1 + b].astype(np.int64)
                      for i in ix.tolist()])
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if device.startswith("cuda"):
            # pin_memory + non_blocking overlaps the host->device copy with
            # compute on the previous batch.
            xt = xt.pin_memory().to(device, non_blocking=True)
            yt = yt.pin_memory().to(device, non_blocking=True)
        else:
            xt, yt = xt.to(device), yt.to(device)
        return xt, yt

    def stats(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "n_tokens": self.n_tokens,
            "block_size": self.block_size,
            "dtype": str(self.dtype),
            "n_windows": len(self),
        }


def load_tokenization_metadata(path: str) -> Dict[str, object]:
    """Read the Phase 1 ``tokenization_metadata.json`` beside the bins."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def steps_for_one_epoch(n_tokens: int, tokens_per_step: int) -> int:
    """Optimizer steps for a single pass over ``n_tokens``.

    Hindi and Nepali have different corpus sizes (648,012,574 vs.
    659,499,440 tokens), so each language's config must derive its own
    ``max_steps`` -- equal step counts would NOT be equal data.
    """
    return n_tokens // tokens_per_step
