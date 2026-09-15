"""Encode (prompt, answer) pairs into loss-masked training sequences.

Course mechanics (L9 pp. 6-11, L10 pp. 3-5): SFT sums cross-entropy only
over the answer/completion positions —

    L_SFT = -(1/|A|) * sum_{t in A} log p_theta(x_t | x_<t)

— while the model still *attends to* every token (prompt + answer) in the
forward pass; only the loss is selective. This project's ``DecoderLM``
already supports exactly this without any model change:
``forward()`` computes ``F.cross_entropy(..., ignore_index=config.pad_id)``
(shared/model/transformer.py), so masking a position out of the loss is
just a matter of setting that position's *target* label to ``pad_id`` —
the input token itself is untouched, so the model still sees and attends
to it.

Sequence layout per example:

    <bos> <prompt tokens...> <answer tokens...> <eos>

Targets are the inputs shifted left by one (identical convention to
pretraining's ``TokenDataset``), with every target position that falls
inside the prompt span (including the shift of <bos> that predicts the
first prompt token) replaced by ``pad_id`` so it contributes zero loss.
The <eos> at the end IS part of the answer span and DOES carry loss —
without it the model has no trained signal for when to stop generating
(course, L9 p.9: "without an explicit stop token the model does not know
to stop").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class EncodedExample:
    input_ids: List[int]      # bos + prompt + answer + eos
    labels: List[int]          # same length; pad_id on every non-answer position
    prompt_len: int             # for bookkeeping / eval (where the answer starts)


def encode_example(
    tokenizer,
    prompt: str,
    answer: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
) -> Optional[EncodedExample]:
    """Encode one (prompt, answer) pair with an answer-only loss mask.

    Returns None if the encoded sequence would exceed ``max_length`` — the
    caller drops such examples rather than truncating, since truncating
    the *answer* would silently corrupt the label and truncating the
    *prompt* would silently change the question being asked.
    """
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    answer_ids = tokenizer.encode(answer, out_type=int)

    input_ids = [bos_id] + prompt_ids + answer_ids + [eos_id]
    if len(input_ids) > max_length:
        return None

    # Targets are inputs shifted left by one (same convention as
    # shared/train/dataset.py's TokenDataset): targets[t] predicts what
    # comes after input_ids[t]. The label at the LAST position has no
    # "next token" in this single example, so it is masked too (it plays
    # no role: nothing after it in this sequence to predict).
    labels = input_ids[1:] + [pad_id]

    # A position's label is real (unmasked) iff the token it predicts is
    # part of the answer span (the answer tokens themselves, plus the
    # trailing <eos>). prompt_len counts bos+prompt, i.e. everything
    # before the first answer token.
    prompt_len = 1 + len(prompt_ids)
    answer_span_len = len(answer_ids) + 1   # + eos

    for t in range(len(labels)):
        # labels[t] is the token at input position t+1. That position is
        # "in the answer span" iff t+1 falls within
        # [prompt_len, prompt_len + answer_span_len).
        if not (prompt_len <= t + 1 < prompt_len + answer_span_len):
            labels[t] = pad_id

    return EncodedExample(input_ids=input_ids, labels=labels, prompt_len=prompt_len)


class ReasoningDataset:
    """In-memory dataset of loss-masked reasoning examples for one split.

    Small enough (a few thousand short sequences) to hold entirely in
    memory as padded tensors, unlike pretraining's memory-mapped
    multi-hundred-million-token corpus — a fundamentally different scale,
    which is itself the point the course makes about SFT vs. pretraining
    data volume (L10 p.8, LIMA: ~200K SFT tokens vs. ~140B pretraining
    tokens for a comparable model).
    """

    def __init__(
        self,
        examples: Sequence[Dict],
        tokenizer,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_length: int,
    ) -> None:
        self.pad_id = pad_id
        self.max_length = max_length
        self.raw_examples = list(examples)

        encoded = []
        dropped = 0
        for ex in self.raw_examples:
            enc = encode_example(
                tokenizer, ex["prompt"], ex["answer"],
                bos_id, eos_id, pad_id, max_length,
            )
            if enc is None:
                dropped += 1
                continue
            encoded.append(enc)
        self.encoded = encoded
        self.n_dropped = dropped

    def __len__(self) -> int:
        return len(self.encoded)

    def _pad_batch(self, batch: List[EncodedExample]):
        max_len = max(len(e.input_ids) for e in batch)
        input_ids = np.full((len(batch), max_len), self.pad_id, dtype=np.int64)
        labels = np.full((len(batch), max_len), self.pad_id, dtype=np.int64)
        for i, e in enumerate(batch):
            n = len(e.input_ids)
            input_ids[i, :n] = e.input_ids
            labels[i, :n] = e.labels
        return input_ids, labels

    def get_batch(self, batch_size: int, device: str, rng: np.random.Generator):
        """Sample a random batch with replacement, right-padded to the
        longest sequence in the batch (not a fixed block_size — reasoning
        prompts are short, so padding to 512 every time would waste >90%
        of every batch on <pad>)."""
        idx = rng.integers(0, len(self.encoded), size=batch_size)
        batch = [self.encoded[i] for i in idx]
        input_ids, labels = self._pad_batch(batch)
        return (torch.from_numpy(input_ids).to(device),
                torch.from_numpy(labels).to(device))

    def iter_batches(self, batch_size: int, device: str):
        """Iterate the whole split once, in order, for evaluation."""
        for start in range(0, len(self.encoded), batch_size):
            batch = self.encoded[start: start + batch_size]
            input_ids, labels = self._pad_batch(batch)
            yield (torch.from_numpy(input_ids).to(device),
                   torch.from_numpy(labels).to(device))

    def stats(self) -> Dict[str, object]:
        lengths = [len(e.input_ids) for e in self.encoded]
        return {
            "n_examples": len(self.encoded),
            "n_dropped_too_long": self.n_dropped,
            "max_length_cap": self.max_length,
            "mean_sequence_length": float(np.mean(lengths)) if lengths else 0.0,
            "max_sequence_length_seen": int(np.max(lengths)) if lengths else 0,
        }
