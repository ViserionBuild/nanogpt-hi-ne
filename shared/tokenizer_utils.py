"""Thin wrapper around SentencePiece for training/loading per-language
tokenizers from scratch. No pretrained tokenizers are used anywhere.

Special-token IDs are canonical and identical to every other stage of the
pipeline (config ``special_token_ids``): pad=0, unk=1, bos=2, eos=3 with
pieces ``<pad>``/``<unk>``/``<bos>``/``<eos>``.
"""
import os
import sys
from itertools import zip_longest
from typing import Any, Dict

import sentencepiece as spm

from shared.line_provenance import has_sidecar, iter_sidecar_rows, sidecar_path


def train_tokenizer(input_file, model_prefix, vocab_size=32000, model_type="bpe",
                    character_coverage=0.9995, byte_fallback=True):
    """Trains a SentencePiece tokenizer from scratch on `input_file`.
    model_type: 'bpe' or 'unigram'. character_coverage should be high
    (0.9995-1.0) for languages with large character sets (Indic scripts)."""
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
    )
    print(f"[tokenizer] trained -> {model_prefix}.model / .vocab")


def load_tokenizer(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def count_tokens_streaming(sp, input_path, logger=None, progress_interval=1_000_000):
    """Stream *input_path* through the already-loaded tokenizer *sp* and
    count BPE CONTENT tokens (``EncodeAsIds`` never adds BOS/EOS). O(1)
    memory — used to measure actual per-source/combined token counts with
    the single final tokenizer (no separate temporary tokenizer)."""
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_lines = 0

    if not os.path.exists(input_path):
        return {"total_lines": 0, "total_tokens": 0, "total_words": 0,
                "total_chars": 0, "fertility": 0.0}

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.rstrip("\r\n")
            if line:
                ids = sp.EncodeAsIds(line)
                total_tokens += len(ids)
                total_words += len(line.split())
                total_chars += len(line)
            total_lines += 1

            if logger and (idx + 1) % progress_interval == 0:
                logger.info(
                    f"  {idx + 1:>12,} lines | {total_tokens:>14,} content tokens"
                )

    fertility = total_tokens / max(total_words, 1)
    return {
        "total_lines": total_lines,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "fertility": round(fertility, 4),
    }


def count_tokens_streaming_with_provenance(
    sp: "spm.SentencePieceProcessor",
    input_path: str,
    logger=None,
    progress_interval: int = 1_000_000,
) -> Dict[str, Any]:
    """Like count_tokens_streaming, but if *input_path* has a
    line-provenance sidecar (manual only — see shared/line_provenance.py),
    ALSO tallies token/word/char counts by source_type ('wikipedia'/'news')
    and by source_name (e.g. 'dainik_jagran') in the same single pass — one
    tokenizer pass over the file regardless of whether a breakdown is
    available. A sidecar row-count mismatch is a data-integrity bug, never
    silently ignored.

    Returns {'total': {...same shape as count_tokens_streaming...},
             'by_source_type': {...}, 'by_source_name': {...},
             'has_provenance': bool}.
    """
    empty_total = {"total_lines": 0, "total_tokens": 0, "total_words": 0,
                    "total_chars": 0, "fertility": 0.0}
    if not os.path.exists(input_path):
        return {"total": empty_total, "by_source_type": {}, "by_source_name": {}, "has_provenance": False}

    if not has_sidecar(input_path):
        return {
            "total": count_tokens_streaming(sp, input_path, logger, progress_interval),
            "by_source_type": {}, "by_source_name": {}, "has_provenance": False,
        }

    def _bump(tally: Dict[str, Dict[str, Any]], key: str, tokens: int, words: int, chars: int) -> None:
        entry = tally.setdefault(key, {"lines": 0, "tokens": 0, "words": 0, "chars": 0})
        entry["lines"] += 1
        entry["tokens"] += tokens
        entry["words"] += words
        entry["chars"] += chars

    total_tokens = total_words = total_chars = total_lines = 0
    by_type: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    _MISSING = object()

    with open(input_path, "r", encoding="utf-8") as fin:
        text_gen = (raw.rstrip("\r\n") for raw in fin)
        prov_gen = iter_sidecar_rows(sidecar_path(input_path))
        for idx, (line, prov) in enumerate(zip_longest(text_gen, prov_gen, fillvalue=_MISSING)):
            if line is _MISSING or prov is _MISSING:
                if logger:
                    logger.error(
                        f"Line-provenance sidecar row count does not match "
                        f"{input_path}'s line count — refusing to silently "
                        f"misattribute token counts."
                    )
                sys.exit(1)
            source_type, source_name, _document_id = prov
            if line:
                ids = sp.EncodeAsIds(line)
                n_tokens = len(ids)
                n_words = len(line.split())
                n_chars = len(line)
                total_tokens += n_tokens
                total_words += n_words
                total_chars += n_chars
                _bump(by_type, source_type, n_tokens, n_words, n_chars)
                _bump(by_name, source_name, n_tokens, n_words, n_chars)
            total_lines += 1

            if logger and (idx + 1) % progress_interval == 0:
                logger.info(f"  {idx + 1:>12,} lines | {total_tokens:>14,} content tokens")

    fertility = total_tokens / max(total_words, 1)
    total = {
        "total_lines": total_lines, "total_tokens": total_tokens,
        "total_words": total_words, "total_chars": total_chars,
        "fertility": round(fertility, 4),
    }
    return {"total": total, "by_source_type": by_type, "by_source_name": by_name, "has_provenance": True}


def tokenizer_stats(sp, sample_text_path, out_json_path=None):
    """Computes vocab size, avg chars/token, and UNK rate on a held-out
    sample — streaming, so it works on files of any size."""
    import json
    total_tokens = 0
    total_chars = 0
    unk_count = 0
    pieces = []
    unk_id = sp.unk_id()

    with open(sample_text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            total_tokens += len(ids)
            total_chars += len(line)
            unk_count += sum(1 for i in ids if i == unk_id)
            if len(pieces) < 30 and pieces is not None:
                pieces.extend(sp.encode(line, out_type=str))
                pieces = pieces[:30]

    stats = {
        "vocab_size": sp.get_piece_size(),
        "num_tokens_in_sample": total_tokens,
        "num_chars_in_sample": total_chars,
        "avg_chars_per_token": total_chars / max(total_tokens, 1),
        "unk_rate": unk_count / max(total_tokens, 1),
        "example_tokenization": pieces,
    }
    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats
