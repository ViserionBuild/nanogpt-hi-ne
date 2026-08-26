"""Shared entry points for the per-language Phase 2 scripts.

The ``<language>/train/`` and ``<language>/eval/`` scripts are thin
wrappers that pass their own config path to these functions. Keeping the
logic here means Model H and Model L run identical code with different
configs -- so any difference between them is data and language, never an
implementation divergence.

Language-agnostic by construction: every path arrives from the config.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml


def setup_logging(name: str = "phase2", log_file: Optional[str] = None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(repo_root: Path, rel: str) -> str:
    """Config paths are repo-root-relative; make them absolute."""
    p = Path(rel)
    return str(p if p.is_absolute() else repo_root / p)


def build_model(cfg: Dict, logger=None):
    """Instantiate the model described by a language's config."""
    from shared.model import DecoderLM, GPTConfig
    mc = GPTConfig.from_dict(cfg["model"])
    model = DecoderLM(mc)
    if logger:
        b = model.parameter_breakdown()
        logger.info(f"model: {b['total']:,} parameters "
                    f"({b['total_non_embedding']:,} non-embedding)")
    return model, mc


def print_parameter_report(cfg_path: str, repo_root: Path,
                           expect: Optional[int] = None) -> Dict:
    """Parameter breakdown for the report's Table T2.

    ``expect`` asserts an exact integer so a config edit cannot silently
    drift the model off the ~25M target.
    """
    cfg = load_config(cfg_path)
    model, mc = build_model(cfg)
    b = model.parameter_breakdown()

    print(f"\n{'=' * 62}")
    print(f"Parameter breakdown — {cfg.get('language', '?')}")
    print(f"{'=' * 62}")
    rows = [
        ("Token embedding", f"{mc.vocab_size} x {mc.d_model}",
         b["token_embedding"]),
        ("Positional embedding", f"{mc.block_size} x {mc.d_model}",
         b["positional_embedding"]),
        ("Per block: attention", "qkv + out proj", b["per_block_attention"]),
        ("Per block: feed-forward",
         f"{mc.d_model}->{mc.d_ff}->{mc.d_model}", b["per_block_feedforward"]),
        ("Per block: layernorms", "2 x LayerNorm", b["per_block_layernorms"]),
        ("Per block total", "", b["per_block_total"]),
        (f"All blocks (x{mc.n_layer})", "", b["blocks_total"]),
        ("Final LayerNorm", f"{mc.d_model}", b["final_layernorm"]),
        ("Output head", "tied" if mc.tie_weights else "untied", b["lm_head"]),
    ]
    for name, shape, n in rows:
        print(f"  {name:<28} {shape:<22} {n:>12,}")
    print(f"  {'-' * 60}")
    print(f"  {'TOTAL':<28} {'':<22} {b['total']:>12,}")
    print(f"  {'non-embedding':<28} {'':<22} "
          f"{b['total_non_embedding']:>12,}")
    if mc.tie_weights:
        print(f"\n  Weight tying saves {b['weight_tying_saved']:,} parameters "
              f"({100 * b['weight_tying_saved'] / (b['total'] + b['weight_tying_saved']):.1f}% "
              f"of the untied total).")

    if expect is not None:
        assert b["total"] == expect, \
            f"expected {expect:,} parameters, got {b['total']:,}"
        print(f"\n  Target {expect:,} — CONFIRMED")
    return b


def run_pretraining(cfg_path: str, repo_root: Path,
                    resume: bool = True, override: Optional[Dict] = None):
    """Pretrain one model from its config."""
    from shared.model import TrainConfig
    from shared.train import TokenDataset, Trainer

    cfg = load_config(cfg_path)
    lang = cfg["language"]
    paths = cfg["paths"]

    logger = setup_logging(
        f"pretrain_{lang}",
        resolve(repo_root, os.path.join(paths["logs"], f"pretrain_{lang}.log")),
    )
    logger.info(f"config: {cfg_path}")

    tc = TrainConfig.from_dict({**cfg["training"], **(override or {})})
    model, mc = build_model(cfg, logger)

    torch.manual_seed(tc.seed)

    train_ds = TokenDataset(resolve(repo_root, paths["train_bin"]), mc.block_size)
    val_ds = TokenDataset(resolve(repo_root, paths["validation_bin"]), mc.block_size)
    logger.info(f"train {train_ds.n_tokens:,} tokens | "
                f"validation {val_ds.n_tokens:,} tokens")
    logger.info(f"{tc.tokens_per_step(mc.block_size):,} tokens/step x "
                f"{tc.max_steps:,} steps = "
                f"{tc.tokens_per_step(mc.block_size) * tc.max_steps:,} tokens "
                f"({tc.tokens_per_step(mc.block_size) * tc.max_steps / train_ds.n_tokens:.2f} epochs)")

    trainer = Trainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        model_config=mc, train_config=tc,
        checkpoint_dir=resolve(repo_root, paths["checkpoints"]),
        log_dir=resolve(repo_root, paths["logs"]),
        language=lang, logger=logger,
    )
    return trainer.train(resume=resume)


def run_causal_verification(cfg_path: str, repo_root: Path,
                            checkpoint: Optional[str] = None,
                            seq_len: int = 128) -> Dict:
    """Causal-masking verification -- the spec's mandatory check."""
    from shared.eval import verify_causal_masking, verify_mask_structure
    from shared.train import load_checkpoint

    cfg = load_config(cfg_path)
    lang = cfg["language"]
    logger = setup_logging(f"causal_{lang}")
    model, mc = build_model(cfg, logger)

    if checkpoint and os.path.exists(checkpoint):
        load_checkpoint(checkpoint, model, device="cpu")
        logger.info(f"loaded weights from {checkpoint}")
    else:
        logger.info("using randomly-initialized weights "
                    "(masking is a structural property, not a learned one)")

    sweep = verify_causal_masking(model, seq_len=seq_len, device="cpu")
    struct = verify_mask_structure(model, seq_len=min(32, seq_len))

    logger.info(f"perturbation sweep : {sweep['positions_tested']} positions, "
                f"max delta {sweep['max_abs_delta_past']}, "
                f"passed={sweep['passed']}")
    logger.info(f"lower-triangular   : max future weight "
                f"{struct['max_future_attention_weight']}, "
                f"passed={struct['passed']}")

    out = {"language": lang, "checkpoint": checkpoint,
           "perturbation_sweep": sweep, "mask_structure": struct,
           "passed": bool(sweep["passed"] and struct["passed"])}

    dest = resolve(repo_root, os.path.join(
        cfg["paths"]["eval_results"], "causal_mask_verification.json"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"wrote {dest}")
    return out
