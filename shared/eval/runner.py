"""Shared evaluation entry points for the per-language eval scripts.

Produces every Phase 2 result the spec's deliverables 5-7 require:
PPL/BPB tables, BLEU/chrF/ROUGE-L, generated samples with diversity and
repetition stats, and attention entropy/distance summaries plus heatmaps.

Language-agnostic: paths, tokenizer, and byte counts all arrive from the
calling language's config.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from shared.train.runner import build_model, load_config, resolve, setup_logging


def _load_tokenizer(path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def _load_weights(model, checkpoint: Optional[str], logger, device: str):
    from shared.train import load_checkpoint
    if checkpoint and os.path.exists(checkpoint):
        st = load_checkpoint(checkpoint, model, device=device)
        logger.info(f"loaded {checkpoint} (step {st['step']:,})")
        return st
    logger.warning(
        f"no checkpoint at {checkpoint!r} — evaluating an UNTRAINED model. "
        f"Numbers will be at the uniform-entropy baseline."
    )
    return {"step": 0}


def evaluate_language_model(
    cfg_path: str,
    repo_root: Path,
    checkpoint: Optional[str] = None,
    splits: List[str] = ("validation",),
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict:
    """CE / PPL / BPB per split -- the report's Table T1."""
    from shared.eval.lm_metrics import full_evaluation, save_metrics
    from shared.train import TokenDataset

    cfg = load_config(cfg_path)
    lang = cfg["language"]
    paths = cfg["paths"]
    logger = setup_logging(f"eval_lm_{lang}")

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.info("CUDA unavailable — using CPU")
        device = "cpu"

    model, mc = build_model(cfg, logger)
    state = _load_weights(model, checkpoint, logger, device)
    model.to(device).eval()

    results = {
        "language": lang,
        "checkpoint": checkpoint,
        "checkpoint_step": state.get("step", 0),
        "parameters": model.count_parameters(),
        "vocab_size": mc.vocab_size,
        "block_size": mc.block_size,
        "splits": {},
    }

    for split in splits:
        ds = TokenDataset(resolve(repo_root, paths[f"{split}_bin"]), mc.block_size)
        txt = resolve(repo_root, paths.get(f"{split}_txt", ""))

        m = full_evaluation(
            model, ds, txt, batch_size=cfg["training"]["micro_batch_size"],
            device=device, split_name=split, n_batches=max_batches,
            amp=device.startswith("cuda"),
        )

        # The .txt files are gitignored (they live on Drive), so fall back to
        # the byte counts recorded in the config -- BPB must not silently
        # vanish just because the corpus is not mounted.
        if m.get("bits_per_byte") is None and cfg.get("byte_counts", {}).get(split):
            from shared.eval.lm_metrics import bits_per_byte
            total_bytes = cfg["byte_counts"][split]
            frac = m["n_tokens_evaluated"] / max(m["n_tokens_in_split"], 1)
            m["n_bytes"] = total_bytes
            m["bits_per_byte"] = round(bits_per_byte(
                m["cross_entropy_nats"], m["n_tokens_evaluated"],
                int(total_bytes * frac)), 6)
            m["bytes_source"] = "config byte_counts (corpus .txt not mounted)"

        results["splits"][split] = m
        logger.info(
            f"{split}: CE {m['cross_entropy_nats']:.4f} nats | "
            f"PPL {m['perplexity']:.2f} | BPB {m.get('bits_per_byte')}"
        )

    dest = resolve(repo_root, os.path.join(paths["eval_results"],
                                           "lm_metrics.json"))
    save_metrics(results, dest)
    logger.info(f"wrote {dest}")
    return results


def evaluate_generation(
    cfg_path: str,
    repo_root: Path,
    checkpoint: Optional[str] = None,
    split: str = "validation",
    device: str = "cuda",
    n_samples: Optional[int] = None,
) -> Dict:
    """Generation metrics for all four decoding settings -- Tables T4/T5."""
    from shared.eval.gen_metrics import evaluate_generations, try_sacrebleu
    from shared.eval.generate import run_all_settings, save_samples
    from shared.train import TokenDataset

    cfg = load_config(cfg_path)
    lang = cfg["language"]
    paths, ecfg = cfg["paths"], cfg.get("evaluation", {})
    logger = setup_logging(f"eval_gen_{lang}")

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, mc = build_model(cfg, logger)
    _load_weights(model, checkpoint, logger, device)
    model.to(device).eval()

    tok = _load_tokenizer(resolve(repo_root, paths["tokenizer_model"]))
    ds = TokenDataset(resolve(repo_root, paths[f"{split}_bin"]), mc.block_size)

    n = n_samples or ecfg.get("n_generation_samples", 500)
    logger.info(f"generating {n} continuations x 4 decoding settings")

    gens = run_all_settings(
        model, tok, ds,
        n_samples=n,
        prefix_tokens=ecfg.get("prefix_tokens", 32),
        max_new_tokens=ecfg.get("max_new_tokens", 128),
        device=device, seed=42,
        batch_size=ecfg.get("generation_batch_size", 16),
        eos_id=mc.eos_id,
    )

    # Overlap metrics are computed on SentencePiece pieces, not whitespace:
    # Devanagari agglutination would otherwise understate real overlap.
    def tokenize_fn(text: str) -> List[str]:
        return tok.encode(text, out_type=str)

    metrics = {}
    for name, r in gens.items():
        m = evaluate_generations(r["hypotheses"], r["references"], tokenize_fn)
        extra = try_sacrebleu(r["hypotheses"], r["references"])
        if extra:
            m.update(extra)
        metrics[name] = m
        logger.info(
            f"{name:>9}: BLEU-4 {m['bleu_4']:.3f} | chrF {m['chrf']:.2f} | "
            f"ROUGE-L {m['rouge_l_f']:.2f} | D-1 {m['distinct_1']:.4f} | "
            f"D-2 {m['distinct_2']:.4f} | rep {m['repetition_rate_3gram']:.4f}"
        )

    out = {"language": lang, "checkpoint": checkpoint, "split": split,
           "n_samples": n, "settings": metrics}

    base = resolve(repo_root, paths["eval_results"])
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "generation_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    save_samples(gens, os.path.join(base, "generation_samples.json"),
                 n_examples=ecfg.get("n_qualitative_examples", 10))
    logger.info(f"wrote {base}/generation_metrics.json and "
                f"generation_samples.json")
    return out


def analyze_attention(
    cfg_path: str,
    repo_root: Path,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    example_text: Optional[str] = None,
    split: str = "validation",
) -> Dict:
    """Attention entropy/distance + heatmaps -- Table T6, figures F12-F19."""
    from shared.eval.attention_tools import (
        analyze_all_layers, capture_attention, get_token_pieces, save_analysis,
    )
    from shared.eval import plots
    from shared.train import TokenDataset

    cfg = load_config(cfg_path)
    lang = cfg["language"]
    paths, ecfg = cfg["paths"], cfg.get("evaluation", {})
    logger = setup_logging(f"eval_attn_{lang}")

    model, mc = build_model(cfg, logger)
    _load_weights(model, checkpoint, logger, device)
    model.to(device).eval()

    tok = _load_tokenizer(resolve(repo_root, paths["tokenizer_model"]))
    seq_len = ecfg.get("attention_seq_len", 24)

    # Prefer a real example sentence in the model's own language, as the
    # spec requires; fall back to a held-out sequence from the corpus.
    if example_text:
        ids = tok.encode(example_text, out_type=int)[:seq_len]
    else:
        ds = TokenDataset(resolve(repo_root, paths[f"{split}_bin"]), mc.block_size)
        raw = ds.data[1000: 1000 + seq_len * 3].astype(np.int64)
        ids = [int(t) for t in raw if t > 3][:seq_len]

    input_ids = torch.tensor([ids], dtype=torch.long)
    pieces = get_token_pieces(tok, ids)
    logger.info(f"example ({len(ids)} tokens): {tok.decode(ids)[:80]}")

    analysis = analyze_all_layers(model, input_ids, device=device)
    analysis["language"] = lang
    analysis["checkpoint"] = checkpoint
    analysis["example_text"] = tok.decode(ids)
    analysis["example_pieces"] = pieces

    base = resolve(repo_root, paths["eval_results"])
    save_analysis(analysis, os.path.join(base, "attention_analysis.json"))

    # ---- figures ----
    font = plots.configure_devanagari_font()
    if font and not plots.verify_devanagari_rendering():
        logger.warning("selected font cannot render Devanagari — labels will "
                       "appear as boxes")

    figdir = resolve(repo_root, paths.get("figures", "report/figures"))
    # Figure suffix and display label come from the config, not from a
    # language branch here -- shared/ must stay language-agnostic.
    code = cfg.get("figure_code", lang[:3])
    label = cfg.get("model_label", lang.title())
    attns = capture_attention(model, input_ids, device)

    for layer in ecfg.get("attention_layers", [0, mc.n_layer - 1]):
        if layer >= len(attns):
            continue
        plots.plot_attention_heatmap(
            attns[layer][0].cpu().numpy(), pieces, layer, label,
            f"attention_heatmap_{code}_L{layer}.png", figdir,
        )

    ent = np.array(analysis["entropy_by_layer_head"])
    dist = np.array(analysis["distance_by_layer_head"])
    plots.plot_by_layer(ent, f"{label} — attention entropy by layer",
                        "Attention entropy (nats)",
                        f"attention_entropy_by_layer_{code}.png", figdir)
    plots.plot_by_layer(dist, f"{label} — mean attention distance by layer",
                        "Mean attention distance (tokens)",
                        f"attention_distance_by_layer_{code}.png", figdir)

    logger.info(f"head types: {analysis['head_type_counts']}")
    return analysis
