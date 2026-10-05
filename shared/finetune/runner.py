"""Shared entry points for the per-language Phase 3 finetuning scripts.

Mirrors ``shared/train/runner.py``'s pattern exactly: the
``<language>/finetune/`` scripts are thin wrappers that pass their own
config path to these functions, so Model H and Model L run identical code
on different data/checkpoints.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch

from shared.train.runner import load_config, resolve, setup_logging


def _load_tokenizer(path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def run_finetuning(
    cfg_path: str,
    model_cfg_path: str,
    repo_root: Path,
    resume: bool = True,
) -> Dict[str, object]:
    """Finetune one model on its language's synthetic reasoning data."""
    from shared.model import GPTConfig, DecoderLM
    from shared.finetune import ReasoningDataset, FinetuneTrainer
    from shared.reasoning import load_split

    cfg = load_config(cfg_path)
    model_cfg = load_config(model_cfg_path)
    lang = cfg["language"]
    paths = cfg["paths"]
    fcfg = cfg["finetune"]

    logger = setup_logging(
        f"finetune_{lang}",
        resolve(repo_root, os.path.join(paths["logs"], f"finetune_{lang}.log")),
    )
    logger.info(f"config: {cfg_path}")

    mc = GPTConfig.from_dict(model_cfg["model"])
    model = DecoderLM(mc)
    tok = _load_tokenizer(resolve(repo_root, model_cfg["paths"]["tokenizer_model"]))

    data_dir = resolve(repo_root, paths["reasoning_data"])
    train_raw = load_split(data_dir, "train")
    val_raw = load_split(data_dir, "validation")

    train_ds = ReasoningDataset(train_raw, tok, mc.bos_id, mc.eos_id, mc.pad_id,
                                max_length=fcfg["max_length"])
    val_ds = ReasoningDataset(val_raw, tok, mc.bos_id, mc.eos_id, mc.pad_id,
                              max_length=fcfg["max_length"])
    logger.info(f"train: {train_ds.stats()}")
    logger.info(f"validation: {val_ds.stats()}")

    # Small fixed sample of test_unseen_entity for the per-epoch probe
    # (FinetuneTrainer's probe_fn) — the WHOLE 320-example split is what
    # run_reasoning_evaluation scores at the end, so 40 here is plenty to
    # catch "generalization accuracy is stuck at zero" without spending
    # the ~15 minutes/epoch a full-split greedy-decode pass would cost.
    from shared.eval.reasoning_metrics import evaluate_split_accuracy
    probe_examples = load_split(data_dir, "test_unseen_entity")[:40]

    def probe_fn(m) -> float:
        m.to(fcfg["device"] if torch.cuda.is_available() else "cpu")
        r = evaluate_split_accuracy(
            m, tok, probe_examples, mc.bos_id, mc.eos_id,
            max_new_tokens=cfg.get("evaluation", {}).get("max_new_tokens", 12),
            device=fcfg["device"] if torch.cuda.is_available() else "cpu",
            batch_size=len(probe_examples), seed=0,
        )
        return r["accuracy"]

    trainer = FinetuneTrainer(
        model=model, train_dataset=train_ds, validation_dataset=val_ds,
        model_config=mc,
        pretrained_checkpoint=resolve(repo_root, paths["pretrained_checkpoint"]),
        checkpoint_dir=resolve(repo_root, paths["checkpoints"]),
        log_dir=resolve(repo_root, paths["logs"]),
        language=lang,
        learning_rate=fcfg["learning_rate"],
        min_learning_rate=fcfg["min_learning_rate"],
        weight_decay=fcfg["weight_decay"],
        warmup_ratio=fcfg["warmup_ratio"],
        n_epochs=fcfg["n_epochs"],
        batch_size=fcfg["batch_size"],
        grad_clip=fcfg["grad_clip"],
        eval_every_n_steps=fcfg["eval_every_n_steps"],
        seed=fcfg["seed"],
        device=fcfg["device"],
        logger=logger,
        probe_fn=probe_fn,
    )
    return trainer.train(resume=resume)


def run_reasoning_evaluation(
    cfg_path: str,
    model_cfg_path: str,
    repo_root: Path,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, object]:
    """3-way accuracy + pretraining-perplexity-before/after — Phase 3 §3.1 deliverable.

    Evaluates BOTH the pretrained-only checkpoint and the finetuned
    checkpoint on the same three test splits, so the report can show the
    pretrained-vs-finetuned delta the spec asks for ("report pretrained vs.
    finetuned accuracy"), plus the course-required perplexity-before/after
    pair (L9 p.34) computed on the ORIGINAL Phase 2 validation set.
    """
    import importlib

    from shared.model import GPTConfig, DecoderLM
    from shared.eval.error_provenance import (
        error_provenance_by_task_family, error_provenance_table,
    )
    from shared.eval.reasoning_metrics import (
        evaluate_pretraining_perplexity, evaluate_three_way, save_reasoning_results,
    )
    from shared.reasoning import load_split
    from shared.train import TokenDataset, load_checkpoint

    cfg = load_config(cfg_path)
    model_cfg = load_config(model_cfg_path)
    lang = cfg["language"]
    paths = cfg["paths"]
    ecfg = cfg["evaluation"]
    logger = setup_logging(f"eval_reasoning_{lang}")

    mc = GPTConfig.from_dict(model_cfg["model"])
    tok = _load_tokenizer(resolve(repo_root, model_cfg["paths"]["tokenizer_model"]))

    data_dir = resolve(repo_root, paths["reasoning_data"])
    splits = {
        name: load_split(data_dir, name)
        for name in ("test_seen", "test_unseen_entity", "test_unseen_wording")
    }

    pretrain_val_ds = TokenDataset(
        resolve(repo_root, model_cfg["paths"]["validation_bin"]), mc.block_size
    )

    def _eval_checkpoint(ckpt_path: str, tag: str) -> Dict[str, object]:
        model = DecoderLM(mc)
        load_checkpoint(ckpt_path, model=model, device=device)
        model.to(device).eval()
        logger.info(f"[{tag}] evaluating 3-way accuracy from {ckpt_path}")
        acc = evaluate_three_way(
            model, tok, splits, mc.bos_id, mc.eos_id,
            max_new_tokens=ecfg["max_new_tokens"], device=device,
            batch_size=ecfg["eval_batch_size"], seeds=ecfg["accuracy_seeds"],
        )
        ppl = evaluate_pretraining_perplexity(
            model, pretrain_val_ds, batch_size=ecfg["eval_batch_size"],
            device=device, n_batches=ecfg["pretrain_ppl_batches"],
        )
        for split_name, r in acc.items():
            logger.info(f"[{tag}] {split_name}: mean accuracy "
                       f"{r['mean_accuracy']:.3f} (seeds {r['accuracies_per_seed']})")
        logger.info(f"[{tag}] pretraining-val perplexity: {ppl['perplexity']:.2f}")
        return {"three_way_accuracy": acc, "pretraining_perplexity": ppl}

    pretrained_ckpt = resolve(repo_root, paths["pretrained_checkpoint"])
    finetuned_ckpt = checkpoint or resolve(
        repo_root, os.path.join(paths["checkpoints"], "best.pt")
    )

    results = {
        "language": lang,
        "pretrained_checkpoint": pretrained_ckpt,
        "finetuned_checkpoint": finetuned_ckpt,
        "before_finetuning": _eval_checkpoint(pretrained_ckpt, "pretrained-only"),
        "after_finetuning": _eval_checkpoint(finetuned_ckpt, "finetuned"),
    }

    # Error-provenance table (spec 3.3 "error analysis" deliverable): for
    # every wrong prediction on the two unseen-entity splits, classify
    # where the predicted name actually came from (this language's own
    # entity pools, via that language's reasoning_data/lexicon.py, reused
    # rather than duplicated here). Pooled across those two splits since
    # the underlying question — "does the model default to a familiar
    # training-pool name instead of reading this prompt's own entities?" —
    # is the same one for both.
    lexicon = importlib.import_module(f"{lang}.reasoning_data.lexicon")
    pools = (lexicon.PERSON_POOL, lexicon.OBJECT_POOL)
    unseen_examples_after = results["after_finetuning"]["three_way_accuracy"]
    pooled_unseen = (
        unseen_examples_after["test_unseen_entity"]["qualitative_examples"]
        + unseen_examples_after["test_unseen_wording"]["qualitative_examples"]
    )
    results["error_provenance"] = {
        "unseen_entity_and_wording_pooled": error_provenance_table(pooled_unseen, *pools),
        "by_task_family": error_provenance_by_task_family(pooled_unseen, *pools),
    }
    for bucket, table in results["error_provenance"]["by_task_family"].items():
        logger.info(f"[error-provenance] {bucket}: {table['n_wrong']} wrong — "
                   f"{table['percentages']}")

    dest = resolve(repo_root, os.path.join(paths["eval_results"], "reasoning_metrics.json"))
    save_reasoning_results(results, dest)
    logger.info(f"wrote {dest}")
    return results


def run_attention_comparison(
    cfg_path: str,
    model_cfg_path: str,
    repo_root: Path,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    example_text: Optional[str] = None,
) -> Dict[str, object]:
    """Pretrained-vs-finetuned attention comparison — Phase 3 §3.2.

    Reuses the Phase 2 attention toolkit unchanged (analyze_all_layers,
    capture_attention, plot_attention_heatmap) on BOTH checkpoints for the
    SAME example sentence, so any difference in the heatmaps is
    attributable to finetuning rather than to a different probe input.
    course PDFs do not define attention entropy/distance/head taxonomy
    formally (utility/phase3_plan.md §2.6) — this is Phase 2's own toolkit,
    reused here on new (finetuned) weights, not new theory.
    """
    from shared.eval import plots
    from shared.eval.attention_tools import (
        analyze_all_layers, capture_attention, get_token_pieces, save_analysis,
    )
    from shared.model import GPTConfig, DecoderLM
    from shared.reasoning import load_split
    from shared.train import load_checkpoint

    cfg = load_config(cfg_path)
    model_cfg = load_config(model_cfg_path)
    lang = cfg["language"]
    paths = cfg["paths"]
    ecfg_model = model_cfg.get("evaluation", {})
    logger = setup_logging(f"eval_attn_ft_{lang}")

    mc = GPTConfig.from_dict(model_cfg["model"])
    tok = _load_tokenizer(resolve(repo_root, model_cfg["paths"]["tokenizer_model"]))

    # Prefer a real comparative-reasoning sentence (post-finetune attention
    # on exactly the task type finetuning targeted is what the spec asks
    # to examine — "especially on comparative-reasoning prompts").
    if example_text is None:
        test_examples = load_split(resolve(repo_root, paths["reasoning_data"]), "test_seen")
        example_text = test_examples[0]["prompt"]

    seq_len = ecfg_model.get("attention_seq_len", 24)
    ids = tok.encode(example_text, out_type=int)[:seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long)
    pieces = get_token_pieces(tok, ids)

    pretrained_ckpt = resolve(repo_root, paths["pretrained_checkpoint"])
    finetuned_ckpt = checkpoint or resolve(
        repo_root, os.path.join(paths["checkpoints"], "best.pt")
    )

    def _analyze(ckpt_path: str, tag: str):
        model = DecoderLM(mc)
        load_checkpoint(ckpt_path, model=model, device=device)
        model.to(device).eval()
        analysis = analyze_all_layers(model, input_ids, device=device)
        analysis["example_text"] = tok.decode(ids)
        analysis["checkpoint"] = ckpt_path
        analysis["tag"] = tag
        attns = capture_attention(model, input_ids, device=device)
        logger.info(f"[{tag}] mean entropy {analysis['overall_mean_entropy']:.4f} | "
                   f"mean distance {analysis['overall_mean_distance']:.4f}")
        return analysis, attns

    analysis_before, attns_before = _analyze(pretrained_ckpt, "pretrained")
    analysis_after, attns_after = _analyze(finetuned_ckpt, "finetuned")

    base = resolve(repo_root, paths["eval_results"])
    save_analysis(analysis_before, os.path.join(base, "attention_pretrained.json"))
    save_analysis(analysis_after, os.path.join(base, "attention_finetuned.json"))

    # Devanagari glyph coverage: matplotlib's default DejaVu Sans has none,
    # so heatmap tick labels (the token pieces) render as empty boxes
    # unless a Devanagari-capable font is registered first — same fix
    # Phase 2's analyze_attention applies, needed again here since this is
    # a separate figure-generating entry point.
    font = plots.configure_devanagari_font()
    if font and not plots.verify_devanagari_rendering():
        logger.warning("selected font cannot render Devanagari — labels will "
                       "appear as boxes")

    figdir = resolve(repo_root, paths.get("figures", "report/figures"))
    code = model_cfg.get("figure_code", lang[:3])
    label = model_cfg.get("model_label", lang.title())
    layers = ecfg_model.get("attention_layers", [0, mc.n_layer - 1])

    for layer in layers:
        if layer >= len(attns_before):
            continue
        plots.plot_attention_heatmap(
            attns_before[layer][0].cpu().numpy(), pieces, layer, f"{label} (pretrained)",
            f"attention_heatmap_{code}_pretrained_L{layer}.png", figdir,
        )
        plots.plot_attention_heatmap(
            attns_after[layer][0].cpu().numpy(), pieces, layer, f"{label} (finetuned)",
            f"attention_heatmap_{code}_finetuned_L{layer}.png", figdir,
        )

    plots.plot_entropy_distance_before_after(
        analysis_before, analysis_after,
        f"attention_entropy_distance_before_after_{code}.png", figdir, label,
    )
    plots.plot_head_taxonomy_shift(
        analysis_before, analysis_after,
        f"attention_head_shift_{code}.png", figdir, label,
    )

    comparison = {
        "language": lang,
        "example_text": tok.decode(ids),
        "pretrained_checkpoint": pretrained_ckpt,
        "finetuned_checkpoint": finetuned_ckpt,
        "before": {"overall_mean_entropy": analysis_before["overall_mean_entropy"],
                  "overall_mean_distance": analysis_before["overall_mean_distance"],
                  "head_type_counts": analysis_before["head_type_counts"]},
        "after": {"overall_mean_entropy": analysis_after["overall_mean_entropy"],
                 "overall_mean_distance": analysis_after["overall_mean_distance"],
                 "head_type_counts": analysis_after["head_type_counts"]},
        "entropy_delta": (analysis_after["overall_mean_entropy"]
                         - analysis_before["overall_mean_entropy"]),
        "distance_delta": (analysis_after["overall_mean_distance"]
                          - analysis_before["overall_mean_distance"]),
    }
    dest = os.path.join(base, "attention_comparison.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info(f"wrote {dest}")
    return comparison
