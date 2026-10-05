"""Language-agnostic evaluation utilities for Phase 2/3."""
from .attention_tools import (
    analyze_all_layers, attention_entropy, capture_attention,
    get_token_pieces, mean_attention_distance, save_analysis,
)
from .causal_check import verify_causal_masking, verify_mask_structure
from .gen_metrics import (
    chrf, corpus_bleu, distinct_n, evaluate_generations,
    repetition_rate, rouge_l,
)
from .generate import DECODING_SETTINGS, run_all_settings, save_samples
from .lm_metrics import (
    bits_per_byte, count_file_bytes, evaluate_loss, full_evaluation, save_metrics,
)
from .reasoning_metrics import (
    evaluate_pretraining_perplexity, evaluate_split_accuracy, evaluate_three_way,
    extract_predicted_answer, save_reasoning_results,
)

__all__ = [
    "verify_causal_masking", "verify_mask_structure",
    "evaluate_loss", "full_evaluation", "bits_per_byte", "count_file_bytes",
    "save_metrics", "corpus_bleu", "chrf", "rouge_l", "distinct_n",
    "repetition_rate", "evaluate_generations", "run_all_settings",
    "save_samples", "DECODING_SETTINGS", "capture_attention",
    "attention_entropy", "mean_attention_distance", "analyze_all_layers",
    "get_token_pieces", "save_analysis",
    "evaluate_three_way", "evaluate_split_accuracy", "extract_predicted_answer",
    "evaluate_pretraining_perplexity", "save_reasoning_results",
]
