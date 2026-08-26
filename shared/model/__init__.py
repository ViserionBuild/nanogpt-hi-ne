"""Language-agnostic decoder-only Transformer implementation.

Nothing in this package refers to a specific language: per-language
behavior comes entirely from the GPTConfig passed in, and the two models
share no data, tokenizer, vocabulary, or weights.
"""
from .attention import MultiHeadCausalSelfAttention
from .block import TransformerBlock
from .config import GPTConfig, TrainConfig
from .feedforward import FeedForward, SwiGLU, build_feedforward
from .positional import (
    LearnedPositional, NoPositional, SinusoidalPositional, build_positional,
)
from .transformer import DecoderLM

__all__ = [
    "MultiHeadCausalSelfAttention", "TransformerBlock",
    "GPTConfig", "TrainConfig",
    "FeedForward", "SwiGLU", "build_feedforward",
    "LearnedPositional", "SinusoidalPositional", "NoPositional",
    "build_positional", "DecoderLM",
]
