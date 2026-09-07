"""Model configuration.

``GPTConfig`` is the single source of truth for architecture shape. It is
serialized into every checkpoint (spec 2.2: "Each checkpoint must contain
model weights, optimizer state, scheduler state, current training step,
and a configuration file"), so a checkpoint is always self-describing —
loading one never depends on the YAML that produced it still existing or
still saying the same thing.

Language-agnostic by construction: no field names or defaults refer to
Hindi or Nepali. Per-language values arrive from
``<language>/configs/<language>_model.yaml``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class GPTConfig:
    """Architecture hyperparameters for the decoder-only Transformer.

    The defaults are the Phase 2 target configuration: 25,321,472
    trainable parameters at ``vocab_size=12000`` with a tied output head.
    See ``count_parameters()`` in ``transformer.py`` for the breakdown.
    """

    # --- vocabulary (from Phase 1; identical size for both languages, but
    # the two vocabularies themselves are entirely separate) ---
    vocab_size: int = 12000

    # --- shape ---
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048          # 4 x d_model
    block_size: int = 512     # max sequence length -- see positional note below

    # --- regularization ---
    dropout: float = 0.1
    attn_dropout: float = 0.1
    embd_dropout: float = 0.1

    # --- structural choices (each is a spec "state which you chose and why") ---
    positional: str = "learned"   # learned | sinusoidal | none
    norm_position: str = "pre"    # pre | post
    activation: str = "gelu"      # gelu | swiglu
    tie_weights: bool = True
    bias: bool = True

    # --- special token ids (canonical across the whole project, Phase 1) ---
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    # --- init ---
    init_std: float = 0.02

    # --- provenance: which language/tokenizer this instance was built for.
    # Recorded so a checkpoint identifies itself; never used to branch logic. ---
    language: str = ""
    tokenizer_model: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail loudly on a configuration that cannot produce a valid model."""
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head "
                f"({self.n_head}) — each head takes a d_model/n_head slice."
            )
        if self.positional not in ("learned", "sinusoidal", "none"):
            raise ValueError(f"unknown positional scheme: {self.positional!r}")
        if self.norm_position not in ("pre", "post"):
            raise ValueError(f"unknown norm_position: {self.norm_position!r}")
        if self.activation not in ("gelu", "swiglu"):
            raise ValueError(f"unknown activation: {self.activation!r}")
        for name in ("dropout", "attn_dropout", "embd_dropout"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {v}")
        if self.vocab_size <= max(self.pad_id, self.unk_id,
                                 self.bos_id, self.eos_id):
            raise ValueError("vocab_size must exceed every special token id")

    @property
    def head_dim(self) -> int:
        """Per-head subspace width, d_k = d_model / n_head."""
        return self.d_model // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        """Rebuild from a dict, ignoring keys this version does not know.

        Tolerating unknown keys means an older checkpoint still loads after
        a new field is added to this dataclass.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "GPTConfig":
        """Load the ``model:`` block of a language's YAML config."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw.get("model", raw))


@dataclass
class TrainConfig:
    """Optimization / schedule hyperparameters.

    Serialized alongside ``GPTConfig`` in each checkpoint so a run is
    fully reproducible from the checkpoint alone.
    """

    # --- optimizer (spec 2.2 requires AdamW) ---
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- schedule (spec 2.2 requires learning-rate scheduling) ---
    warmup_ratio: float = 0.02
    max_steps: int = 13184        # one epoch; set per language from token count
    decay_style: str = "cosine"

    # --- batching. effective tokens/step = micro_batch * block * accum ---
    micro_batch_size: int = 24
    grad_accum_steps: int = 4

    # --- evaluation / checkpointing ---
    eval_interval: int = 500
    eval_batches: int = 100
    checkpoint_interval: int = 1000
    log_interval: int = 10

    # --- runtime ---
    seed: int = 42
    amp: bool = True
    device: str = "cuda"
    # Use every visible GPU via DataParallel when more than one is present
    # (Kaggle's "GPU T4 x2"). Set false to pin training to a single device.
    data_parallel: bool = True
    compile_model: bool = False
    num_workers: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw.get("training", raw))

    def tokens_per_step(self, block_size: int) -> int:
        """Effective tokens consumed per optimizer step."""
        return self.micro_batch_size * block_size * self.grad_accum_steps
