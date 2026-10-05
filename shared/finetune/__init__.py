"""Supervised finetuning on the synthetic reasoning data — Phase 3 §3.1.

Language-agnostic by construction, same rule as every other package under
``shared/`` (enforced by
``tests/test_phase3.py::test_shared_finetune_never_names_a_language``).
Reuses Phase 2's model, checkpoint, and scheduler code unchanged; the only
new piece is the dataset (prompt/answer -> loss-masked token sequence) and
a thin finetuning loop that starts from a pretrained checkpoint instead of
random initialization.
"""
from .dataset import ReasoningDataset, encode_example
from .trainer import FinetuneTrainer

__all__ = ["ReasoningDataset", "encode_example", "FinetuneTrainer"]

# NOTE: shared.finetune.runner is imported directly by callers (it imports
# shared.reasoning, shared.eval, shared.train — pulling it in here would
# create a needless import cycle risk for no benefit, since only the
# per-language finetune scripts need it).
