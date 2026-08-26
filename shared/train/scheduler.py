"""Learning-rate schedule: linear warmup then cosine decay.

Spec 2.2 requires "learning-rate scheduling". Two phases:

**Warmup** (first ``warmup_steps``): LR rises linearly from ~0 to the peak.
Adam's second-moment estimate ``v`` starts at zero and is badly conditioned
for the first few dozen steps, so a full-size step taken immediately can
land the model somewhere it never recovers from. Warmup lets the moment
estimates stabilize before large steps are taken.

**Cosine decay** (remainder): LR follows a half-cosine from peak down to
``min_lr``. The smooth, gradually-flattening decay lets the model take
large exploratory steps early and fine ones late, and unlike a step
schedule it introduces no discontinuity that shows up as a visible kink in
the loss curve.

Implemented as a plain function of ``step`` rather than a stateful
``_LRScheduler``, so resuming is exact: the LR at step N is the same
whether the run reached N continuously or across three Kaggle sessions.
The ``state_dict``/``load_state_dict`` pair exists to satisfy the spec's
"scheduler state" checkpoint requirement.
"""
from __future__ import annotations

import math
from typing import Dict


class CosineWarmupScheduler:
    """Stateless-by-construction LR schedule with a checkpointable wrapper."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
        decay_style: str = "cosine",
    ) -> None:
        if warmup_steps >= max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < max_steps ({max_steps})"
            )
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.decay_style = decay_style
        self.last_step = 0

    def get_lr(self, step: int) -> float:
        """LR for ``step`` -- a pure function, so resume is exact."""
        if step < self.warmup_steps:
            # linear warmup; +1 so step 0 is not exactly zero LR
            return self.max_lr * (step + 1) / max(self.warmup_steps, 1)

        if step >= self.max_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / max(
            self.max_steps - self.warmup_steps, 1
        )
        if self.decay_style == "cosine":
            # 1 -> 0 as a half cosine
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.decay_style == "linear":
            coeff = 1.0 - progress
        elif self.decay_style == "constant":
            coeff = 1.0
        else:
            raise ValueError(f"unknown decay_style {self.decay_style!r}")

        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self, step: int) -> float:
        """Set the LR on every param group for ``step``; return it."""
        lr = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_step = step
        return lr

    def state_dict(self) -> Dict[str, object]:
        """Spec 2.2 requires scheduler state in every checkpoint."""
        return {
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "decay_style": self.decay_style,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        for k, v in state.items():
            setattr(self, k, v)
