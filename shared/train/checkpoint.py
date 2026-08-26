"""Checkpoint save/load with full training-state recovery.

Spec 2.2 is explicit and marks this mandatory: "Each checkpoint must
contain model weights, optimizer state, scheduler state, current training
step, and a configuration file", and "A run that cannot resume after
interruption will lose marks."

Saved state, and why each piece is required for an EXACT resume:

``model_state_dict``   the weights themselves.
``optimizer_state_dict``  AdamW's per-parameter first/second moment
    estimates. Dropping these restarts Adam with zeroed moments, which
    produces a visible loss spike at the resume point.
``scheduler_state_dict``  the schedule's parameters; combined with
    ``step`` it reproduces the exact LR.
``scaler_state_dict``  the AMP GradScaler's loss scale. Resuming with a
    fresh scaler re-runs the scale-discovery process and throws away
    several steps to overflow.
``step`` / ``tokens_seen``  where the run was.
``config``  the full GPTConfig + TrainConfig, so a checkpoint is
    self-describing and does not depend on the YAML still existing.
``rng_state``  torch/numpy RNG, so the data stream continues rather than
    restarting from the same windows.

Two files are maintained: ``latest.pt`` (rolling, for resume) and
``best.pt`` (lowest validation loss, for evaluation).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    config: Dict[str, Any],
    tokens_seen: int = 0,
    best_val_loss: float = float("inf"),
    extra: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> str:
    """Write a full training checkpoint atomically.

    Written to a ``.tmp`` file then renamed: a Kaggle session killed
    mid-write would otherwise leave a truncated checkpoint that is not
    loadable, destroying the run it was meant to protect.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "extra": extra or {},
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic on POSIX

    # A sidecar JSON so the config is readable without loading a 300 MB
    # tensor file -- the spec asks for "a configuration file" per checkpoint.
    with open(Path(path).with_suffix(".config.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "tokens_seen": tokens_seen,
                   "best_val_loss": best_val_loss, "config": config},
                  f, indent=2, ensure_ascii=False)

    if is_best:
        best = str(Path(path).parent / "best.pt")
        shutil.copyfile(path, best)
        shutil.copyfile(Path(path).with_suffix(".config.json"),
                        Path(best).with_suffix(".config.json"))
    return path


def load_checkpoint(
    path: str,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Restore training state from ``path``.

    Any of model/optimizer/scheduler/scaler may be None to load only part
    of the state (e.g. evaluation needs the weights alone).

    Returns the raw payload minus the large tensors, so the caller can read
    ``step``, ``tokens_seen``, ``best_val_loss`` and ``config``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    # weights_only=False: the payload includes numpy RNG state and config
    # dicts, not just tensors. The file is one we wrote ourselves.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if ckpt.get("torch_rng_state") is not None:
        # RNG state must be a CPU ByteTensor; map_location may have moved it.
        torch.set_rng_state(ckpt["torch_rng_state"].cpu()
                            if hasattr(ckpt["torch_rng_state"], "cpu")
                            else ckpt["torch_rng_state"])
    if ckpt.get("numpy_rng_state") is not None:
        np.random.set_state(ckpt["numpy_rng_state"])
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        except Exception:
            pass    # different GPU count on resume is not fatal

    return {
        "step": ckpt.get("step", 0),
        "tokens_seen": ckpt.get("tokens_seen", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "config": ckpt.get("config", {}),
        "extra": ckpt.get("extra", {}),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return ``latest.pt`` in *checkpoint_dir*, or None if absent.

    Lets a training script be re-run with the same command after a session
    dies and pick up automatically where it left off.
    """
    p = Path(checkpoint_dir) / "latest.pt"
    return str(p) if p.exists() else None
