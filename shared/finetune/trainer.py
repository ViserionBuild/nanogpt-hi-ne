"""Finetuning loop: start from a pretrained checkpoint, train on masked SFT data.

Course-grounded hyperparameter choices (utility/phase3_plan.md §2):

  * **Learning rate far below pretraining's.** Pretraining used 6e-4
    (hindi/nepali_model.yaml); the SFT Ultimate Guide's rule of thumb for
    finetuning a pretrained model is 1e-4-2e-4 (p.32, p.60). This project
    uses a peak SFT LR of 1.5e-4 by default — roughly a quarter of the
    pretraining peak — with a short warmup and cosine decay, reusing
    ``shared/train/scheduler.py`` unchanged.
  * **Few epochs, watched closely.** The course's own worked example (L9
    p.22) trained 3 epochs on 3,000 examples and reached 0.000 exact-match
    on unseen names — a memorization failure, not a schedule that needed
    more epochs. This trainer evaluates 3-way accuracy every epoch and
    keeps the best-validation-loss checkpoint (mirroring Phase 2's
    ``is_best`` convention exactly), so overfitting past the optimum is
    visible in the logged curve rather than silently shipped.
  * **Perplexity before/after is a required diagnostic, not a proxy for
    task success** (L9 p.22 gives 1.33 -> 194.7 as its own example of a
    swing that must NOT be read as "finetuning made the model worse").
    ``evaluate_pretraining_perplexity`` in ``shared/eval/reasoning_metrics.py``
    is what a caller runs before AND after this trainer's ``train()`` to
    produce that pair of numbers; this module does not compute it itself
    since it needs the ORIGINAL pretraining validation set, not the
    reasoning data this trainer trains on.

Reuses, unchanged: ``shared/train/checkpoint.py`` (identical resumable
format — model weights, optimizer state, scheduler state, step, config),
``shared/train/scheduler.py`` (same warmup+cosine schedule code).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from shared.train.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from shared.train.scheduler import CosineWarmupScheduler


class FinetuneTrainer:
    """Drives reasoning-SFT for one model, starting from its Phase 2 checkpoint."""

    def __init__(
        self,
        model,
        train_dataset,
        validation_dataset,
        model_config,
        pretrained_checkpoint: str,
        checkpoint_dir: str,
        log_dir: str,
        language: str = "",
        learning_rate: float = 1.5e-4,
        min_learning_rate: float = 1.5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        n_epochs: int = 6,
        batch_size: int = 16,
        grad_clip: float = 1.0,
        eval_every_n_steps: int = 50,
        seed: int = 42,
        device: str = "cuda",
        logger=None,
        probe_fn: Optional[Callable[[object], float]] = None,
    ) -> None:
        """
        ``probe_fn``, if given, is called once per epoch as
        ``probe_fn(model)`` and must return a single float — this project
        passes an unseen-entity greedy-accuracy probe (see
        ``shared/finetune/runner.py``). It exists because validation LOSS
        alone cannot detect the memorization failure this trainer's own
        docstring warns about: ``val_ds`` is drawn from the SAME train-side
        entity pool as ``train_dataset``, so a model that has memorized
        per-name answers instead of the comparison logic still shows a
        healthy-looking validation loss. The probe is logged for visibility
        only — it does NOT affect ``best_val_loss``/checkpoint selection,
        which stays on validation loss for consistency with Phase 2's
        checkpointing convention; a human (or the report) reads the logged
        probe curve to catch a val-loss-looks-fine-but-generalization-failed
        run before shipping it.
        """
        self.model = model
        self.train_ds = train_dataset
        self.val_ds = validation_dataset
        self.mcfg = model_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.language = language
        self.logger = logger
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.eval_every_n_steps = eval_every_n_steps
        self.n_epochs = n_epochs
        self.probe_fn = probe_fn

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self._log("CUDA requested but unavailable — falling back to CPU")
            self.device = "cpu"
        self.model.to(self.device)

        self.steps_per_epoch = max(1, len(train_dataset) // batch_size)
        self.max_steps = self.steps_per_epoch * n_epochs

        self.optimizer = model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(0.9, 0.95),
            device_type="cuda" if self.device.startswith("cuda") else "cpu",
        )
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=max(1, int(warmup_ratio * self.max_steps)),
            max_steps=self.max_steps,
            max_lr=learning_rate,
            min_lr=min_learning_rate,
            decay_style="cosine",
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.log_path = self.log_dir / f"finetune_log_{language or 'model'}.csv"
        self._init_log()

        self._load_pretrained(pretrained_checkpoint)

    # ------------------------------------------------------------- logging

    def _log(self, msg: str) -> None:
        (self.logger.info if self.logger else print)(msg)

    def _init_log(self) -> None:
        if self.log_path.exists():
            return
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "step", "epoch", "learning_rate", "train_loss",
                "val_loss", "val_perplexity", "grad_norm", "elapsed_sec",
                "unseen_entity_accuracy_probe",
            ])

    def _append_log(self, **row) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                row.get("step", ""), row.get("epoch", ""),
                f"{row['learning_rate']:.8f}" if row.get("learning_rate") is not None else "",
                f"{row['train_loss']:.6f}" if row.get("train_loss") is not None else "",
                f"{row['val_loss']:.6f}" if row.get("val_loss") is not None else "",
                f"{row['val_perplexity']:.4f}" if row.get("val_perplexity") is not None else "",
                f"{row['grad_norm']:.4f}" if row.get("grad_norm") is not None else "",
                f"{row['elapsed_sec']:.1f}" if row.get("elapsed_sec") is not None else "",
                f"{row['unseen_entity_accuracy_probe']:.4f}"
                if row.get("unseen_entity_accuracy_probe") is not None else "",
            ])

    # ---------------------------------------------------------- pretrained init

    def _load_pretrained(self, checkpoint: str) -> None:
        """Load ONLY the model weights from the Phase 2 checkpoint.

        Deliberately not the optimizer/scheduler state: SFT is a new
        optimization problem with its own (lower) learning rate and its
        own short schedule, and reusing pretraining's Adam moment
        estimates (tuned for a 6e-4 peak LR and a totally different loss
        landscape — masked, tiny, high-variance batches) would misapply
        step-size information computed for a different regime entirely.
        Spec 3.1's own protocol is "start from that language's own
        pretrained checkpoint. Keep tokenizer/vocabulary fixed." — weights
        and vocabulary carry over; the optimization loop is new.
        """
        state = load_checkpoint(checkpoint, model=self.model, device=self.device)
        self._log(f"[init] loaded pretrained weights from {checkpoint} "
                  f"(pretrain step {state['step']:,})")
        self.pretrain_step = state["step"]

    # ---------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Mean masked loss over the WHOLE validation split, in order."""
        self.model.eval()
        total_loss, total_answer_tokens = 0.0, 0
        for x, y in self.val_ds.iter_batches(self.batch_size, self.device):
            out = self.model(x, y)
            loss = out["loss"]
            n_answer_tokens = int((y != self.mcfg.pad_id).sum().item())
            total_loss += loss.item() * n_answer_tokens
            total_answer_tokens += n_answer_tokens
        self.model.train()
        mean = total_loss / max(total_answer_tokens, 1)
        import math
        return {
            "val_loss": mean,
            "val_perplexity": float(math.exp(min(mean, 20.0))),
            "n_answer_tokens": total_answer_tokens,
        }

    # ------------------------------------------------------------ training

    def train(self, resume: bool = True) -> Dict[str, object]:
        if resume:
            ckpt = find_latest_checkpoint(str(self.checkpoint_dir))
            if ckpt:
                state = load_checkpoint(
                    ckpt, self.model, self.optimizer, self.scheduler, device=self.device,
                )
                self.step = state["step"]
                self.best_val_loss = state["best_val_loss"]
                self._log(f"[resume] from {ckpt} at step {self.step:,}")

        self.model.train()
        start = time.time()

        self._log(
            f"[finetune] {self.language or 'model'} | device={self.device} | "
            f"{len(self.train_ds)} train examples ({self.train_ds.n_dropped} dropped "
            f"as too long) | {self.steps_per_epoch} steps/epoch x {self.n_epochs} "
            f"epochs = {self.max_steps} steps"
        )

        while self.step < self.max_steps:
            lr = self.scheduler.step(self.step)
            x, y = self.train_ds.get_batch(self.batch_size, self.device, self.rng)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x, y)
            loss = out["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()

            self.step += 1
            epoch = self.step / self.steps_per_epoch

            if self.step % 10 == 0:
                self._append_log(step=self.step, epoch=round(epoch, 3),
                                 learning_rate=lr, train_loss=loss.item(),
                                 grad_norm=float(grad_norm),
                                 elapsed_sec=time.time() - start)

            if self.step % self.eval_every_n_steps == 0 or self.step == self.max_steps:
                m = self.evaluate()
                is_best = m["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = m["val_loss"]

                # Epoch-boundary probe (visibility only — see __init__'s
                # docstring on why this exists and why it does NOT drive
                # checkpoint selection): only run at whole-epoch steps,
                # since it involves greedy generation and is much more
                # expensive than the loss-only evaluate() above.
                probe_acc = None
                is_epoch_boundary = self.step % self.steps_per_epoch == 0
                if self.probe_fn is not None and is_epoch_boundary:
                    probe_acc = self.probe_fn(self.model)
                    self._log(f"  [probe] unseen-entity accuracy: {probe_acc:.3f}")

                self._append_log(step=self.step, epoch=round(epoch, 3),
                                 learning_rate=lr, train_loss=loss.item(),
                                 val_loss=m["val_loss"], val_perplexity=m["val_perplexity"],
                                 elapsed_sec=time.time() - start,
                                 unseen_entity_accuracy_probe=probe_acc)
                self._log(f"  step {self.step:>5}/{self.max_steps} "
                         f"(epoch {epoch:.2f}) | train_loss {loss.item():.4f} | "
                         f"val_loss {m['val_loss']:.4f} | val_ppl "
                         f"{m['val_perplexity']:.2f}{'  <- best' if is_best else ''}")
                self._save(is_best=is_best)

        final = self.evaluate()
        elapsed = time.time() - start
        summary = {
            "language": self.language,
            "final_step": self.step,
            "n_epochs": self.n_epochs,
            "pretrain_step": self.pretrain_step,
            "final_val_loss": final["val_loss"],
            "final_val_perplexity": final["val_perplexity"],
            "best_val_loss": self.best_val_loss,
            "elapsed_seconds": elapsed,
        }
        with open(self.log_dir / f"finetune_summary_{self.language}.json",
                 "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._log(f"[done] {self.step} steps | val_loss {final['val_loss']:.4f} | "
                 f"val_ppl {final['val_perplexity']:.2f} | {elapsed:.1f}s")
        return summary

    def _save(self, is_best: bool) -> None:
        save_checkpoint(
            path=str(self.checkpoint_dir / "latest.pt"),
            model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=None,
            step=self.step,
            config={"model": self.mcfg.to_dict(), "language": self.language,
                    "pretrain_step": self.pretrain_step},
            tokens_seen=0,
            best_val_loss=self.best_val_loss,
            is_best=is_best,
        )
