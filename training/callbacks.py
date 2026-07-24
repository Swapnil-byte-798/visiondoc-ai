"""Trainer callbacks: early stopping, GPU-memory logging, and throughput.

Design rationale
----------------
The HF ``Trainer`` exposes a callback protocol so cross-cutting concerns
(observability, early stopping) can be added without subclassing the trainer or
touching the training loop. We keep three small, single-responsibility callbacks
here rather than one god-callback so each can be reused/tested in isolation and
toggled independently:

* :func:`build_early_stopping` — halts a run once the tracked metric stops
  improving, saving GPU hours on a plateaued LoRA fine-tune.
* :class:`GPUMemoryCallback` — records CUDA memory at every evaluation, which is
  the single most useful signal for right-sizing ``max_pixels`` / batch size on
  a document VLM (image resolution dominates VRAM here).
* :class:`ThroughputCallback` — turns wall-clock deltas between log events into a
  samples/second figure, so a config change's cost is visible in the logs
  without an external profiler.

Why import ``transformers`` at module top level: these callbacks *are*
``TrainerCallback`` subclasses, so the base class must exist at class-definition
time. The project only ever imports this module when it is actually about to
train (i.e. transformers is installed), so there is no lightweight-import
requirement to honor here.
"""

from __future__ import annotations

import time
from typing import Any

from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from configs.config import ProjectConfig
from utils.device import gpu_memory_stats
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_early_stopping(config: ProjectConfig) -> EarlyStoppingCallback:
    """Construct an :class:`~transformers.EarlyStoppingCallback` from the config.

    The callback watches ``TrainingArguments.metric_for_best_model`` (set by the
    trainer builder) and stops training after ``early_stopping_patience``
    consecutive evaluations without an improvement of at least
    ``early_stopping_threshold``.

    We surface both knobs through the typed config (rather than hard-coding them)
    because the right patience is dataset-dependent: a small extraction set
    (CORD/SROIE) overfits fast and wants a short patience, whereas DocVQA's noisy
    validation curve needs a couple of extra evals before we trust a plateau.
    """
    return EarlyStoppingCallback(
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_threshold=config.training.early_stopping_threshold,
    )


class GPUMemoryCallback(TrainerCallback):
    """Log CUDA memory (allocated / reserved / peak) at each evaluation.

    Evaluation is the natural sampling point: it runs periodically, and its
    forward passes (often with a larger eval batch) usually mark the run's memory
    high-water line. On non-CUDA devices :func:`gpu_memory_stats` returns zeros,
    so the callback is a harmless no-op on CPU/MPS rather than a special case.

    The stats are both logged *and* injected into the ``metrics`` dict when one is
    provided, so they land in ``trainer_state.json``'s ``log_history`` and become
    plottable after the fact (see ``visualization.plots.plot_training_curves``).
    """

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record GPU memory once the periodic evaluation has finished."""
        stats = gpu_memory_stats()
        logger.info(
            "GPU memory @ step %d: allocated=%.2f GB | reserved=%.2f GB | peak=%.2f GB",
            state.global_step,
            stats["allocated_gb"],
            stats["reserved_gb"],
            stats["max_allocated_gb"],
        )
        # Fold the readings into the eval metrics so they persist in the trainer
        # state. We namespace with a ``gpu_`` prefix so they never collide with a
        # real quality metric or get mistaken for one by ``metric_for_best_model``.
        if metrics is not None:
            metrics["gpu_allocated_gb"] = round(stats["allocated_gb"], 3)
            metrics["gpu_reserved_gb"] = round(stats["reserved_gb"], 3)
            metrics["gpu_peak_gb"] = round(stats["max_allocated_gb"], 3)


class ThroughputCallback(TrainerCallback):
    """Estimate training throughput (samples/second) from log-event timing.

    Rationale for a *windowed* estimate: the ``Trainer`` already reports a
    ``train_samples_per_second`` averaged over the whole run at the very end, but
    that single number hides warm-up cost and mid-run regressions (e.g. a
    checkpoint save stalling the pipeline). By measuring the delta between two
    consecutive ``on_log`` events we get a rolling figure that makes such stalls
    visible while training is still going.

    The sample count per optimizer step is
    ``per_device_train_batch_size * gradient_accumulation_steps * world_size`` —
    gradient accumulation means one logged step corresponds to several micro-
    batches, and ``world_size`` accounts for data-parallel replicas. We read
    ``world_size`` defensively because it is only meaningful once the distributed
    backend is initialized.
    """

    def __init__(self) -> None:
        """Initialize the rolling-window anchors (reset again at train start)."""
        self._t0: float = 0.0
        self._last_time: float = 0.0
        self._last_step: int = 0
        #: Last computed windowed throughput; exposed for external inspection/tests.
        self.last_samples_per_second: float = 0.0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Anchor the timing window at the start of training."""
        now = time.perf_counter()
        self._t0 = now
        self._last_time = now
        self._last_step = int(state.global_step)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Compute samples/second over the window since the previous log event."""
        now = time.perf_counter()
        delta_t = now - self._last_time
        delta_steps = int(state.global_step) - self._last_step

        # Only emit once we have both elapsed time and completed optimizer steps;
        # the first ``on_log`` (or a zero-time window) would divide by ~0.
        if delta_t > 0 and delta_steps > 0:
            world_size = _safe_world_size(args)
            samples = (
                delta_steps
                * int(args.per_device_train_batch_size)
                * int(args.gradient_accumulation_steps)
                * max(world_size, 1)
            )
            samples_per_second = samples / delta_t
            self.last_samples_per_second = samples_per_second
            logger.info(
                "Throughput @ step %d: %.2f samples/s (%.0f samples in %.1fs)",
                state.global_step,
                samples_per_second,
                samples,
                delta_t,
            )
            # Enrich the current log record so the figure is captured in the
            # trainer's ``log_history`` alongside loss/lr for later plotting.
            if logs is not None:
                logs["throughput_samples_per_second"] = round(samples_per_second, 2)

        self._last_time = now
        self._last_step = int(state.global_step)


def _safe_world_size(args: TrainingArguments) -> int:
    """Return the data-parallel world size, defaulting to 1 off a distributed run.

    ``TrainingArguments.world_size`` touches the (possibly uninitialized)
    distributed backend, so we guard it: a throughput log is never worth raising
    inside a callback and aborting a training run.
    """
    try:
        return int(getattr(args, "world_size", 1) or 1)
    except Exception:  # pragma: no cover - depends on distributed init state
        return 1


__all__ = ["build_early_stopping", "GPUMemoryCallback", "ThroughputCallback"]
