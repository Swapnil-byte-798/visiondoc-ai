"""Training package: HF ``Trainer`` wiring, callbacks, and the training CLI.

Public API::

    from training import (
        build_trainer,              # assemble a configured transformers.Trainer
        build_training_arguments,   # TrainingConfig -> transformers.TrainingArguments
        make_compute_metrics,       # eval-loop metric fn (None -> select on eval_loss)
        build_early_stopping,       # EarlyStoppingCallback from config
        GPUMemoryCallback, ThroughputCallback,  # observability callbacks
        train_main,                 # python -m training.train entry point
    )

Importing this package pulls in ``transformers`` (the callbacks subclass
``TrainerCallback``), which is by design — anything that touches this package is
about to train and therefore needs the training stack installed.
"""

from __future__ import annotations

from training.callbacks import (
    GPUMemoryCallback,
    ThroughputCallback,
    build_early_stopping,
)
from training.train import main as train_main
from training.trainer import (
    build_trainer,
    build_training_arguments,
    make_compute_metrics,
)

__all__ = [
    "build_trainer",
    "build_training_arguments",
    "make_compute_metrics",
    "build_early_stopping",
    "GPUMemoryCallback",
    "ThroughputCallback",
    "train_main",
]
