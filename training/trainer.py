"""HF ``Trainer`` / ``TrainingArguments`` construction for LoRA fine-tuning.

Design rationale
----------------
This module is the single translation layer between the project's typed
:class:`~configs.config.TrainingConfig` and Hugging Face's ``TrainingArguments``
/ ``Trainer``. Keeping the mapping here (rather than sprinkled through
``train.py``) means the training *policy* lives in config and the *wiring* lives
in one auditable place.

Three deliberate, non-obvious choices are encoded below:

1. ``remove_unused_columns=False`` and ``label_names=["labels"]``. Our datasets
   yield raw ``DocSample.to_record()`` dicts (image, question, answer, ...) that
   the model's ``collate_train`` turns into tensors. The default Trainer would
   strip any column not matching the model's ``forward`` signature *before* the
   collator runs, deleting exactly the columns we need; disabling that is
   mandatory. ``label_names`` tells the Trainer where the supervision lives so it
   still computes eval loss.

2. Version-robust argument construction. ``TrainingArguments`` renamed
   ``evaluation_strategy`` -> ``eval_strategy`` and added
   ``gradient_checkpointing_kwargs`` across releases. We build a superset kwargs
   dict and reconcile it against the *installed* signature, so the same code runs
   on a range of ``transformers`` versions instead of hard-failing on one.

3. Checkpoint-selection metric fallback. Generation-based QA metrics (ANLS/EM)
   cannot be computed cheaply inside the teacher-forced eval loop (see
   :func:`make_compute_metrics`), so when no ``compute_metrics`` is supplied we
   transparently switch ``metric_for_best_model`` to ``eval_loss`` — otherwise
   ``load_best_model_at_end`` would look for a metric that never gets produced
   and crash at the first evaluation.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from transformers import Trainer, TrainingArguments

from configs.config import ProjectConfig
from models.base import VisionDocModel
from utils.logging_utils import get_logger

from training.callbacks import (
    GPUMemoryCallback,
    ThroughputCallback,
    build_early_stopping,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TrainingArguments
# ---------------------------------------------------------------------------
def build_training_arguments(config: ProjectConfig) -> TrainingArguments:
    """Map :class:`TrainingConfig` onto ``transformers.TrainingArguments``.

    Every field of ``TrainingConfig`` is forwarded (precision, scheduler, eval /
    save cadence, best-model selection, logging, seed, dataloader workers). See
    the module docstring for the two load-bearing overrides
    (``remove_unused_columns`` / ``label_names``) and the version reconciliation.
    """
    tcfg = config.training

    # bf16 and fp16 are mutually exclusive; if a config sets both (e.g. copied
    # from two examples), prefer bf16 — it has the wider dynamic range and is the
    # right default on Ampere+ where these LoRA runs are expected to live.
    bf16 = bool(tcfg.bf16)
    fp16 = bool(tcfg.fp16)
    if bf16 and fp16:
        logger.warning("Both bf16 and fp16 requested; using bf16 and disabling fp16.")
        fp16 = False

    kwargs: dict[str, Any] = dict(
        output_dir=tcfg.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=tcfg.num_train_epochs,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        learning_rate=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        warmup_ratio=tcfg.warmup_ratio,
        lr_scheduler_type=tcfg.lr_scheduler_type,  # "cosine" by default
        max_grad_norm=tcfg.max_grad_norm,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=tcfg.gradient_checkpointing,
        logging_steps=tcfg.logging_steps,
        logging_dir=str(Path(tcfg.output_dir) / "logs"),
        eval_strategy=tcfg.eval_strategy,
        eval_steps=tcfg.eval_steps,
        save_strategy=tcfg.save_strategy,
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,
        load_best_model_at_end=tcfg.load_best_model_at_end,
        metric_for_best_model=tcfg.metric_for_best_model,
        greater_is_better=tcfg.greater_is_better,
        report_to=_normalize_report_to(tcfg.report_to),
        run_name=tcfg.run_name,
        seed=tcfg.seed,
        dataloader_num_workers=tcfg.dataloader_num_workers,
        # -- load-bearing overrides (see module docstring) --
        remove_unused_columns=False,
        label_names=["labels"],
        # Safetensors checkpoints are safer (no pickle) and faster to load; the
        # adapter is tiny so there is no downside to always using them.
        save_safetensors=True,
    )

    # ``use_reentrant=False`` is the modern, correct autograd path for gradient
    # checkpointing and is required for it to cooperate with PEFT + frozen base
    # weights; only pass it when checkpointing is actually on.
    if tcfg.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return _construct_training_arguments(kwargs)


def _normalize_report_to(report_to: str) -> str | list[str]:
    """Normalize the config's ``report_to`` to something HF accepts.

    ``transformers`` understands the string ``"none"`` (disable all trackers) as
    well as individual integration names. We lower-case and pass through; the
    only translation is mapping empty/falsey values to ``"none"`` so a blank
    config never silently defaults to "all" (which would try to spin up every
    installed tracker).
    """
    value = (report_to or "none").strip().lower()
    return value or "none"


def _construct_training_arguments(kwargs: dict[str, Any]) -> TrainingArguments:
    """Instantiate ``TrainingArguments`` reconciled with the installed signature.

    Handles two forms of cross-version drift:

    * ``eval_strategy`` (new) vs ``evaluation_strategy`` (old) — rename if only
      the legacy name is accepted.
    * Unknown kwargs (e.g. ``gradient_checkpointing_kwargs`` on very old builds)
      — dropped with a warning instead of raising ``TypeError``, so a slightly
      older/newer ``transformers`` degrades gracefully rather than crashing.
    """
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    accepted = set(params)
    accepts_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())

    # Reconcile the eval-strategy parameter name with the installed version.
    if "eval_strategy" in kwargs and "eval_strategy" not in accepted:
        if "evaluation_strategy" in accepted:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        elif not accepts_var_kwargs:
            logger.warning("This transformers build accepts neither eval_strategy nor evaluation_strategy; dropping it.")
            kwargs.pop("eval_strategy", None)

    # Drop anything this version does not understand (unless it takes **kwargs).
    if not accepts_var_kwargs:
        for key in [k for k in kwargs if k not in accepted]:
            logger.warning("TrainingArguments has no %r on this transformers version; dropping it.", key)
            kwargs.pop(key)

    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
def make_compute_metrics(
    model: VisionDocModel, config: ProjectConfig
) -> Callable[[Any], dict[str, float]] | None:
    """Return a ``compute_metrics`` for the eval loop, or ``None`` (the default).

    We intentionally return ``None``. Here is the reasoning, because it drives the
    checkpoint-selection fallback in :func:`build_trainer`:

    The vanilla ``Trainer`` eval loop is **teacher-forced** and hands
    ``compute_metrics`` the model's raw *logits* over the full vocabulary — shape
    ``(batch, seq_len, |V|)``. For a VLM like Qwen2.5-VL that tensor is enormous
    and accumulating it across the eval set reliably OOMs. Worse, the argmax of
    teacher-forced logits measures next-token accuracy, **not** the free-running
    ANLS/exact-match we actually care about — those require autoregressive
    generation, which the base ``Trainer`` does not perform (that needs
    ``Seq2SeqTrainer`` + ``predict_with_generate``, which does not apply to this
    multimodal causal-LM setup).

    So during training we select checkpoints on ``eval_loss`` (a perfectly good
    proxy that is cheap and always available), and compute the *real* generation
    metrics exactly once, post-hoc, via :func:`evaluation.evaluate_model` — which
    performs true generation on a held-out subset at the end of ``train.py``.
    Returning ``None`` keeps the loop cheap and the reported numbers honest.

    The signature still takes ``model``/``config`` so a future backbone that *can*
    afford in-loop generation metrics can override this without touching callers.
    """
    logger.info(
        "compute_metrics=None: selecting checkpoints on eval_loss; generation "
        "metrics (ANLS/EM) are computed post-hoc by evaluation.evaluate_model."
    )
    return None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def build_trainer(
    model: VisionDocModel,
    train_dataset: Any,
    eval_dataset: Any,
    config: ProjectConfig,
    compute_metrics: Callable[[Any], dict[str, float]] | None = None,
) -> Trainer:
    """Assemble a ``Trainer`` wired for LoRA fine-tuning of a VisionDoc model.

    Wiring:

    * ``model=model.model`` — the underlying (PEFT-wrapped) ``nn.Module``; the
      :class:`VisionDocModel` wrapper is not itself a module.
    * ``data_collator=model.collate_train`` — the backbone-specific tensorizer
      that turns raw record dicts into supervised batches.
    * callbacks: throughput + GPU-memory observability always, early stopping when
      evaluation is enabled and a positive patience is configured.
    * the processor is attached (as ``processing_class`` on new ``transformers``,
      else ``tokenizer``) so saved checkpoints are self-contained and reloadable
      for inference without re-fetching the base processor.
    """
    args = build_training_arguments(config)

    # Checkpoint-selection safety net: without a generation-aware compute_metrics
    # the eval loop only emits ``eval_loss``. A config asking for e.g.
    # ``eval_anls`` would then make ``load_best_model_at_end`` search for a metric
    # that is never produced and raise at the first evaluation. Fall back to
    # eval_loss (lower is better) in that case.
    if compute_metrics is None and args.load_best_model_at_end:
        wanted = (args.metric_for_best_model or "").replace("eval_", "")
        if wanted != "loss":
            logger.warning(
                "No compute_metrics supplied; switching metric_for_best_model "
                "from %r to 'eval_loss' (generation metrics are computed post-hoc).",
                args.metric_for_best_model,
            )
            args.metric_for_best_model = "eval_loss"
        # Only eval_loss is produced without compute_metrics, so lower is always
        # better. Force it even when the config already named a loss metric while
        # leaving greater_is_better=True — transformers only auto-derives that
        # flag from the metric name when it is None, not when it was explicitly set.
        args.greater_is_better = False

    # Observability first; these are cheap and safe on any device.
    callbacks: list[Any] = [ThroughputCallback(), GPUMemoryCallback()]
    # Early stopping only makes sense when we actually evaluate and a patience is
    # set — otherwise the callback would either never fire or warn every step.
    if (
        config.training.eval_strategy != "no"
        and config.training.early_stopping_patience > 0
        and config.training.load_best_model_at_end
    ):
        callbacks.append(build_early_stopping(config))
    elif config.training.early_stopping_patience > 0 and not config.training.load_best_model_at_end:
        # EarlyStoppingCallback.on_train_begin asserts load_best_model_at_end;
        # skip it (rather than crash the run) when best-model tracking is off.
        logger.warning(
            "early_stopping_patience > 0 but load_best_model_at_end is False; "
            "skipping EarlyStoppingCallback (it requires load_best_model_at_end=True)."
        )

    hf_model = getattr(model, "model", model)  # unwrap to the trainable nn.Module

    trainer_kwargs: dict[str, Any] = dict(
        model=hf_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=model.collate_train,
        callbacks=callbacks,
    )
    if compute_metrics is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics

    processor = getattr(model, "processor", None)
    if processor is not None:
        _attach_processor(trainer_kwargs, processor)

    trainer = Trainer(**trainer_kwargs)
    logger.info(
        "Trainer ready: %d train / %d eval examples | effective batch=%d | callbacks=%s",
        _safe_len(train_dataset),
        _safe_len(eval_dataset),
        args.per_device_train_batch_size * args.gradient_accumulation_steps,
        [type(cb).__name__ for cb in callbacks],
    )
    return trainer


def _attach_processor(trainer_kwargs: dict[str, Any], processor: Any) -> None:
    """Attach the processor under whichever keyword the Trainer version supports.

    ``transformers`` migrated the argument from ``tokenizer`` to
    ``processing_class`` (the latter is correct for multimodal processors).
    Passing the wrong name would raise ``TypeError``, so we probe the signature.
    """
    accepted = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in accepted:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in accepted:
        trainer_kwargs["tokenizer"] = processor
    else:  # pragma: no cover - both names absent would be a very unusual build
        logger.debug("Trainer accepts neither processing_class nor tokenizer; skipping processor attach.")


def _safe_len(dataset: Any) -> int:
    """Best-effort ``len`` for logging (streaming/iterable datasets have none)."""
    try:
        return len(dataset)
    except Exception:  # pragma: no cover - iterable datasets
        return -1


__all__ = ["build_training_arguments", "build_trainer", "make_compute_metrics"]
