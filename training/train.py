"""End-to-end training CLI: LoRA fine-tune a VisionDoc backbone on documents.

    python -m training.train --config configs/default.yaml
    python -m training.train --config configs/cord_qwen.yaml --resume outputs/.../checkpoint-500

Pipeline (kept identical in order to the evaluation CLI so numbers stay
comparable): load config -> seed -> device summary -> build model -> apply LoRA
-> obtain splits -> wrap in datasets -> build trainer -> train -> save adapter +
resolved config -> final generation-based evaluation + report.

Design notes
------------
* **Heavy imports are deferred into functions.** Importing this module (which the
  package ``__init__`` does to re-export :func:`main`) must stay cheap; torch /
  transformers / datasets are only needed when a run actually starts.
* **Splits -> ``DocumentDataset``.** We standardize on the torch map-style
  ``DocumentDataset`` (augment on the train split, deterministic on eval) rather
  than an HF ``Dataset.with_transform`` — both work with our custom collator, but
  the map-style dataset keeps image decoding lazy and the augmentation policy in
  one obvious place.
* **Two evaluation regimes, on purpose.** The in-loop eval selects checkpoints on
  ``eval_loss`` (cheap, teacher-forced); the *final* evaluation runs true
  autoregressive generation via :func:`evaluation.evaluate_model` to produce the
  real ANLS/EM/latency numbers and an HTML report. See
  ``training.trainer.make_compute_metrics`` for why the loop metric is loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # type-only; avoids importing heavy deps at module import time.
    from configs.config import ProjectConfig
    from models.base import VisionDocModel
    from preprocessing.schema import DocSample

logger = get_logger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m training.train``."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a VisionDoc model (Qwen2.5-VL / Donut) on a document dataset.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (defaults to $VISIONDOC_CONFIG or configs/default.yaml).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Checkpoint directory to resume from. Overrides "
            "config.training.resume_from_checkpoint when given."
        ),
    )
    return parser


def _configure_experiment_tracking(config: "ProjectConfig") -> None:
    """Make experiment tracking non-interactive and network-safe by default.

    ``report_to=wandb`` will otherwise block on ``wandb login`` (or fail) on a CI
    box / sandbox with no credentials. Defaulting ``WANDB_MODE=offline`` lets the
    run log locally and be synced later, without ever prompting. We use
    ``setdefault`` so an operator who *has* configured W&B online is respected.
    """
    import os

    tracker = (config.training.report_to or "none").strip().lower()
    if tracker == "wandb":
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_PROJECT", config.name)
        logger.info("Weights & Biases logging enabled (WANDB_MODE=%s).", os.environ["WANDB_MODE"])
    elif tracker == "tensorboard":
        logger.info("TensorBoard logging enabled (event files under the run's logs/ dir).")


def _obtain_splits(config: "ProjectConfig") -> dict[str, list["DocSample"]]:
    """Return ``{"train","validation","test"} -> list[DocSample]``.

    Resolution order mirrors the project contract: reuse a fingerprinted cache if
    present, otherwise build + cache (so the next run is fast), and as a last
    resort build in memory. Cached rows are re-hydrated to :class:`DocSample` via
    ``from_record`` so the rest of the pipeline is agnostic to how the data was
    obtained.
    """
    from preprocessing import build_and_cache, build_splits, load_cached
    from preprocessing.schema import DocSample

    cached = load_cached(config)
    if cached is not None:
        logger.info(
            "Using cached splits: %s",
            {name: len(cached[name]) for name in cached},
        )
        return {name: [DocSample.from_record(row) for row in cached[name]] for name in cached}

    logger.info("No dataset cache found; building splits from source (and caching for reuse).")
    try:
        dataset_dict = build_and_cache(config)
        return {
            name: [DocSample.from_record(row) for row in dataset_dict[name]]
            for name in dataset_dict
        }
    except Exception:
        # Caching (save_to_disk / Image casting) can fail on exotic environments;
        # an in-memory build still lets training proceed, just without a cache.
        logger.exception("build_and_cache failed; falling back to an in-memory build_splits().")
        return build_splits(config)


def _final_evaluation(
    model: "VisionDocModel", eval_samples: Sequence["DocSample"], config: "ProjectConfig"
) -> None:
    """Run true generation-based evaluation on a capped subset and write a report.

    This is the run's headline quality number. We cap the subset (via
    ``max_eval_samples`` or a sane default) because full-generation ANLS over a
    large validation set is expensive and this is a post-training sanity check,
    not the formal benchmark (``python -m evaluation.evaluate`` does the full
    split). The report goes under the run directory so each run's artifacts stay
    together and a standalone eval report is never clobbered.
    """
    from evaluation import build_report, evaluate_model

    if not eval_samples:
        logger.warning("No evaluation samples available; skipping final evaluation.")
        return

    cap = config.data.max_eval_samples or 200
    subset = list(eval_samples[:cap])
    logger.info("Final generation-based evaluation on %d validation samples...", len(subset))

    model.to_eval()
    results = evaluate_model(model, subset, config, profile=True)
    # Stamp provenance so the JSON/HTML are self-describing after the fact.
    results["split"] = "validation"
    results["adapter_path"] = str(Path(config.training.output_dir) / "adapter")

    report_dir = str(Path(config.training.output_dir) / "final_eval")
    html_path = build_report(results, config, report_dir)
    metrics = results.get("metrics", {})
    logger.info(
        "Final eval | EM=%.3f ANLS=%.3f token_f1=%.3f | report=%s",
        metrics.get("exact_match", 0.0),
        metrics.get("anls", 0.0),
        metrics.get("token_f1", 0.0),
        html_path,
    )


def main(argv: Sequence[str] | None = None) -> str:
    """Run the full training pipeline; return the saved adapter directory path.

    Heavy modules are imported here (not at module top) so importing
    ``training.train`` — which the package ``__init__`` does — stays lightweight.
    """
    # Deferred heavy imports (torch / transformers / datasets pulled transitively).
    from configs import load_config
    from models import build_model, format_parameter_summary
    from preprocessing import DocumentDataset, build_augmenter
    from utils.device import log_device_summary
    from utils.seed import set_seed

    from training.trainer import build_trainer, make_compute_metrics

    args = _build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    # Reproducibility, then a one-line environment banner. deterministic=False:
    # bit-exact determinism disables fast kernels and materially slows LoRA runs;
    # we accept run-to-run jitter for training and reserve strict determinism for
    # evaluation/debugging (see utils.seed.set_seed).
    set_seed(config.seed, deterministic=False)
    log_device_summary(config.device, config.model.torch_dtype)
    _configure_experiment_tracking(config)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Model + LoRA -------------------------------------------------------
    logger.info("Building model %s (type=%s)", config.model.model_id, config.model.model_type)
    model = build_model(config, load=True)
    model.apply_lora()
    logger.info("Parameter budget: %s", format_parameter_summary(model))

    # -- Data ---------------------------------------------------------------
    splits = _obtain_splits(config)
    augmenter = build_augmenter(config)  # None when config.data.augment is False
    train_dataset = DocumentDataset(splits["train"], augmenter=augmenter)
    # Prefer the validation split for in-loop eval; fall back to test if a corpus
    # only defines the latter, so early stopping / best-model selection still work.
    eval_samples = splits.get("validation") or splits.get("test") or []
    eval_dataset = DocumentDataset(eval_samples, augmenter=None)
    logger.info(
        "Datasets ready: train=%d | eval=%d (augment=%s)",
        len(train_dataset),
        len(eval_dataset),
        augmenter is not None,
    )

    # -- Trainer ------------------------------------------------------------
    compute_metrics = make_compute_metrics(model, config)
    trainer = build_trainer(model, train_dataset, eval_dataset, config, compute_metrics=compute_metrics)

    # -- Train --------------------------------------------------------------
    resume_from = args.resume or config.training.resume_from_checkpoint
    logger.info("Starting training%s...", f" (resume from {resume_from})" if resume_from else "")
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # Persist training metrics + trainer state (log_history feeds the plots).
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # -- Persist artifacts --------------------------------------------------
    adapter_dir = output_dir / "adapter"
    model.save_adapter(adapter_dir)
    config.save(output_dir / "run_config.yaml")
    logger.info("Saved LoRA adapter -> %s", adapter_dir)
    logger.info("Saved resolved run config -> %s", output_dir / "run_config.yaml")

    # -- Final (real) evaluation -------------------------------------------
    _final_evaluation(model, eval_samples, config)

    logger.info("Training complete. All artifacts under %s", output_dir)
    return str(adapter_dir)


if __name__ == "__main__":
    main()
