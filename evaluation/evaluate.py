"""End-to-end evaluation driver for a :class:`VisionDocModel`.

Why this module looks the way it does
-------------------------------------
Training produces an adapter; this module answers the only question that
matters afterwards — *is it any good, and what does it cost to serve?* It runs
the model over a held-out split and rolls three orthogonal signals into one
JSON-serializable result:

* **Quality** — DocVQA-style QA metrics (EM / ANLS / token-F1 / BLEU / ROUGE)
  from :mod:`evaluation.metrics`, plus micro field P/R/F1 for structured
  extraction samples so a single run scores both answer shapes the product
  supports.
* **Cost** — latency percentiles, serial throughput and peak GPU memory from
  :class:`evaluation.profiler.InferenceProfiler`, because a fine-tune that
  cannot meet the API's latency budget is not shippable regardless of accuracy.
* **Footprint** — trainable/total parameter counts, so the report can show the
  LoRA parameter efficiency next to the quality numbers.

Design choices worth calling out:

* **Batched generation.** We call ``model.generate`` on lists of images sized
  by ``config.inference.batch_size`` rather than one sample at a time. Batching
  is how the served endpoint actually amortizes the vision encoder, so
  measuring per-batch and dividing gives a per-sample latency that reflects
  production, not an artificially slow one-at-a-time loop.
* **Heavy imports are deferred.** ``models`` / ``utils.image_utils`` pull in
  torch/PIL. They are imported *inside* the functions that need them so the
  package's ``__init__`` (and metrics-only unit tests) stay importable in a
  torch-less environment. ``inference.extract`` is imported defensively because
  it is a sibling leaf module that may be built after this one.
* **Never crash a long eval on one bad sample.** A single un-decodable image or
  generation error is logged and recorded as an empty prediction rather than
  aborting a multi-hour run over the whole test set.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from evaluation.metrics import aggregate_qa_metrics, structured_field_metrics
from evaluation.profiler import InferenceProfiler
from preprocessing.schema import DocSample
from utils.logging_utils import get_logger

if TYPE_CHECKING:  # imported only for type checkers; avoids a runtime torch pull.
    from configs.config import ProjectConfig
    from models.base import VisionDocModel

logger = get_logger(__name__)

# Task label (see DocSample.task) that switches a sample into structured
# field-extraction scoring in addition to the free-form QA metrics.
_EXTRACTION_TASK = "extraction"


def _chunk(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """Yield consecutive slices of ``items`` of length ``size`` (last may be short).

    A plain generator (rather than materializing all batches) keeps memory flat
    when evaluating tens of thousands of documents.
    """
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _sample_refs(sample: DocSample) -> list[str]:
    """All acceptable reference strings for a sample.

    ``DocSample.__post_init__`` already folds the primary ``answer`` into
    ``answers``, so ``answers`` is normally the complete set; we fall back to
    ``[answer]`` defensively in case a sample was constructed by hand.
    """
    if sample.answers:
        return list(sample.answers)
    return [sample.answer] if sample.answer else []


def _sample_gold(sample: DocSample) -> str:
    """The single canonical gold string shown in per-sample report rows."""
    if sample.answer:
        return sample.answer
    return sample.answers[0] if sample.answers else ""


def _extraction_field_metrics(
    samples: Sequence[DocSample], predictions: Sequence[str]
) -> dict[str, float] | None:
    """Micro field P/R/F1 over the subset of samples tagged as extraction.

    Both the model output and the gold answer are parsed as JSON via
    :func:`inference.extract.parse_json_answer` (robust to code fences / prose
    around the object). Returns ``None`` — signalling "no extraction scoring" —
    when there are no extraction samples or the parser is unavailable, so the
    caller only surfaces these keys when they are meaningful.
    """
    ext_indices = [i for i, s in enumerate(samples) if s.task == _EXTRACTION_TASK]
    if not ext_indices:
        return None

    # ``inference.extract`` is a sibling leaf module; import it lazily and
    # degrade gracefully if it has not been built yet rather than hard-failing
    # a QA evaluation that happens to contain a few extraction rows.
    try:
        from inference.extract import parse_json_answer
    except Exception:  # pragma: no cover - only when the sibling module is absent.
        logger.warning(
            "inference.extract.parse_json_answer unavailable; "
            "skipping structured field metrics for %d extraction samples.",
            len(ext_indices),
        )
        return None

    pred_fields: list[dict] = []
    gold_fields: list[dict] = []
    for i in ext_indices:
        pred_fields.append(parse_json_answer(predictions[i]) or {})
        # Gold extraction answers are stored as a serialized JSON object in
        # ``answer``; parse them the same way so key/value normalization matches.
        gold_fields.append(parse_json_answer(samples[i].answer) or {})

    metrics = structured_field_metrics(pred_fields, gold_fields)
    metrics["n_extraction"] = float(len(ext_indices))
    return metrics


def evaluate_model(
    model: "VisionDocModel",
    samples: list[DocSample],
    config: "ProjectConfig",
    profile: bool = True,
) -> dict:
    """Evaluate ``model`` over ``samples`` and return a report-ready result dict.

    The returned dict has a stable schema (see the project contract)::

        {
          "metrics":     {... aggregate_qa_metrics (+ field_* for extraction)},
          "profile":     ProfileResult.to_dict(),
          "predictions": [{sample_id, question, prediction, gold, confidence}, ...],
          "params":      count_parameters(model),
          "device":      "<device str>",
        }

    ``profile`` toggles GPU peak-memory tracking (the profiler's context manager
    resets/reads CUDA counters). Per-sample *latency* is always measured because
    the QA report is far less useful without a serving-cost figure; disabling
    ``profile`` merely skips the memory window on environments where the reset
    would be noise (e.g. a shared MPS laptop).
    """
    # Deferred heavy import: parameter counting lives in ``models`` which pulls
    # torch. Keeping it here means metrics-only callers never import torch.
    from models import count_parameters

    device_label = str(getattr(model, "device", "cpu"))
    batch_size = max(1, int(config.inference.batch_size))

    model.to_eval()  # dropout off + eval-mode norms; idempotent, cheap, correct.

    predictions: list[str] = []
    confidences: list[float] = []
    prediction_records: list[dict[str, Any]] = []

    profiler = InferenceProfiler(getattr(model, "device", device_label))
    # When profiling is requested we enter the profiler's context so peak GPU
    # memory is attributed to this window only; otherwise we still record
    # latencies (nullcontext) but leave peak-memory at 0.
    memory_ctx = profiler if profile else contextlib.nullcontext()

    with memory_ctx:
        for batch in _chunk(samples, batch_size):
            images = _load_batch_images(batch)
            questions = [s.question for s in batch]

            start = time.perf_counter()
            try:
                outputs = model.generate(images, questions)
            except Exception as exc:  # keep a long run alive on a single failure.
                logger.exception("Generation failed for a batch of %d; recording empties.", len(batch))
                outputs = None
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                _record_failed_batch(batch, elapsed_ms, profiler, predictions, confidences, prediction_records)
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # ``generate`` returns a single object for a scalar question and a
            # list for a list; we always pass a list, but normalize defensively.
            batch_outputs = list(outputs) if isinstance(outputs, list) else [outputs]
            # Spread the measured batch wall-clock evenly across its samples so
            # throughput/latency reflect the amortized per-document cost.
            per_sample_ms = elapsed_ms / max(len(batch), 1)

            for sample, gen in zip(batch, batch_outputs):
                text = (gen.text or "").strip()
                conf = float(gen.confidence)
                predictions.append(text)
                confidences.append(conf)
                profiler.record(per_sample_ms)
                prediction_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "prediction": text,
                        "gold": _sample_gold(sample),
                        "confidence": round(conf, 4),
                    }
                )

    # -- Quality metrics ----------------------------------------------------
    refs = [_sample_refs(s) for s in samples[: len(predictions)]]
    metrics = aggregate_qa_metrics(predictions, refs, confidences)

    field_metrics = _extraction_field_metrics(samples[: len(predictions)], predictions)
    if field_metrics is not None:
        # Merge structured extraction scores alongside the QA metrics so the
        # report renders one flat table; keys are disjoint (field_* / n_extraction).
        metrics.update(field_metrics)

    profile_result = profiler.result()

    logger.info(
        "Evaluated %d samples | EM=%.3f ANLS=%.3f token_f1=%.3f | %.1f ms/sample | %.2f samples/s",
        int(metrics.get("n", 0)),
        metrics.get("exact_match", 0.0),
        metrics.get("anls", 0.0),
        metrics.get("token_f1", 0.0),
        profile_result.latency_ms_mean,
        profile_result.throughput_samples_s,
    )

    return {
        "metrics": metrics,
        "profile": profile_result.to_dict(),
        "predictions": prediction_records,
        "params": count_parameters(model),
        "device": device_label,
    }


def _load_batch_images(batch: Sequence[DocSample]) -> list[Any]:
    """Load a batch of sample images to PIL, tolerating per-sample decode errors.

    ``DocSample.image`` may be a PIL image or a path/bytes/base64 source, so we
    route everything through :func:`utils.image_utils.load_image` for a uniform
    RGB PIL result. A failed load yields a tiny blank RGB placeholder so the
    batch stays aligned with ``questions`` (dropping it would misalign
    predictions with references).
    """
    from PIL import Image  # local: keep module import PIL-free for lightweight callers.

    from utils.image_utils import load_image

    images: list[Any] = []
    for sample in batch:
        try:
            images.append(load_image(sample.image))
        except Exception:
            logger.warning("Failed to load image for sample %r; using blank placeholder.", sample.sample_id)
            images.append(Image.new("RGB", (32, 32), color=(255, 255, 255)))
    return images


def _record_failed_batch(
    batch: Sequence[DocSample],
    elapsed_ms: float,
    profiler: InferenceProfiler,
    predictions: list[str],
    confidences: list[float],
    prediction_records: list[dict[str, Any]],
) -> None:
    """Record empty predictions for a batch whose generation raised.

    Keeping the samples in the result (as empty, zero-confidence predictions)
    means the metric denominators still reflect the true dataset size — a crash
    on hard documents should count as a miss, not silently shrink the test set.
    """
    per_sample_ms = elapsed_ms / max(len(batch), 1)
    for sample in batch:
        predictions.append("")
        confidences.append(0.0)
        profiler.record(per_sample_ms)
        prediction_records.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "prediction": "",
                "gold": _sample_gold(sample),
                "confidence": 0.0,
            }
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m evaluation.evaluate``."""
    parser = argparse.ArgumentParser(
        description="Evaluate a VisionDoc model (base or LoRA-adapted) on a split.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (defaults to $VISIONDOC_CONFIG or configs/default.yaml).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter path; overrides config.inference.adapter_path.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate: train | validation | test (default: test).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on the number of evaluated samples (default: config.data.max_eval_samples).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Directory for the HTML/JSON report (default: config.report_dir).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> str:
    """Load a model, evaluate it, and write an HTML+JSON report; return html path.

    Wiring order mirrors the training pipeline so numbers are comparable:
    ``load_config`` -> seed -> device summary -> build model -> (optional)
    adapter -> load split -> evaluate -> report. Heavy modules are imported
    here (not at module top) so this file compiles and imports without torch.
    """
    # Deferred imports: these pull torch / datasets and are only needed for a
    # real evaluation run, not for importing ``evaluate_model`` as a library.
    from configs import load_config
    from evaluation.report import build_report
    from models import build_model
    from preprocessing.datasets import load_samples
    from utils.device import log_device_summary
    from utils.seed import set_seed

    from dataclasses import replace

    args = _build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    # CLI adapter wins over the config so one config can benchmark many runs.
    # ProjectConfig is a *frozen* dataclass, so we rebuild it immutably rather
    # than assigning (which would raise FrozenInstanceError).
    if args.adapter is not None:
        config = replace(config, inference=replace(config.inference, adapter_path=args.adapter))

    set_seed(config.seed)
    log_device_summary(config.device, config.model.torch_dtype)

    model = build_model(config)  # load=True: weights + processor on device.
    if config.inference.adapter_path:
        logger.info("Attaching LoRA adapter from %s", config.inference.adapter_path)
        model.load_adapter(config.inference.adapter_path)
    model.to_eval()

    max_samples = args.max_samples if args.max_samples is not None else config.data.max_eval_samples
    samples = load_samples(config, args.split, max_samples)
    if not samples:
        logger.warning("No samples loaded for split %r; the report will be empty.", args.split)

    results = evaluate_model(model, samples, config, profile=True)
    # Stamp provenance so the report/JSON are self-describing after the fact.
    results["split"] = args.split
    results["adapter_path"] = config.inference.adapter_path

    output_dir = args.report or config.report_dir
    html_path = build_report(results, config, output_dir)
    logger.info("Wrote evaluation report to %s", html_path)

    # A terse stdout line so shell wrappers / CI logs show the headline numbers.
    metrics = results["metrics"]
    print(
        json.dumps(
            {
                "split": args.split,
                "n": metrics.get("n"),
                "exact_match": round(metrics.get("exact_match", 0.0), 4),
                "anls": round(metrics.get("anls", 0.0), 4),
                "token_f1": round(metrics.get("token_f1", 0.0), 4),
                "report": html_path,
            },
            indent=2,
        )
    )
    return html_path


if __name__ == "__main__":
    main()
