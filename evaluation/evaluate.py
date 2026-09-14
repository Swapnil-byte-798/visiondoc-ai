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
* **Never crash a long eval on one bad sample — but never hide it either.** A
  single un-decodable image or generation error is logged and recorded as an
  empty prediction rather than aborting a multi-hour run. Those empties are
  then *counted* and published under ``results["failures"]`` with the offending
  sample ids, because an empty prediction scores exactly like a confidently
  wrong one: a run where 30% of generations crashed produces a plausible-looking
  metric table that is not a measurement of the model at all. The counter is
  what makes the difference visible to a reader.
* **Per-sample score vectors are published, not just means.** ``results``
  carries ``per_sample`` (sample ids zipped to EM / ANLS / token-F1 / field-F1)
  so a reader can recompute the headline numbers, re-derive the confidence
  intervals, run a paired test between two arms, or do error analysis without
  re-running inference. Aggregates alone are not checkable.
* **Intervals, always.** Quality metrics are requested with ``with_ci=True`` so
  every published number arrives with ``n`` and a bootstrap 95% interval. On a
  200-sample eval set the intervals are wide, and two arms whose intervals
  overlap have *not* been shown to differ — that caveat belongs next to the
  number, which is only possible if the number ships with its interval.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from evaluation.metrics import (
    aggregate_qa_metrics,
    per_sample_field_scores,
    per_sample_qa_scores,
    structured_field_metrics,
)
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

# Nominal coverage of every published interval. Fixed here (rather than plumbed
# through the CLI) so two arms of the same comparison cannot accidentally be
# reported at different confidence levels — which would make the intervals
# incomparable and the comparison meaningless.
_CI_CONFIDENCE = 0.95

# Per-sample vectors are rounded before serialization purely to keep the JSON
# small; 6 decimals is far below the resolution of any bootstrap over n=200, so
# recomputing a mean or an interval from the published vectors reproduces the
# published figure. It is a size choice, never a presentation choice.
_PER_SAMPLE_ROUND = 6

# Sentinel distinguishing "the backend returned no ``text`` field at all"
# (a broken result object) from "the backend returned an empty answer"
# (a real, scorable prediction). ``None`` cannot do that job: it is a legal
# value of ``GenerationResult.text``.
_MISSING = object()


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
    samples: Sequence[DocSample],
    predictions: Sequence[str],
    ci_seed: int = 0,
) -> tuple[dict[str, float] | None, list[float | None] | None]:
    """Micro field P/R/F1 (with CIs) plus a per-document field-F1 vector.

    Both the model output and the gold answer are parsed as JSON via
    :func:`inference.extract.parse_json_answer` (robust to code fences / prose
    around the object).

    Returns ``(metrics, per_sample_field_f1)`` where both members are ``None``
    — signalling "no extraction scoring" — when there are no extraction samples
    or the parser is unavailable, so the caller only surfaces these keys when
    they are meaningful.

    ``per_sample_field_f1`` is aligned 1:1 with ``samples`` and holds ``None``
    for every non-extraction row, so it can be zipped straight onto the sample
    ids in ``results["per_sample"]`` without a second index list.

    Why this metric carries the headline: CORD gold answers are serialized JSON
    objects, so EM/ANLS/token-F1 over a whole JSON blob are structurally pinned
    near zero and cannot separate the two arms. The field-level scores are the
    only ones that move, which is why they are computed ``with_ci=True`` and
    published rather than left as a debug aside.

    Caveat encoded below: the returned *vector* is macro (one score per
    document) while the published ``field_f1`` is micro (over all key/value
    pairs). Their means legitimately differ; the vector is for error analysis
    and paired tests, not for reproducing the headline number.
    """
    ext_indices = [i for i, s in enumerate(samples) if s.task == _EXTRACTION_TASK]
    if not ext_indices:
        return None, None

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
        return None, None

    pred_fields: list[dict] = []
    gold_fields: list[dict] = []
    for i in ext_indices:
        pred_fields.append(parse_json_answer(predictions[i]) or {})
        # Gold extraction answers are stored as a serialized JSON object in
        # ``answer``; parse them the same way so key/value normalization matches.
        gold_fields.append(parse_json_answer(samples[i].answer) or {})

    metrics = structured_field_metrics(
        pred_fields,
        gold_fields,
        with_ci=True,
        ci_confidence=_CI_CONFIDENCE,
        ci_seed=ci_seed,
    )
    # ``structured_field_metrics(with_ci=True)`` returns its own ``n`` (number of
    # extraction documents) and ``ci_level``. Both collide with the QA roll-up's
    # keys when the caller merges the two dicts into one flat table, and the QA
    # ``n`` (total scored samples) is the one a reader expects next to EM/ANLS.
    # Rename the count and drop the duplicate level — the two bootstraps run at
    # the same ``_CI_CONFIDENCE``, so the surviving ``ci_level`` describes both.
    extraction_n = metrics.pop("n", float(len(ext_indices)))
    metrics.pop("ci_level", None)
    metrics["n_extraction"] = float(extraction_n)

    # Scatter the per-document vector back over the full sample list so it lines
    # up with the sample ids; non-extraction rows stay None rather than 0.0,
    # which would be indistinguishable from "extracted nothing correctly".
    doc_scores = per_sample_field_scores(pred_fields, gold_fields)["field_f1"]
    field_f1_vector: list[float | None] = [None] * len(samples)
    for position, index in enumerate(ext_indices):
        field_f1_vector[index] = round(doc_scores[position], _PER_SAMPLE_ROUND)

    return metrics, field_f1_vector


def evaluate_model(
    model: "VisionDocModel",
    samples: list[DocSample],
    config: "ProjectConfig",
    profile: bool = True,
) -> dict:
    """Evaluate ``model`` over ``samples`` and return a report-ready result dict.

    The returned dict has a stable schema (see the project contract)::

        {
          "metrics":     {... aggregate_qa_metrics (+ *_ci_low/_ci_high, n,
                              ci_level, and field_* for extraction)},
          "profile":     ProfileResult.to_dict(),
          "predictions": [{sample_id, question, prediction, gold, confidence}, ...],
          "params":      count_parameters(model),
          "device":      "<device str>",
          "failures":    {"generation_failures": int, "failed_sample_ids": [...],
                          "image_load_failures": int,
                          "image_load_failed_sample_ids": [...]},
          "per_sample":  {"sample_id": [...], "exact_match": [...],
                          "anls": [...], "token_f1": [...],
                          "field_f1": [...]  # extraction runs only
                         },
        }

    The first five keys are the historical contract and are unchanged;
    ``failures`` and ``per_sample`` are additive, and every previously emitted
    metric key still appears with the same meaning (``with_ci=True`` only adds
    keys). Callers written against the old schema keep working.

    ``failures`` exists because this function deliberately does not abort on a
    bad sample: a crashed generation is scored as an empty string, which is
    indistinguishable in the metric table from a model that answered and got it
    wrong. Publishing the count (and the ids) is what lets a reader tell a
    measured number from a broken run. A non-zero count is logged at WARNING.

    ``per_sample`` exists because a mean without its underlying vector is not
    checkable: with it, a reader can recompute the aggregate, re-derive the
    bootstrap interval, or run a *paired* baseline-vs-LoRA test — which is far
    more sensitive than comparing two independent intervals.

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
    # Failure bookkeeping. ``failed_sample_ids`` is kept alongside the count so
    # a reader can go and look at the offending documents instead of guessing
    # whether the failures were random or concentrated on one document type.
    failed_sample_ids: list[str] = []
    image_failed_sample_ids: list[str] = []

    profiler = InferenceProfiler(getattr(model, "device", device_label))
    # When profiling is requested we enter the profiler's context so peak GPU
    # memory is attributed to this window only; otherwise we still record
    # latencies (nullcontext) but leave peak-memory at 0.
    memory_ctx = profiler if profile else contextlib.nullcontext()

    with memory_ctx:
        for batch in _chunk(samples, batch_size):
            images, image_failures = _load_batch_images(batch)
            image_failed_sample_ids.extend(image_failures)
            questions = [s.question for s in batch]

            start = time.perf_counter()
            try:
                outputs = model.generate(images, questions)
            except Exception:  # keep a long run alive on a single failure.
                logger.exception("Generation failed for a batch of %d; recording empties.", len(batch))
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                failed_sample_ids.extend(
                    _record_failed_batch(
                        batch, elapsed_ms, profiler, predictions, confidences, prediction_records
                    )
                )
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # ``generate`` returns a single object for a scalar question and a
            # list for a list; we always pass a list, but normalize defensively.
            batch_outputs = list(outputs) if isinstance(outputs, list) else [outputs]
            # Spread the measured batch wall-clock evenly across its samples so
            # throughput/latency reflect the amortized per-document cost.
            per_sample_ms = elapsed_ms / max(len(batch), 1)

            # Pad a short return with ``None`` instead of zipping it away: a
            # backend that silently drops outputs would otherwise shorten
            # ``predictions`` and slide every later sample's references out of
            # alignment (``samples[: len(predictions)]`` below), quietly scoring
            # each prediction against the wrong gold answer. Which samples were
            # dropped is unknowable from a bare list, so the shortfall is
            # charged to the tail of the batch and every padded row counts as a
            # failure — the ids may be approximate, the count is not.
            if len(batch_outputs) != len(batch):
                logger.error(
                    "generate() returned %d outputs for a batch of %d; "
                    "padding the shortfall as generation failures.",
                    len(batch_outputs),
                    len(batch),
                )
                batch_outputs = (batch_outputs + [None] * len(batch))[: len(batch)]

            for sample, gen in zip(batch, batch_outputs):
                text, conf, ok = _unpack_generation(gen, sample.sample_id)
                if not ok:
                    failed_sample_ids.append(sample.sample_id)
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
    scored_samples = samples[: len(predictions)]
    refs = [_sample_refs(s) for s in scored_samples]
    # Seed the bootstrap from the run's own seed so the published interval is
    # reproducible *and* identical in construction across the two arms of a
    # comparison (same config -> same resampling), rather than being one more
    # uncontrolled source of difference between baseline and LoRA.
    ci_seed = int(getattr(config, "seed", 0) or 0)
    metrics = aggregate_qa_metrics(
        predictions,
        refs,
        confidences,
        with_ci=True,
        ci_confidence=_CI_CONFIDENCE,
        ci_seed=ci_seed,
    )

    field_metrics, field_f1_vector = _extraction_field_metrics(
        scored_samples, predictions, ci_seed=ci_seed
    )
    if field_metrics is not None:
        # Merge structured extraction scores alongside the QA metrics so the
        # report renders one flat table; keys are disjoint (field_* / n_extraction).
        metrics.update(field_metrics)

    profile_result = profiler.result()

    # -- Per-sample vectors --------------------------------------------------
    # Recomputed through the public metrics helper rather than threaded out of
    # the aggregator: ``evaluation.metrics`` scores both through one private
    # vector function, so these are the *same* numbers the means and intervals
    # were built from and cannot drift from the published aggregate.
    qa_vectors = per_sample_qa_scores(predictions, refs)
    per_sample: dict[str, list[Any]] = {
        "sample_id": [record["sample_id"] for record in prediction_records],
        "exact_match": [round(v, _PER_SAMPLE_ROUND) for v in qa_vectors["exact_match"]],
        "anls": [round(v, _PER_SAMPLE_ROUND) for v in qa_vectors["anls"]],
        "token_f1": [round(v, _PER_SAMPLE_ROUND) for v in qa_vectors["token_f1"]],
    }
    if field_f1_vector is not None:
        per_sample["field_f1"] = field_f1_vector

    failures = {
        "generation_failures": len(failed_sample_ids),
        "failed_sample_ids": failed_sample_ids,
        "image_load_failures": len(image_failed_sample_ids),
        "image_load_failed_sample_ids": image_failed_sample_ids,
    }

    logger.info(
        "Evaluated %d samples | EM=%.3f ANLS=%.3f token_f1=%.3f | %.1f ms/sample | %.2f samples/s",
        int(metrics.get("n", 0)),
        metrics.get("exact_match", 0.0),
        metrics.get("anls", 0.0),
        metrics.get("token_f1", 0.0),
        profile_result.latency_ms_mean,
        profile_result.throughput_samples_s,
    )
    if "field_f1" in metrics:
        logger.info(
            "Field-level (headline for structured extraction) | n=%d "
            "P=%.3f [%.3f, %.3f] R=%.3f [%.3f, %.3f] F1=%.3f [%.3f, %.3f] (%.0f%% CI)",
            int(metrics.get("n_extraction", 0)),
            metrics.get("field_precision", 0.0),
            metrics.get("field_precision_ci_low", 0.0),
            metrics.get("field_precision_ci_high", 0.0),
            metrics.get("field_recall", 0.0),
            metrics.get("field_recall_ci_low", 0.0),
            metrics.get("field_recall_ci_high", 0.0),
            metrics.get("field_f1", 0.0),
            metrics.get("field_f1_ci_low", 0.0),
            metrics.get("field_f1_ci_high", 0.0),
            _CI_CONFIDENCE * 100,
        )
    _log_failure_summary(failures, len(predictions))

    return {
        "metrics": metrics,
        "profile": profile_result.to_dict(),
        "predictions": prediction_records,
        "params": count_parameters(model),
        "device": device_label,
        "failures": failures,
        "per_sample": per_sample,
    }


def _unpack_generation(gen: Any, sample_id: str) -> tuple[str, float, bool]:
    """Pull ``(text, confidence, ok)`` out of one generation result.

    ``ok`` is False when the backend produced nothing usable for this sample —
    a ``None`` padding entry from a short batch, or an object without the
    expected ``text``/``confidence`` fields. Those cases are recorded as empty
    predictions (so the metric denominator still counts the document) *and*
    flagged so the caller can report them.

    An empty-but-well-formed generation is **not** a failure: a model that
    legitimately answers with "" is being measured correctly and scores zero on
    its own merits. Conflating the two would inflate the failure count and let a
    genuinely bad model hide behind an "infrastructure problem" label.
    """
    if gen is None:
        return "", 0.0, False
    raw_text = getattr(gen, "text", _MISSING)
    if raw_text is _MISSING:
        # No ``text`` field at all is a broken result object, not an empty
        # answer — flag it rather than letting it masquerade as a zero score.
        logger.warning(
            "Generation result for sample %r has no 'text' field (%s); recording empty.",
            sample_id,
            type(gen).__name__,
        )
        return "", 0.0, False
    try:
        text = (raw_text or "").strip()
        conf = float(getattr(gen, "confidence", 0.0) or 0.0)
    except Exception:  # pragma: no cover - a malformed backend result object.
        logger.warning("Unreadable generation result for sample %r; recording empty.", sample_id)
        return "", 0.0, False
    return text, conf, True


def _log_failure_summary(failures: dict[str, Any], n_scored: int) -> None:
    """Log a WARNING naming the failures so they cannot pass unnoticed.

    Emitted at WARNING (not INFO) and placed after the metrics line on purpose:
    the failure count is the first thing that invalidates a metric table, so it
    must be visible in a log a reader skims, right under the numbers it
    qualifies. Sample ids are truncated to keep the line readable; the full list
    is always in ``results["failures"]``.
    """
    gen_failures = int(failures.get("generation_failures", 0))
    img_failures = int(failures.get("image_load_failures", 0))
    if not gen_failures and not img_failures:
        return

    def _preview(ids: Sequence[str], limit: int = 10) -> str:
        head = ", ".join(str(i) for i in ids[:limit])
        return f"{head}, ... (+{len(ids) - limit} more)" if len(ids) > limit else head

    if gen_failures:
        share = (gen_failures / n_scored * 100.0) if n_scored else 0.0
        logger.warning(
            "%d/%d samples (%.1f%%) failed generation and were scored as EMPTY "
            "predictions — treat the metrics above as a LOWER BOUND and report "
            "this count alongside them. Failed ids: [%s]",
            gen_failures,
            n_scored,
            share,
            _preview(failures.get("failed_sample_ids", [])),
        )
    if img_failures:
        logger.warning(
            "%d sample image(s) failed to decode and were replaced with a blank "
            "placeholder; their predictions are not a measurement of the model. "
            "Ids: [%s]",
            img_failures,
            _preview(failures.get("image_load_failed_sample_ids", [])),
        )


def _load_batch_images(batch: Sequence[DocSample]) -> tuple[list[Any], list[str]]:
    """Load a batch of sample images to PIL, tolerating per-sample decode errors.

    ``DocSample.image`` may be a PIL image or a path/bytes/base64 source, so we
    route everything through :func:`utils.image_utils.load_image` for a uniform
    RGB PIL result. A failed load yields a tiny blank RGB placeholder so the
    batch stays aligned with ``questions`` (dropping it would misalign
    predictions with references).

    Returns ``(images, failed_sample_ids)``. The ids are returned — not just
    logged — because a blank placeholder still produces a *scored* prediction:
    the model is being graded on a white square, which is a data failure
    masquerading as a model error. The caller publishes the count so the
    distinction survives into the report.
    """
    from PIL import Image  # local: keep module import PIL-free for lightweight callers.

    from utils.image_utils import load_image

    images: list[Any] = []
    failed_sample_ids: list[str] = []
    for sample in batch:
        try:
            images.append(load_image(sample.image))
        except Exception:
            logger.warning("Failed to load image for sample %r; using blank placeholder.", sample.sample_id)
            images.append(Image.new("RGB", (32, 32), color=(255, 255, 255)))
            failed_sample_ids.append(sample.sample_id)
    return images, failed_sample_ids


def _record_failed_batch(
    batch: Sequence[DocSample],
    elapsed_ms: float,
    profiler: InferenceProfiler,
    predictions: list[str],
    confidences: list[float],
    prediction_records: list[dict[str, Any]],
) -> list[str]:
    """Record empty predictions for a batch whose generation raised.

    Keeping the samples in the result (as empty, zero-confidence predictions)
    means the metric denominators still reflect the true dataset size — a crash
    on hard documents should count as a miss, not silently shrink the test set.

    Returns the ids of the affected samples so the caller can count them into
    ``results["failures"]``: the denominator is only honest if the reader is
    also told how much of it is made of crashes rather than answers.
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
    return [sample.sample_id for sample in batch]


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
    # Every quality figure ships with its interval and the sample count, and the
    # generation-failure count rides along: a number printed without them is not
    # publishable, so it is not printed without them here either.
    metrics = results["metrics"]
    # Existing keys keep their exact legacy names, types and values (bare
    # rounded floats) so anything already parsing this stream keeps working;
    # the intervals arrive as *additional* ``<metric>_ci`` [low, high] pairs.
    summary: dict[str, Any] = {
        "split": args.split,
        "n": metrics.get("n"),
        "exact_match": round(metrics.get("exact_match", 0.0), 4),
        "anls": round(metrics.get("anls", 0.0), 4),
        "token_f1": round(metrics.get("token_f1", 0.0), 4),
        "report": html_path,
        "ci_level": metrics.get("ci_level"),
        "generation_failures": results["failures"]["generation_failures"],
    }
    for key in ("exact_match", "anls", "token_f1"):
        _add_ci(summary, metrics, key)
    if "field_f1" in metrics:
        # Headline metric for structured extraction (CORD): EM/ANLS/token-F1 are
        # computed over a whole serialized JSON answer and are structurally
        # pinned near zero, so they are the *secondary* figures here despite
        # appearing first for backwards compatibility.
        summary["n_extraction"] = metrics.get("n_extraction")
        for key in ("field_precision", "field_recall", "field_f1"):
            summary[key] = round(metrics.get(key, 0.0), 4)
            _add_ci(summary, metrics, key)
    print(json.dumps(summary, indent=2))
    return html_path


def _add_ci(summary: dict[str, Any], metrics: dict[str, Any], key: str) -> None:
    """Attach ``<key>_ci: [low, high]`` to ``summary`` when the interval exists.

    Additive by construction: the point estimate stays a bare float under its
    original key, so this can never change the type of an existing field for a
    downstream parser. A metrics dict built with ``with_ci=False`` simply gets
    no ``_ci`` entry rather than a fabricated one.
    """
    low = metrics.get(f"{key}_ci_low")
    high = metrics.get(f"{key}_ci_high")
    if low is None or high is None:
        return
    summary[f"{key}_ci"] = [round(float(low), 4), round(float(high), 4)]


if __name__ == "__main__":
    main()
