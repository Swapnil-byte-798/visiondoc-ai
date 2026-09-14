"""Base vs. LoRA-fine-tuned model comparison for VisionDoc AI.

Why this module exists
----------------------
Training produces a LoRA adapter; :mod:`evaluation.evaluate` scores *one* model
in isolation. The question a stakeholder actually asks is comparative — "did the
fine-tune earn its keep?" — and answering it *defensibly* means putting the base
(zero-shot) backbone and the adapted model side by side on the **same** held-out
sample list, with the **same** decoding settings, the **same** metric/latency/
memory machinery, and publishing every quality number with its sample size and
bootstrap confidence interval.

Design choices worth calling out
--------------------------------
* **We load the backbone exactly once.** A Qwen2.5-VL-3B checkpoint is several
  gigabytes; instantiating two ``VisionDocModel`` objects would pin ~2x the host
  and GPU memory and double the (slow) weight load. Instead we build the base
  model, evaluate it, then attach the LoRA adapter *in place*
  (:meth:`VisionDocModel.load_adapter` wraps the already-resident base weights in
  a ``PeftModel``) and evaluate again. This is why the evaluation order is fixed:
  **base first, then adapter** — ``load_adapter`` is a one-way wrap, so we must
  measure the untouched base before the adapter is attached.
* **Both arms see the identical sample list, and we assert it.** The eval set is
  materialized once and reused. After both runs we compare the two arms'
  ``sample_id`` sequences element-by-element and refuse to publish if they
  differ: a comparison over two different sample lists is not a comparison, and
  the failure mode (a loader that reshuffles, a short batch that drops rows) is
  silent unless something checks.
* **A missing adapter is a hard failure, never a silent downgrade.** The previous
  behaviour — fall back to base-only, still write ``comparison.*``, exit 0 — is
  how a "comparison" that is secretly one model gets published. If an adapter was
  requested and cannot be loaded or evaluated we log an error and exit non-zero.
  A deliberate baseline-only run is a different intent and must be *opted into*
  with ``--base-only``, which writes an artifact clearly labelled as such.
* **Field-level P/R/F1 is the headline, not EM/ANLS/token-F1.** On CORD the gold
  answer is a serialized JSON object. Whole-string metrics over a JSON blob are
  structurally pinned near zero — they cannot separate the two arms no matter how
  much better one of them is at the actual task. The field-level scores are the
  ones that move, so they are listed first and marked PRIMARY; EM/ANLS/token-F1
  are still published, labelled secondary, with the reason stated inline.
* **``reports/results.json`` is the canonical artifact.** It is the single file
  ``scripts/update_readme.py`` consumes to regenerate the README table, so its
  schema is fixed and versioned (``schema_version``). Everything under
  ``environment`` (latency, VRAM, training wall-clock) is *machine-dependent* and
  must never enter the CI-checked README block — it is recorded here so the run
  is auditable, not so it can be diffed across runners.
* **Training time is recovered, not re-measured.** The wall-clock of a multi-hour
  fine-tune is not something a comparison script re-runs; the Hugging Face
  ``Trainer`` already persisted it in ``trainer_state.json``.
* **Heavy imports are deferred.** ``models`` / ``evaluation.evaluate`` /
  ``preprocessing.datasets`` pull torch/datasets, and pandas/matplotlib are only
  needed for the CLI's table+chart. They are imported inside the functions that
  use them, so this module's own top-level imports add nothing heavier than the
  shared logger — the pure helpers (training-time recovery, row building,
  results-JSON assembly, Markdown rendering) can be exercised without loading a
  model or the plotting stack.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # type-only: keeps runtime import torch/pandas-free.
    import pandas as pd

    from configs.config import ProjectConfig
    from models.base import VisionDocModel
    from preprocessing.schema import DocSample

logger = get_logger(__name__)

# Version of the reports/results.json contract. Bump on any breaking change to
# the shape below; consumers (scripts/update_readme.py) key off it so an old
# results file cannot be silently rendered by a newer, incompatible reader.
RESULTS_SCHEMA_VERSION = 1

# Default location of the canonical artifact, relative to ``config.report_dir``.
RESULTS_JSON_NAME = "results.json"

# Why EM/ANLS/token-F1 are demoted on this benchmark. Stated once, rendered into
# both comparison.md and (as a metric-tier label) results.json so a reader never
# sees a near-zero EM and concludes the model is broken.
SECONDARY_METRIC_NOTE = (
    "Secondary. CORD-style targets are *serialized JSON objects*, so Exact Match, "
    "ANLS and token-F1 are computed over a whole JSON blob and are structurally "
    "pinned near zero for both arms — they cannot separate base from fine-tuned. "
    "The field-level scores above are the headline metric."
)


class AdapterUnavailableError(RuntimeError):
    """Raised when an adapter was requested but could not be loaded/evaluated.

    Exists as its own type so the CLI can turn it into a non-zero exit while
    library callers can catch exactly this condition. It is *never* downgraded to
    a warning: the whole point of the comparison is the second arm, and an
    artifact that silently contains only the base model is worse than no
    artifact — it looks like a result.
    """


# ---------------------------------------------------------------------------
# Row specification for the comparison table
# ---------------------------------------------------------------------------
# A single source of truth for which numbers appear, their display label, where
# to read them from a result dict, how to format them, and whether higher is
# better. We keep it as data so the CSV/Markdown/chart/results.json all stay in
# lockstep and the order is trivial to tweak in one place.
#
# ``kind`` controls formatting: ratio (0..1), ms, gb, millions (M params),
# seconds. ``higher_better`` is None for footprint rows where "better" is not a
# meaningful direction. ``metric_key``/``primary`` are set only on the quality
# rows that belong in ``results.json`` under ``metrics`` (latency/memory live in
# ``environment`` instead, because they are machine-dependent). ``ci`` marks rows
# whose evaluation result carries ``<key>_ci_low`` / ``<key>_ci_high``.
#
# Order matters and is deliberate: the PRIMARY field-level metrics come first,
# because on serialized-JSON targets they are the only ones that measure the task.
_ROW_SPECS: list[dict[str, Any]] = [
    {
        "label": "Field F1 (PRIMARY)",
        "section": "metrics",
        "key": "field_f1",
        "kind": "ratio",
        "higher_better": True,
        "metric_key": "field_f1",
        "primary": True,
        "ci": True,
    },
    {
        "label": "Field Precision",
        "section": "metrics",
        "key": "field_precision",
        "kind": "ratio",
        "higher_better": True,
        "metric_key": "field_precision",
        "primary": False,
        "ci": True,
    },
    {
        "label": "Field Recall",
        "section": "metrics",
        "key": "field_recall",
        "kind": "ratio",
        "higher_better": True,
        "metric_key": "field_recall",
        "primary": False,
        "ci": True,
    },
    {
        "label": "Exact Match",
        "section": "metrics",
        "key": "exact_match",
        "kind": "ratio",
        "higher_better": True,
        "metric_key": "exact_match",
        "primary": False,
        "ci": True,
        "secondary": True,
    },
    {
        "label": "ANLS",
        "section": "metrics",
        "key": "anls",
        "kind": "ratio",
        "higher_better": True,
        "metric_key": "anls",
        "primary": False,
        "ci": True,
        "secondary": True,
    },
    {
        "label": "Token F1",
        "section": "metrics",
        "key": "token_f1",
        "kind": "ratio",
        "higher_better": True,
        "metric_key": "token_f1",
        "primary": False,
        "ci": True,
        "secondary": True,
    },
    {"label": "Avg latency (ms)", "section": "profile", "key": "latency_ms_mean", "kind": "ms", "higher_better": False},
    {"label": "Peak memory (GB)", "section": "profile", "key": "peak_memory_gb", "kind": "gb", "higher_better": False},
    # Params + training time are filled in specially (see _build_rows) because
    # they are derived from cross-model differences / the trainer state, not a
    # single result-dict lookup.
    {"label": "Trainable LoRA params (M)", "section": "_derived", "key": "trainable_lora_m", "kind": "millions", "higher_better": None},
    {"label": "Total params (M)", "section": "_derived", "key": "total_params_m", "kind": "millions", "higher_better": None},
    {"label": "Training time (s)", "section": "_derived", "key": "training_time_s", "kind": "seconds", "higher_better": None},
]

# Metric labels that are all on a comparable [0, 1] scale — the subset we hand to
# the bar chart so it does not try to plot milliseconds next to a 0.3 ANLS. The
# primary field metrics lead so the chart reads left-to-right as "the metric that
# matters, then the context".
_CHART_METRIC_LABELS = [
    spec["label"] for spec in _ROW_SPECS if spec.get("metric_key") is not None
]

# Ordered metric keys published under results.json["metrics"]. Derived from the
# row specs so the table, the chart and the canonical artifact can never drift.
_RESULTS_METRIC_KEYS = [
    spec["metric_key"] for spec in _ROW_SPECS if spec.get("metric_key") is not None
]


# ---------------------------------------------------------------------------
# Eval split resolution
# ---------------------------------------------------------------------------
def parse_split_spec(spec: str) -> list[str]:
    """Split a spec like ``"validation+test"`` into ``["validation", "test"]``.

    The published experiment evaluates on validation **and** test pooled into one
    fixed 200-document set, but :func:`preprocessing.datasets.load_samples` only
    understands a single canonical split. Rather than teach the loader about
    unions (and risk changing training-side behaviour), the union is expressed
    here, in the one place that needs it. ``+`` and ``,`` are both accepted so the
    spec is comfortable in a shell, a YAML string, and a notebook.
    """
    parts = [p.strip() for chunk in spec.split("+") for p in chunk.split(",")]
    resolved = [p for p in parts if p]
    if not resolved:
        raise ValueError(f"Empty eval split spec: {spec!r}")
    return resolved


def resolve_eval_split(config: "ProjectConfig", split: str | None) -> str:
    """Pick the eval split spec: CLI > ``data.eval_split`` > ``data.test_split``.

    ``data.eval_split`` is read with ``getattr`` because it is an optional field
    that a config schema may or may not define; falling back to ``test_split``
    keeps every existing config working unchanged.
    """
    if split:
        return split
    configured = getattr(config.data, "eval_split", None)
    return str(configured or config.data.test_split or "test")


def load_eval_samples(
    config: "ProjectConfig", split_spec: str, max_samples: int | None
) -> list["DocSample"]:
    """Materialize the (possibly pooled) evaluation set exactly once.

    Duplicate ``sample_id``s across the pooled splits are dropped with a warning:
    scoring the same document twice would silently double its weight in every
    micro-averaged metric and in the bootstrap.

    ``max_samples`` is a cap on the *combined* list. It is also pushed down into
    each individual load so a capped smoke run does not decode a whole split into
    PIL images before slicing it away (the memory bug that used to OOM Colab).
    """
    from preprocessing.datasets import load_samples

    parts = parse_split_spec(split_spec)
    pooled: list["DocSample"] = []
    seen: set[str] = set()
    duplicates = 0
    for part in parts:
        for sample in load_samples(config, part, max_samples):
            sid = str(sample.sample_id)
            if sid in seen:
                duplicates += 1
                continue
            seen.add(sid)
            pooled.append(sample)
            if max_samples is not None and len(pooled) >= max_samples:
                break
        if max_samples is not None and len(pooled) >= max_samples:
            break

    if duplicates:
        logger.warning(
            "Dropped %d duplicate sample_id(s) while pooling split %r; "
            "a document scored twice would be double-weighted in every metric.",
            duplicates,
            split_spec,
        )
    logger.info("Eval split %r -> %d samples (cap=%s)", split_spec, len(pooled), max_samples)
    return pooled


def decoding_signature(config: "ProjectConfig") -> dict[str, Any]:
    """The generation settings both arms must share, as a comparable dict.

    Recorded and asserted rather than assumed: if the two arms decode with
    different ``max_new_tokens``/sampling/beams, the delta measures the decoder,
    not the adapter. Both arms run from this one frozen config object, so the
    assertion is cheap — and it is exactly the kind of invariant that quietly
    breaks the day someone adds a per-arm override.
    """
    inference = config.inference
    return {
        "max_new_tokens": int(inference.max_new_tokens),
        "do_sample": bool(inference.do_sample),
        "temperature": float(inference.temperature),
        "top_p": float(inference.top_p),
        "num_beams": int(inference.num_beams),
        "batch_size": int(inference.batch_size),
        "seed": int(config.seed),
    }


def _sample_ids(result: dict | None) -> list[str]:
    """The ordered ``sample_id``s an evaluation result actually scored."""
    if not result:
        return []
    per_sample = result.get("per_sample") or {}
    ids = per_sample.get("sample_id")
    if ids:
        return [str(i) for i in ids]
    return [str(rec.get("sample_id")) for rec in (result.get("predictions") or [])]


def _assert_same_sample_list(
    samples: Sequence["DocSample"], base_result: dict, finetuned_result: dict | None
) -> None:
    """Fail loudly unless both arms scored the same documents in the same order.

    A paired comparison (and the paired reading of two intervals) is only valid
    over identical inputs. Mismatches here are always a bug — a reshuffling
    loader, a dropped batch, a partially-failed run — and publishing through one
    would produce numbers that look fine and mean nothing.
    """
    expected = [str(s.sample_id) for s in samples]
    base_ids = _sample_ids(base_result)
    if base_ids != expected:
        raise RuntimeError(
            "Base arm scored a different sample list than was loaded "
            f"({len(base_ids)} scored vs {len(expected)} loaded); refusing to publish."
        )
    if finetuned_result is None:
        return
    ft_ids = _sample_ids(finetuned_result)
    if ft_ids != base_ids:
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(zip(base_ids, ft_ids))
                if a != b
            ),
            min(len(base_ids), len(ft_ids)),
        )
        raise RuntimeError(
            "Base and fine-tuned arms scored different sample lists "
            f"(base n={len(base_ids)}, finetuned n={len(ft_ids)}, first difference at "
            f"index {first_diff}); refusing to publish a comparison that is not paired."
        )


# ---------------------------------------------------------------------------
# Training-time recovery from trainer_state.json
# ---------------------------------------------------------------------------
def _candidate_state_dirs(config: "ProjectConfig", adapter_path: str | None) -> list[Path]:
    """Ordered, de-duplicated directories that may hold a ``trainer_state.json``.

    The training entry point saves the adapter under ``<output_dir>/adapter`` and
    the trainer writes its state under ``<output_dir>`` (and inside each
    ``checkpoint-*/``). So the adapter's parent directory and the configured
    ``output_dir`` are the two most likely homes; we also try ``output_root/
    run_name`` for setups that key runs by name.
    """
    roots: list[Path] = []
    if adapter_path:
        ap = Path(adapter_path)
        roots.append(ap)  # unlikely, but the adapter dir itself is cheap to check
        roots.append(ap.parent)  # the usual case: <output_dir>/adapter -> <output_dir>
    roots.append(Path(config.training.output_dir))
    roots.append(Path(config.output_root) / config.training.run_name)

    # Preserve order while dropping duplicates (Path is hashable).
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def _find_trainer_state(config: "ProjectConfig", adapter_path: str | None) -> Path | None:
    """Locate the most authoritative ``trainer_state.json`` for the run, if any.

    Preference within a directory: the top-level ``trainer_state.json`` (written
    by ``Trainer.save_state`` at the end of training and therefore the final,
    load-best-model-aware record) over a ``checkpoint-*/`` copy. Across
    checkpoints we pick the highest step, which is the latest snapshot.
    """
    for root in _candidate_state_dirs(config, adapter_path):
        top = root / "trainer_state.json"
        if top.is_file():
            return top
        # Fall back to the newest checkpoint's state file.
        checkpoints = sorted(
            root.glob("checkpoint-*/trainer_state.json"),
            key=lambda p: _checkpoint_step(p.parent.name),
        )
        if checkpoints:
            return checkpoints[-1]
    return None


def _checkpoint_step(dir_name: str) -> int:
    """Extract the integer step from a ``checkpoint-<step>`` directory name.

    Returns ``-1`` for anything that does not parse so odd names sort first and
    never shadow a real, higher-numbered checkpoint.
    """
    _, _, tail = dir_name.partition("-")
    try:
        return int(tail)
    except ValueError:
        return -1


def read_training_time(config: "ProjectConfig", adapter_path: str | None) -> dict[str, Any] | None:
    """Recover training wall-clock (and FLOPs/epochs/steps) from trainer state.

    Returns ``None`` when no ``trainer_state.json`` exists (e.g. comparing against
    an externally supplied adapter, or before any training has run) so callers can
    render "n/a" instead of a fabricated number. The HF ``Trainer`` appends a
    terminal record to ``log_history`` on ``train()`` completion containing
    ``train_runtime`` — we scan from the end for the last such record and fall
    back to any top-level keys for robustness across transformers versions.
    """
    state_path = _find_trainer_state(config, adapter_path)
    if state_path is None:
        logger.info("No trainer_state.json found; training time will be reported as n/a.")
        return None

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # unreadable/corrupt state file
        logger.warning("Could not parse %s (%s); skipping training-time recovery.", state_path, exc)
        return None

    log_history = data.get("log_history", []) or []
    # The terminal summary is the last log entry that carries "train_runtime".
    summary: dict[str, Any] = {}
    for entry in reversed(log_history):
        if isinstance(entry, dict) and "train_runtime" in entry:
            summary = entry
            break

    def pick(*keys: str) -> Any:
        """First present value across the summary record then the top-level dict."""
        for src in (summary, data):
            for k in keys:
                if k in src and src[k] is not None:
                    return src[k]
        return None

    train_runtime = pick("train_runtime")
    result = {
        "train_runtime_s": float(train_runtime) if train_runtime is not None else None,
        "train_runtime_str": _format_duration(train_runtime),
        "total_flos": pick("total_flos"),
        "epochs": pick("epoch"),
        "steps": pick("step", "global_step"),
        "train_samples_per_second": pick("train_samples_per_second"),
        "train_loss": pick("train_loss"),
        "source": str(state_path),
    }
    logger.info(
        "Recovered training time %s (%s) from %s",
        result["train_runtime_str"],
        f"{result['steps']} steps" if result["steps"] is not None else "unknown steps",
        state_path,
    )
    return result


def _format_duration(seconds: Any) -> str:
    """Render a second count as a compact ``H:MM:SS`` / ``M:SS`` human string."""
    if seconds is None:
        return "n/a"
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "n/a"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------
def compare_models(
    config: "ProjectConfig",
    max_samples: int | None = None,
    *,
    split: str | None = None,
    base_only: bool = False,
) -> dict:
    """Evaluate the base and LoRA-adapted models on one fixed set and diff them.

    Returns a JSON-serializable dict::

        {
          "base":          <evaluate_model result>,
          "finetuned":     <evaluate_model result | None>,   # None only if base_only
          "delta":         {metric: finetuned - base, ...},
          "training_time": {train_runtime_s, ...} | None,
          "adapter_path":  str | None,
          "split":         "<eval split spec>",
          "n_samples":     int,
          "base_only":     bool,
          "decoding":      {max_new_tokens, do_sample, num_beams, ...},
          "sample_ids":    [...],   # the exact, shared eval list
        }

    **Behaviour change (deliberate):** when an adapter is configured but missing,
    or its evaluation raises, this now raises :class:`AdapterUnavailableError`
    instead of quietly returning a base-only result. The old fallback made it
    possible to publish a "comparison" containing one model. A baseline-only run
    is still supported — it just has to be *asked for* via ``base_only=True``
    (``--base-only`` on the CLI), which labels the artifact accordingly.
    """
    # Deferred heavy imports (torch / datasets) — see module docstring.
    from evaluation.evaluate import evaluate_model
    from models import build_model

    adapter_path = config.inference.adapter_path
    if not base_only and not adapter_path:
        raise AdapterUnavailableError(
            "No adapter_path configured: there is nothing to compare the base model "
            "against. Pass --adapter / set config.inference.adapter_path, or run with "
            "--base-only to publish a clearly-labelled baseline-only artifact."
        )
    if base_only and adapter_path:
        logger.warning(
            "--base-only requested while adapter_path=%r is set; the adapter will NOT "
            "be evaluated and the artifact will be labelled base-only.",
            adapter_path,
        )

    # One eval set, shared by both arms: a fair comparison must score the exact
    # same documents, especially under a --max-samples cap.
    split_spec = resolve_eval_split(config, split)
    samples = load_eval_samples(config, split_spec, max_samples)
    if not samples:
        raise RuntimeError(
            f"Eval split {split_spec!r} produced 0 samples; refusing to publish an "
            "empty comparison."
        )

    # Both arms decode with these settings; captured once so the artifact records
    # what was actually used rather than what the reader assumes.
    decoding = decoding_signature(config)
    logger.info("Decoding settings (identical for both arms): %s", decoding)

    # -- Base model (zero-shot) --------------------------------------------
    # build_model(load=True) puts weights + processor on the device. We do NOT
    # apply/attach any adapter here; this is the untouched pretrained backbone.
    model: "VisionDocModel" = build_model(config)
    logger.info("Evaluating BASE model (zero-shot, no adapter) on %d samples...", len(samples))
    base_result = evaluate_model(model, samples, config, profile=True)
    base_result["variant"] = "base"

    # -- Fine-tuned model (base + LoRA adapter) ----------------------------
    finetuned_result: dict | None = None
    if not base_only:
        if not Path(str(adapter_path)).exists():
            raise AdapterUnavailableError(
                f"Adapter path {adapter_path!r} does not exist. Train an adapter (or fix "
                "the path) and re-run; refusing to write a base-only file that would "
                "read as a base-vs-fine-tuned comparison."
            )
        # Reuse the resident base weights: load_adapter wraps `model.model` in a
        # PeftModel in place, so no second multi-GB backbone copy is loaded. This
        # is exactly why base was evaluated first (the wrap is one-way).
        logger.info("Attaching LoRA adapter from %s and re-evaluating...", adapter_path)
        try:
            model.load_adapter(adapter_path)
            finetuned_result = evaluate_model(model, samples, config, profile=True)
        except Exception as exc:  # adapter load or second-arm evaluation failed
            raise AdapterUnavailableError(
                f"Failed to load/evaluate the adapter at {adapter_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finetuned_result["variant"] = "finetuned"

        # The decoding settings come from the same frozen config for both arms;
        # assert it anyway so a future per-arm override cannot slip through.
        if decoding_signature(config) != decoding:
            raise RuntimeError(
                "Decoding settings changed between the base and fine-tuned arms; "
                "the delta would measure the decoder, not the adapter."
            )

    # Paired-comparison integrity: identical documents, identical order.
    _assert_same_sample_list(samples, base_result, finetuned_result)

    delta = _compute_delta(base_result, finetuned_result)
    training_time = read_training_time(config, adapter_path)

    return {
        "base": base_result,
        "finetuned": finetuned_result,
        "delta": delta,
        "training_time": training_time,
        "adapter_path": adapter_path if not base_only else None,
        "split": split_spec,
        "n_samples": len(samples),
        "base_only": bool(base_only),
        "decoding": decoding,
        "sample_ids": [str(s.sample_id) for s in samples],
    }


def _compute_delta(base_result: dict, finetuned_result: dict | None) -> dict[str, float]:
    """Fine-tuned minus base on the headline metric/profile axes.

    Empty when there is no fine-tuned model to diff against. Kept separate from
    the table builder so the raw numbers are available in the returned dict for
    notebooks/tests without re-parsing the DataFrame.
    """
    if finetuned_result is None:
        return {}
    axes: dict[str, tuple[str, str]] = {
        key: ("metrics", key) for key in _RESULTS_METRIC_KEYS
    }
    axes["latency_ms_mean"] = ("profile", "latency_ms_mean")
    axes["peak_memory_gb"] = ("profile", "peak_memory_gb")

    delta: dict[str, float] = {}
    for name, (section, key) in axes.items():
        base_v = _safe_float(base_result.get(section, {}).get(key))
        ft_v = _safe_float(finetuned_result.get(section, {}).get(key))
        if base_v is not None and ft_v is not None:
            delta[name] = ft_v - base_v
    return delta


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------
def _safe_float(value: Any) -> float | None:
    """Coerce to float, returning ``None`` for missing/NaN/unparseable values."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _safe_int(value: Any) -> int | None:
    """Coerce to int, returning ``None`` rather than inventing a 0 for missing."""
    f = _safe_float(value)
    return None if f is None else int(round(f))


def _ci_pair(result: dict | None, key: str) -> tuple[float | None, float | None]:
    """Read ``<key>_ci_low`` / ``<key>_ci_high`` out of an evaluation result.

    Returns ``(None, None)`` when the evaluator did not publish an interval, so
    rendering degrades to a bare point estimate instead of fabricating bounds. A
    missing interval is a fact about the run, not something to paper over.
    """
    if not result:
        return None, None
    metrics = result.get("metrics") or {}
    return _safe_float(metrics.get(f"{key}_ci_low")), _safe_float(metrics.get(f"{key}_ci_high"))


def _build_rows(comparison: dict) -> list[dict[str, Any]]:
    """Turn a comparison dict into per-row records (label, base, finetuned, delta).

    Values are kept numeric (``float`` or ``None`` for missing) so the CSV is
    machine-readable; formatting to strings happens only for the Markdown/stdout
    views. Delta is ``finetuned - base`` for the two-model rows and is left blank
    for footprint/training rows where a signed difference is not meaningful.
    Quality rows additionally carry their bootstrap interval per arm.
    """
    base = comparison.get("base") or {}
    finetuned = comparison.get("finetuned")
    training_time = comparison.get("training_time") or {}

    base_total_m = _safe_float((base.get("params") or {}).get("total_millions"))
    ft_total_m = (
        _safe_float((finetuned.get("params") or {}).get("total_millions"))
        if finetuned
        else None
    )
    trainable_m = _safe_float(_trainable_params(base, finetuned))
    trainable_m = trainable_m / 1e6 if trainable_m is not None else None

    derived = {
        "trainable_lora_m": {"base": 0.0 if finetuned else None, "finetuned": trainable_m},
        "total_params_m": {"base": base_total_m, "finetuned": ft_total_m},
        "training_time_s": {
            "base": None,  # a zero-shot base is not trained
            "finetuned": _safe_float(training_time.get("train_runtime_s")) if finetuned else None,
        },
    }

    rows: list[dict[str, Any]] = []
    for spec in _ROW_SPECS:
        if spec["section"] == "_derived":
            pair = derived[spec["key"]]
            base_v = pair["base"]
            ft_v = pair["finetuned"]
            base_ci: tuple[float | None, float | None] = (None, None)
            ft_ci: tuple[float | None, float | None] = (None, None)
        else:
            base_v = _safe_float(base.get(spec["section"], {}).get(spec["key"]))
            ft_v = (
                _safe_float(finetuned.get(spec["section"], {}).get(spec["key"]))
                if finetuned
                else None
            )
            if spec.get("ci"):
                base_ci = _ci_pair(base, spec["key"])
                ft_ci = _ci_pair(finetuned, spec["key"])
            else:
                base_ci = (None, None)
                ft_ci = (None, None)

        # Delta only where a signed difference is meaningful: both operands present
        # AND the axis has a defined better-direction (metrics/latency/memory).
        if base_v is not None and ft_v is not None and spec["higher_better"] is not None:
            delta_v: float | None = ft_v - base_v
        else:
            delta_v = None

        rows.append(
            {
                "label": spec["label"],
                "kind": spec["kind"],
                "higher_better": spec["higher_better"],
                "metric_key": spec.get("metric_key"),
                "primary": bool(spec.get("primary", False)),
                "secondary": bool(spec.get("secondary", False)),
                "base": base_v,
                "finetuned": ft_v,
                "delta": delta_v,
                "base_ci_low": base_ci[0],
                "base_ci_high": base_ci[1],
                "finetuned_ci_low": ft_ci[0],
                "finetuned_ci_high": ft_ci[1],
            }
        )
    return rows


def _trainable_params(base: dict, finetuned: dict | None) -> int | None:
    """Number of parameters the LoRA fine-tune actually trained.

    Two sources, in order:

    1. ``count_parameters(finetuned)["trainable"]`` — correct when the adapter was
       attached in a trainable state (PEFT keeps ``requires_grad`` on the LoRA
       matrices).
    2. ``finetuned.total - base.total`` — the fallback, because
       :meth:`VisionDocModel.load_adapter` may attach the adapter for *inference*
       (``is_trainable=False``), which zeroes every ``requires_grad`` and makes
       source 1 report 0. The size difference between the wrapped and unwrapped
       model is exactly the low-rank matrices the fine-tune added.

    ``None`` when there is no fine-tuned arm. Note that a *freshly loaded base*
    model reports all weights as trainable, which is meaningless for a zero-shot
    baseline — that number is deliberately never used here.
    """
    if not finetuned:
        return None
    ft_params = finetuned.get("params") or {}
    trainable = _safe_int(ft_params.get("trainable"))
    if trainable:
        return trainable
    base_total = _safe_int((base.get("params") or {}).get("total"))
    ft_total = _safe_int(ft_params.get("total"))
    if base_total is not None and ft_total is not None:
        return max(0, ft_total - base_total)
    return None


def build_comparison_dataframe(comparison: dict) -> "pd.DataFrame":
    """Tidy DataFrame with one row per metric and Base / Fine-tuned / Delta columns.

    Numeric and machine-readable (missing cells are ``NaN``) so it round-trips to
    CSV cleanly and can be handed straight to
    :func:`visualization.plots.plot_model_comparison`. The four CI columns are
    appended *after* the historical four so existing readers (the Streamlit
    research tab, the plotting helper) keep working by name while the CSV still
    carries the intervals — a point estimate without its interval is not a
    publishable number.
    """
    import pandas as pd

    rows = _build_rows(comparison)

    def cell(value: Any) -> float:
        return value if value is not None else float("nan")

    records = [
        {
            "Metric": r["label"],
            "Base": cell(r["base"]),
            "Fine-tuned": cell(r["finetuned"]),
            "Delta": cell(r["delta"]),
            "Base CI low": cell(r["base_ci_low"]),
            "Base CI high": cell(r["base_ci_high"]),
            "Fine-tuned CI low": cell(r["finetuned_ci_low"]),
            "Fine-tuned CI high": cell(r["finetuned_ci_high"]),
        }
        for r in rows
    ]
    return pd.DataFrame(
        records,
        columns=[
            "Metric",
            "Base",
            "Fine-tuned",
            "Delta",
            "Base CI low",
            "Base CI high",
            "Fine-tuned CI low",
            "Fine-tuned CI high",
        ],
    )


# ---------------------------------------------------------------------------
# Canonical artifact: reports/results.json
# ---------------------------------------------------------------------------
def _package_version(name: str) -> str | None:
    """Installed version of ``name`` via importlib.metadata, or ``None``.

    Versions are provenance, not a dependency: this must never raise just because
    an optional package is absent in the environment writing the report.
    """
    try:
        from importlib.metadata import version

        return str(version(name))
    except Exception:  # noqa: BLE001 - missing dist / metadata backport quirks
        return None


def _torch_runtime_info() -> tuple[str | None, str | None]:
    """``(torch_version, gpu_name)`` read from a live torch, best-effort.

    ``torch.__version__`` beats the dist metadata (it reflects the CUDA build
    actually in use, e.g. ``2.4.0+cu121``), and the GPU name is the single most
    useful line of provenance for a run whose latency/VRAM numbers are
    machine-dependent. Both degrade to ``None`` on a torch-less box so this module
    stays importable without torch.
    """
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001 - torch absent (CI, docs build, this repo's lint env)
        return _package_version("torch"), None

    version = str(getattr(torch, "__version__", "")) or _package_version("torch")
    gpu_name: str | None = None
    try:
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            gpu_name = str(torch.cuda.get_device_name(0))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - driver hiccup must not sink the report
        gpu_name = None
    return version, gpu_name


def _quantization_label(config: "ProjectConfig") -> str:
    """How the backbone was loaded: ``"4bit"`` / ``"8bit"`` / ``"none"``.

    Part of provenance because a 4-bit QLoRA number and an fp16 number are not
    the same measurement, and the difference is invisible in the metric table.
    """
    if getattr(config.model, "load_in_4bit", False):
        return "4bit"
    if getattr(config.model, "load_in_8bit", False):
        return "8bit"
    return "none"


def _params_block(comparison: dict) -> dict[str, Any]:
    """The ``params`` section of results.json, with its counting caveat attached.

    ``counting_note`` is taken from :func:`models.count_parameters` when it
    supplies one (it knows whether the count had to undo bitsandbytes' packed
    ``Params4bit`` storage, which otherwise under-reports a 4-bit model by ~2x)
    and falls back to a note that states the same caveat, so the number is never
    published bare.
    """
    base = comparison.get("base") or {}
    finetuned = comparison.get("finetuned")
    base_params = base.get("params") or {}
    ft_params = (finetuned or {}).get("params") or {}

    base_total = _safe_int(base_params.get("total"))
    ft_total = _safe_int(ft_params.get("total"))
    trainable = _trainable_params(base, finetuned)

    trainable_pct = _safe_float(ft_params.get("trainable_pct"))
    if (trainable_pct is None or trainable_pct == 0.0) and trainable and ft_total:
        trainable_pct = 100.0 * trainable / ft_total

    note = ft_params.get("counting_note") or base_params.get("counting_note")
    if not note:
        note = (
            "Counts are nn.Parameter element counts on the loaded module. Under "
            "bitsandbytes 4-bit (Params4bit) storage a raw numel() under-reports the "
            "true parameter count roughly 2x because two weights share one byte; "
            "trainable is the LoRA adapter's own parameters (falling back to "
            "finetuned_total - base_total when the adapter is attached for inference "
            "and therefore reports requires_grad=False)."
        )

    return {
        "base_total": base_total,
        "finetuned_total": ft_total,
        "trainable": trainable,
        "trainable_pct": trainable_pct,
        "counting_note": str(note),
    }


def build_results_json(comparison: dict, config: "ProjectConfig") -> dict[str, Any]:
    """Assemble the canonical ``reports/results.json`` payload.

    This is the one artifact ``scripts/update_readme.py`` reads, so its shape is
    a contract (see ``schema_version``). Three properties are load-bearing:

    * **Every quality number carries its interval.** ``metrics[k]`` holds
      ``value``/``ci_low``/``ci_high`` per arm plus the signed ``delta``. Two arms
      whose intervals overlap have *not* been shown to differ, and the reader can
      only apply that rule if the bounds travel with the value.
    * **Primary vs. secondary is recorded, not implied.** ``field_f1`` is flagged
      ``primary``; EM/ANLS/token-F1 are not, because on serialized-JSON targets
      they are structurally near zero for both arms.
    * **Machine-dependent figures are quarantined under ``environment``.**
      Latency, VRAM and training wall-clock differ per runner; the README's
      CI-checked block must never contain them or the check becomes untenable and
      gets disabled — which is how unchecked numbers get into a README.
    """
    base = comparison.get("base") or {}
    finetuned = comparison.get("finetuned")
    rows_by_key = {r["metric_key"]: r for r in _build_rows(comparison) if r["metric_key"]}

    metrics: dict[str, Any] = {}
    for spec in _ROW_SPECS:
        key = spec.get("metric_key")
        if key is None:
            continue
        row = rows_by_key[key]
        metrics[key] = {
            "base": {
                "value": row["base"],
                "ci_low": row["base_ci_low"],
                "ci_high": row["base_ci_high"],
            },
            "finetuned": {
                "value": row["finetuned"],
                "ci_low": row["finetuned_ci_low"],
                "ci_high": row["finetuned_ci_high"],
            },
            "delta": row["delta"],
            "higher_is_better": bool(spec["higher_better"]),
            "primary": bool(spec.get("primary", False)),
        }

    torch_version, gpu_name = _torch_runtime_info()
    decoding = comparison.get("decoding") or decoding_signature(config)
    base_profile = base.get("profile") or {}
    ft_profile = (finetuned or {}).get("profile") or {}
    training_time = comparison.get("training_time") or {}

    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        # Explicit label so a baseline-only artifact can never be mistaken for a
        # comparison: with base_only=True every "finetuned" value is null.
        "base_only": bool(comparison.get("base_only", finetuned is None)),
        "provenance": {
            "model_id": config.model.model_id,
            "model_type": config.model.model_type,
            "dataset_name": config.data.dataset_name,
            "dataset_id": config.data.dataset_id,
            "eval_split": comparison.get("split"),
            "n_eval": int(comparison.get("n_samples", 0)),
            "seed": int(config.seed),
            "dtype": config.model.torch_dtype,
            "quantization": _quantization_label(config),
            "max_new_tokens": int(decoding.get("max_new_tokens", config.inference.max_new_tokens)),
            "do_sample": bool(decoding.get("do_sample", config.inference.do_sample)),
            "num_beams": int(decoding.get("num_beams", config.inference.num_beams)),
            "adapter_path": comparison.get("adapter_path"),
            "torch_version": torch_version,
            "transformers_version": _package_version("transformers"),
            "peft_version": _package_version("peft"),
            "gpu_name": gpu_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": metrics,
        "params": _params_block(comparison),
        # MACHINE-DEPENDENT — never render these into the CI-checked README block.
        "environment": {
            "latency_ms_base": _safe_float(base_profile.get("latency_ms_mean")),
            "latency_ms_finetuned": _safe_float(ft_profile.get("latency_ms_mean")),
            "peak_vram_gb_base": _safe_float(base_profile.get("peak_memory_gb")),
            "peak_vram_gb_finetuned": _safe_float(ft_profile.get("peak_memory_gb")),
            "train_runtime_s": _safe_float(training_time.get("train_runtime_s")),
        },
        "failures": {
            "base_generation_failures": int(
                (base.get("failures") or {}).get("generation_failures", 0)
            ),
            "finetuned_generation_failures": int(
                ((finetuned or {}).get("failures") or {}).get("generation_failures", 0)
            ),
        },
    }


def write_results_json(comparison: dict, config: "ProjectConfig", path: Path) -> Path:
    """Serialize :func:`build_results_json` to ``path`` and return it."""
    payload = build_results_json(comparison, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    logger.info(
        "Wrote canonical results artifact %s (schema_version=%d, n_eval=%d, base_only=%s)",
        path,
        payload["schema_version"],
        payload["provenance"]["n_eval"],
        payload["base_only"],
    )
    return path


# ---------------------------------------------------------------------------
# Human-readable rendering (Markdown + stdout)
# ---------------------------------------------------------------------------
def _format_cell(value: Any, kind: str, signed: bool = False) -> str:
    """Format one numeric cell for display according to its ``kind``.

    ``signed`` prefixes a ``+`` on non-negative numbers so the Delta column reads
    as a clear gain/loss. Missing values render as an em dash.
    """
    v = _safe_float(value)
    if v is None:
        return "—"
    if kind == "ratio":
        return f"{v:+.4f}" if signed else f"{v:.4f}"
    if kind == "ms":
        return f"{v:+.1f}" if signed else f"{v:.1f}"
    if kind == "gb":
        return f"{v:+.3f}" if signed else f"{v:.3f}"
    if kind == "millions":
        return f"{v:+.3f}" if signed else f"{v:.3f}"
    if kind == "seconds":
        return f"{v:+.1f}" if signed else f"{v:.1f}"
    return f"{v:+.4f}" if signed else f"{v:.4f}"


def _format_with_ci(value: Any, ci_low: Any, ci_high: Any, kind: str) -> str:
    """``"0.6123 [0.5710, 0.6534]"`` — the point estimate and its interval.

    Falls back to the bare estimate when the evaluator published no interval,
    which is honest about what is known rather than implying a precision the run
    did not measure.
    """
    point = _format_cell(value, kind)
    low = _safe_float(ci_low)
    high = _safe_float(ci_high)
    if point == "—" or low is None or high is None:
        return point
    return f"{point} [{_format_cell(low, kind)}, {_format_cell(high, kind)}]"


def _delta_annotation(row: dict[str, Any]) -> str:
    """A ▲/▼ improvement marker appended to the Delta cell in Markdown/stdout.

    Uses the axis's ``higher_better`` flag so a latency *drop* and a field-F1
    *rise* both read as improvements, which a bare signed number cannot convey.
    """
    delta = _safe_float(row["delta"])
    if delta is None or row["higher_better"] is None or delta == 0.0:
        return ""
    improved = (delta > 0) == bool(row["higher_better"])
    return " ▲" if improved else " ▼"


def _overlap_note(row: dict[str, Any]) -> str:
    """Flag a delta whose two 95% intervals overlap as not-yet-demonstrated.

    Overlapping intervals are the single most common way a small-n result gets
    over-claimed. Marking it in the table itself means the caveat cannot be
    dropped when someone quotes the row.
    """
    lo_b, hi_b = _safe_float(row["base_ci_low"]), _safe_float(row["base_ci_high"])
    lo_f, hi_f = _safe_float(row["finetuned_ci_low"]), _safe_float(row["finetuned_ci_high"])
    if None in (lo_b, hi_b, lo_f, hi_f):
        return ""
    overlaps = not (hi_b < lo_f or hi_f < lo_b)  # type: ignore[operator]
    return " (CIs overlap)" if overlaps else ""


def _render_markdown(comparison: dict, rows: list[dict[str, Any]]) -> str:
    """Full ``comparison.md`` document: context header, table, training details."""
    base = comparison.get("base") or {}
    finetuned = comparison.get("finetuned")
    training_time = comparison.get("training_time")
    device = base.get("device") or (base.get("profile") or {}).get("device", "unknown")
    base_only = bool(comparison.get("base_only", finetuned is None))
    decoding = comparison.get("decoding") or {}

    lines: list[str] = []
    lines.append("# VisionDoc AI — Base vs. Fine-tuned Comparison")
    lines.append("")
    if base_only:
        lines.append(
            "> **BASELINE-ONLY RUN.** No adapter was evaluated (`--base-only`). "
            "Every Fine-tuned cell below is empty by construction. This file is "
            "**not** a base-vs-fine-tuned comparison."
        )
        lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Eval split | {comparison.get('split', '?')} |")
    lines.append(f"| Samples (n) | {comparison.get('n_samples', '?')} |")
    lines.append(f"| Adapter | {comparison.get('adapter_path') or '(base only, no adapter)'} |")
    lines.append(f"| Device | {device} |")
    lines.append(f"| Fine-tuned evaluated | {'no (base-only run)' if base_only else 'yes'} |")
    if decoding:
        lines.append(
            "| Decoding (both arms) | "
            f"max_new_tokens={decoding.get('max_new_tokens')}, "
            f"do_sample={decoding.get('do_sample')}, "
            f"num_beams={decoding.get('num_beams')}, "
            f"seed={decoding.get('seed')} |"
        )
    lines.append(
        f"| Generation failures | base={_failure_count(base)}, "
        f"fine-tuned={_failure_count(finetuned)} |"
    )
    lines.append("")

    # Metric table. Quality rows render "value [ci_low, ci_high]".
    lines.append(f"| Metric | Base (n={comparison.get('n_samples', '?')}) | Fine-tuned | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for r in rows:
        if r["metric_key"]:
            base_cell = _format_with_ci(r["base"], r["base_ci_low"], r["base_ci_high"], r["kind"])
            ft_cell = _format_with_ci(
                r["finetuned"], r["finetuned_ci_low"], r["finetuned_ci_high"], r["kind"]
            )
            delta_cell = (
                _format_cell(r["delta"], r["kind"], signed=True)
                + _delta_annotation(r)
                + _overlap_note(r)
            )
        else:
            base_cell = _format_cell(r["base"], r["kind"])
            ft_cell = _format_cell(r["finetuned"], r["kind"])
            delta_cell = _format_cell(r["delta"], r["kind"], signed=True) + _delta_annotation(r)
        lines.append(f"| {r['label']} | {base_cell} | {ft_cell} | {delta_cell} |")
    lines.append("")
    lines.append(
        "_Quality cells are `point estimate [95% bootstrap CI]` over the n samples "
        "above. **Field F1 is the PRIMARY metric.** "
        + SECONDARY_METRIC_NOTE
        + "_"
    )
    lines.append("")
    lines.append(
        "_▲ marks an improvement in the metric's preferred direction "
        "(higher field/EM/ANLS/F1, lower latency/memory). `(CIs overlap)` means the "
        "two arms' intervals intersect: at this sample size the difference has **not** "
        "been demonstrated, whatever the sign of the delta. Trainable LoRA params = "
        "weights the adapter trained; the base is scored zero-shot. Latency, memory "
        "and training time are machine-dependent and are excluded from the README._"
    )
    lines.append("")

    # Training-run provenance.
    lines.append("## Training run")
    if training_time:
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Wall-clock | {training_time.get('train_runtime_str', 'n/a')} |")
        runtime_s = training_time.get("train_runtime_s")
        lines.append(f"| Wall-clock (s) | {runtime_s if runtime_s is not None else 'n/a'} |")
        lines.append(f"| Epochs | {training_time.get('epochs', 'n/a')} |")
        lines.append(f"| Steps | {training_time.get('steps', 'n/a')} |")
        total_flos = training_time.get("total_flos")
        if total_flos:
            # 1 PFLOP = 1e15 FLOPs; a compact figure for the training budget.
            lines.append(f"| Total compute | {float(total_flos) / 1e15:.2f} PFLOPs |")
        lines.append(f"| Source | `{training_time.get('source', '?')}` |")
    else:
        lines.append("")
        lines.append(
            "_No `trainer_state.json` was found for this run, so training time "
            "could not be recovered._"
        )
    lines.append("")
    return "\n".join(lines)


def _failure_count(result: dict | None) -> str:
    """Generation-failure count for a result, or ``n/a`` when the arm is absent.

    Surfaced in the Markdown header because failed generations are scored as
    empty predictions: they are indistinguishable from confident wrong answers in
    the metric table, and a run with many of them is not a measurement.
    """
    if not result:
        return "n/a"
    return str(int((result.get("failures") or {}).get("generation_failures", 0)))


def _render_stdout_table(rows: list[dict[str, Any]]) -> str:
    """A monospaced, aligned table for the terminal (no pandas display deps)."""
    headers = ("Metric", "Base", "Fine-tuned", "Delta")
    table_rows: list[tuple[str, str, str, str]] = []
    for r in rows:
        if r["metric_key"]:
            base_cell = _format_with_ci(r["base"], r["base_ci_low"], r["base_ci_high"], r["kind"])
            ft_cell = _format_with_ci(
                r["finetuned"], r["finetuned_ci_low"], r["finetuned_ci_high"], r["kind"]
            )
        else:
            base_cell = _format_cell(r["base"], r["kind"])
            ft_cell = _format_cell(r["finetuned"], r["kind"])
        table_rows.append(
            (
                r["label"],
                base_cell,
                ft_cell,
                _format_cell(r["delta"], r["kind"], signed=True) + _delta_annotation(r),
            )
        )

    # Column widths from the widest cell (header included) for clean alignment.
    cols = list(zip(headers, *table_rows)) if table_rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt_row(cells: Sequence[str]) -> str:
        first = str(cells[0]).ljust(widths[0])
        rest = "  ".join(str(c).rjust(widths[i + 1]) for i, c in enumerate(cells[1:]))
        return f"{first}  {rest}"

    sep = "-" * (sum(widths) + 2 * len(widths))
    out = [fmt_row(headers), sep]
    out += [fmt_row(tr) for tr in table_rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m research.compare``."""
    parser = argparse.ArgumentParser(
        description="Compare a base VLM against its LoRA fine-tune on a fixed eval set.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (defaults to $VISIONDOC_CONFIG or configs/default.yaml).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter path; overrides config.inference.adapter_path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on evaluated samples (default: config.data.max_eval_samples).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Eval split spec; '+' or ',' pools splits into one fixed set, e.g. "
            "'validation+test'. Default: config.data.eval_split, else "
            "config.data.test_split."
        ),
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help=(
            "Deliberately evaluate ONLY the base model and write a clearly-labelled "
            "baseline-only artifact. Without this flag a missing/unloadable adapter is "
            "a hard error, never a silent downgrade to base-only."
        ),
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help=(
            "Path for the canonical machine-readable artifact "
            f"(default: <report_dir>/{RESULTS_JSON_NAME})."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, str | None]:
    """Run the comparison, write the artifacts under ``reports/``, print the table.

    Returns a dict of output artifact paths (``csv``/``md``/``chart``/``json``/
    ``results``) so shell wrappers and tests can locate them; ``chart`` is ``None``
    if the plotting dependency/module was unavailable. Mirrors
    :func:`evaluation.evaluate.main`'s wiring order (load_config -> seed -> device
    summary -> work -> report) so the numbers are directly comparable to a
    standalone evaluation run.

    Raises ``SystemExit(2)`` when an adapter was requested but could not be loaded
    or evaluated. That non-zero exit is the whole point: a CI job or a notebook
    cell must not carry on and commit a results file that is secretly base-only.
    """
    # Deferred imports: torch/datasets/pandas are only needed for a real run.
    from configs import load_config
    from utils.device import log_device_summary
    from utils.seed import set_seed

    from dataclasses import replace

    args = _build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    # CLI adapter wins over the config so one config can benchmark many adapters.
    # ProjectConfig is frozen -> rebuild immutably instead of assigning.
    if args.adapter is not None:
        config = replace(config, inference=replace(config.inference, adapter_path=args.adapter))

    set_seed(config.seed)
    log_device_summary(config.device, config.model.torch_dtype)

    max_samples = args.max_samples if args.max_samples is not None else config.data.max_eval_samples

    try:
        comparison = compare_models(
            config, max_samples, split=args.split, base_only=args.base_only
        )
    except AdapterUnavailableError as exc:
        # Loud, non-zero, and no artifact written: the previous behaviour (warn,
        # write comparison.*, exit 0) published a one-model file that looked like
        # a two-model result.
        logger.error("Comparison aborted: %s", exc)
        logger.error(
            "No comparison artifacts were written. Re-run with a valid --adapter, or "
            "pass --base-only if you genuinely want a baseline-only report."
        )
        raise SystemExit(2) from exc

    rows = _build_rows(comparison)
    dataframe = build_comparison_dataframe(comparison)

    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    md_path = out_dir / "comparison.md"
    chart_path = out_dir / "comparison_bars.png"
    results_path = (
        Path(args.results_json) if args.results_json else out_dir / RESULTS_JSON_NAME
    )

    # results.json first: it is the canonical artifact every downstream consumer
    # (scripts/update_readme.py) reads, so it must exist even if the optional
    # rendering steps below hit a snag.
    write_results_json(comparison, config, results_path)
    # CSV: the machine-readable table (raw numeric cells + CI columns).
    dataframe.to_csv(csv_path, index=False)
    # Markdown: the human-readable, review-friendly artifact.
    md_path.write_text(_render_markdown(comparison, rows), encoding="utf-8")
    # Also persist the full comparison dict so downstream tooling never has to
    # re-run two evaluations to get at the raw metrics/predictions.
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )

    saved_chart = _maybe_plot(dataframe, chart_path)

    # Print the aligned table so CI logs / interactive runs show the headline.
    print(_render_stdout_table(rows))
    print(
        f"\nPRIMARY metric: field_f1 (n={comparison.get('n_samples')}). "
        f"{SECONDARY_METRIC_NOTE}"
    )
    if comparison.get("base_only"):
        print("\nBASE-ONLY RUN: no adapter was evaluated; this is not a comparison.")
    print(f"\nWrote: {results_path}\n       {csv_path}\n       {md_path}")
    if saved_chart:
        print(f"       {saved_chart}")

    logger.info(
        "Comparison complete: results=%s csv=%s md=%s chart=%s",
        results_path,
        csv_path,
        md_path,
        saved_chart,
    )
    return {
        "csv": str(csv_path),
        "md": str(md_path),
        "chart": str(saved_chart) if saved_chart else None,
        "json": str(out_dir / "comparison.json"),
        "results": str(results_path),
    }


def _maybe_plot(dataframe: "pd.DataFrame", chart_path: Path) -> Path | None:
    """Render the grouped base-vs-finetuned bar chart, degrading gracefully.

    ``visualization.plots`` is a sibling leaf module that pulls matplotlib; both
    it and its backend may be absent on a slim image. We therefore import it
    lazily and treat *any* failure (missing module, missing matplotlib, a plotting
    error) as non-fatal — results.json/CSV/Markdown are the artifacts that matter
    and a completed comparison must not be lost to a charting hiccup.
    """
    try:
        from visualization.plots import plot_model_comparison

        # Chart only the comparable [0, 1] quality metrics; latency/memory/params
        # live on wildly different scales and would flatten the bars.
        return plot_model_comparison(dataframe, str(chart_path), metrics=_CHART_METRIC_LABELS)
    except Exception as exc:  # noqa: BLE001 - charting is best-effort by design.
        logger.warning(
            "Skipping comparison bar chart (%s: %s). results.json/CSV/Markdown were "
            "still written.",
            type(exc).__name__,
            exc,
        )
        return None


__all__ = [
    "AdapterUnavailableError",
    "RESULTS_SCHEMA_VERSION",
    "build_comparison_dataframe",
    "build_results_json",
    "compare_models",
    "decoding_signature",
    "load_eval_samples",
    "main",
    "parse_split_spec",
    "read_training_time",
    "resolve_eval_split",
    "write_results_json",
]


if __name__ == "__main__":
    main()
