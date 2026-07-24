"""Base vs. LoRA-fine-tuned model comparison for VisionDoc AI.

Why this module exists
----------------------
Training produces a LoRA adapter; :mod:`evaluation.evaluate` scores *one* model
in isolation. The question a stakeholder actually asks is comparative — "did the
fine-tune earn its keep?" — and answering it well means putting the base
(zero-shot) backbone and the adapted model side by side on the **same** held-out
split, with the **same** metric/latency/memory machinery, and quantifying the
delta on every axis that matters: answer quality (EM / ANLS / token-F1), serving
cost (latency, peak memory), parameter footprint (how tiny the trained adapter
is relative to the frozen backbone), and how long the run took to produce it.

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
* **Both models see identical samples.** The test split is loaded once and reused
  for both evaluations so the comparison is not confounded by different subsets
  when ``--max-samples`` caps the set.
* **Training time is recovered, not re-measured.** The wall-clock of a multi-hour
  fine-tune is not something a comparison script re-runs; the Hugging Face
  ``Trainer`` already persisted it. At the end of training it writes
  ``trainer_state.json`` whose ``log_history`` contains a terminal summary record
  with ``train_runtime`` (seconds), ``total_flos``, the final ``epoch`` and
  ``step``. We locate that file next to the adapter / under the configured
  ``output_dir`` (falling back to the newest ``checkpoint-*/`` state) and read the
  numbers straight out of it — no heavy imports, no re-training.
* **Heavy imports are deferred.** ``models`` / ``evaluation.evaluate`` /
  ``preprocessing.datasets`` pull torch/datasets, and pandas/matplotlib are only
  needed for the CLI's table+chart. They are imported inside the functions that
  use them, so this module's own top-level imports add nothing heavier than the
  shared logger — the pure helpers (training-time recovery, table/Markdown
  rendering) can be exercised without loading a model or the plotting stack.
  ``visualization.plots`` is a sibling leaf module that may be built
  after this one, so it is imported defensively and its absence degrades to a
  warning rather than aborting a completed comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # type-only: keeps runtime import torch/pandas-free.
    import pandas as pd

    from configs.config import ProjectConfig
    from models.base import VisionDocModel

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Row specification for the comparison table
# ---------------------------------------------------------------------------
# A single source of truth for which numbers appear, their display label, where
# to read them from a result dict, how to format them, and whether higher is
# better (used only to annotate the Delta direction in the Markdown/stdout). We
# keep it as data so the CSV/Markdown/chart all stay in lockstep and the order is
# trivial to tweak in one place.
#
# ``kind`` controls formatting: ratio (0..1), ms, gb, millions (M params),
# seconds. ``higher_better`` is None for footprint rows where "better" is not a
# meaningful direction.
_ROW_SPECS: list[dict[str, Any]] = [
    {"label": "Exact Match", "section": "metrics", "key": "exact_match", "kind": "ratio", "higher_better": True},
    {"label": "ANLS", "section": "metrics", "key": "anls", "kind": "ratio", "higher_better": True},
    {"label": "Token F1", "section": "metrics", "key": "token_f1", "kind": "ratio", "higher_better": True},
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
# the bar chart so it does not try to plot milliseconds next to a 0.3 ANLS.
_CHART_METRIC_LABELS = ["Exact Match", "ANLS", "Token F1"]


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
def compare_models(config: "ProjectConfig", max_samples: int | None = None) -> dict:
    """Evaluate the base and LoRA-adapted models on the test split and diff them.

    Returns a JSON-serializable dict::

        {
          "base":          <evaluate_model result>,
          "finetuned":     <evaluate_model result | None>,   # None if no adapter
          "delta":         {metric: finetuned - base, ...},
          "training_time": {train_runtime_s, ...} | None,
          "adapter_path":  str | None,
          "split":         "test",
          "n_samples":     int,
        }

    The adapter is taken from ``config.inference.adapter_path``. When it is unset
    (or points at a non-existent path) only the base model is evaluated and
    ``finetuned`` is ``None`` — a partial comparison is still a useful artifact
    (it shows the zero-shot baseline) and is far better than crashing.
    """
    # Deferred heavy imports (torch / datasets) — see module docstring.
    from evaluation.evaluate import evaluate_model
    from models import build_model
    from preprocessing.datasets import load_samples

    # One test split, shared by both models: a fair comparison must score the
    # exact same documents, especially under a --max-samples cap.
    split = config.data.test_split or "test"
    samples = load_samples(config, split, max_samples)
    logger.info("Loaded %d samples from split %r for comparison.", len(samples), split)

    # -- Base model (zero-shot) --------------------------------------------
    # build_model(load=True) puts weights + processor on the device. We do NOT
    # apply/attach any adapter here; this is the untouched pretrained backbone.
    model: "VisionDocModel" = build_model(config)
    logger.info("Evaluating BASE model (zero-shot, no adapter)...")
    base_result = evaluate_model(model, samples, config, profile=True)
    base_result["variant"] = "base"

    # -- Fine-tuned model (base + LoRA adapter) ----------------------------
    adapter_path = config.inference.adapter_path
    finetuned_result: dict | None = None
    if adapter_path and Path(adapter_path).exists():
        # Reuse the resident base weights: load_adapter wraps `model.model` in a
        # PeftModel in place, so no second multi-GB backbone copy is loaded. This
        # is exactly why base was evaluated first (the wrap is one-way).
        logger.info("Attaching LoRA adapter from %s and re-evaluating...", adapter_path)
        model.load_adapter(adapter_path)
        finetuned_result = evaluate_model(model, samples, config, profile=True)
        finetuned_result["variant"] = "finetuned"
    elif adapter_path:
        logger.warning(
            "Adapter path %r does not exist; reporting base-only comparison.", adapter_path
        )
    else:
        logger.warning(
            "No adapter_path configured; reporting base-only comparison. "
            "Pass --adapter or set config.inference.adapter_path to compare a fine-tune."
        )

    # -- Deltas on the headline axes ---------------------------------------
    delta = _compute_delta(base_result, finetuned_result)

    training_time = read_training_time(config, adapter_path)

    return {
        "base": base_result,
        "finetuned": finetuned_result,
        "delta": delta,
        "training_time": training_time,
        "adapter_path": adapter_path,
        "split": split,
        "n_samples": len(samples),
    }


def _compute_delta(base_result: dict, finetuned_result: dict | None) -> dict[str, float]:
    """Fine-tuned minus base on the headline metric/profile axes.

    Empty when there is no fine-tuned model to diff against. Kept separate from
    the table builder so the raw numbers are available in the returned dict for
    notebooks/tests without re-parsing the DataFrame.
    """
    if finetuned_result is None:
        return {}
    axes = {
        "exact_match": ("metrics", "exact_match"),
        "anls": ("metrics", "anls"),
        "token_f1": ("metrics", "token_f1"),
        "latency_ms_mean": ("profile", "latency_ms_mean"),
        "peak_memory_gb": ("profile", "peak_memory_gb"),
    }
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


def _build_rows(comparison: dict) -> list[dict[str, Any]]:
    """Turn a comparison dict into per-row records (label, base, finetuned, delta).

    Values are kept numeric (``float`` or ``float('nan')`` for missing) so the CSV
    is machine-readable; formatting to strings happens only for the Markdown/stdout
    views. Delta is ``finetuned - base`` for the two-model rows and is left blank
    for footprint/training rows where a signed difference is not meaningful.
    """
    base = comparison.get("base") or {}
    finetuned = comparison.get("finetuned")
    training_time = comparison.get("training_time") or {}

    # Parameter footprint: count_parameters(base) reflects a freshly loaded model
    # whose weights all carry requires_grad, which is NOT a meaningful "trainable"
    # figure for a *zero-shot* baseline — nothing is trained. So we define the base
    # trainable count as 0 and derive the fine-tuned trainable count as the number
    # of parameters the adapter ADDED (finetuned.total - base.total): that
    # difference is exactly the LoRA low-rank matrices, i.e. the only weights the
    # fine-tune actually updated. This tells the true parameter-efficiency story.
    base_total_m = _safe_float((base.get("params") or {}).get("total_millions"))
    ft_total_m = (
        _safe_float((finetuned.get("params") or {}).get("total_millions"))
        if finetuned
        else None
    )
    adapter_m: float | None = None
    if base_total_m is not None and ft_total_m is not None:
        adapter_m = max(0.0, ft_total_m - base_total_m)

    derived = {
        "trainable_lora_m": {"base": 0.0 if finetuned else None, "finetuned": adapter_m},
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
        else:
            base_v = _safe_float(base.get(spec["section"], {}).get(spec["key"]))
            ft_v = (
                _safe_float(finetuned.get(spec["section"], {}).get(spec["key"]))
                if finetuned
                else None
            )

        # Delta only where a signed difference is meaningful: both operands present
        # AND the axis has a defined better-direction (metrics/latency/memory).
        if (
            base_v is not None
            and ft_v is not None
            and spec["higher_better"] is not None
        ):
            delta_v: float | None = ft_v - base_v
        else:
            delta_v = None

        rows.append(
            {
                "label": spec["label"],
                "kind": spec["kind"],
                "higher_better": spec["higher_better"],
                "base": base_v,
                "finetuned": ft_v,
                "delta": delta_v,
            }
        )
    return rows


def build_comparison_dataframe(comparison: dict) -> "pd.DataFrame":
    """Tidy DataFrame with one row per metric and Base / Fine-tuned / Delta columns.

    Numeric and machine-readable (missing cells are ``NaN``) so it round-trips to
    CSV cleanly and can be handed straight to
    :func:`visualization.plots.plot_model_comparison`. Human formatting lives in
    the Markdown/stdout renderers, not here.
    """
    import pandas as pd

    rows = _build_rows(comparison)
    records = [
        {
            "Metric": r["label"],
            "Base": r["base"] if r["base"] is not None else float("nan"),
            "Fine-tuned": r["finetuned"] if r["finetuned"] is not None else float("nan"),
            "Delta": r["delta"] if r["delta"] is not None else float("nan"),
        }
        for r in rows
    ]
    return pd.DataFrame(records, columns=["Metric", "Base", "Fine-tuned", "Delta"])


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


def _delta_annotation(row: dict[str, Any]) -> str:
    """A ▲/▼ improvement marker appended to the Delta cell in Markdown/stdout.

    Uses the axis's ``higher_better`` flag so a latency *drop* and an ANLS *rise*
    both read as improvements, which a bare signed number cannot convey.
    """
    delta = _safe_float(row["delta"])
    if delta is None or row["higher_better"] is None or delta == 0.0:
        return ""
    improved = (delta > 0) == bool(row["higher_better"])
    return " ▲" if improved else " ▼"


def _render_markdown(comparison: dict, rows: list[dict[str, Any]]) -> str:
    """Full ``comparison.md`` document: context header, table, training details."""
    base = comparison.get("base") or {}
    finetuned = comparison.get("finetuned")
    training_time = comparison.get("training_time")
    device = base.get("device") or (base.get("profile") or {}).get("device", "unknown")

    lines: list[str] = []
    lines.append("# VisionDoc AI — Base vs. Fine-tuned Comparison")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Split | {comparison.get('split', '?')} |")
    lines.append(f"| Samples | {comparison.get('n_samples', '?')} |")
    lines.append(f"| Adapter | {comparison.get('adapter_path') or '(base only, no adapter)'} |")
    lines.append(f"| Device | {device} |")
    lines.append(f"| Fine-tuned evaluated | {'yes' if finetuned else 'no'} |")
    lines.append("")

    # Metric table.
    lines.append("| Metric | Base | Fine-tuned | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for r in rows:
        base_cell = _format_cell(r["base"], r["kind"])
        ft_cell = _format_cell(r["finetuned"], r["kind"])
        delta_cell = _format_cell(r["delta"], r["kind"], signed=True) + _delta_annotation(r)
        lines.append(f"| {r['label']} | {base_cell} | {ft_cell} | {delta_cell} |")
    lines.append("")
    lines.append(
        "_▲ marks an improvement in the metric's preferred direction "
        "(higher EM/ANLS/F1, lower latency/memory). "
        "Trainable LoRA params = weights the adapter added, i.e. what the "
        "fine-tune actually trained; the base is scored zero-shot._"
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


def _render_stdout_table(rows: list[dict[str, Any]]) -> str:
    """A monospaced, aligned table for the terminal (no pandas display deps)."""
    headers = ("Metric", "Base", "Fine-tuned", "Delta")
    table_rows: list[tuple[str, str, str, str]] = []
    for r in rows:
        table_rows.append(
            (
                r["label"],
                _format_cell(r["base"], r["kind"]),
                _format_cell(r["finetuned"], r["kind"]),
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
        description="Compare a base VLM against its LoRA fine-tune on the test split.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, str | None]:
    """Run the comparison, write CSV+Markdown+chart under ``reports/``, print it.

    Returns a dict of output artifact paths (``csv``/``md``/``chart``) so shell
    wrappers and tests can locate them; ``chart`` is ``None`` if the plotting
    dependency/module was unavailable. Mirrors :func:`evaluation.evaluate.main`'s
    wiring order (load_config -> seed -> device summary -> work -> report) so the
    numbers are directly comparable to a standalone evaluation run.
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
    comparison = compare_models(config, max_samples)

    rows = _build_rows(comparison)
    dataframe = build_comparison_dataframe(comparison)

    out_dir = Path(config.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    md_path = out_dir / "comparison.md"
    chart_path = out_dir / "comparison_bars.png"

    # CSV: the machine-readable source of truth (raw numeric cells).
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
    print(f"\nWrote: {csv_path}\n       {md_path}")
    if saved_chart:
        print(f"       {saved_chart}")

    logger.info("Comparison complete: csv=%s md=%s chart=%s", csv_path, md_path, saved_chart)
    return {
        "csv": str(csv_path),
        "md": str(md_path),
        "chart": str(saved_chart) if saved_chart else None,
    }


def _maybe_plot(dataframe: "pd.DataFrame", chart_path: Path) -> Path | None:
    """Render the grouped base-vs-finetuned bar chart, degrading gracefully.

    ``visualization.plots`` is a sibling leaf module that pulls matplotlib; both
    it and its backend may be absent on a slim image. We therefore import it
    lazily and treat *any* failure (missing module, missing matplotlib, a plotting
    error) as non-fatal — the CSV/Markdown are the primary artifacts and a
    completed comparison must not be lost to a charting hiccup.
    """
    try:
        from visualization.plots import plot_model_comparison

        # Chart only the comparable [0, 1] quality metrics; latency/memory/params
        # live on wildly different scales and would flatten the bars.
        return plot_model_comparison(dataframe, str(chart_path), metrics=_CHART_METRIC_LABELS)
    except Exception as exc:  # noqa: BLE001 - charting is best-effort by design.
        logger.warning(
            "Skipping comparison bar chart (%s: %s). CSV/Markdown were still written.",
            type(exc).__name__,
            exc,
        )
        return None


if __name__ == "__main__":
    main()
