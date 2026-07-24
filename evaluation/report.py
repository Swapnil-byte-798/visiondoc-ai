"""Self-contained HTML (+ JSON) evaluation report rendering.

Why this module looks the way it does
-------------------------------------
An evaluation is only useful if a human can read it. :func:`build_report` turns
the dict from :func:`evaluation.evaluate.evaluate_model` into a single
inline-styled HTML file that opens anywhere — no web server, no external CSS/JS,
no CDN — which matters because these reports get emailed, dropped into PR
comments, and archived under ``reports/`` next to the run. Alongside the HTML we
always dump the raw ``results.json`` so downstream tooling (the research
comparison table, notebooks, dashboards) has a machine-readable source of truth
that never has to scrape the HTML.

Design choices worth calling out:

* **Zero hard dependencies to *render*.** pandas is imported lazily and only for
  :func:`results_to_dataframe`; the HTML itself is built with stdlib string
  formatting so a report can be produced on a slim image that lacks pandas.
* **Everything is HTML-escaped.** Predictions and gold answers are model / dataset
  text and can contain ``<``/``&``/quotes; we escape every interpolated value so
  the report can never be broken (or XSS'd) by document content.
* **Graceful with partial data.** Missing sections (no predictions, no profile)
  degrade to a short note rather than raising, because a failed/empty run should
  still produce a readable artifact explaining that it was empty.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # type-only import; keeps this module torch/pandas-free to load.
    import pandas as pd

    from configs.config import ProjectConfig

logger = get_logger(__name__)

# Human-friendly labels + display ordering for the flat metrics dict. Keeping
# this table here (rather than scattering formatting at call sites) means the
# report column order is stable across runs and easy to tweak in one place.
_METRIC_LABELS: list[tuple[str, str]] = [
    ("exact_match", "Exact Match"),
    ("anls", "ANLS"),
    ("token_f1", "Token F1"),
    ("precision", "Token Precision"),
    ("recall", "Token Recall"),
    ("bleu", "BLEU"),
    ("rouge1", "ROUGE-1"),
    ("rouge2", "ROUGE-2"),
    ("rougeL", "ROUGE-L"),
    ("field_precision", "Field Precision"),
    ("field_recall", "Field Recall"),
    ("field_f1", "Field F1"),
    ("avg_confidence", "Avg Confidence"),
    ("n_extraction", "Extraction Samples"),
    ("n", "Samples (n)"),
]

# Maximum number of per-sample prediction rows to render inline. The full set is
# always available in the JSON dump; the HTML shows a representative head so the
# file stays small enough to open instantly even after a 10k-sample eval.
_MAX_PREDICTION_ROWS = 10


def results_to_dataframe(results: dict) -> "pd.DataFrame":
    """Return the flat metrics as a two-column ``(metric, value)`` DataFrame.

    A tidy long-form frame is the most useful shape for the report table and for
    ad-hoc analysis (``df.set_index('metric')``), and it round-trips cleanly to
    CSV for the research comparison. pandas is imported lazily so importing this
    module never requires it.
    """
    import pandas as pd

    metrics: dict[str, Any] = dict(results.get("metrics", {}))
    # Preserve the curated label order, then append any unexpected extra keys so
    # nothing a future metric adds silently disappears from the table.
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, label in _METRIC_LABELS:
        if key in metrics:
            rows.append({"metric": label, "value": metrics[key]})
            seen.add(key)
    for key, value in metrics.items():
        if key not in seen:
            rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_report(results: dict, config: "ProjectConfig", output_dir: str) -> str:
    """Render ``results`` to ``<output_dir>/evaluation_report.html`` and return its path.

    Also writes ``evaluation_results.json`` beside it (the raw, machine-readable
    results). ``config`` supplies the header context (model id/type, dataset,
    adapter). The directory is created if needed so callers never have to
    pre-make ``reports/``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_results.json"
    html_path = out_dir / "evaluation_report.html"

    # Persist the raw results first: even if HTML assembly hit an edge case, the
    # authoritative numbers are already safely on disk. ``default=str`` guards
    # against any stray non-JSON value (e.g. a Path) slipping into results.
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    html_doc = _render_html(results, config)
    html_path.write_text(html_doc, encoding="utf-8")

    logger.info("Report written: %s (raw JSON: %s)", html_path, json_path)
    return str(html_path)


# ---------------------------------------------------------------------------
# HTML assembly (stdlib only)
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    """Format a metric/profile value for display.

    Floats get 4 significant decimals (metrics live in ``[0, 1]`` or small ms
    ranges where trailing precision is noise); ints and strings pass through.
    """
    if isinstance(value, bool):  # bool is an int subclass — handle before int.
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(rows: list[tuple[str, Any]], header: tuple[str, str] = ("Metric", "Value")) -> str:
    """Build a two-column HTML table from ``(label, value)`` rows (all escaped)."""
    if not rows:
        return "<p class='empty'>No data.</p>"
    body = "\n".join(
        f"<tr><td>{html.escape(str(label))}</td>"
        f"<td class='num'>{html.escape(_fmt(value))}</td></tr>"
        for label, value in rows
    )
    return (
        "<table>\n"
        f"<thead><tr><th>{html.escape(header[0])}</th>"
        f"<th>{html.escape(header[1])}</th></tr></thead>\n"
        f"<tbody>\n{body}\n</tbody>\n</table>"
    )


def _metrics_section(results: dict) -> str:
    """Metrics table, rendered in the curated order with friendly labels."""
    metrics: dict[str, Any] = dict(results.get("metrics", {}))
    if not metrics:
        return "<p class='empty'>No metrics were computed (empty evaluation set).</p>"
    seen: set[str] = set()
    rows: list[tuple[str, Any]] = []
    for key, label in _METRIC_LABELS:
        if key in metrics:
            rows.append((label, metrics[key]))
            seen.add(key)
    for key, value in metrics.items():  # surface any unlabeled extras too.
        if key not in seen:
            rows.append((key, value))
    return _table(rows)


def _profile_section(results: dict) -> str:
    """Serving-cost table from the ProfileResult dict."""
    profile: dict[str, Any] = dict(results.get("profile", {}))
    if not profile:
        return "<p class='empty'>Profiling was not recorded.</p>"
    label_map = {
        "latency_ms_mean": "Latency mean (ms)",
        "latency_ms_p50": "Latency p50 (ms)",
        "latency_ms_p90": "Latency p90 (ms)",
        "throughput_samples_s": "Throughput (samples/s)",
        "peak_memory_gb": "Peak GPU memory (GB)",
        "n_samples": "Samples profiled",
        "device": "Device",
    }
    rows = [(label_map.get(k, k), profile[k]) for k in label_map if k in profile]
    # Include any keys the profiler adds later that we did not anticipate.
    rows += [(k, v) for k, v in profile.items() if k not in label_map]
    return _table(rows)


def _params_section(results: dict) -> str:
    """LoRA parameter-footprint table from count_parameters()."""
    params: dict[str, Any] = dict(results.get("params", {}))
    if not params:
        return "<p class='empty'>Parameter counts unavailable.</p>"
    rows = [
        ("Trainable", params.get("trainable")),
        ("Total", params.get("total")),
        ("Trainable %", params.get("trainable_pct")),
        ("Trainable (M)", params.get("trainable_millions")),
        ("Total (M)", params.get("total_millions")),
    ]
    rows = [(label, value) for label, value in rows if value is not None]
    return _table(rows)


def _predictions_section(results: dict) -> str:
    """Up to ~10 per-sample rows: question, gold, prediction, confidence.

    A ``✓/✗`` hint compares the (case-insensitively) stripped strings for a
    quick visual read; it is only a hint (the real scoring is SQuAD-normalized
    in :mod:`evaluation.metrics`), which the column header makes clear.
    """
    predictions: list[dict[str, Any]] = list(results.get("predictions", []))
    if not predictions:
        return "<p class='empty'>No predictions were produced.</p>"

    shown = predictions[:_MAX_PREDICTION_ROWS]
    rows_html: list[str] = []
    for rec in shown:
        question = html.escape(str(rec.get("question", "")))
        gold = html.escape(str(rec.get("gold", "")))
        pred = html.escape(str(rec.get("prediction", "")))
        conf = rec.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.0
        # Loose visual match hint (not the scored metric — see docstring).
        hit = str(rec.get("prediction", "")).strip().lower() == str(rec.get("gold", "")).strip().lower()
        mark = "<span class='ok'>&#10003;</span>" if hit else "<span class='bad'>&#10007;</span>"
        # Confidence bar width is clamped to [0, 100]% so a stray value can never
        # blow out the layout.
        pct = max(0.0, min(1.0, conf_f)) * 100.0
        conf_cell = (
            f"<div class='bar'><div class='bar-fill' style='width:{pct:.1f}%'></div>"
            f"<span class='bar-label'>{conf_f:.3f}</span></div>"
        )
        rows_html.append(
            "<tr>"
            f"<td>{question}</td>"
            f"<td>{gold}</td>"
            f"<td>{pred}</td>"
            f"<td class='center'>{mark}</td>"
            f"<td>{conf_cell}</td>"
            "</tr>"
        )

    note = ""
    if len(predictions) > len(shown):
        note = (
            f"<p class='note'>Showing {len(shown)} of {len(predictions)} predictions "
            "(full set in evaluation_results.json).</p>"
        )
    return (
        "<table class='preds'>\n"
        "<thead><tr><th>Question</th><th>Gold</th><th>Prediction</th>"
        "<th>Match*</th><th>Confidence</th></tr></thead>\n"
        f"<tbody>\n{''.join(rows_html)}\n</tbody>\n</table>\n"
        f"{note}"
        "<p class='note'>*Match is a loose case-insensitive string hint; "
        "scored metrics use SQuAD-style normalization.</p>"
    )


def _header_context(results: dict, config: "ProjectConfig") -> list[tuple[str, Any]]:
    """Key/value run-context rows for the report header (model, data, device)."""
    device = results.get("device", results.get("profile", {}).get("device", "unknown"))
    rows: list[tuple[str, Any]] = [
        ("Project", getattr(config, "name", "visiondoc-ai")),
        ("Model", getattr(config.model, "model_id", "?")),
        ("Backbone", getattr(config.model, "model_type", "?")),
        ("Dataset", getattr(config.data, "dataset_name", "?")),
        ("Split", results.get("split", "?")),
        ("Adapter", results.get("adapter_path") or getattr(config.inference, "adapter_path", None) or "(base, none)"),
        ("Device", device),
        ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
    ]
    return rows


def _render_html(results: dict, config: "ProjectConfig") -> str:
    """Assemble the full self-contained HTML document string."""
    project = html.escape(str(getattr(config, "name", "visiondoc-ai")))
    header_rows = _header_context(results, config)
    header_table = _table(header_rows, header=("Field", "Value"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Evaluation Report &mdash; {project}</title>
<style>
{_CSS}
</style>
</head>
<body>
<main>
  <h1>VisionDoc AI &mdash; Evaluation Report</h1>
  <section>
    <h2>Run context</h2>
    {header_table}
  </section>
  <section>
    <h2>Quality metrics</h2>
    {_metrics_section(results)}
  </section>
  <section>
    <h2>Inference profile</h2>
    {_profile_section(results)}
  </section>
  <section>
    <h2>Parameter footprint</h2>
    {_params_section(results)}
  </section>
  <section>
    <h2>Sample predictions</h2>
    {_predictions_section(results)}
  </section>
  <footer>Generated by evaluation.report &middot; self-contained, no external assets.</footer>
</main>
</body>
</html>
"""


# Inline stylesheet: deliberately small, system-font, and theme-neutral so the
# report reads well printed, in an email client, or on a laptop with no network.
_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5; color: #1c1e21; background: #f5f6f8;
}
main { max-width: 1000px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 1.25rem; }
h2 { font-size: 1.15rem; margin: 0 0 .5rem; color: #2d3748; }
section {
  background: #fff; border: 1px solid #e2e5e9; border-radius: 10px;
  padding: 1rem 1.25rem; margin-bottom: 1.25rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
table { width: 100%; border-collapse: collapse; font-size: .92rem; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid #edf0f2; vertical-align: top; }
th { font-weight: 600; color: #4a5568; background: #fafbfc; }
td.num, td.center { text-align: right; font-variant-numeric: tabular-nums; }
td.center { text-align: center; }
tr:last-child td { border-bottom: none; }
table.preds td { max-width: 340px; overflow-wrap: anywhere; }
.ok { color: #1a7f37; font-weight: 700; }
.bad { color: #c0362c; font-weight: 700; }
.bar { position: relative; background: #edf0f2; border-radius: 6px; height: 1.2rem; min-width: 90px; }
.bar-fill { position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg,#3b82f6,#2563eb); border-radius: 6px; }
.bar-label { position: relative; z-index: 1; font-size: .78rem; padding-left: .4rem; color: #0b1b3a; }
.empty, .note { color: #718096; font-size: .85rem; margin: .5rem 0 0; }
footer { color: #a0aec0; font-size: .8rem; text-align: center; margin-top: 1rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e4e6eb; background: #18191a; }
  section { background: #242526; border-color: #3a3b3c; box-shadow: none; }
  h2 { color: #cbd5e0; }
  th { color: #a0aec0; background: #2d2e30; }
  th, td { border-bottom-color: #3a3b3c; }
  .bar { background: #3a3b3c; }
  .bar-label { color: #e4e6eb; }
}
"""


__all__ = ["results_to_dataframe", "build_report"]
