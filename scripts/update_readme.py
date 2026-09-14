#!/usr/bin/env python3
"""Rewrite the README's results table from ``reports/results.json`` — or fail if it is stale.

    python scripts/update_readme.py --write    # regenerate the block in README.md
    python scripts/update_readme.py --check    # exit 1 with a diff if it is stale

This is the mechanism that makes a hand-typed number impossible to merge. CI runs
``--check`` on every push: it regenerates the block between ``<!-- EVAL:BEGIN -->``
and ``<!-- EVAL:END -->`` from ``reports/results.json`` (the canonical artifact
written by :mod:`research.compare`) and fails the build if what is committed
differs by a single character. There is no path by which a number reaches the
README except through a real evaluation run.

Three deliberate properties make that check meaningful rather than flaky:

*   **The block contains nothing machine-dependent.** No latency, no VRAM, no
    timestamp, no training wall-clock, no GPU or library version. Those live in
    ``reports/results.json`` under ``environment``/``provenance``. A wall-clock
    number in a committed table would fail the build on a slower CI runner, and
    the usual fix for that is to make the check toothless — so the number never
    goes in.
*   **The block is a pure function of the results file.** Same JSON in, same
    bytes out, on any machine, with no clock and no environment lookup. That is
    what lets a GPU-less ``ubuntu-latest`` runner verify a table produced on a
    Colab T4.
*   **A missing results file is rendered honestly, not filled with placeholders.**
    On a fresh clone with no ``reports/results.json`` the block says, in as many
    words, that no evaluation has been published yet and names the command that
    produces one. ``--check`` is green in that state (CI passes on a fresh clone)
    but still fails the moment the committed block disagrees with whatever state
    the JSON is actually in — including someone pasting plausible-looking
    ``0.xx`` numbers into a repo that has never run an evaluation.

The consumed contract is ``schema_version: 1`` of ``reports/results.json``; see
:func:`research.compare.build_results_json`, which is the only writer.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import sys
from pathlib import Path
from typing import Any

# Repo root = the parent of this ``scripts/`` directory. Defaults are resolved
# against it (not the cwd) so ``make readme-check`` behaves identically whether
# it is invoked from the repo root, a subdirectory, or a CI runner's shell.
REPO_ROOT = Path(__file__).resolve().parents[1]

BEGIN = "<!-- EVAL:BEGIN -->"
END = "<!-- EVAL:END -->"

DEFAULT_RESULTS = REPO_ROOT / "reports" / "results.json"
DEFAULT_README = REPO_ROOT / "README.md"

# The results.json contract this renderer understands. A newer writer must bump
# its ``schema_version``; we refuse to render an unknown one rather than guess at
# a moved field and publish a silently wrong table.
SUPPORTED_SCHEMA_VERSION = 1

# The exact command that produces the artifact. Named in the "no run yet" block
# and in every error message, because a gate that fails without telling you how
# to satisfy it is the kind of gate people delete.
PRODUCE_CMD = "make results"
PRODUCE_CMD_LONG = (
    "python -m research.compare --config configs/benchmark_cord.yaml "
    "--adapter outputs/.../adapter --split validation+test --max-samples 200"
)

# Display labels for the published metric keys, in publication order: the
# field-level scores first because on serialized-JSON targets they are the only
# ones that measure the task. Kept as an ordered mapping so the table order is a
# property of this file rather than of dict ordering in the JSON.
METRIC_LABELS: dict[str, str] = {
    "field_f1": "Field F1",
    "field_precision": "Field Precision",
    "field_recall": "Field Recall",
    "exact_match": "Exact Match",
    "anls": "ANLS",
    "token_f1": "Token F1",
}

# Why EM/ANLS/token-F1 are demoted on this benchmark. Mirrors
# ``research.compare.SECONDARY_METRIC_NOTE`` in substance: a reader who sees a
# near-zero Exact Match must be told *why* in the same eyeful, or they will
# conclude the model is broken.
SECONDARY_NOTE = (
    "Exact Match, ANLS and token-F1 are marked † and are **secondary**: CORD targets are "
    "serialized JSON objects, so whole-string metrics score a JSON blob against a JSON blob "
    "and are structurally pinned near zero for *both* arms — they cannot separate base from "
    "fine-tuned. The field-level scores are the ones that move, which is why Field F1 is the "
    "primary metric."
)

# Provenance line stating this block is generated. Emitted inside the block so it
# travels with the numbers into any diff view.
GENERATED_BANNER = [
    "<!-- Written by scripts/update_readme.py from reports/results.json.",
    "     Do not edit by hand: CI runs `--check` and fails the build when this",
    "     block is stale or hand-edited. Machine-dependent figures (latency,",
    "     VRAM, training wall-clock) are deliberately absent — see results.json. -->",
]


class ReadmeError(RuntimeError):
    """A condition the operator has to fix: bad JSON, bad schema, missing markers.

    Its own type so :func:`main` can turn any of them into exit 1 with a readable
    message instead of a traceback, while a caller embedding this module can
    catch exactly this class.
    """


# ---------------------------------------------------------------------------
# Reading the canonical artifact
# ---------------------------------------------------------------------------
def load_results(path: Path) -> dict[str, Any] | None:
    """Return the parsed ``results.json``, or ``None`` when the file is absent.

    Absence is a *state*, not an error: a fresh clone has never run an
    evaluation, and the honest rendering of that state (see
    :func:`build_unpublished_block`) is what lets the README ship with no
    placeholder numbers in it. Corrupt content and an unknown schema version, by
    contrast, are hard errors — rendering them would mean guessing.
    """
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadmeError(f"{path} exists but could not be read as JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ReadmeError(f"{path} must contain a JSON object, got {type(payload).__name__}")

    version = payload.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ReadmeError(
            f"{path} has schema_version={version!r}, but this renderer understands "
            f"schema_version={SUPPORTED_SCHEMA_VERSION}. Regenerate the artifact with "
            f"`{PRODUCE_CMD}` (or update scripts/update_readme.py to the new contract)."
        )
    return payload


# ---------------------------------------------------------------------------
# Formatting helpers — deterministic, no locale, no clock
# ---------------------------------------------------------------------------
def _safe_float(value: Any) -> float | None:
    """Coerce to ``float``, mapping ``None``/junk/NaN/inf to ``None``.

    Metrics arrive from JSON where a missing arm is ``null`` and a degenerate
    bootstrap can be ``NaN``; both must render as an em dash rather than as
    ``nan``, which in a published table reads like a measured value.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, *, signed: bool = False) -> str:
    """A single metric number at fixed precision, or ``—`` when unknown.

    Three decimals everywhere: enough to separate two arms on a 200-sample set,
    few enough that the table does not imply precision the sample size cannot
    support. ``signed`` forces the ``+`` on deltas so a gain reads as a gain.
    """
    out = _safe_float(value)
    if out is None:
        return "—"
    return f"{out:+.3f}" if signed else f"{out:.3f}"


def _arm_cell(arm: Any) -> str:
    """``"0.412 [0.351, 0.474]"`` — point estimate with its 95% interval.

    Falls back to the bare estimate if the run published no interval, which is
    honest about what was measured instead of inventing bounds.
    """
    if not isinstance(arm, dict):
        return "—"
    point = _fmt(arm.get("value"))
    if point == "—":
        return "—"
    low, high = _safe_float(arm.get("ci_low")), _safe_float(arm.get("ci_high"))
    if low is None or high is None:
        return point
    return f"{point} [{_fmt(low)}, {_fmt(high)}]"


def _delta_cell(metric: dict[str, Any]) -> str:
    """Signed delta, a direction marker, and the CI-overlap caveat.

    ``▲``/``▼`` respects ``higher_is_better`` so the marker means "improved",
    not "went up". ``(CIs overlap)`` is the load-bearing part: overlapping 95%
    intervals mean the difference has **not** been demonstrated at this sample
    size, whatever the sign, and printing that caveat inside the cell means it
    survives being quoted out of context.
    """
    delta = _safe_float(metric.get("delta"))
    if delta is None:
        return "—"

    cell = _fmt(delta, signed=True)
    if delta != 0.0:
        improved = (delta > 0) == bool(metric.get("higher_is_better", True))
        cell += " ▲" if improved else " ▼"

    base, finetuned = metric.get("base"), metric.get("finetuned")
    if isinstance(base, dict) and isinstance(finetuned, dict):
        lo_b, hi_b = _safe_float(base.get("ci_low")), _safe_float(base.get("ci_high"))
        lo_f, hi_f = _safe_float(finetuned.get("ci_low")), _safe_float(finetuned.get("ci_high"))
        if None not in (lo_b, hi_b, lo_f, hi_f) and not (hi_b < lo_f or hi_f < lo_b):  # type: ignore[operator]
            cell += " (CIs overlap)"
    return cell


def _metric_order(metrics: dict[str, Any]) -> list[str]:
    """Known keys in publication order, then any unknown keys the writer added.

    Unknown keys are appended rather than dropped: a metric that
    :mod:`research.compare` publishes but this renderer has never heard of must
    still reach the table, otherwise a schema addition would be invisible to the
    drift gate.
    """
    known = [key for key in METRIC_LABELS if key in metrics]
    extra = [key for key in metrics if key not in METRIC_LABELS]
    return known + extra


def _label(key: str, metric: dict[str, Any]) -> str:
    """Row label: display name, a **primary** marker, or a † secondary marker."""
    name = METRIC_LABELS.get(key, key.replace("_", " "))
    if metric.get("primary"):
        return f"**{name}** *(primary)*"
    if key in {"exact_match", "anls", "token_f1"}:
        return f"{name} †"
    return name


def _int_or_none(value: Any) -> int | None:
    """Coerce to ``int`` for count-like fields, or ``None`` when unusable."""
    out = _safe_float(value)
    return int(out) if out is not None else None


def _millions(count: int | None) -> str:
    """``3,754,750,208`` → ``"3,754.8M"``. Compact, still exact in the raw count."""
    if count is None:
        return "unknown"
    return f"{count / 1e6:,.1f}M"


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------
def build_header(provenance: dict[str, Any], base_only: bool) -> str:
    """The one-line "what was measured" header above the table.

    Everything in it comes from the run's own config — model, dataset, split,
    n, dtype, quantization. Deliberately *not* the GPU name, torch version or
    timestamp, which are equally true provenance but machine-dependent, and so
    belong in results.json rather than in a CI-checked table.
    """
    model = provenance.get("model_id") or "unknown model"
    dataset = provenance.get("dataset_name") or "unknown dataset"
    dataset_id = provenance.get("dataset_id")
    dataset_str = f"{dataset} (`{dataset_id}`)" if dataset_id else str(dataset)
    split = provenance.get("eval_split") or "unknown split"
    n_eval = _int_or_none(provenance.get("n_eval"))
    dtype = provenance.get("dtype") or "unknown dtype"
    quant = provenance.get("quantization") or "none"
    quant_str = "no quantization" if quant == "none" else f"{quant} quantization"

    # The tail clause differs by arm count: claiming "identical decoding for both
    # arms" on a run that evaluated one arm would be a false claim about the
    # experiment, not just clumsy phrasing.
    arms = (
        "**baseline only**: no LoRA adapter was evaluated in this run"
        if base_only
        else "frozen base (zero-shot) vs. the LoRA fine-tuned adapter, on an identical "
        "sample list with identical decoding"
    )
    return (
        f"**`{model}`** on **{dataset_str}**, eval split `{split}`, "
        f"**n = {n_eval if n_eval is not None else '?'}** documents — {arms}; "
        f"{dtype} / {quant_str}."
    )


def build_table(metrics: dict[str, Any]) -> list[str]:
    """The metric table rows. Primary metrics first (see :func:`_metric_order`)."""
    lines = [
        "| Metric | Base (zero-shot) | LoRA fine-tuned | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in _metric_order(metrics):
        metric = metrics.get(key)
        if not isinstance(metric, dict):
            continue
        lines.append(
            f"| {_label(key, metric)} | {_arm_cell(metric.get('base'))} "
            f"| {_arm_cell(metric.get('finetuned'))} | {_delta_cell(metric)} |"
        )
    return lines


def build_caption(results: dict[str, Any]) -> str:
    """The italic caption: n, the interval method, decoding, seed, and the caveat.

    This is the part that makes the numbers *readable as evidence* rather than as
    marketing: a score without its sample size and interval cannot be argued
    with, and a near-zero Exact Match without the serialized-JSON explanation
    will be read as a broken model.
    """
    provenance = results.get("provenance") or {}
    failures = results.get("failures") or {}
    n_eval = _int_or_none(provenance.get("n_eval"))
    seed = _int_or_none(provenance.get("seed"))
    max_new = _int_or_none(provenance.get("max_new_tokens"))
    do_sample = bool(provenance.get("do_sample"))
    num_beams = _int_or_none(provenance.get("num_beams"))
    base_fail = _int_or_none(failures.get("base_generation_failures"))
    ft_fail = _int_or_none(failures.get("finetuned_generation_failures"))

    decoding = (
        f"`max_new_tokens={max_new if max_new is not None else '?'}`, "
        f"`do_sample={str(do_sample).lower()}`, "
        f"`num_beams={num_beams if num_beams is not None else '?'}`"
    )
    # Two-arm phrasing is only true of a two-arm run; a baseline-only artifact
    # must not carry sentences that imply a comparison happened.
    if results.get("base_only"):
        setup = (
            f"_n = {n_eval if n_eval is not None else '?'} documents (baseline arm only). "
            f"Bracketed ranges are **percentile-bootstrap 95% confidence intervals** over "
            f"those n documents. Decoding: {decoding}, seed "
            f"`{seed if seed is not None else '?'}`. "
        )
        fail = f"Generation failures (scored as empty predictions): {base_fail if base_fail is not None else '?'}. "
    else:
        setup = (
            f"_n = {n_eval if n_eval is not None else '?'} documents, the **same fixed sample "
            f"list** for both arms. Bracketed ranges are **percentile-bootstrap 95% confidence "
            f"intervals** over those n documents — where the two arms' intervals overlap, the "
            f"difference has **not** been demonstrated at this sample size, whatever the sign of "
            f"the delta. Decoding was identical for both arms ({decoding}) at seed "
            f"`{seed if seed is not None else '?'}`. "
        )
        fail = (
            f"Generation failures are scored as empty predictions and counted: base "
            f"{base_fail if base_fail is not None else '?'}, fine-tuned "
            f"{ft_fail if ft_fail is not None else '?'}. "
        )
    return f"{setup}{fail}{SECONDARY_NOTE}_"


def build_params_line(results: dict[str, Any]) -> str:
    """The honest parameter line, carrying ``params.counting_note`` verbatim.

    The note is not decoration. Under bitsandbytes 4-bit storage a raw
    ``numel()`` under-reports a model by roughly 2x because two weights share a
    byte, so "3.75B params, 0.8% trainable" is only checkable if the counting
    method travels with the number.
    """
    params = results.get("params") or {}
    trainable = _int_or_none(params.get("trainable"))
    total = _int_or_none(params.get("finetuned_total")) or _int_or_none(params.get("base_total"))
    pct = _safe_float(params.get("trainable_pct"))
    note = str(params.get("counting_note") or "").strip()

    if results.get("base_only") or trainable is None:
        # Nothing was trained in this artifact (or the counter could not tell us
        # how much was): report the footprint, and do not imply an adapter.
        head = (
            f"**Parameters:** {_millions(total)} total in the evaluated model; no trainable "
            f"LoRA parameters are reported for this run."
        )
    else:
        pct_str = f" (**{pct:.3f}%**)" if pct is not None else ""
        head = (
            f"**Trainable parameters:** {_millions(trainable)} of {_millions(total)} "
            f"total{pct_str} — the LoRA adapter only; the backbone stays frozen."
        )
    return f"_{head} {note}_" if note else f"_{head}_"


def build_published_block(results: dict[str, Any]) -> str:
    """The markdown between the two markers for a run that exists."""
    provenance = results.get("provenance") or {}
    metrics = results.get("metrics") or {}
    base_only = bool(results.get("base_only", False))

    lines: list[str] = ["", *GENERATED_BANNER, ""]
    lines.append(build_header(provenance, base_only))
    lines.append("")

    if base_only:
        # A baseline-only artifact is a legitimate run, but it is not a
        # comparison; saying so in bold inside the block is the only thing
        # stopping an empty column from being read as "the fine-tune tied".
        lines.append(
            "> **Baseline-only run.** Every *LoRA fine-tuned* cell below is empty by "
            "construction — no adapter was evaluated. This table is **not** a "
            "base-vs-fine-tuned comparison."
        )
        lines.append("")

    if metrics:
        lines.extend(build_table(metrics))
    else:
        lines.append(
            "**No metrics were published in `reports/results.json`.** The run produced an "
            "artifact with an empty `metrics` object, so there is nothing to report."
        )
    lines.append("")
    lines.append(build_caption(results))
    lines.append("")
    lines.append(build_params_line(results))
    lines.append("")
    lines.append(
        "_Latency, peak VRAM and training wall-clock are machine-dependent and are "
        "deliberately **excluded** from this checked block; they are recorded under "
        "`environment` in [`reports/results.json`](reports/results.json), together with the "
        "GPU name, library versions and run timestamp._"
    )
    lines.append("")
    return "\n".join(lines)


def build_unpublished_block(results_path: Path) -> str:
    """The markdown for "no evaluation has been run yet".

    This block exists so the README can ship with **zero** placeholder numbers.
    The alternative — leaving ``0.xx`` examples in the table "until the real run
    lands" — is how a fabricated number ends up quoted as a result. It also keeps
    ``--check`` meaningful on a fresh clone: the committed block must match this
    text exactly, so nobody can paste numbers into a repo that has never produced
    a results file.
    """
    try:
        rel = results_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = results_path

    return "\n".join(
        [
            "",
            *GENERATED_BANNER,
            "",
            f"**No evaluation run has been published yet.** `{rel.as_posix()}` does not exist in "
            "this checkout, so there are no numbers to show — and this project does not print "
            "placeholder ones.",
            "",
            "To produce the artifact this table is generated from:",
            "",
            "```bash",
            f"{PRODUCE_CMD}      # {PRODUCE_CMD_LONG}",
            "make readme       # regenerate this block from reports/results.json",
            "```",
            "",
            f"Commit the resulting `{rel.as_posix()}` alongside the regenerated block. CI runs "
            "`make readme-check` and fails if the two ever disagree, so the table cannot drift "
            "from the run — and cannot be written by hand.",
            "",
        ]
    )


def build_block(results: dict[str, Any] | None, results_path: Path) -> str:
    """Dispatch to the published or the unpublished rendering."""
    if results is None:
        return build_unpublished_block(results_path)
    return build_published_block(results)


def render_readme(current: str, block: str) -> str:
    """Replace the marked block, leaving every other byte of the README alone."""
    start = current.find(BEGIN)
    end = current.find(END)
    if start == -1 or end == -1 or end < start:
        raise ReadmeError(
            f"README.md must contain the markers {BEGIN} and {END} (in that order) "
            "around the results table"
        )
    head = current[: start + len(BEGIN)]
    tail = current[end:]
    return f"{head}\n{block}\n{tail}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite (or verify) the README results table from reports/results.json. "
            "--check is the CI drift gate."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Rewrite the block in README.md.")
    mode.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 with a diff when the committed block is stale. Used by CI.",
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help=f"Canonical results artifact (default: {DEFAULT_RESULTS.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--readme",
        default=str(DEFAULT_README),
        help=f"README to rewrite (default: {DEFAULT_README.relative_to(REPO_ROOT)}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exit 0 = up to date / written, 1 = stale or operator error."""
    args = _build_arg_parser().parse_args(argv)
    results_path = Path(args.results)
    readme_path = Path(args.readme)

    try:
        results = load_results(results_path)
        if not readme_path.is_file():
            raise ReadmeError(f"{readme_path} does not exist")
        current = readme_path.read_text(encoding="utf-8")
        updated = render_readme(current, build_block(results, results_path))
    except ReadmeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    state = (
        "no published run"
        if results is None
        else f"n={(results.get('provenance') or {}).get('n_eval', '?')}"
        + (", BASELINE-ONLY" if results.get("base_only") else "")
    )

    if args.write:
        if updated == current:
            print(f"{readme_path} already up to date with {results_path} ({state})")
            return 0
        readme_path.write_text(updated, encoding="utf-8")
        print(f"wrote {readme_path} from {results_path} ({state})")
        return 0

    if updated == current:
        print(f"{readme_path} is up to date with {results_path} ({state})")
        return 0

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"{readme_path} (committed)",
        tofile=f"{readme_path} (regenerated from {results_path})",
    )
    sys.stdout.writelines(diff)
    print(
        f"\nSTALE: the {BEGIN} block in {readme_path} does not match {results_path} ({state}). "
        "Run `make readme` and commit the result. Never edit the block by hand.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
