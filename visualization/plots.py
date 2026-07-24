"""Matplotlib plotting helpers for VisionDoc AI (training, evaluation, research).

Why this module exists
----------------------
Numbers in a ``trainer_state.json`` or an evaluation dict do not persuade anyone
on their own — the story of a fine-tune is told in *pictures*: the loss curves
that show the model actually learning, the validation-metric curve that justifies
early stopping, the confidence histogram that reveals whether the model *knows*
when it is right, the attention overlay that makes "region highlighting" tangible,
and the grouped bar chart that answers the only question a stakeholder cares about
("did the LoRA adapter beat the frozen backbone?"). Centralizing every plot here
means the notebooks, the research comparison CLI (:mod:`research.compare`), and the
report layer all render with one consistent visual language instead of each
re-inventing axes and colors.

Design choices worth calling out
--------------------------------
* **Headless by construction.** The Agg backend is selected *at import time,
  before* ``pyplot`` is imported. Every downstream consumer (Docker image, CI
  runner, a Streamlit worker with no display) then gets a non-interactive backend
  automatically — we never depend on a ``$DISPLAY``. This is why the ``use("Agg")``
  call sits at the very top and not inside a function.
* **Optional deps degrade, never crash.** seaborn (nicer default theme), scipy and
  sklearn are all *optional*: seaborn is imported lazily in :func:`set_style`, and
  where a heavy dependency would normally be reached for (a KDE, a confusion
  matrix) we ship a small NumPy fallback so a plot is *always* produced. A missing
  ``seaborn`` must not stop a training run from writing its loss curve.
* **Every function is a pure "data -> file" transform.** Each creates its output
  parent directory, writes a single tight PNG at 150 dpi, closes the figure (so a
  long batch never leaks figure handles / memory), and returns the ``Path`` it
  wrote. Callers therefore never have to manage matplotlib state.
* **Best-effort on genuinely hard inputs.** Attention tensors from a multimodal VLM
  have no single canonical 2D shape; :func:`plot_attention_map` reduces *whatever*
  it is handed to a 2D saliency map and documents the caveat, and renders an
  explanatory placeholder (rather than raising) when attention was not captured.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

# The backend MUST be chosen before pyplot is imported, so that importing this
# module on a headless box (CI/Docker) never tries to open a GUI backend.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection is intentional)
import numpy as np  # noqa: E402

from utils.logging_utils import get_logger  # noqa: E402

if TYPE_CHECKING:  # type-only imports keep module load free of pandas/PIL.
    import pandas as pd
    from PIL import Image as PILImage

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared visual language
# ---------------------------------------------------------------------------
# A single palette keeps "Base" vs "Fine-tuned" (and train vs eval) colors stable
# across every figure, so a reader learns the encoding once. Colors are the
# color-blind-friendly seaborn "deep"/"muted" hues chosen so red/green
# correctness cues in the prediction grid remain distinguishable.
_C_TRAIN = "#4C72B0"   # train loss / "Base" model
_C_EVAL = "#DD8452"    # eval loss / "Fine-tuned" model
_C_METRIC = "#55A868"  # validation metric curve
_C_GOOD = "#2E8B57"    # correct prediction (green)
_C_BAD = "#C44E52"     # incorrect prediction (red)
_C_ACCENT = "#8172B3"  # secondary accent (mean/median markers)

_DPI = 150  # crisp enough for reports/PR comments without bloating file size.

# Applying seaborn's theme repeatedly is wasteful; guard the one-time import.
_STYLE_APPLIED = False

# Trainer log_history keys that are timing/bookkeeping noise, not learning-quality
# metrics — excluded from the validation-metric curve so we plot ANLS/EM, not
# "eval_samples_per_second".
_EVAL_SKIP_KEYS = {
    "eval_loss",
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "eval_jit_compilation_time",
    "eval_num_input_tokens_seen",
    "eval_progress",
}


def set_style() -> None:
    """Apply a clean, consistent theme for every plot in the project.

    seaborn gives the nicest defaults, but it is optional (a slim inference image
    may omit it), so we import it lazily and fall back to hand-tuned ``rcParams``.
    Either way the *rcParams* below are applied so fonts/grid/spines match whether
    or not seaborn is installed — the goal is that two figures made in different
    environments still look like they belong to the same project.
    """
    global _STYLE_APPLIED
    if not _STYLE_APPLIED:
        try:
            import seaborn as sns

            # whitegrid reads well on the white report background; "notebook"
            # context sizes fonts for embedding in HTML/markdown at ~medium size.
            sns.set_theme(style="whitegrid", context="notebook")
        except Exception:  # pragma: no cover - seaborn optional at runtime
            logger.debug("seaborn unavailable; using matplotlib defaults for style.")
        _STYLE_APPLIED = True

    # Applied unconditionally (and idempotently) so the look is identical with or
    # without seaborn, and so a caller that mutated rcParams gets reset.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 10,
            "legend.frameon": False,
            "figure.autolayout": False,  # we use bbox_inches="tight" on save instead
        }
    )


# ---------------------------------------------------------------------------
# Small internal helpers (kept private; not part of the public contract)
# ---------------------------------------------------------------------------
def _finalize(fig: "plt.Figure", out_path: str | Path) -> Path:
    """Persist ``fig`` to ``out_path`` (tight, 150 dpi), close it, return the path.

    Closing the figure here — rather than trusting the caller — is what makes it
    safe to call these functions in a loop over hundreds of samples without
    leaking Matplotlib figure managers (a classic slow memory bleed).
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug("Wrote figure -> %s", path)
    return path


def _placeholder_figure(text: str, out_path: str | Path, figsize=(8, 4.5)) -> Path:
    """Render a centered explanatory note as an image.

    Used whenever there is genuinely nothing to plot (e.g. attention was not
    captured, or a log had no usable records). Returning a *readable artifact*
    that explains the emptiness is far more debuggable than an exception or a
    zero-byte file, especially when these PNGs are archived unattended.
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        wrap=True,
        fontsize=12,
        color="#444444",
        transform=ax.transAxes,
    )
    return _finalize(fig, out_path)


def _num(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` for NaN / non-numeric / missing.

    Comparison DataFrames use NaN for "metric not available for this model", and
    treating that as ``None`` lets the plotters annotate "n/a" instead of drawing
    a misleading zero-height bar.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _normalize_text(s: Any) -> str:
    """Normalize a string for correctness comparison in the prediction grid.

    Prefers the project's canonical :func:`evaluation.metrics.normalize_text` so
    "grid correct == metric correct", but falls back to a lightweight lower/strip
    so this module never hard-depends on the evaluation package being importable.
    """
    try:
        from evaluation.metrics import normalize_text

        return normalize_text(str(s))
    except Exception:  # pragma: no cover - defensive fallback
        return " ".join(str(s).lower().split())


def _as_display_array(image: Any) -> np.ndarray:
    """Turn any image-like input into an HxWx3 uint8 array for ``imshow``.

    Delegates decoding (path / bytes / base64 / PIL / ndarray) to the shared
    :func:`utils.image_utils.load_image` so behavior matches the rest of the app,
    and degrades to a neutral gray tile if the image cannot be loaded — a broken
    thumbnail should not abort a whole grid of otherwise-valid predictions.
    """
    try:
        from utils.image_utils import load_image

        return np.asarray(load_image(image))
    except Exception:
        logger.debug("Could not load image for display; using placeholder tile.")
        return np.full((96, 96, 3), 224, dtype=np.uint8)


def _truncate(text: Any, limit: int = 60) -> str:
    """Shorten long strings so grid/label captions never overflow their axes."""
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def _load_log_history(log_history: "list[dict] | str | Path") -> list[dict]:
    """Accept either an in-memory log list or a path to ``trainer_state.json``.

    The Hugging Face ``Trainer`` persists a ``trainer_state.json`` whose
    ``log_history`` is a flat list of per-log dicts (train steps carry ``loss``,
    eval steps carry ``eval_loss`` + ``eval_<metric>``). Supporting both the raw
    list *and* the on-disk state file means callers can plot mid-run (from
    ``trainer.state.log_history``) or after the fact (from the saved JSON) with
    the same function.
    """
    if isinstance(log_history, (str, Path)):
        data = json.loads(Path(log_history).read_text(encoding="utf-8"))
        # The file may be the whole TrainerState (dict with "log_history") or an
        # already-extracted list; handle both shapes.
        if isinstance(data, dict):
            return list(data.get("log_history", []))
        if isinstance(data, list):
            return data
        return []
    return list(log_history or [])


def _rec_x(rec: dict, idx: int, use_step: bool) -> float:
    """Choose an x-coordinate for a log record.

    Steps are the most faithful x-axis (uniform training progress); we fall back
    to ``epoch`` and finally the record index so a log that predates HF adding a
    ``step`` key still plots monotonically instead of collapsing to a single x.
    """
    if use_step:
        v = rec.get("step", rec.get("global_step"))
        if v is not None:
            return float(v)
    if rec.get("epoch") is not None:
        return float(rec["epoch"])
    return float(idx)


def plot_training_curves(
    log_history: "list[dict] | str | Path",
    out_dir: str | Path,
) -> list[Path]:
    """Plot train/eval loss and the validation metric(s) from a training log.

    Produces up to two figures under ``out_dir``: a loss curve (train + eval on
    one axis so their gap — the generalization signal — is obvious) and, when the
    log contains evaluation metrics (e.g. ``eval_anls``), a validation-metric
    curve. Returns the list of written paths. If the log is empty/unusable, a
    single annotated placeholder is written and returned, so the function's
    contract ("always returns Paths to real files") holds even for degenerate
    inputs.
    """
    set_style()
    out_dir = Path(out_dir)
    records = _load_log_history(log_history)

    if not records:
        return [_placeholder_figure(
            "No training log records found.\n(trainer_state.json had an empty log_history)",
            out_dir / "training_curves.png",
        )]

    # Steps are preferred only if *some* record actually carries one; otherwise we
    # commit to epochs for the whole axis to avoid mixing incompatible scales.
    use_step = any(("step" in r or "global_step" in r) for r in records)
    x_label = "Training step" if use_step else "Epoch"

    train_x: list[float] = []
    train_loss: list[float] = []
    eval_x: list[float] = []
    eval_loss: list[float] = []
    # metric_key -> (xs, ys); a dict-of-lists so metrics logged on different
    # schedules (or added mid-run) each keep their own aligned x/y pairs.
    metric_series: dict[str, tuple[list[float], list[float]]] = {}

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        x = _rec_x(rec, idx, use_step)

        # A pure training log carries "loss" but never "eval_loss".
        if "loss" in rec and rec["loss"] is not None and "eval_loss" not in rec:
            lv = _num(rec["loss"])
            if lv is not None:
                train_x.append(x)
                train_loss.append(lv)

        # An eval record is any that reports at least one "eval_*" quantity.
        is_eval = any(k.startswith("eval_") for k in rec)
        if is_eval:
            ev = _num(rec.get("eval_loss"))
            if ev is not None:
                eval_x.append(x)
                eval_loss.append(ev)
            for k, v in rec.items():
                if k.startswith("eval_") and k not in _EVAL_SKIP_KEYS:
                    val = _num(v)
                    if val is None:
                        continue
                    xs, ys = metric_series.setdefault(k, ([], []))
                    xs.append(x)
                    ys.append(val)

    paths: list[Path] = []

    # --- Figure 1: loss curves -------------------------------------------------
    if train_loss or eval_loss:
        fig, ax = plt.subplots(figsize=(8, 5))
        if train_loss:
            ax.plot(train_x, train_loss, color=_C_TRAIN, lw=1.8, label="Train loss")
        if eval_loss:
            # Markers on eval because it is logged sparsely (once per eval_steps);
            # a marker-per-point reads better than a near-flat sparse line.
            ax.plot(
                eval_x, eval_loss, color=_C_EVAL, lw=1.8, marker="o", ms=4,
                label="Validation loss",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend(loc="upper right")
        paths.append(_finalize(fig, out_dir / "training_loss_curve.png"))

    # --- Figure 2: validation metric(s) ---------------------------------------
    if metric_series:
        fig, ax = plt.subplots(figsize=(8, 5))
        # A stable palette cycle so re-runs color the same metric the same way.
        cycle = [_C_METRIC, _C_ACCENT, _C_EVAL, _C_TRAIN, _C_BAD]
        for i, (key, (xs, ys)) in enumerate(sorted(metric_series.items())):
            label = key[len("eval_"):].replace("_", " ").upper()
            ax.plot(
                xs, ys, color=cycle[i % len(cycle)], lw=1.8, marker="o", ms=4,
                label=label,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Metric value")
        ax.set_title("Validation Metrics")
        ax.legend(loc="best")
        paths.append(_finalize(fig, out_dir / "training_eval_metrics.png"))

    if not paths:
        # Records existed but had no loss/metric fields (e.g. only LR logs).
        paths.append(_placeholder_figure(
            "Training log contained no loss or metric values to plot.",
            out_dir / "training_curves.png",
        ))
    return paths


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def _confusion_counts(
    y_true: Sequence, y_pred: Sequence, labels: Sequence
) -> np.ndarray:
    """Confusion counts, preferring sklearn but degrading to a NumPy fallback.

    sklearn guarantees identical label ordering/semantics to the rest of the
    evaluation stack, but classification is a *bonus* task here (CORD/SROIE field
    typing) and sklearn may be absent on a slim image — so we hand-roll the same
    count matrix rather than fail to draw the plot.
    """
    try:
        from sklearn.metrics import confusion_matrix

        return np.asarray(confusion_matrix(list(y_true), list(y_pred), labels=list(labels)))
    except Exception:
        index = {lbl: i for i, lbl in enumerate(labels)}
        m = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in index and p in index:
                m[index[t], index[p]] += 1
        return m


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence,
    out_path: str | Path,
) -> Path:
    """Draw a labeled confusion-matrix heatmap for classification-style tasks.

    Cell text is colored for contrast against the (Blues) background so counts
    stay legible in both the light and dark cells — a small detail that makes the
    difference between a figure people read and one they squint at.
    """
    set_style()
    labels = list(labels)
    cm = _confusion_counts(y_true, y_pred, labels)

    fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * len(labels) + 2),
                                    max(4.0, 0.7 * len(labels) + 1.5)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([_truncate(l, 18) for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([_truncate(l, 18) for l in labels])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    # The plot's own grid would clutter a heatmap; disable it for this axes only.
    ax.grid(False)

    # Annotate each cell; switch text color past the mid-point of the color scale
    # so dark cells get white numerals and light cells get dark ones.
    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "#222222",
                fontsize=9,
            )
    return _finalize(fig, out_path)


# ---------------------------------------------------------------------------
# Confidence distribution
# ---------------------------------------------------------------------------
def _gaussian_kde(samples: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """A dependency-free Gaussian KDE (Silverman bandwidth).

    scipy's ``gaussian_kde`` is the obvious tool, but scipy is a heavy optional
    dep; a few lines of NumPy give the same "smoothed density" curve we want for a
    confidence plot without adding it. Returns ``None`` when there is too little
    data (or zero variance) for a KDE to be meaningful — the caller then just
    shows the histogram.
    """
    s = np.asarray(samples, dtype=float)
    if s.size < 2:
        return None
    std = s.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        return None
    # Silverman's rule of thumb; floored so a very tight cluster still gets a
    # visible (not needle-thin) curve.
    bw = max(1.06 * std * s.size ** (-1 / 5), 1e-3)
    u = (grid[:, None] - s[None, :]) / bw
    kernel = np.exp(-0.5 * u**2) / math.sqrt(2 * math.pi)
    return kernel.sum(axis=1) / (s.size * bw)


def plot_confidence_distribution(
    confidences: Sequence[float],
    out_path: str | Path,
) -> Path:
    """Histogram (+ smoothed KDE) of per-prediction confidence scores.

    A calibrated model's confidences should spread across [0, 1], not pile up at
    1.0; this plot is the quickest visual read on that. Mean and median markers
    are drawn because a long left tail (a few very-unsure predictions) moves the
    mean but not the median, and seeing both tells you which is happening.
    """
    set_style()
    vals = np.asarray([c for c in (_num(x) for x in confidences) if c is not None], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    if vals.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No confidence scores to plot.", ha="center", va="center",
                fontsize=12, color="#444444", transform=ax.transAxes)
        return _finalize(fig, out_path)

    # Confidence is a probability, so fix the domain to [0, 1]; a fixed bin edge
    # set makes histograms from different runs directly comparable.
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(vals, bins=bins, density=True, color=_C_TRAIN, alpha=0.55,
            edgecolor="white", label="Histogram")

    grid = np.linspace(0.0, 1.0, 200)
    kde = _gaussian_kde(vals, grid)
    if kde is not None:
        ax.plot(grid, kde, color=_C_EVAL, lw=2.0, label="Density (KDE)")

    mean_v, median_v = float(vals.mean()), float(np.median(vals))
    ax.axvline(mean_v, color=_C_BAD, ls="--", lw=1.5, label=f"Mean = {mean_v:.3f}")
    ax.axvline(median_v, color=_C_ACCENT, ls=":", lw=1.5, label=f"Median = {median_v:.3f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Prediction Confidence Distribution (n={vals.size})")
    ax.legend(loc="upper left")
    return _finalize(fig, out_path)


# ---------------------------------------------------------------------------
# Attention map
# ---------------------------------------------------------------------------
def _tensor_to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor / array-like to a float32 NumPy array.

    Duck-typed (``detach``/``cpu``) rather than importing torch, so this module
    stays torch-free to import while still handling live attention tensors.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float32)


def _attention_to_2d(attentions: Any) -> np.ndarray | None:
    """Reduce arbitrary attention output to a single 2D saliency map (best-effort).

    A multimodal VLM's attentions have no canonical shape: HF returns a tuple over
    layers of ``(batch, heads, q, k)`` tensors, generation returns a nested
    per-step structure, and custom hooks return anything. Rather than special-case
    each, we collapse *whatever* we get by averaging away leading axes until two
    remain; a 1D result (an attention-over-tokens vector) is folded into the
    nearest square grid. This is deliberately approximate — see the public
    function's docstring for the caveat — but it reliably yields *a* map to overlay.
    """
    if attentions is None:
        return None

    # Normalize container types (tuple/list of per-layer tensors) into one array.
    arr: np.ndarray | None
    if isinstance(attentions, (list, tuple)):
        mats: list[np.ndarray] = []
        for a in attentions:
            try:
                mats.append(_tensor_to_numpy(a))
            except Exception:
                continue
        if not mats:
            return None
        ref_shape = mats[-1].shape
        same = [m for m in mats if m.shape == ref_shape]
        try:
            arr = np.mean(np.stack(same), axis=0) if same else mats[-1]
        except Exception:
            arr = mats[-1]
    else:
        try:
            arr = _tensor_to_numpy(attentions)
        except Exception:
            return None

    if arr is None or arr.size == 0:
        return None

    # Average away every leading dimension (batch, heads, layers, query positions)
    # until a 2D map remains.
    while arr.ndim > 2:
        arr = arr.mean(axis=0)

    if arr.ndim == 1:
        n = arr.shape[0]
        side = int(round(math.sqrt(n)))
        if side < 1:
            return None
        # Trim/pad to a perfect square so the vector can be laid out as a grid.
        if side * side > n:
            arr = np.pad(arr, (0, side * side - n), mode="constant")
        else:
            arr = arr[: side * side]
        arr = arr.reshape(side, side)

    if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).any():
        return None
    return arr


def plot_attention_map(
    image: "PILImage.Image | str | Any",
    attentions: Any,
    out_path: str | Path,
) -> Path:
    """Overlay a reduced attention/saliency map on the document image.

    Caveat (by design): a VLM's attention is a high-dimensional
    ``layers x heads x query x key`` object with no single "correct" projection to
    image space. This function makes a *best-effort* 2D reduction (see
    :func:`_attention_to_2d`) and blends it over the image with
    :func:`utils.image_utils.overlay_heatmap`; it is a qualitative "where did the
    model look" aid, not a rigorous saliency method. When ``attentions`` is
    ``None`` (attention capture disabled for speed/memory), an annotated
    placeholder explaining that is written instead of failing.
    """
    set_style()
    heat = _attention_to_2d(attentions)

    if heat is None:
        return _placeholder_figure(
            "Attention was not captured for this prediction.\n"
            "Enable `return_attention` (config.inference) to visualize it.",
            out_path,
        )

    fig, ax = plt.subplots(figsize=(7, 8))
    try:
        from utils.image_utils import load_image, overlay_heatmap

        base = load_image(image)
        overlaid = overlay_heatmap(base, heat, alpha=0.5)
        ax.imshow(np.asarray(overlaid))
    except Exception:
        # If the image cannot be loaded, still show the raw heatmap so the user
        # gets *something* interpretable rather than an error.
        logger.debug("overlay_heatmap failed; showing raw attention grid.")
        ax.imshow(heat, cmap="jet")
    ax.axis("off")
    ax.set_title("Attention Overlay")
    return _finalize(fig, out_path)


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------
def plot_prediction_grid(
    items: list[dict],
    out_path: str | Path,
    max_items: int = 6,
) -> Path:
    """Thumbnail grid of predictions with pred/gold/confidence captions.

    Each cell's frame and caption are green when the prediction matches the gold
    answer and red otherwise (using the *same* normalization as the metrics), so a
    reviewer can scan a page of qualitative results and immediately see the error
    pattern — which document types fail, whether wrong answers are confidently
    wrong, etc. Missing images degrade to a gray tile rather than dropping the row.
    """
    set_style()
    items = list(items or [])[:max_items]

    if not items:
        return _placeholder_figure("No prediction samples to display.", out_path)

    # At most 3 columns keeps thumbnails large enough to read the document text.
    n = len(items)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.8 * nrows))
    # Normalize axes to a flat list regardless of the 1x1 / 1xN / NxM shape.
    axes_flat = np.atleast_1d(axes).ravel().tolist()

    for ax, item in zip(axes_flat, items):
        ax.imshow(_as_display_array(item.get("image")))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        pred = item.get("prediction", "")
        gold = item.get("gold", "")
        conf = _num(item.get("confidence"))
        correct = bool(gold) and _normalize_text(pred) == _normalize_text(gold)
        color = _C_GOOD if correct else _C_BAD

        # A colored frame is the quickest correctness cue; thicken it so it reads
        # even in a small thumbnail.
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        question = item.get("question")
        conf_txt = f"{conf:.2f}" if conf is not None else "n/a"
        caption_lines = []
        if question:
            caption_lines.append(f"Q: {_truncate(question, 48)}")
        caption_lines.append(f"Pred: {_truncate(pred, 48)}")
        caption_lines.append(f"Gold: {_truncate(gold, 48)}")
        caption_lines.append(f"conf={conf_txt}  {'✓' if correct else '✗'}")
        ax.set_xlabel("\n".join(caption_lines), fontsize=8, color=color, loc="left")

    # Blank out any unused cells in the final row so we don't show empty axes.
    for ax in axes_flat[len(items):]:
        ax.axis("off")

    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    return _finalize(fig, out_path)


# ---------------------------------------------------------------------------
# Model comparison (base vs. fine-tuned)
# ---------------------------------------------------------------------------
def plot_model_comparison(
    df: "pd.DataFrame",
    out_path: str | Path,
    metrics: list[str] | None = None,
) -> Path:
    """Grouped bar chart of Base vs. Fine-tuned for the selected metrics.

    Consumes the tidy comparison DataFrame from
    :func:`research.compare.build_comparison_dataframe` (columns ``Metric`` /
    ``Base`` / ``Fine-tuned`` with numeric or NaN cells). ``metrics`` restricts and
    orders the rows shown; the default of quality metrics (EM/ANLS/Token-F1) keeps
    all bars on a comparable [0, 1] scale so we never plot milliseconds next to a
    ratio. Missing (NaN) values are drawn as zero-height bars annotated "n/a" so a
    base-only comparison (no adapter yet) still renders honestly.
    """
    set_style()

    # Build a metric -> (base, finetuned) lookup from the DataFrame rows. We use
    # the pandas API directly (the caller already has pandas since df is a
    # DataFrame) but tolerate absent columns defensively.
    lookup: dict[str, tuple[float | None, float | None]] = {}
    order: list[str] = []
    for _, row in df.iterrows():
        name = str(row.get("Metric")) if hasattr(row, "get") else str(row["Metric"])
        base = _num(row["Base"]) if "Base" in row else None
        ft = _num(row["Fine-tuned"]) if "Fine-tuned" in row else None
        lookup[name] = (base, ft)
        order.append(name)

    # Selection order: caller-provided metric list (filtered to those present),
    # else every row that has at least one numeric value.
    if metrics:
        selected = [m for m in metrics if m in lookup]
    else:
        selected = [m for m in order if any(v is not None for v in lookup[m])]

    if not selected:
        return _placeholder_figure(
            "No comparable metrics available for the base-vs-fine-tuned chart.",
            out_path,
        )

    base_vals = [lookup[m][0] for m in selected]
    ft_vals = [lookup[m][1] for m in selected]
    # None -> 0.0 for bar height, but remember which were missing to annotate them.
    base_plot = [v if v is not None else 0.0 for v in base_vals]
    ft_plot = [v if v is not None else 0.0 for v in ft_vals]

    x = np.arange(len(selected))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(selected) + 2), 5.5))
    bars_base = ax.bar(x - width / 2, base_plot, width, label="Base", color=_C_TRAIN)
    bars_ft = ax.bar(x + width / 2, ft_plot, width, label="Fine-tuned", color=_C_EVAL)

    # Value labels on top of each bar; "n/a" where the underlying value was NaN.
    def _label(bars, raw_vals):
        for bar, raw in zip(bars, raw_vals):
            txt = f"{raw:.3f}" if raw is not None else "n/a"
            ax.annotate(
                txt,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    _label(bars_base, base_vals)
    _label(bars_ft, ft_vals)

    ax.set_xticks(x)
    ax.set_xticklabels([_truncate(m, 22) for m in selected], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Base vs. Fine-tuned Model")
    ax.legend(loc="best")
    # A little headroom above the tallest bar so value labels are not clipped.
    heights = base_plot + ft_plot
    top = max(heights) if heights else 1.0
    ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
    return _finalize(fig, out_path)


__all__ = [
    "set_style",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_confidence_distribution",
    "plot_attention_map",
    "plot_prediction_grid",
    "plot_model_comparison",
]
