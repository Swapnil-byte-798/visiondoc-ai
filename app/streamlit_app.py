"""Interactive Streamlit dashboard for VisionDoc AI.

This is the demo/operator surface over :class:`inference.predictor.DocumentPredictor`
— the same facade the FastAPI service uses, so a question answered here and the
same question answered via ``POST /ask`` go through byte-for-byte identical code
and return identical results. The dashboard adds only *presentation*: it uploads
a document, asks a question, and renders the answer with a confidence meter, a
latency figure, region highlighting, and a live device/GPU panel.

Design constraints that shape this file
---------------------------------------
* **Streamlit re-runs the whole script on every interaction.** That makes two
  things expensive if done naively: (1) loading the multi-GB model, and (2) the
  heavy imports (torch/transformers/PIL) themselves. We solve (1) with
  ``@st.cache_resource`` keyed on ``(model_choice, adapter_path)`` so the model
  is built once per distinct configuration and reused across every re-run and
  across browser sessions. We solve (2) by importing the ML stack *inside*
  functions, never at module top — so a bare ``import`` of this module (or a
  syntax check) never needs torch, and the page shell can render even when the
  ML stack or a GPU is missing.
* **Nothing here may hard-crash the page.** A missing config, an un-loadable
  model, no Tesseract, no GPU, a corrupt upload — each is caught and surfaced as
  an ``st.error``/``st.warning`` so the operator sees *why* rather than a raw
  traceback. Graceful degradation is a first-class requirement of the demo.
* **Base vs. fine-tuned is a first-class toggle.** The whole point of the
  project is to *show* the LoRA adapter helping, so the sidebar lets you flip
  between the zero-shot base model and the fine-tuned adapter, and a dedicated
  research tab renders the offline ``reports/comparison.csv`` produced by
  ``make compare`` (``research/compare.py``).

The script is organized as small, individually-testable helpers plus a
``main()`` that lays out the page. ``main()`` runs only under Streamlit's
``__main__`` execution (``streamlit run``), so importing this module for tests
or tooling has no side effects.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only imports: never trigger the heavy ML stack at runtime.
    from configs.config import ProjectConfig
    from inference.predictor import DocumentPredictor, PredictionResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
# Radio labels are used as cache keys and as the branch that decides whether the
# adapter path is honoured, so they are defined once here and referenced by name
# rather than re-typed as string literals (a typo would silently break caching).
MODEL_CHOICE_BASE = "Base (zero-shot)"
MODEL_CHOICE_LORA = "LoRA fine-tuned"

# A curated menu of document-QA prompts covering the field types the model is
# trained on (invoices/receipts/forms/IDs). The sentinel keeps the "type my own"
# path explicit instead of overloading an empty string. A free-text box below
# always overrides the menu, so power users are never boxed in.
CUSTOM_QUESTION_SENTINEL = "✏️  Custom question…"
SAMPLE_QUESTIONS: list[str] = [
    "What is the total amount?",
    "What is the invoice number?",
    "What is the invoice date?",
    "What is the due date?",
    "Who is the vendor?",
    "What is the tax amount?",
    "What is the merchant name?",
    "What is the name on this document?",
    "What is the document number?",
    "What is the date on this document?",
]

# Session-state keys, namespaced to avoid collisions with widget-owned keys.
_STATE_RESULT = "vd_last_result"
_STATE_IMAGE = "vd_last_image"
_STATE_QUESTION = "vd_last_question"
_STATE_MODEL = "vd_last_model_choice"


# ---------------------------------------------------------------------------
# Cached, lazily-built resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_app_config() -> "ProjectConfig":
    """Load the project config once per server process.

    Cached with ``cache_resource`` (not ``cache_data``) because a
    ``ProjectConfig`` is a live, non-serializable domain object we want shared by
    reference — not hashed/pickled — across every re-run. Any load failure is
    left to propagate so the caller can render a precise ``st.error`` (the sidebar
    still needs *some* config for its defaults, so we do not swallow it here).
    """
    from configs import load_config

    return load_config()


@st.cache_resource(show_spinner="Loading model — first run downloads/initializes weights…")
def get_predictor(model_choice: str, adapter_path: str | None) -> "DocumentPredictor":
    """Build and cache a :class:`DocumentPredictor` for one model configuration.

    Keyed on ``(model_choice, adapter_path)`` so flipping between base and
    fine-tuned — or pointing at a different adapter — builds a *separate* cached
    predictor while re-running the page for unrelated widget changes reuses the
    already-loaded weights. This is the single most important performance lever
    in the app: without it every checkbox toggle would reload the model.

    ``adapter_path`` is expected to already be *resolved* by the caller: for the
    base choice it is ``None`` (so the cache key ignores whatever text sits in the
    adapter box and we never attach an adapter), and for the LoRA choice it is the
    concrete path. We rebuild the config with that adapter and hand it to the
    predictor, which loads the base weights and best-effort-attaches the adapter.
    """
    config = load_app_config()
    # ``ProjectConfig``/``InferenceConfig`` are frozen dataclasses, so we produce a
    # modified *copy* rather than mutating shared state — important because the
    # cached base config is reused for the other model choice too.
    inference = replace(config.inference, adapter_path=adapter_path)
    scoped_config = replace(config, inference=inference)

    from inference.predictor import DocumentPredictor

    # ``from_config`` builds the model and (when adapter_path is set) attaches the
    # adapter best-effort; a bad adapter path degrades to the base model rather
    # than raising, which is exactly the resilience we want in a live demo.
    return DocumentPredictor.from_config(scoped_config)


# ---------------------------------------------------------------------------
# Small pure-ish helpers
# ---------------------------------------------------------------------------
def _effective_adapter_path(model_choice: str, adapter_text: str) -> str | None:
    """Resolve the adapter path the predictor should actually use.

    The base model must *never* honour an adapter — otherwise the whole "base vs.
    fine-tuned" comparison is meaningless — so we force ``None`` for that choice
    regardless of the text box. This also stabilizes the cache key: editing the
    adapter path while "Base" is selected does not thrash the cached base model.
    """
    if model_choice != MODEL_CHOICE_LORA:
        return None
    cleaned = (adapter_text or "").strip()
    return cleaned or None  # empty box => still no adapter (base weights only)


def _load_upload_to_image(uploaded_file: Any) -> "Any":
    """Turn an uploaded image *or* PDF into a single RGB PIL image.

    PDFs are common in document intelligence, but a QA turn answers one page at a
    time, so we rasterize and take the first page — the pragmatic default for a
    demo (multi-page PDFs are handled programmatically by
    ``DocumentPredictor.predict_pdf``). Heavy imports (PIL/pdf backend) are
    deferred here so the page shell loads without them.
    """
    data = uploaded_file.getvalue()
    name = (getattr(uploaded_file, "name", "") or "").lower()
    is_pdf = name.endswith(".pdf") or getattr(uploaded_file, "type", "") == "application/pdf"

    if is_pdf:
        from utils.image_utils import pdf_to_images

        # ``max_pages=1``: we only display/answer the first page here, so there is
        # no reason to pay to rasterize the rest.
        pages = pdf_to_images(data, max_pages=1)
        if not pages:
            raise ValueError("The PDF produced no rasterizable pages.")
        return pages[0]

    from utils.image_utils import load_image

    # ``load_image`` accepts raw bytes and normalizes to RGB.
    return load_image(data)


def _result_display_image(result: "PredictionResult", fallback_image: "Any") -> "Any":
    """Pick the image to show for a result: highlighted page if we have one.

    When OCR localized the answer, the predictor returns a base64 PNG with the
    region boxed; decoding and showing that gives the operator visual grounding.
    Otherwise we fall back to the original upload so the layout never shows a
    blank — grounding is a bonus, not a precondition for rendering.
    """
    if result.highlighted_image_b64:
        try:
            from utils.image_utils import base64_to_pil

            return base64_to_pil(result.highlighted_image_b64)
        except Exception:
            # A decode hiccup must not blank the page; log and fall back.
            logger.debug("Failed to decode highlighted image; showing original.", exc_info=True)
    return fallback_image


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(config: "ProjectConfig | None") -> dict[str, Any]:
    """Render every input control and return the collected selections.

    Returning a plain dict (rather than reading widget state elsewhere) keeps the
    layout code declarative and makes the run flow easy to follow: the sidebar is
    the *only* place that defines inputs, ``main`` consumes them.
    """
    st.sidebar.header("Document & question")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a document",
        type=["png", "jpg", "jpeg", "pdf"],
        help="An image (PNG/JPG) or a PDF. For PDFs, the first page is used.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Model")

    # Base vs. fine-tuned is the headline toggle of the whole app.
    model_choice = st.sidebar.radio(
        "Which model?",
        options=[MODEL_CHOICE_BASE, MODEL_CHOICE_LORA],
        index=0,
        help=(
            "Base runs the raw Vision-Language model zero-shot. "
            "LoRA fine-tuned attaches the trained adapter at the path below."
        ),
    )

    # Default the adapter path from config so a configured deployment 'just works'
    # without the operator hunting for the path. Disabled for the base choice to
    # make it visually obvious the field is inert there.
    default_adapter = ""
    if config is not None and config.inference.adapter_path:
        default_adapter = config.inference.adapter_path
    adapter_text = st.sidebar.text_input(
        "Adapter path",
        value=default_adapter,
        disabled=(model_choice != MODEL_CHOICE_LORA),
        help="Filesystem path to the saved LoRA adapter directory (…/adapter).",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Question")

    sample_question = st.sidebar.selectbox(
        "Sample questions",
        options=[CUSTOM_QUESTION_SENTINEL, *SAMPLE_QUESTIONS],
        index=1,  # default to a useful concrete question, not the sentinel
        help="Pick a common prompt, or choose 'Custom question' and type your own below.",
    )
    free_text = st.sidebar.text_input(
        "…or type your own question",
        value="",
        placeholder="e.g. What is the grand total including tax?",
        help="Anything typed here overrides the sample question above.",
    )

    # Free text always wins when present; otherwise use the sample (unless the
    # user explicitly selected the 'custom' sentinel, in which case there is no
    # effective question until they type one).
    if free_text.strip():
        question = free_text.strip()
    elif sample_question != CUSTOM_QUESTION_SENTINEL:
        question = sample_question
    else:
        question = ""

    highlight = st.sidebar.checkbox(
        "Highlight answer region",
        value=True,
        help="Uses OCR to box where the answer appears on the page (needs Tesseract).",
    )

    st.sidebar.divider()
    run_clicked = st.sidebar.button("Run", type="primary", use_container_width=True)

    # The device panel lives in the sidebar and is recomputed on every re-run, so
    # it always reflects current GPU memory after the latest inference.
    render_device_panel(config)

    return {
        "uploaded_file": uploaded_file,
        "model_choice": model_choice,
        "adapter_text": adapter_text,
        "question": question,
        "highlight": highlight,
        "run_clicked": run_clicked,
    }


def render_device_panel(config: "ProjectConfig | None") -> None:
    """Render the compute/GPU panel in the sidebar (best-effort).

    Uses :func:`utils.device.get_device_info` for the static picture (device,
    dtype, GPU count, total VRAM) and :func:`utils.device.gpu_memory_stats` for
    the live allocation. Both are imported lazily because they pull torch; if
    torch is missing (e.g. the shell that only syntax-checks) we degrade to a
    short notice instead of erroring.
    """
    st.sidebar.divider()
    st.sidebar.subheader("Compute")

    preference = config.device if config is not None else "auto"
    dtype = config.model.torch_dtype if config is not None else "bfloat16"

    try:
        from utils.device import get_device_info, gpu_memory_stats

        info = get_device_info(preference, dtype)
        mem = gpu_memory_stats()
    except Exception as exc:
        # No torch / no device access: say so plainly rather than crash the page.
        st.sidebar.info(f"Device info unavailable: {exc}")
        return

    # Compact metrics: device + dtype on one row, VRAM on the next.
    top = st.sidebar.columns(2)
    top[0].metric("Device", info.device.upper())
    top[1].metric("dtype", info.dtype)
    st.sidebar.caption(f"{info.device_name} · {info.n_gpus} GPU(s)")

    if info.device == "cuda":
        # Only meaningful on CUDA; gpu_memory_stats returns zeros elsewhere.
        bottom = st.sidebar.columns(2)
        bottom[0].metric("Allocated", f"{mem['allocated_gb']:.2f} GB")
        bottom[1].metric("Reserved", f"{mem['reserved_gb']:.2f} GB")
        if info.total_memory_gb > 0:
            frac = min(max(mem["reserved_gb"] / info.total_memory_gb, 0.0), 1.0)
            st.sidebar.progress(frac, text=f"VRAM {mem['reserved_gb']:.1f}/{info.total_memory_gb:.1f} GB")
    else:
        st.sidebar.caption("Running on CPU/MPS — GPU memory stats not applicable.")


# ---------------------------------------------------------------------------
# Inference run + result rendering
# ---------------------------------------------------------------------------
def run_inference(selections: dict[str, Any]) -> None:
    """Execute one QA turn from the sidebar selections and store the result.

    Validation, model build, image load, and generation are each wrapped so a
    failure produces a targeted ``st.error`` and leaves any prior result intact.
    The result is stashed in ``st.session_state`` so it survives the re-runs
    triggered by unrelated widgets (e.g. the device panel refreshing).
    """
    uploaded_file = selections["uploaded_file"]
    question = selections["question"]
    model_choice = selections["model_choice"]

    if uploaded_file is None:
        st.warning("Upload a document (PNG/JPG/PDF) in the sidebar first.")
        return
    if not question:
        st.warning("Enter a question (pick a sample or type your own) before running.")
        return

    adapter_path = _effective_adapter_path(model_choice, selections["adapter_text"])
    if model_choice == MODEL_CHOICE_LORA and adapter_path is None:
        st.warning(
            "LoRA selected but no adapter path is set — the base model will answer. "
            "Set an adapter path in the sidebar to use the fine-tuned model."
        )

    # 1) Build (or fetch cached) predictor. The spinner covers the potentially
    #    slow first-time weight load; subsequent runs hit the cache instantly.
    try:
        predictor = get_predictor(model_choice, adapter_path)
    except Exception as exc:
        logger.exception("Failed to build predictor")
        st.error(
            f"Could not load the model: {exc}\n\n"
            "Check that the model weights (and any adapter path) are available and "
            "that the ML dependencies are installed."
        )
        return

    # 2) Decode the upload to a single RGB page.
    try:
        image = _load_upload_to_image(uploaded_file)
    except Exception as exc:
        logger.exception("Failed to decode upload")
        st.error(f"Could not read the uploaded file: {exc}")
        return

    # 3) Generate. This is the only genuinely heavy call; keep it under a spinner.
    try:
        with st.spinner("Thinking about your document…"):
            result = predictor.answer(image, question, highlight=selections["highlight"])
    except Exception as exc:
        logger.exception("Inference failed")
        st.error(f"Inference failed: {exc}")
        return

    # Persist for rendering across subsequent re-runs.
    st.session_state[_STATE_RESULT] = result
    st.session_state[_STATE_IMAGE] = image
    st.session_state[_STATE_QUESTION] = question
    st.session_state[_STATE_MODEL] = model_choice


def render_results() -> None:
    """Render the most recent result (if any) in a two-column layout.

    Reads from ``st.session_state`` rather than a fresh run so the answer stays on
    screen while the operator tweaks unrelated controls. When nothing has been run
    yet, shows a friendly placeholder instead of an empty page.
    """
    result: "PredictionResult | None" = st.session_state.get(_STATE_RESULT)
    image = st.session_state.get(_STATE_IMAGE)

    if result is None or image is None:
        st.info(
            "Upload a document, choose a model, pick or type a question, and press "
            "**Run** in the sidebar to see an answer here."
        )
        return

    # Image on the left (highlighted when grounded), answer + meters on the right.
    left, right = st.columns([3, 2], gap="large")

    with left:
        display_image = _result_display_image(result, image)
        caption = "Answer region highlighted" if result.regions else "Uploaded document"
        st.image(display_image, caption=caption, use_container_width=True)

    with right:
        question = st.session_state.get(_STATE_QUESTION, "")
        model_choice = st.session_state.get(_STATE_MODEL, "")
        if question:
            st.markdown(f"**Question**  \n{question}")
        if model_choice:
            st.caption(f"Model: {model_choice}")

        # The answer itself — the headline output — gets prominent styling.
        answer_text = result.answer or "_(no answer produced)_"
        st.markdown("**Answer**")
        st.markdown(f"### {answer_text}")

        # Confidence as both a labelled metric and a bar for at-a-glance reading.
        confidence = float(result.confidence)
        conf_pct = max(0.0, min(confidence, 1.0))
        meters = st.columns(2)
        meters[0].metric("Confidence", f"{confidence * 100:.1f}%")
        meters[1].metric("Latency", f"{result.latency_ms:.0f} ms")
        st.progress(conf_pct, text="Model confidence")

        if not result.regions:
            # Explain the absence rather than leaving the operator guessing.
            st.caption(
                "No answer region highlighted (OCR unavailable or the answer text "
                "was not located on the page)."
            )

        # Diagnostics tucked away for the curious without cluttering the main view.
        with st.expander("Raw generation details"):
            st.json(result.raw)


# ---------------------------------------------------------------------------
# Research tab: offline base-vs-fine-tuned comparison
# ---------------------------------------------------------------------------
def render_research_tab(config: "ProjectConfig | None") -> None:
    """Render the offline comparison produced by ``make compare``.

    ``research/compare.py`` writes ``reports/comparison.csv`` with one row per
    metric and ``Base``/``Fine-tuned``/``Delta`` columns. We render the full table
    plus a grouped bar chart of base vs. fine-tuned. When the file is absent we
    show actionable guidance rather than an error — the comparison is an optional
    offline artifact, not a runtime dependency.
    """
    st.subheader("Base vs. Fine-tuned")
    st.caption(
        "Offline evaluation on the held-out test split. Generated by "
        "`research/compare.py` (run `make compare`)."
    )

    report_dir = Path(config.report_dir) if config is not None else Path("reports")
    csv_path = report_dir / "comparison.csv"

    if not csv_path.exists():
        st.info(
            f"No comparison report found at `{csv_path}`.\n\n"
            "Train an adapter, then generate the comparison with:\n\n"
            "```bash\nmake compare\n```\n\n"
            "or directly:\n\n"
            "```bash\npython -m research.compare --config configs/default.yaml \\\n"
            "    --adapter outputs/<run>/adapter\n```"
        )
        return

    # pandas is deferred (it is a heavy-ish optional dep for this view). A missing
    # pandas or a malformed CSV degrades to a clear message, never a crash.
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"Could not read `{csv_path}`: {exc}")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Grouped bar chart of Base vs Fine-tuned per metric. We only chart when the
    # expected columns exist so an out-of-shape CSV still shows the table above.
    if {"Metric", "Base", "Fine-tuned"}.issubset(df.columns):
        try:
            chart_df = df.set_index("Metric")[["Base", "Fine-tuned"]]
            st.bar_chart(chart_df)
            # Metrics have mixed units (ratios vs. ms vs. GB), so warn the reader
            # not to compare bar heights across unrelated rows.
            st.caption(
                "Note: metrics use different scales (accuracy ratios, latency ms, "
                "memory GB). Compare Base vs. Fine-tuned within each metric, not across."
            )
        except Exception as exc:
            st.warning(f"Could not render the comparison chart: {exc}")
    else:
        st.warning(
            "Comparison CSV is missing expected columns "
            "(Metric / Base / Fine-tuned); showing the raw table only."
        )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Compose the full page: header, sidebar controls, and the two tabs."""
    # Wide layout gives the document image room to breathe next to the answer.
    st.set_page_config(
        page_title="VisionDoc AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📄 VisionDoc AI")
    st.caption(
        "Document intelligence with a locally-served Vision-Language model — "
        "document QA, confidence scoring, and answer-region highlighting. "
        "All inference runs on-device; no hosted LLM APIs are called."
    )

    # Load config up front for sidebar defaults + the research tab. A failure here
    # is non-fatal: the app still renders with sane fallbacks and a visible notice.
    config: "ProjectConfig | None"
    try:
        config = load_app_config()
    except Exception as exc:
        logger.exception("Config load failed")
        st.error(
            f"Failed to load configuration: {exc}. Using built-in defaults for the UI. "
            "Set VISIONDOC_CONFIG or ensure configs/default.yaml exists."
        )
        config = None

    selections = render_sidebar(config)

    # Run inference *before* laying out tabs so the freshly-computed result is the
    # one rendered on this same script pass (Streamlit runs top-to-bottom).
    qa_tab, research_tab = st.tabs(["Document QA", "Research: Base vs Fine-tuned"])

    with qa_tab:
        if selections["run_clicked"]:
            run_inference(selections)
        render_results()

    with research_tab:
        render_research_tab(config)


# Streamlit executes the target script as ``__main__`` (via ``streamlit run``),
# so this guard both drives the app under Streamlit and keeps a plain
# ``import app.streamlit_app`` (tests, tooling, syntax checks) side-effect free.
if __name__ == "__main__":
    main()
