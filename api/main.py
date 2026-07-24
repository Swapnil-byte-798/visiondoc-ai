"""FastAPI service exposing VisionDoc AI for document QA and field extraction.

This is the HTTP surface over :class:`inference.predictor.DocumentPredictor`.
The whole file is organized around one constraint that shapes every decision
here: **the model is huge and slow to load, so it must not load at import time.**

How that constraint is honoured
-------------------------------
* **Lazy, thread-safe predictor singleton.** :func:`get_predictor` builds the
  predictor exactly once, on the first request that actually needs it, guarded
  by a lock with double-checked locking. Consequences: the process boots
  instantly (good for container health checks and autoscalers), ``/health``
  answers without a GPU or any weights, and a machine with no CUDA can still
  import and serve metadata.
* **Deferred heavy imports.** ``inference`` (torch/PIL) and ``utils.device``
  (torch) are imported *inside* the functions that use them, never at module
  top. Importing :mod:`api.main` — e.g. to generate the OpenAPI schema in a
  test — therefore costs nothing heavy.
* **Config is loaded lazily and cached** (:func:`get_config`) for the same
  reason and so the app object can be constructed even when the working
  directory has no ``configs/default.yaml`` yet.

Error handling contract: bad client input (undecodable image, empty question)
returns ``400``; a failure *inside* inference returns ``500`` with a short
message. We never leak a traceback to the client, but we always log it.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AskRequest,
    AskResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    PredictResponse,
)
from configs import load_config
from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only: never triggers the heavy import at runtime.
    from configs.config import ProjectConfig
    from inference.predictor import DocumentPredictor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application version
# ---------------------------------------------------------------------------
def _resolve_version() -> str:
    """Return the installed package version, falling back to a literal.

    Reading the version from installed metadata keeps the OpenAPI ``version``
    in lock-step with ``pyproject.toml`` when the package is ``pip install``-ed,
    while the fallback keeps the app runnable straight from a source checkout
    (where distribution metadata may be absent).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("visiondoc-ai")
    except Exception:  # PackageNotFoundError or a broken metadata cache.
        return "0.1.0"


app = FastAPI(
    title="VisionDoc AI",
    version=_resolve_version(),
    description=(
        "Document intelligence over a locally-served Vision-Language model: "
        "document QA, structured field extraction, confidence scoring, and "
        "region highlighting. All inference runs on the locally-loaded model — "
        "no hosted LLM APIs are called."
    ),
)

# CORS: wide-open for the demo so the Streamlit app / a browser client on any
# origin can call the API without preflight friction. In production this MUST
# be restricted to the known frontend origin(s) — an open API with credentials
# is a data-exfiltration risk. Kept permissive here deliberately and documented.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PROD: replace with an explicit allowlist.
    allow_credentials=False,  # must stay False while allow_origins is "*".
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazy singletons (config + predictor)
# ---------------------------------------------------------------------------
# Module globals guarded by locks. We use plain globals + double-checked locking
# rather than FastAPI's dependency cache because these must be shared across
# *all* requests and workers-within-a-process, and building them twice (a torn
# read under concurrency) would load the multi-GB model twice.
_config: "ProjectConfig | None" = None
_config_lock = threading.Lock()

_predictor: "DocumentPredictor | None" = None
_predictor_lock = threading.Lock()


def get_config() -> "ProjectConfig":
    """Load the project config once and cache it.

    Lazy (not import-time) so importing this module never touches the filesystem
    and never fails on a missing/instance-specific YAML. Env overrides
    (``VISIONDOC_CONFIG`` for the path, ``VISIONDOC_ADAPTER_PATH`` for the served
    adapter) are applied by :func:`load_config`/``from_yaml``.
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
                logger.info(
                    "Loaded config: model_id=%s type=%s adapter=%s device=%s",
                    _config.model.model_id,
                    _config.model.model_type,
                    _config.inference.adapter_path,
                    _config.device,
                )
    return _config


def get_predictor() -> "DocumentPredictor":
    """Build the :class:`DocumentPredictor` on first use, then reuse it.

    Double-checked locking: the ``is None`` fast path avoids taking the lock on
    the hot path once the predictor exists, while the lock + re-check guarantees
    exactly one construction under concurrent first requests (so the model
    weights load a single time). The heavy import lives here, not at module top,
    so the ML stack is only required the moment inference is first requested.
    """
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                # Deferred heavy import: pulls torch/transformers/PIL.
                from inference.predictor import DocumentPredictor

                logger.info("Building DocumentPredictor (first request) ...")
                _predictor = DocumentPredictor.from_config(get_config())
                logger.info("DocumentPredictor ready.")
    return _predictor


# ---------------------------------------------------------------------------
# Small input helpers (kept out of the handlers so error mapping is uniform)
# ---------------------------------------------------------------------------
def _decode_base64_image(image_base64: str):
    """Decode a base64 image string to a PIL image, mapping failure to 400.

    A malformed image is *client* error, so we translate any decode exception
    into an HTTP 400 with a short reason rather than letting it bubble to a 500.
    """
    from utils.image_utils import base64_to_pil

    if not image_base64 or not image_base64.strip():
        raise HTTPException(status_code=400, detail="image_base64 must not be empty.")
    try:
        return base64_to_pil(image_base64)
    except Exception as exc:  # bad base64, unknown format, truncated bytes ...
        raise HTTPException(
            status_code=400, detail=f"Could not decode image_base64: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness + config snapshot. Never forces the model to load.

    We report the *configured* model/adapter and the resolved device so this
    endpoint stays fast and works on a machine with no GPU and cold weights —
    exactly what a container/orchestrator readiness probe needs.
    """
    config = get_config()
    # Deferred torch import; resolving the device does not allocate the model.
    from utils.device import get_device_info

    info = get_device_info(config.device, config.model.torch_dtype)
    return HealthResponse(
        status="ok",
        model_id=config.model.model_id,
        model_type=config.model.model_type,
        # Reflects configured intent (weights load lazily); an adapter path in
        # config means the served model will be LoRA-adapted on first request.
        adapter_loaded=bool(config.inference.adapter_path),
        device=info.device,
    )


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(request: AskRequest) -> AskResponse:
    """Answer a natural-language question about a base64 image.

    Input decoding errors -> 400; inference errors -> 500. The predictor loads
    on the first call to this (or any inference) endpoint.
    """
    image = _decode_base64_image(request.image_base64)
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    try:
        predictor = get_predictor()
        result = predictor.answer(image, request.question, highlight=request.highlight)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/ask inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        highlighted_image_base64=result.highlighted_image_b64,
        # PredictionResult.regions are float 4-tuples; JSON wants nested lists.
        regions=[list(box) for box in result.regions],
    )


@app.post("/extract", response_model=ExtractResponse, tags=["extraction"])
def extract(request: ExtractRequest) -> ExtractResponse:
    """Extract structured fields (by explicit list or by document type).

    ``DocumentPredictor.extract`` returns ``fields``/``confidence``/``raw`` but
    not latency, so we time the call here to populate ``latency_ms`` — the same
    wall-clock the client actually waited.
    """
    image = _decode_base64_image(request.image_base64)

    try:
        predictor = get_predictor()
        start = time.perf_counter()
        out = predictor.extract(image, fields=request.fields, doc_type=request.doc_type)
        latency_ms = (time.perf_counter() - start) * 1000.0
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/extract inference failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    # Defensive ``.get`` reads: the extractor contract guarantees these keys, but
    # tolerating a partial dict keeps a degraded generation from becoming a 500.
    return ExtractResponse(
        fields=out.get("fields", {}) or {},
        confidence=float(out.get("confidence", 0.0) or 0.0),
        latency_ms=latency_ms,
        raw=str(out.get("raw", "") or ""),
    )


@app.post("/predict", response_model=PredictResponse, tags=["qa"])
def predict(
    file: UploadFile = File(..., description="Document image to answer about."),
    question: str = Form(..., description="Question to answer about the uploaded file."),
) -> PredictResponse:
    """Multipart "upload a file + ask" endpoint.

    A sync ``def`` on purpose: FastAPI runs it in a worker thread, so the heavy,
    blocking generate() call does not stall the event loop. We read the upload
    via the underlying ``file.file`` object (synchronous) rather than the async
    ``await file.read()`` to stay in the threadpool model. Bytes are handed to
    :func:`utils.image_utils.load_image`, which accepts raw image bytes.
    """
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    try:
        contents = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}") from exc
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    from utils.image_utils import load_image

    try:
        image = load_image(contents)
    except Exception as exc:  # unknown/corrupt image bytes -> client error.
        raise HTTPException(
            status_code=400, detail=f"Could not decode uploaded image: {exc}"
        ) from exc

    try:
        predictor = get_predictor()
        # ``highlight=False``: this lightweight endpoint returns text only, so we
        # skip the OCR grounding pass (clients wanting regions use JSON /ask).
        result = predictor.answer(image, question, highlight=False)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/predict inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(
        answer=result.answer,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# Dev/entrypoint runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Local run: `python -m api.main`. Host/port come from env so the same entry
    # point works in a container (0.0.0.0) and on a laptop. We pass the import
    # string so uvicorn owns the app lifecycle (and reload/workers stay possible).
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
