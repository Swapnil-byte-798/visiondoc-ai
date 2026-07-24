"""Pydantic v2 request/response models for the VisionDoc AI HTTP API.

Why a dedicated schema module
-----------------------------
The FastAPI handlers in :mod:`api.main` stay thin and readable only because the
*shape* of every request and response lives here, in one place. Keeping the
models separate from the app also lets tests and client-code import the schemas
without importing the app (which would transitively pull the ML stack the first
time a request is served).

Design choices worth calling out:

* **Self-documenting OpenAPI.** Every field carries a ``description`` and each
  model carries a ``json_schema_extra`` ``examples`` block. FastAPI surfaces
  these verbatim in ``/docs`` (Swagger) and ``/redoc``, so the interactive docs
  are usable without reading this file — the single biggest usability win for a
  demo API.
* **Base64 for image transport in JSON.** ``/ask`` and ``/extract`` accept the
  image as a base64 string rather than multipart because they are JSON
  endpoints (easy to call from any language / the Streamlit app), and base64 is
  the lossless, dependency-free way to carry bytes inside JSON. The multipart
  ``/predict`` endpoint exists for the common "just upload a file" case and does
  not use these schemas for its request body.
* **Permissive value types on responses.** ``confidence`` is a plain ``float``
  (not constrained to ``[0, 1]``) and ``fields`` is an open ``dict`` on purpose:
  a response model must never *reject* a value the model legitimately produced,
  because a response-validation error would turn a successful inference into an
  opaque 500. Input models are where we validate; output models only shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# A short, obviously-truncated stand-in for a real base64 PNG. Kept tiny so the
# generated OpenAPI examples stay readable; a real request carries the full
# encoding of the document image.
_EXAMPLE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCA... (base64-encoded PNG)"


class AskRequest(BaseModel):
    """A document-QA request: one image plus one natural-language question."""

    image_base64: str = Field(
        ...,
        description=(
            "The document image, base64-encoded (PNG/JPEG bytes). A bare base64 "
            "string or a data-URI (`data:image/png;base64,...`) is accepted."
        ),
    )
    question: str = Field(
        ...,
        description="Natural-language question to answer about the document.",
        min_length=1,
    )
    highlight: bool = Field(
        True,
        description=(
            "When true, the service attempts to locate the answer text on the "
            "page (via OCR) and returns a highlighted image plus region boxes. "
            "Silently degrades to no highlight when OCR is unavailable."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_base64": _EXAMPLE_IMAGE_B64,
                    "question": "What is the invoice total?",
                    "highlight": True,
                }
            ]
        }
    }


class AskResponse(BaseModel):
    """The answer to an :class:`AskRequest`, with optional grounding."""

    answer: str = Field(..., description="The model's answer text.")
    confidence: float = Field(
        ...,
        description=(
            "Sequence confidence in [0, 1] derived from token probabilities "
            "(see InferenceConfig.confidence_method). A soft signal, not a "
            "calibrated probability."
        ),
    )
    latency_ms: float = Field(
        ..., description="End-to-end inference latency for this call, milliseconds."
    )
    highlighted_image_base64: str | None = Field(
        default=None,
        description=(
            "Base64 PNG of the page with the answer region boxed, or null when "
            "highlighting was not requested / OCR found no match."
        ),
    )
    regions: list[list[float]] = Field(
        default_factory=list,
        description=(
            "Absolute-pixel bounding boxes [x0, y0, x1, y1] the answer was "
            "localized to. Empty when nothing was grounded."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "$1,240.00",
                    "confidence": 0.94,
                    "latency_ms": 812.5,
                    "highlighted_image_base64": None,
                    "regions": [[512.0, 300.0, 640.0, 332.0]],
                }
            ]
        }
    }


class ExtractRequest(BaseModel):
    """A structured-extraction request.

    Callers may pin an exact ``fields`` list, name a ``doc_type`` (whose
    canonical field set is used), or neither (a generic field set is applied).
    Both are optional so the simplest useful call is just an image.
    """

    image_base64: str = Field(
        ...,
        description="The document image, base64-encoded (PNG/JPEG bytes) or data-URI.",
    )
    doc_type: str | None = Field(
        default=None,
        description=(
            "Known document type to use its canonical field set. Recognized "
            "values include 'invoice', 'receipt', 'id_card', and 'form'."
        ),
    )
    fields: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of field keys to extract. Takes precedence over "
            "`doc_type` when both are given."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_base64": _EXAMPLE_IMAGE_B64,
                    "doc_type": "invoice",
                    "fields": None,
                }
            ]
        }
    }


class ExtractResponse(BaseModel):
    """Structured fields extracted from a document."""

    fields: dict[str, Any] = Field(
        ...,
        description=(
            "Extracted field->value mapping. When the request pinned a key set, "
            "every requested key is present (missing values are null)."
        ),
    )
    confidence: float = Field(
        ..., description="Sequence confidence in [0, 1] for the generated JSON."
    )
    latency_ms: float = Field(
        ..., description="End-to-end extraction latency for this call, milliseconds."
    )
    raw: str = Field(
        ...,
        description=(
            "The undecorated model output text, retained for debugging/audit "
            "when the JSON parse is partial."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fields": {
                        "invoice_number": "INV-0042",
                        "total": "$1,240.00",
                        "due_date": "2026-08-01",
                    },
                    "confidence": 0.88,
                    "latency_ms": 905.2,
                    "raw": '{"invoice_number": "INV-0042", "total": "$1,240.00"}',
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Minimal answer payload for the multipart ``/predict`` upload endpoint.

    A slimmer sibling of :class:`AskResponse` (no highlighting/regions) because
    ``/predict`` is the "quick upload a file and ask" path; callers that want
    grounding use JSON ``/ask``.
    """

    answer: str = Field(..., description="The model's answer text.")
    confidence: float = Field(..., description="Sequence confidence in [0, 1].")
    latency_ms: float = Field(
        ..., description="End-to-end inference latency for this call, milliseconds."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"answer": "$1,240.00", "confidence": 0.94, "latency_ms": 812.5}
            ]
        }
    }


class HealthResponse(BaseModel):
    """Liveness + configuration snapshot returned by ``/health``.

    Deliberately answerable *without* loading model weights: it reports what the
    service is configured to serve (model id/type, whether an adapter is wired
    up) and the resolved compute device, so orchestrators can probe readiness
    and humans can sanity-check the deployment before paying the load cost.
    """

    status: str = Field(..., description="Service status; 'ok' when reachable.")
    model_id: str = Field(..., description="Hugging Face model id the service will load.")
    model_type: str = Field(
        ..., description="Backbone family (e.g. 'qwen2_5_vl', 'donut')."
    )
    adapter_loaded: bool = Field(
        ...,
        description=(
            "Whether a LoRA adapter is configured to be served (from config; "
            "reflects intent, since weights load lazily on first request)."
        ),
    )
    device: str = Field(..., description="Resolved compute device: 'cuda' | 'mps' | 'cpu'.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "model_type": "qwen2_5_vl",
                    "adapter_loaded": False,
                    "device": "cuda",
                }
            ]
        }
    }


__all__ = [
    "AskRequest",
    "AskResponse",
    "ExtractRequest",
    "ExtractResponse",
    "PredictResponse",
    "HealthResponse",
]
