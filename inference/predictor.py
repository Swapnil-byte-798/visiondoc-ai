"""High-level inference facade used by the API, the Streamlit app, and batch jobs.

:class:`DocumentPredictor` is the single object every user-facing surface talks
to. It owns the three moving parts an answer needs and hides their wiring:

1. a loaded :class:`VisionDocModel` (base weights, optionally LoRA-adapted),
2. an :class:`OCREngine` for *region grounding* — turning an answer string into
   a highlighted box on the page, and
3. a :class:`FieldExtractor` for structured (JSON) extraction.

Design principles for this file:

* **Everything degrades gracefully.** No GPU, no Tesseract, no PDF backend, or a
  single un-decodable image must never take down a request. Missing OCR simply
  disables highlighting (regions come back empty); a failed page in a PDF is
  logged and returned as an empty answer rather than aborting the document.
* **One result shape everywhere.** :class:`PredictionResult` is what the API
  serializes, what the app renders, and what batch jobs tabulate. Building it in
  one private helper (:meth:`_build_result`) keeps single, batch, and per-page
  paths byte-for-byte consistent.
* **Heavy imports are deferred.** ``models`` / ``utils.image_utils`` / ``utils.ocr``
  pull torch/PIL. They are imported inside methods (or ``__init__``) so that
  merely importing this module — e.g. when the API process is still cold, or in
  a torch-less test — does not require the ML stack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only imports; never executed at runtime.
    from PIL import Image

    from configs.config import ProjectConfig
    from models.base import GenerationOutput, VisionDocModel

logger = get_logger(__name__)

# Absolute-pixel bounding box (x0, y0, x1, y1).
Box = tuple[float, float, float, float]


@dataclass
class PredictionResult:
    """A single answer plus everything the UI/API needs to present it.

    ``regions`` are absolute-pixel boxes the answer was localized to (empty when
    OCR is unavailable or no match was found). ``highlighted_image_b64`` is a
    base64 PNG of the page with those boxes drawn — ``None`` when nothing was
    highlighted, so the caller can fall back to the original upload. ``raw``
    carries the generation's diagnostic dict (token count, latency, confidence).
    """

    answer: str
    confidence: float
    latency_ms: float
    regions: list[Box] = field(default_factory=list)
    highlighted_image_b64: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (boxes as plain lists) for API responses/CSV."""
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "regions": [list(b) for b in self.regions],
            "highlighted_image_b64": self.highlighted_image_b64,
            "raw": self.raw,
        }


class DocumentPredictor:
    """Facade that turns ``(image, question)`` into a :class:`PredictionResult`."""

    def __init__(
        self,
        config: "ProjectConfig",
        model: "VisionDocModel | None" = None,
        use_ocr: bool = True,
    ) -> None:
        """Build (or adopt) the model and optional OCR engine.

        When ``model`` is ``None`` we construct it from ``config`` and attach the
        adapter named by ``config.inference.adapter_path`` (if any) — this is the
        path the API/app take, where the caller only has a config. Passing an
        already-built ``model`` lets callers (research comparisons, tests) inject
        a base or pre-adapted model without paying to load weights twice.
        """
        self.config = config

        # Deferred heavy imports (torch/PIL live behind these).
        from models import build_model

        if model is None:
            built = build_model(config)  # load=True: weights + processor on device
            adapter_path = config.inference.adapter_path
            if adapter_path:
                # Best-effort adapter attach: a bad/missing adapter path should
                # not brick the service — we log and fall back to the base model,
                # which still answers (just zero-shot).
                try:
                    built.load_adapter(adapter_path)
                    logger.info("Loaded LoRA adapter from %s", adapter_path)
                except Exception:
                    logger.exception(
                        "Failed to load adapter %s; serving base model instead.",
                        adapter_path,
                    )
            self.model = built
        else:
            self.model = model

        # Eval mode: disables dropout and switches norms to inference stats. The
        # facade is inference-only, so this is always correct here.
        self.model.to_eval()

        # OCR is optional and self-reporting via ``.available``; construct it
        # eagerly (cheap — it only probes for the tesseract binary) so we can
        # answer ``highlight`` requests without a per-call setup cost.
        self.ocr = None
        if use_ocr:
            try:
                from utils.ocr import OCREngine

                self.ocr = OCREngine()
                if not self.ocr.available:
                    logger.info("OCR engine unavailable; region highlighting disabled.")
            except Exception:
                # Even importing pytesseract can fail on a minimal host; treat
                # any failure as "no OCR" rather than propagating.
                logger.info("OCR engine could not be initialized; highlighting disabled.")
                self.ocr = None

        # Lazily constructed on first ``extract`` call (keeps import light).
        self._extractor: Any = None

    @classmethod
    def from_config(cls, config: "ProjectConfig") -> "DocumentPredictor":
        """Convenience constructor mirroring the rest of the codebase's style."""
        return cls(config)

    # -- Core QA ------------------------------------------------------------

    def answer(self, image: Any, question: str, highlight: bool = True) -> PredictionResult:
        """Answer ``question`` about ``image`` and (optionally) highlight it.

        ``image`` may be anything :func:`utils.image_utils.load_image` accepts
        (path, bytes, base64, ndarray, or a PIL image). Latency is measured here
        rather than trusting the model wrapper so the number reflects the exact
        call the caller made (and includes tokenization overhead).
        """
        from utils.image_utils import load_image

        pil_image = load_image(image)

        start = time.perf_counter()
        output = self.model.generate(pil_image, question)
        latency_ms = (time.perf_counter() - start) * 1000.0
        # ``generate`` returns a single object for a scalar question; normalize
        # defensively in case a backbone hands back a 1-element list.
        if isinstance(output, list):
            output = output[0]

        return self._build_result(pil_image, output, latency_ms, highlight)

    # ``ask`` is a friendlier alias used in notebooks/README examples.
    ask = answer

    def batch_answer(
        self, images: Sequence[Any], questions: Sequence[str]
    ) -> list[PredictionResult]:
        """Answer a batch in a single ``generate`` call.

        Batching amortizes the vision encoder across samples, which is how the
        served endpoint actually gets its throughput — so we pass whole lists to
        the model rather than looping :meth:`answer`. The measured batch
        wall-clock is spread evenly across samples to report an amortized
        per-document latency. Highlighting is applied per result (best-effort),
        matching :meth:`answer`'s default.
        """
        from utils.image_utils import load_image

        images = list(images)
        questions = list(questions)
        if len(images) != len(questions):
            raise ValueError(
                f"images ({len(images)}) and questions ({len(questions)}) length mismatch"
            )
        if not images:
            return []

        pil_images = [load_image(img) for img in images]

        start = time.perf_counter()
        outputs = self.model.generate(pil_images, questions)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Single vs list normalization (a length-1 batch may return one object).
        batch_outputs = list(outputs) if isinstance(outputs, list) else [outputs]
        per_sample_ms = elapsed_ms / max(len(pil_images), 1)

        results: list[PredictionResult] = []
        for pil_image, output in zip(pil_images, batch_outputs):
            results.append(self._build_result(pil_image, output, per_sample_ms, highlight=True))
        return results

    # -- Structured extraction ---------------------------------------------

    def extract(
        self,
        image: Any,
        fields: list[str] | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """Extract structured fields, delegating to :class:`FieldExtractor`.

        The extractor is built lazily and cached so the (cheap) import of
        ``inference.extract`` is paid at most once per predictor, and only when
        extraction is actually used.
        """
        if self._extractor is None:
            from inference.extract import FieldExtractor

            # Pass the predictor's OCR engine so extraction gets page-text
            # grounding (prompt context + per-field hallucination flags).
            self._extractor = FieldExtractor(self.model, ocr=self.ocr)
        return self._extractor.extract(image, fields=fields, doc_type=doc_type)

    # -- PDF ----------------------------------------------------------------

    def predict_pdf(self, pdf_source: Any, question: str) -> list[PredictionResult]:
        """Answer ``question`` on every page of a PDF.

        Pages are rasterized via :func:`utils.image_utils.pdf_to_images` (capped
        at that helper's ``max_pages`` guard). Each page is answered
        independently so a per-page generation failure degrades to an empty
        result for that page instead of aborting the whole document.
        """
        from utils.image_utils import pdf_to_images

        pages = pdf_to_images(pdf_source)
        results: list[PredictionResult] = []
        for page_index, page in enumerate(pages):
            try:
                results.append(self.answer(page, question))
            except Exception:
                logger.exception("Failed to answer PDF page %d; recording empty result.", page_index)
                results.append(
                    PredictionResult(
                        answer="",
                        confidence=0.0,
                        latency_ms=0.0,
                        raw={"error": "generation_failed", "page": page_index},
                    )
                )
        return results

    # -- Region highlighting ------------------------------------------------

    def highlight_regions(self, image: Any, answer: str) -> "Image.Image":
        """Return ``image`` with the answer's location boxed, if OCR can find it.

        Returns the image *unchanged* when OCR is unavailable or the answer text
        is not found on the page — the caller can always render the result
        regardless of whether grounding succeeded.
        """
        from utils.image_utils import draw_boxes, load_image

        pil_image = load_image(image)
        box = self._find_answer_box(pil_image, answer)
        if box is None:
            return pil_image
        # ``normalized=False``: OCR boxes are absolute pixels in the loaded image.
        return draw_boxes(pil_image, [box], normalized=False)

    # -- Internals ----------------------------------------------------------

    def _find_answer_box(self, pil_image: "Image.Image", answer: str) -> Box | None:
        """Locate ``answer`` on the page via OCR, or ``None`` if not groundable.

        Wraps the OCR call in a try/except because OCR is a best-effort add-on:
        any engine hiccup should silently skip highlighting, not fail the answer.
        """
        if self.ocr is None or not getattr(self.ocr, "available", False):
            return None
        if not answer or not answer.strip():
            return None
        try:
            box = self.ocr.find_answer_box(pil_image, answer)
        except Exception:
            logger.debug("OCR find_answer_box failed; skipping highlight.", exc_info=True)
            return None
        if box is None:
            return None
        # Normalize to a float 4-tuple so the result shape is uniform/JSON-safe.
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    def _build_result(
        self,
        pil_image: "Image.Image",
        output: "GenerationOutput",
        latency_ms: float,
        highlight: bool,
    ) -> PredictionResult:
        """Assemble a :class:`PredictionResult`, adding region highlight if asked.

        Shared by :meth:`answer`, :meth:`batch_answer`, and (via ``answer``)
        :meth:`predict_pdf` so every entry point produces an identically-shaped
        result. Highlighting only runs when requested *and* OCR located the
        answer, keeping the common no-OCR path free of extra work.
        """
        from utils.image_utils import draw_boxes, pil_to_base64

        answer_text = (output.text or "").strip()

        regions: list[Box] = []
        highlighted_b64: str | None = None
        if highlight:
            box = self._find_answer_box(pil_image, answer_text)
            if box is not None:
                regions = [box]
                highlighted = draw_boxes(pil_image, [box], normalized=False)
                highlighted_b64 = pil_to_base64(highlighted)

        return PredictionResult(
            answer=answer_text,
            confidence=float(output.confidence),
            latency_ms=float(latency_ms),
            regions=regions,
            highlighted_image_b64=highlighted_b64,
            # ``GenerationOutput.to_dict`` reports confidence + token count, but
            # its latency_ms is unset (generate() doesn't populate it), so we
            # overwrite it with the wall-clock latency actually measured here.
            raw={**output.to_dict(), "latency_ms": round(float(latency_ms), 2)},
        )


__all__ = ["PredictionResult", "DocumentPredictor", "Box"]
