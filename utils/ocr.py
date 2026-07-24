"""OCR fallback (bonus feature).

The VLM answers questions directly from pixels, but for two cases a classical
OCR pass is valuable:

1. **Abstention fallback** — if the model returns an empty / low-confidence
   answer, we can search OCR-extracted text for a candidate span.
2. **Region grounding** — OCR word boxes let us highlight *where* an answer
   token appears on the page even when the model itself is not box-aware.

``pytesseract`` is imported lazily and treated as optional: if Tesseract is not
installed, the fallback degrades to a no-op instead of crashing the request.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger("visiondoc.ocr")

# Absolute pixel box (x0, y0, x1, y1).
Box = tuple[int, int, int, int]


@dataclass
class OCRWord:
    """A single recognized word with its confidence and bounding box."""

    text: str
    confidence: float  # 0..1
    box: Box


class OCREngine:
    """Thin wrapper around Tesseract with graceful degradation.

    Parameters
    ----------
    lang:
        Tesseract language code(s), e.g. ``"eng"``.
    tesseract_cmd:
        Optional explicit path to the tesseract binary (else read from the
        ``TESSERACT_CMD`` env var or the system ``PATH``).
    """

    def __init__(self, lang: str = "eng", tesseract_cmd: str | None = None) -> None:
        self.lang = lang
        self._available = False
        try:
            import pytesseract

            cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
            self._pytesseract = pytesseract
            # Probe the binary; get_tesseract_version raises if missing.
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception as exc:  # pragma: no cover - depends on host install
            logger.info("OCR unavailable (%s). Fallback disabled.", exc.__class__.__name__)
            self._pytesseract = None

    @property
    def available(self) -> bool:
        """Whether a working Tesseract binary was found."""
        return self._available

    def extract_words(self, image: Image.Image, min_confidence: float = 0.3) -> list[OCRWord]:
        """Return recognized words with confidences and boxes.

        Empty list if OCR is unavailable — callers must handle the no-op case.
        """
        if not self._available or self._pytesseract is None:
            return []
        data = self._pytesseract.image_to_data(
            image.convert("RGB"),
            lang=self.lang,
            output_type=self._pytesseract.Output.DICT,
        )
        words: list[OCRWord] = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(data["conf"][i])
            conf = conf / 100.0 if conf >= 0 else 0.0
            if conf < min_confidence:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(OCRWord(text=text, confidence=conf, box=(x, y, x + w, y + h)))
        return words

    def extract_text(self, image: Image.Image) -> str:
        """Return the full plain-text transcription (empty if unavailable)."""
        if not self._available or self._pytesseract is None:
            return ""
        return self._pytesseract.image_to_string(image.convert("RGB"), lang=self.lang).strip()

    def find_answer_box(self, image: Image.Image, answer: str) -> Box | None:
        """Locate ``answer`` in the page and return a bounding box, if found.

        Enables region highlighting for a model-produced answer string by
        matching it against OCR words (case-insensitive, whitespace-tolerant).
        """
        if not answer.strip():
            return None
        words = self.extract_words(image)
        if not words:
            return None
        target = answer.lower().split()
        texts = [w.text.lower() for w in words]
        # Sliding-window match over the OCR word sequence.
        for start in range(len(texts) - len(target) + 1):
            if texts[start : start + len(target)] == target:
                span = words[start : start + len(target)]
                x0 = min(w.box[0] for w in span)
                y0 = min(w.box[1] for w in span)
                x1 = max(w.box[2] for w in span)
                y1 = max(w.box[3] for w in span)
                return (x0, y0, x1, y1)
        return None


__all__ = ["OCRWord", "OCREngine"]
