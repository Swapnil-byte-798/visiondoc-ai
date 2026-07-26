"""Structured field extraction on top of a :class:`VisionDocModel`.

Document QA answers a free-form question; *field extraction* asks the same VLM
to return a machine-readable object (``{"total": "42.00", ...}``) for a known
document type. We implement this as a prompting + parsing + grounding layer
rather than a separate model because the instruction-tuned backbone already
follows a "return only JSON" directive well, and reusing the same weights keeps
the project self-contained (no second model to train or serve).

Quality levers implemented here (each documented at its call site):

* **Schema-aware prompt.** We list the exact keys *with a short description and a
  format hint* and include a worked ``null``-filled skeleton. VLMs honour an
  explicit, described key contract far more reliably than a bare key list — this
  is the single biggest lever for a zero-shot (un-fine-tuned) backbone.
* **OCR grounding.** When an :class:`~utils.ocr.OCREngine` is available we (a)
  paste the page's OCR transcription into the prompt as *reference text*, and
  (b) flag, per field, whether the emitted value actually appears on the page.
  Grounding is what catches hallucinated values (e.g. a merchant name the model
  invented). We *flag* rather than delete, to protect recall against OCR misses.
* **Value normalization.** Money fields are stripped of currency symbols and
  thousands separators so downstream comparison/aggregation is numeric-friendly.
* **Defensive parsing.** :func:`parse_json_answer` never trusts the model's
  formatting: it strips ```json fences, extracts the first *balanced* ``{...}``
  span (brace-counting that respects string literals), and repairs a trailing
  comma before giving up with ``{}`` — so one malformed generation degrades a
  single row, never the batch.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only imports keep this module importable without torch.
    from PIL import Image

    from models.base import VisionDocModel
    from utils.ocr import OCREngine

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical field sets per document type
# ---------------------------------------------------------------------------
# The keys the extractor asks for when the caller passes only a ``doc_type``.
# Intentionally business-oriented (the fields a document-intelligence product
# actually needs) rather than an exhaustive layout dump: a tight, well-known key
# set yields far more reliable JSON than an open-ended "extract everything".
DOC_TYPE_FIELDS: dict[str, list[str]] = {
    "invoice": [
        "invoice_number", "invoice_date", "due_date", "vendor_name",
        "vendor_address", "bill_to", "subtotal", "tax", "total", "currency",
    ],
    "receipt": [
        "merchant_name", "merchant_address", "date", "time", "subtotal",
        "tax", "tip", "total", "payment_method",
    ],
    "id_card": [
        "full_name", "document_number", "date_of_birth", "sex", "nationality",
        "issue_date", "expiry_date", "address",
    ],
    "form": [
        "form_title", "name", "date", "address", "phone", "email",
        "signature_present",
    ],
}

# Short, human descriptions + format hints per key. A described key contract is
# the main reason a zero-shot VLM stops emitting garbage for a bare key name.
# Keys not listed here fall back to a generic instruction.
FIELD_DESCRIPTIONS: dict[str, str] = {
    # invoice / receipt shared money + meta fields
    "invoice_number": "the invoice identifier/number",
    "invoice_date": "the invoice issue date (YYYY-MM-DD if unambiguous)",
    "due_date": "the payment due date (YYYY-MM-DD if unambiguous)",
    "vendor_name": "the name of the company that issued the document",
    "vendor_address": "the vendor's postal address as printed",
    "bill_to": "the customer/entity the document is billed to",
    "merchant_name": "the store or business name, usually printed at the top",
    "merchant_address": "the store's address as printed",
    "date": "the document date (YYYY-MM-DD if unambiguous, else as printed)",
    "time": "the time printed on the document (HH:MM)",
    "subtotal": "the pre-tax subtotal amount, digits only (no currency symbol)",
    "tax": "the tax amount, digits only (no currency symbol)",
    "tip": "the tip/gratuity amount, digits only (no currency symbol)",
    "total": "the grand total amount, digits only (no currency symbol)",
    "currency": "the 3-letter currency code (e.g. USD, EUR) or symbol as printed",
    "payment_method": "how it was paid (e.g. cash, visa, mastercard)",
    # id_card
    "full_name": "the person's full name as printed",
    "document_number": "the ID/document number",
    "date_of_birth": "date of birth (YYYY-MM-DD if unambiguous)",
    "sex": "sex/gender as printed (e.g. M, F)",
    "nationality": "nationality/citizenship as printed",
    "issue_date": "the issue date (YYYY-MM-DD if unambiguous)",
    "expiry_date": "the expiry date (YYYY-MM-DD if unambiguous)",
    "address": "the address as printed",
    # form
    "form_title": "the title/heading of the form",
    "name": "the person's name filled into the form",
    "phone": "the phone number as printed",
    "email": "the email address as printed",
    "signature_present": "true if a handwritten signature is visible, else false",
}

# Keys whose values are monetary and should be normalized to bare numbers.
_MONEY_KEYS = {"subtotal", "tax", "tip", "total", "amount", "price", "due", "balance", "grand_total"}

# A compact, generic key set used when the caller gives neither ``fields`` nor a
# recognized ``doc_type``. We still constrain the schema so output stays parseable.
_GENERIC_FIELDS: list[str] = ["document_type", "date", "total", "names", "key_values"]


def resolve_fields(fields: list[str] | None, doc_type: str | None) -> list[str]:
    """Decide the concrete key set to request.

    Precedence: explicit ``fields`` wins; else the canonical set for a recognized
    ``doc_type``; else a generic fallback. Centralizing this keeps the prompt and
    the parsed-result projection perfectly in sync.
    """
    if fields:
        seen: set[str] = set()
        unique: list[str] = []
        for f in fields:
            key = str(f).strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(key)
        if unique:
            return unique
    if doc_type:
        canonical = DOC_TYPE_FIELDS.get(doc_type.strip().lower())
        if canonical:
            return list(canonical)
    return list(_GENERIC_FIELDS)


def build_extraction_prompt(
    fields: list[str] | None,
    doc_type: str | None,
    ocr_text: str | None = None,
) -> str:
    """Build a schema-aware instruction that asks for ONLY a JSON object.

    Compared with a bare key list, we add (1) a one-line description + format hint
    per key, (2) explicit numeric/date/verbatim rules, (3) a worked ``null``
    skeleton, and (4) optional OCR reference text. Each of these measurably
    reduces hallucination and malformed JSON on an instruction-tuned VLM.
    """
    keys = resolve_fields(fields, doc_type)
    doc_hint = f" from this {doc_type.strip()}" if doc_type else ""

    described = "\n".join(
        f'- "{k}": {FIELD_DESCRIPTIONS.get(k, "the value for " + k + " as printed")}'
        for k in keys
    )
    skeleton = "{" + ", ".join(f'"{k}": null' for k in keys) + "}"

    parts = [
        f"You are a precise document-information extractor. Extract the fields{doc_hint} "
        "and return ONLY one JSON object (no prose, no markdown, no code fences).",
        "",
        "Fields — use exactly these keys:",
        described,
        "",
        "Rules:",
        "- Output valid JSON only, with exactly the keys above and no extra keys.",
        "- Copy each value exactly as printed on the document; never infer or invent.",
        "- If a field is not clearly visible on the document, set it to null.",
        "- Money values: digits and a decimal point only (e.g. 42.50) — no currency "
        "symbols, no thousands separators.",
        "- Dates: use YYYY-MM-DD when unambiguous, otherwise copy exactly as printed.",
        "",
        f"Output template (fill in real values, keep every key):\n{skeleton}",
    ]
    if ocr_text:
        # Truncate so a huge page can't blow the context budget; the image stays
        # authoritative, OCR is only a hint to reduce mis-reads.
        snippet = ocr_text.strip().replace("\n", " ")
        if len(snippet) > 1500:
            snippet = snippet[:1500] + " ..."
        parts += [
            "",
            "Text detected on the document (reference only — the image is authoritative):",
            snippet,
        ]
    parts += ["", "Now output the JSON object."]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Robust JSON extraction from free-form model text (unchanged, well-tested)
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code-fence wrappers, keeping the fenced content."""
    stripped = text.strip()
    if "```" in stripped:
        stripped = _FENCE_OPEN_RE.sub("", stripped, count=1)
        stripped = _FENCE_CLOSE_RE.sub("", stripped)
    return stripped.strip()


def _iter_balanced_objects(text: str):
    """Yield each balanced ``{...}`` substring in the text, in order."""
    idx = 0
    n = len(text)
    while True:
        start = text.find("{", idx)
        if start == -1:
            return
        depth = 0
        in_string = False
        escaped = False
        closed = False
        for i in range(start, n):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    idx = i + 1
                    closed = True
                    break
        if not closed:
            return


def _load_json_object(candidate: str) -> dict | None:
    """Parse ``candidate`` as a JSON object, with a light trailing-comma repair."""
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            obj = json.loads(attempt)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_json_answer(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output (never raises)."""
    if not text or not text.strip():
        return {}
    cleaned = _strip_code_fences(text)
    direct = _load_json_object(cleaned)
    if direct is not None:
        return direct
    for snippet in _iter_balanced_objects(cleaned):
        parsed = _load_json_object(snippet)
        if parsed is not None:
            return parsed
    logger.debug("Could not parse a JSON object from model output: %.120r", text)
    return {}


# ---------------------------------------------------------------------------
# Normalization + grounding helpers
# ---------------------------------------------------------------------------

_CURRENCY_RE = re.compile(r"[^\d.\-]")  # keep digits, dot, minus


def _normalize_money(value: Any) -> Any:
    """Strip currency symbols / thousands separators from a money value.

    ``"$1,234.50"`` -> ``"1234.50"``. Leaves non-string / empty values untouched
    and returns the original string if the result would be empty (so we never
    turn a real-but-odd value into ``""``).
    """
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return value
    # Remove thousands separators first, then anything that isn't a digit/dot/sign.
    candidate = raw.replace(",", "")
    candidate = _CURRENCY_RE.sub("", candidate)
    return candidate or raw


def _norm_for_match(s: str) -> str:
    """Lowercase + keep alphanumerics only, for a whitespace/punct-tolerant match."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _value_grounded(value: Any, ocr_norm: str) -> bool:
    """True if ``value`` (normalized) appears within the normalized OCR text."""
    if not isinstance(value, str) or not value.strip() or not ocr_norm:
        return False
    v = _norm_for_match(value)
    return len(v) >= 2 and v in ocr_norm


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FieldExtractor:
    """Run structured field extraction against a loaded :class:`VisionDocModel`.

    Stateless apart from the model (and optional OCR) reference, so one instance
    can be shared by the predictor, the API, and batch jobs. The model is
    expected to already be loaded (and optionally LoRA-adapted).
    """

    def __init__(self, model: "VisionDocModel", ocr: "OCREngine | None" = None) -> None:
        # Hold the high-level wrapper (not the raw HF module) for its
        # confidence-scored ``generate`` and device handling.
        self.model = model
        # Optional OCR engine enables prompt grounding + value verification. It
        # is used only when present and reporting itself available.
        self.ocr = ocr

    def extract(
        self,
        image: Any,
        fields: list[str] | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """Extract ``fields`` (or a ``doc_type``'s canonical set) from ``image``.

        Returns ``{"fields", "confidence", "raw", "grounded", "ocr_used"}``:

        * ``fields``   — schema-projected result (every requested key present;
          missing -> ``None``; money fields normalized).
        * ``confidence`` — the generation's sequence confidence.
        * ``raw``      — the undecorated model text (audit/debug).
        * ``grounded`` — ``{key: bool}`` whether each value was found in the page
          OCR text (``{}`` when OCR is unavailable). A ``False`` on a non-null
          value is a strong hallucination signal for a reviewer.
        * ``ocr_used`` — whether OCR text was injected into the prompt.
        """
        from utils.image_utils import load_image

        pil_image = load_image(image)

        # OCR reference text (best-effort; disabled/empty degrades gracefully).
        ocr_text = ""
        if self.ocr is not None and getattr(self.ocr, "available", False):
            try:
                ocr_text = self.ocr.extract_text(pil_image) or ""
            except Exception:  # pragma: no cover - OCR must never break extraction
                logger.debug("OCR text extraction failed; continuing without grounding.")
                ocr_text = ""

        prompt = build_extraction_prompt(fields, doc_type, ocr_text=ocr_text or None)

        # Greedy decoding (config default) is what we want for extraction:
        # deterministic and less prone to hallucinated values.
        output = self.model.generate(pil_image, prompt)
        if isinstance(output, list):  # normalize a 1-element batch return
            output = output[0]

        raw_text = output.text or ""
        parsed = parse_json_answer(raw_text)

        # Project onto the requested schema: guarantee every requested key exists
        # (missing -> None), drop hallucinated extra keys, and normalize money.
        requested = resolve_fields(fields, doc_type)
        if fields or doc_type:
            projected = {k: parsed.get(k) for k in requested}
        else:
            projected = dict(parsed)
        for k in list(projected):
            if k in _MONEY_KEYS or any(m in k for m in _MONEY_KEYS):
                projected[k] = _normalize_money(projected[k])

        # Grounding: flag which values actually appear in the page text.
        grounded: dict[str, bool] = {}
        if ocr_text:
            ocr_norm = _norm_for_match(ocr_text)
            for k, v in projected.items():
                if isinstance(v, str) and v.strip():
                    grounded[k] = _value_grounded(v, ocr_norm)
            missing = [k for k, ok in grounded.items() if not ok]
            if missing:
                logger.debug("Ungrounded (possibly hallucinated) fields: %s", missing)

        return {
            "fields": projected,
            "confidence": float(output.confidence),
            "raw": raw_text,
            "grounded": grounded,
            "ocr_used": bool(ocr_text),
        }


__all__ = [
    "DOC_TYPE_FIELDS",
    "FIELD_DESCRIPTIONS",
    "build_extraction_prompt",
    "parse_json_answer",
    "resolve_fields",
    "FieldExtractor",
]
