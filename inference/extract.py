"""Structured field extraction on top of a :class:`VisionDocModel`.

Document QA answers a free-form question; *field extraction* asks the same VLM
to return a machine-readable object (``{"total": "$42.00", ...}``) for a known
document type. We implement this as a thin prompting + parsing layer rather than
a separate model because the instruction-tuned backbone already follows a
"return only JSON" directive well, and reusing the same weights keeps the
project self-contained (no second model to train or serve).

Two design decisions dominate this module:

* **Prompt shape.** We ask for a JSON object whose keys are *exactly* the
  requested fields and instruct the model to emit ``null`` for anything it
  cannot find. Pinning the key set makes downstream parsing schema-stable and
  lets the caller diff predictions against a gold object field-by-field.
* **Parsing must never trust the model's formatting.** Instruction-tuned models
  routinely wrap JSON in ```json fences, add a prose preamble, or emit a
  trailing comma. :func:`parse_json_answer` is therefore defensive: it strips
  fences, extracts the first *balanced* ``{...}`` span (brace-counting that
  respects string literals/escapes), and retries after a light trailing-comma
  repair before giving up with an empty dict. Returning ``{}`` instead of
  raising means a single malformed generation degrades one row, never the batch.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only import: keeps this module importable without torch.
    from PIL import Image

    from models.base import VisionDocModel

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical field sets per document type
# ---------------------------------------------------------------------------
# These are the keys the extractor asks for when the caller passes only a
# ``doc_type``. They are intentionally business-oriented (the fields a document
# intelligence product actually needs) rather than an exhaustive layout dump,
# because a tight, well-known key set yields far more reliable JSON than an
# open-ended "extract everything" request.
DOC_TYPE_FIELDS: dict[str, list[str]] = {
    "invoice": [
        "invoice_number",
        "invoice_date",
        "due_date",
        "vendor_name",
        "vendor_address",
        "bill_to",
        "subtotal",
        "tax",
        "total",
        "currency",
    ],
    "receipt": [
        "merchant_name",
        "merchant_address",
        "date",
        "time",
        "subtotal",
        "tax",
        "tip",
        "total",
        "payment_method",
    ],
    "id_card": [
        "full_name",
        "document_number",
        "date_of_birth",
        "sex",
        "nationality",
        "issue_date",
        "expiry_date",
        "address",
    ],
    "form": [
        "form_title",
        "name",
        "date",
        "address",
        "phone",
        "email",
        "signature_present",
    ],
}

# A compact, generic key set used when the caller gives neither ``fields`` nor a
# recognized ``doc_type``. We still constrain the schema (rather than asking for
# arbitrary keys) so the output stays parseable and comparable across documents.
_GENERIC_FIELDS: list[str] = [
    "document_type",
    "date",
    "total",
    "names",
    "key_values",
]


def resolve_fields(fields: list[str] | None, doc_type: str | None) -> list[str]:
    """Decide the concrete key set to request.

    Precedence: an explicit ``fields`` list wins; else the canonical set for a
    recognized ``doc_type``; else a generic fallback. Centralizing this here
    keeps :func:`build_extraction_prompt` and :meth:`FieldExtractor.extract`
    (which needs the same list to shape its output) perfectly in sync.
    """
    if fields:
        # De-duplicate while preserving order so the prompt and the parsed
        # result share a stable, caller-controlled key ordering.
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


def build_extraction_prompt(fields: list[str] | None, doc_type: str | None) -> str:
    """Build an instruction that asks the model for ONLY a JSON object.

    The prompt is phrased imperatively and lists the exact keys because VLMs
    honour an explicit key contract far more reliably than a vague "extract the
    fields" request. We also spell out the failure convention (``null`` for
    missing values) so absent fields produce a stable key rather than a dropped
    one — critical for field-level precision/recall scoring downstream.
    """
    keys = resolve_fields(fields, doc_type)
    doc_hint = f" from this {doc_type.strip()}" if doc_type else ""
    keys_block = ", ".join(f'"{k}"' for k in keys)

    # A worked skeleton nudges the model toward valid JSON and the right keys.
    skeleton = "{" + ", ".join(f'"{k}": ""' for k in keys) + "}"

    return (
        f"Extract the following fields{doc_hint} and return ONLY a single JSON "
        f"object with exactly these keys: {keys_block}.\n"
        "Rules:\n"
        "- Respond with valid JSON and nothing else (no explanations, no code "
        "fences).\n"
        "- Use the value exactly as written on the document.\n"
        "- If a field is not present, set its value to null.\n"
        f"JSON template to fill in:\n{skeleton}"
    )


# ---------------------------------------------------------------------------
# Robust JSON extraction from free-form model text
# ---------------------------------------------------------------------------

# Matches an opening code fence, optionally language-tagged (```json / ``` ).
_FENCE_OPEN_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```")
# Removes a trailing comma that precedes a closing } or ] — the single most
# common JSON syntax error emitted by language models.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code-fence wrappers, keeping the fenced content."""
    stripped = text.strip()
    if "```" in stripped:
        stripped = _FENCE_OPEN_RE.sub("", stripped, count=1)
        stripped = _FENCE_CLOSE_RE.sub("", stripped)
    return stripped.strip()


def _iter_balanced_objects(text: str):
    """Yield each balanced ``{...}`` substring in the text, in order.

    A naive ``text[text.find('{'):text.rfind('}')+1]`` breaks when the model
    appends a second object or prose containing braces. We brace-count from each
    successive ``{``, skipping over string literals so that braces inside quoted
    values do not affect nesting depth. Yielding *every* candidate (rather than
    only the first) lets the caller skip a leading ``{...}`` that is not valid
    JSON and still recover a valid object appearing later in the output.
    """
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
                # Inside a JSON string: only an unescaped quote ends it; a
                # backslash escapes the next character (so \" stays inside).
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
            return  # unbalanced tail; nothing more to find


def _load_json_object(candidate: str) -> dict | None:
    """Parse ``candidate`` as a JSON object, with a light trailing-comma repair.

    Returns ``None`` (not a raise) when the text is not a JSON object so callers
    can keep trying cheaper-to-more-aggressive strategies.
    """
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            obj = json.loads(attempt)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_json_answer(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output.

    Strategy, cheapest first: strip code fences and try to parse the whole
    string; if that fails, isolate the first balanced ``{...}`` span and parse
    that. Any failure yields ``{}`` so a malformed generation never propagates
    an exception into an evaluation or serving loop.
    """
    if not text or not text.strip():
        return {}

    cleaned = _strip_code_fences(text)

    direct = _load_json_object(cleaned)
    if direct is not None:
        return direct

    # Try each balanced {...} span until one parses as a JSON object, so a
    # leading non-JSON brace group doesn't hide a valid object further along.
    for snippet in _iter_balanced_objects(cleaned):
        parsed = _load_json_object(snippet)
        if parsed is not None:
            return parsed

    logger.debug("Could not parse a JSON object from model output: %.120r", text)
    return {}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FieldExtractor:
    """Run structured field extraction against a loaded :class:`VisionDocModel`.

    Stateless apart from the model reference, so a single instance can be shared
    by the predictor, the API, and batch jobs. The model is expected to already
    be loaded (and optionally LoRA-adapted); we only drive generation here.
    """

    def __init__(self, model: "VisionDocModel") -> None:
        # We hold the high-level wrapper (not the raw HF module) so we get its
        # confidence-scored ``generate`` and device handling for free.
        self.model = model

    def extract(
        self,
        image: Any,
        fields: list[str] | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """Extract ``fields`` (or a ``doc_type``'s canonical set) from ``image``.

        Returns ``{"fields": <dict>, "confidence": <float>, "raw": <str>}``.
        ``confidence`` is the generation's sequence confidence (a proxy for how
        sure the model is about the emitted tokens); ``raw`` is the undecorated
        model text, retained for debugging and for report/audit trails when the
        JSON parse comes back partial.
        """
        # Deferred import: ``utils.image_utils`` pulls PIL/NumPy. Importing it
        # here (not at module top) keeps ``inference.extract`` importable in a
        # lightweight, torch-less environment (e.g. metric-only unit tests).
        from utils.image_utils import load_image

        pil_image = load_image(image)
        prompt = build_extraction_prompt(fields, doc_type)

        # ``generate`` with a single (image, question) returns one
        # GenerationOutput; greedy decoding (config default) is what we want for
        # extraction — deterministic and less prone to hallucinated values.
        output = self.model.generate(pil_image, prompt)
        # Defensive: normalize in case a backbone returns a 1-element list.
        if isinstance(output, list):
            output = output[0]

        raw_text = output.text or ""
        parsed = parse_json_answer(raw_text)

        # When the caller specified an exact key set, project the parsed object
        # onto it: guarantee every requested key exists (missing -> None) and
        # drop any extra hallucinated keys. This gives the caller a schema-
        # stable result regardless of how disciplined the model was.
        requested = resolve_fields(fields, doc_type)
        if fields or doc_type:
            projected = {k: parsed.get(k) for k in requested}
        else:
            projected = parsed

        return {
            "fields": projected,
            "confidence": float(output.confidence),
            "raw": raw_text,
        }


__all__ = [
    "DOC_TYPE_FIELDS",
    "build_extraction_prompt",
    "parse_json_answer",
    "resolve_fields",
    "FieldExtractor",
]
