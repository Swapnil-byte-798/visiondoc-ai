"""Inference package: the user-facing serving layer for VisionDoc AI.

Public API::

    from inference import DocumentPredictor, PredictionResult
    predictor = DocumentPredictor.from_config(load_config())
    result = predictor.answer(image, "What is the invoice total?")
    fields = predictor.extract(image, doc_type="invoice")

    from inference import batch_infer_dir
    df = batch_infer_dir(predictor, "data/inbox", "What is the total?", "out.csv")

Everything here is a thin, robust orchestration layer over the ``models``,
``utils`` and ``evaluation`` foundations — it adds no new model intelligence,
only prompt construction (field extraction), region grounding (OCR), and I/O
convenience (PDF fan-out, directory batches).

The re-exports below pull in torch/PIL indirectly, so importing this package is
only appropriate where the ML stack is available (the API and app do exactly
that on first use); metric/config-only code should import the submodules it
needs directly.
"""

from __future__ import annotations

from inference.batch import batch_infer_dir
from inference.extract import (
    DOC_TYPE_FIELDS,
    FieldExtractor,
    build_extraction_prompt,
    parse_json_answer,
)
from inference.predictor import DocumentPredictor, PredictionResult

__all__ = [
    "DocumentPredictor",
    "PredictionResult",
    "FieldExtractor",
    "DOC_TYPE_FIELDS",
    "batch_infer_dir",
    "build_extraction_prompt",
    "parse_json_answer",
]
