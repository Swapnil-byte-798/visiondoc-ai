"""Batch inference over a directory of documents.

A common product workflow is "run one question across a folder of scanned
invoices and give me a spreadsheet." This module is that: it walks a directory,
drives a :class:`DocumentPredictor` over every supported file, and returns a
tabular result (one row per image, one row per PDF *page*) that can be written
straight to CSV.

Why the shape it has:

* **pandas is imported lazily.** The rest of the inference package stays usable
  without pandas installed; only this convenience function needs it, so the cost
  is paid only when a batch job actually runs.
* **PDF pages become rows.** A multi-page PDF answers per page (via
  :meth:`DocumentPredictor.predict_pdf`), and each page gets its own row with a
  1-based ``page`` number so the output stays flat and joinable.
* **One bad file never sinks the job.** Any per-file error is logged and recorded
  as an empty-answer row with the error message, so a 10k-document run finishes
  and surfaces the failures instead of aborting midway.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only; avoids importing pandas/torch at module load.
    import pandas as pd

    from inference.predictor import DocumentPredictor, PredictionResult

logger = get_logger(__name__)

# Extensions we know how to rasterize/answer. Lower-cased for case-insensitive
# matching (scanners love ``.JPG``). PDFs are fanned out to one row per page.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_PDF_EXTS = {".pdf"}
_SUPPORTED_EXTS = _IMAGE_EXTS | _PDF_EXTS


def _iter_supported_files(input_dir: Path) -> list[Path]:
    """Return sorted supported files directly under ``input_dir``.

    Sorting makes the output deterministic (important for diffing runs and for
    reproducible reports). We scan a single directory level rather than recursing
    because a flat "inbox of documents" is the expected layout and recursion
    would surprise callers with unrelated nested assets.
    """
    files = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
    ]
    return sorted(files, key=lambda p: p.name)


def _row(
    filename: str,
    page: int,
    question: str,
    result: "PredictionResult",
) -> dict[str, Any]:
    """Flatten one :class:`PredictionResult` into a CSV-friendly record."""
    return {
        "file": filename,
        "page": page,
        "question": question,
        "answer": result.answer,
        "confidence": round(result.confidence, 4),
        "latency_ms": round(result.latency_ms, 2),
        # Count only: a full box list bloats the CSV and is rarely needed there;
        # callers wanting geometry should use the predictor's result objects.
        "n_regions": len(result.regions),
    }


def _error_row(filename: str, question: str, error: str) -> dict[str, Any]:
    """A placeholder row for a file that failed, so the job stays complete."""
    return {
        "file": filename,
        "page": 1,
        "question": question,
        "answer": "",
        "confidence": 0.0,
        "latency_ms": 0.0,
        "n_regions": 0,
        "error": error,
    }


def batch_infer_dir(
    predictor: "DocumentPredictor",
    input_dir: str,
    question: str,
    output_csv: str | None = None,
) -> "pd.DataFrame":
    """Answer ``question`` for every supported file in ``input_dir``.

    Parameters
    ----------
    predictor:
        A ready :class:`DocumentPredictor` (model already loaded).
    input_dir:
        Directory to scan (single level) for ``.png/.jpg/.jpeg/.pdf`` files.
    question:
        The question asked of every document/page.
    output_csv:
        If given, the resulting table is also written to this CSV path (parent
        directories are created as needed).

    Returns
    -------
    pandas.DataFrame
        One row per image and one row per PDF page. Empty (with the expected
        columns) when the directory contains no supported files.
    """
    # Lazy import: keep pandas out of the package's import cost.
    import pandas as pd

    directory = Path(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    files = _iter_supported_files(directory)
    if not files:
        logger.warning("No supported documents (%s) found in %s", sorted(_SUPPORTED_EXTS), input_dir)

    records: list[dict[str, Any]] = []
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix in _PDF_EXTS:
                page_results = predictor.predict_pdf(str(path), question)
                if not page_results:
                    # A valid-but-empty PDF still deserves a row so it is visible.
                    records.append(_error_row(path.name, question, "no_pages_rasterized"))
                for page_number, result in enumerate(page_results, start=1):
                    records.append(_row(path.name, page_number, question, result))
            else:
                result = predictor.answer(str(path), question)
                records.append(_row(path.name, page=1, question=question, result=result))
        except Exception as exc:  # never let one document abort the batch.
            logger.exception("Inference failed for %s; recording an error row.", path.name)
            records.append(_error_row(path.name, question, f"{type(exc).__name__}: {exc}"))

    # Stable column order even when every row succeeded (no ``error`` key). We
    # build the frame then reindex so the CSV header is predictable, with the
    # optional ``error`` column appended only when at least one row set it.
    base_columns = ["file", "page", "question", "answer", "confidence", "latency_ms", "n_regions"]
    has_error = any("error" in r for r in records)
    columns = base_columns + (["error"] if has_error else [])
    frame = pd.DataFrame(records, columns=columns)

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        logger.info("Wrote %d rows to %s", len(frame), out_path)

    return frame


__all__ = ["batch_infer_dir"]
