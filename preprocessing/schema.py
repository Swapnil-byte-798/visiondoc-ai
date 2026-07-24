"""Uniform sample schema shared across every dataset loader and model.

The whole project is decoupled by this one contract: dataset loaders in
``preprocessing/datasets.py`` convert DocVQA / CORD / FUNSD / SROIE into a list
of :class:`DocSample`, and model adapters in ``models/`` know how to turn a
:class:`DocSample` into tensors. Neither side needs to know about the other's
concrete format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# A normalized bounding box in [0, 1] fractions: (x0, y0, x1, y1).
NormBox = tuple[float, float, float, float]


class Split(str, Enum):
    """Canonical split names used throughout the project."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class DocSample:
    """A single document-understanding example.

    Attributes
    ----------
    image:
        Either an in-memory ``PIL.Image`` or a path/URL string. We keep it lazy
        (path) where possible so large datasets are not held in RAM; the
        collator loads pixels only for the current batch.
    question:
        The natural-language question (for QA datasets) or an extraction prompt
        synthesized for field-extraction datasets (CORD/SROIE/FUNSD).
    answer:
        The gold answer string. For structured datasets this is a serialized
        JSON string of the target fields.
    answers:
        Optional list of *all* acceptable answers (DocVQA provides several);
        used by metrics like ANLS that score against the best match.
    boxes / box_labels:
        Optional normalized bounding boxes and their labels, used for region
        highlighting and layout-aware datasets (FUNSD).
    sample_id:
        Stable unique id for caching, error analysis, and report joins.
    task:
        "qa" or "extraction" — lets a model adapter pick the right prompt
        template.
    """

    image: Any
    question: str
    answer: str
    answers: list[str] = field(default_factory=list)
    boxes: list[NormBox] = field(default_factory=list)
    box_labels: list[str] = field(default_factory=list)
    sample_id: str = ""
    task: str = "qa"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Guarantee `answers` always contains the primary answer for metrics.
        if self.answer and self.answer not in self.answers:
            self.answers = [self.answer, *self.answers]

    def to_record(self) -> dict[str, Any]:
        """Serialize to a plain dict for a Hugging Face ``datasets.Dataset``.

        The image is stored as-is (PIL or path); ``datasets`` handles PIL via
        its Image feature, and paths remain lazy.
        """
        return {
            "image": self.image,
            "question": self.question,
            "answer": self.answer,
            "answers": self.answers,
            "boxes": self.boxes,
            "box_labels": self.box_labels,
            "sample_id": self.sample_id,
            "task": self.task,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> DocSample:
        """Inverse of :meth:`to_record` (used when reading a cached dataset)."""
        return cls(
            image=record["image"],
            question=record.get("question", ""),
            answer=record.get("answer", ""),
            answers=list(record.get("answers", []) or []),
            boxes=[tuple(b) for b in record.get("boxes", []) or []],
            box_labels=list(record.get("box_labels", []) or []),
            sample_id=record.get("sample_id", ""),
            task=record.get("task", "qa"),
            metadata=dict(record.get("metadata", {}) or {}),
        )


__all__ = ["DocSample", "Split", "NormBox"]
