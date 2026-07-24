"""Tests for the shared sample schema (``preprocessing/schema.py``).

``DocSample`` is the single contract every dataset loader and model adapter
agrees on, so its serialization round-trip and normalization rules are
load-bearing for the whole project. The schema module itself is stdlib-only, but
importing it goes through ``preprocessing/__init__`` which eagerly pulls the
torch-backed dataset classes; we therefore ``importorskip`` the module so this
file *runs its assertions when torch is present and skips cleanly when it is
not*, per the suite's lightweight-CI contract.
"""

from __future__ import annotations

import pytest

# Guard: importing preprocessing.schema triggers preprocessing/__init__, which
# imports the torch-backed DocumentDataset. A no-torch environment therefore
# raises ModuleNotFoundError on import -> importorskip converts that into a skip.
schema = pytest.importorskip("preprocessing.schema")

DocSample = schema.DocSample
Split = schema.Split


def test_docsample_record_roundtrip() -> None:
    """to_record() -> from_record() reconstructs an equivalent sample."""
    original = DocSample(
        image="documents/invoice_001.png",
        question="What is the invoice total?",
        answer="42.00",
        answers=["42.00", "$42.00"],
        boxes=[(0.10, 0.20, 0.30, 0.40)],
        box_labels=["total"],
        sample_id="s-001",
        task="qa",
        metadata={"source": "unit-test"},
    )
    record = original.to_record()
    # to_record must be a plain dict (HF datasets / JSON friendly), not the obj.
    assert isinstance(record, dict)
    assert record["question"] == original.question

    restored = DocSample.from_record(record)
    assert restored.question == original.question
    assert restored.answer == original.answer
    assert restored.sample_id == "s-001"
    assert restored.task == "qa"
    assert restored.box_labels == ["total"]
    assert restored.metadata == {"source": "unit-test"}


def test_primary_answer_prepended_to_answers() -> None:
    """__post_init__ guarantees the primary answer heads the answers list."""
    sample = DocSample(image="x", question="capital?", answer="Paris", answers=["Lutetia"])
    # Metrics like ANLS score against `answers`; the gold `answer` must always
    # be present (and first) so it can never be missed during scoring.
    assert sample.answers[0] == "Paris"
    assert "Lutetia" in sample.answers


def test_answers_do_not_duplicate_primary() -> None:
    """The primary answer is not re-added when already in ``answers`` (dedupe)."""
    sample = DocSample(image="x", question="q", answer="Paris", answers=["Paris", "paris"])
    # Only the exact-duplicate primary is suppressed; case variants are kept.
    assert sample.answers.count("Paris") == 1
    assert "paris" in sample.answers


def test_empty_answer_leaves_answers_untouched() -> None:
    """A falsy answer must not inject an empty string into ``answers``."""
    sample = DocSample(image="x", question="q", answer="")
    # `if self.answer` short-circuits, so no "" leaks in to pollute metrics.
    assert sample.answers == []


def test_from_record_coerces_boxes_to_tuples() -> None:
    """Boxes deserialized from JSON (lists) are coerced back to tuples."""
    # Cached datasets store boxes as JSON arrays; from_record must restore the
    # NormBox tuple type so downstream tuple-unpacking (x0, y0, x1, y1) works.
    record = {
        "image": "x",
        "question": "q",
        "answer": "a",
        "boxes": [[0.0, 0.1, 0.2, 0.3]],
        "box_labels": ["field"],
    }
    sample = DocSample.from_record(record)
    assert isinstance(sample.boxes[0], tuple)
    assert sample.boxes[0] == (0.0, 0.1, 0.2, 0.3)


def test_from_record_tolerates_missing_optional_fields() -> None:
    """from_record fills sane defaults for absent optional keys."""
    # A minimal record (only image) must not KeyError — loaders build partial
    # records for datasets that lack boxes/answers.
    sample = DocSample.from_record({"image": "only.png"})
    assert sample.image == "only.png"
    assert sample.question == ""
    assert sample.boxes == []
    assert sample.metadata == {}


def test_split_enum_is_str_valued() -> None:
    """Split is a str-enum whose members equal their canonical string values."""
    assert Split.TRAIN.value == "train"
    assert Split.VALIDATION.value == "validation"
    assert Split.TEST.value == "test"
    # str-mixin: members compare equal to the raw string, which lets callers use
    # Split.TRAIN interchangeably with "train" in dict keys / HF split names.
    assert Split.TRAIN == "train"
