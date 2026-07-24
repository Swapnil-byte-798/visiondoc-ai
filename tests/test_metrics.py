"""Tests for the evaluation metrics core (``evaluation/metrics.py``).

The functions asserted here — ``normalize_text``, ``exact_match``, ``anls`` and
``token_level_prf`` — are the stdlib-only heart of the evaluator (ANLS uses an
optional C-accelerated Levenshtein but falls back to a pure-Python DP, so no
heavy dep is required for correctness). We therefore assert exact numeric
behaviour on hand-computed cases.

Importing ``evaluation.metrics`` pulls ``utils.logging_utils``, which routes
through ``utils/__init__`` and its torch-backed device module; so we
``importorskip`` the module to skip cleanly when torch is absent while still
running the full assertions whenever it is present.
"""

from __future__ import annotations

import pytest

# Guarded import: transitively needs torch via utils/__init__ -> utils.device.
metrics = pytest.importorskip("evaluation.metrics")


# ---------------------------------------------------------------------------
# normalize_text — SQuAD/DocVQA-style normalization
# ---------------------------------------------------------------------------
def test_normalize_text_strips_articles_and_punctuation() -> None:
    """Lowercase, punctuation->space, article removal, whitespace collapse."""
    # "The Total: $4.00!" -> lower -> punct-to-space -> drop "the" -> collapse.
    assert metrics.normalize_text("The Total: $4.00!") == "total 4 00"


def test_normalize_text_removes_only_standalone_articles() -> None:
    """Articles are stripped as *words*, not as substrings."""
    # "a"/"an"/"the" gone, but "an" inside "banana" survives (word boundaries).
    assert metrics.normalize_text("The banana and an apple") == "banana and apple"


def test_normalize_text_handles_none() -> None:
    """``None`` (a possible generation failure) coerces to empty string."""
    assert metrics.normalize_text(None) == ""


# ---------------------------------------------------------------------------
# exact_match — normalized set membership
# ---------------------------------------------------------------------------
def test_exact_match_is_normalization_aware() -> None:
    """Case + article differences must not defeat an exact match."""
    assert metrics.exact_match("Paris", ["paris"]) == 1.0
    assert metrics.exact_match("The Paris", ["paris"]) == 1.0  # article stripped
    assert metrics.exact_match("Paris", ["London", "Berlin"]) == 0.0


def test_exact_match_empty_refs_is_zero() -> None:
    """No references => nothing to match => 0.0 (never a crash)."""
    assert metrics.exact_match("anything", []) == 0.0


# ---------------------------------------------------------------------------
# anls — Average Normalized Levenshtein Similarity (the DocVQA metric)
# ---------------------------------------------------------------------------
def test_anls_identical_is_one() -> None:
    """Identical (post-normalization) strings score a perfect 1.0."""
    assert metrics.anls("invoice total", ["invoice total"]) == 1.0


def test_anls_very_different_gated_to_zero() -> None:
    """A string share below the 0.5 threshold is gated to exactly 0.0."""
    # 8 distinct chars vs 16 z's: NL == 1.0 -> similarity 0.0 -> below threshold.
    assert metrics.anls("abcdefgh", ["zzzzzzzzzzzzzzzz"]) == 0.0


def test_anls_near_miss_gets_partial_credit() -> None:
    """A one-edit near miss stays above threshold and earns partial credit."""
    # "pariss" vs "paris": edit distance 1 over length 6 -> similarity ~0.833.
    score = metrics.anls("Pariss", ["Paris"])
    assert 0.5 < score < 1.0


def test_anls_empty_refs_is_zero() -> None:
    """No references => 0.0."""
    assert metrics.anls("x", []) == 0.0


# ---------------------------------------------------------------------------
# token_level_prf — SQuAD token-overlap precision/recall/F1
# ---------------------------------------------------------------------------
def test_token_level_prf_partial_overlap_is_sane() -> None:
    """Prediction fully contained in the reference: precision 1, recall < 1."""
    prf = metrics.token_level_prf("New York", "New York City")
    # 2 overlapping tokens; pred has 2 -> precision 1.0; ref has 3 -> recall 2/3.
    assert prf["precision"] == pytest.approx(1.0)
    assert prf["recall"] == pytest.approx(2 / 3)
    assert prf["f1"] == pytest.approx(0.8)


def test_token_level_prf_perfect_match() -> None:
    """Exact token agreement scores 1.0 across the board."""
    prf = metrics.token_level_prf("total amount due", "total amount due")
    assert prf["precision"] == 1.0
    assert prf["recall"] == 1.0
    assert prf["f1"] == 1.0


def test_token_level_prf_disjoint_is_zero() -> None:
    """No shared tokens => zeros (not a division-by-zero crash)."""
    prf = metrics.token_level_prf("apple", "orange")
    assert prf == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ---------------------------------------------------------------------------
# aggregate_qa_metrics — stable schema even when heavy metric deps degrade
# ---------------------------------------------------------------------------
def test_aggregate_qa_metrics_schema_and_core_values() -> None:
    """The roll-up always exposes every key; core (stdlib) metrics are exact."""
    out = metrics.aggregate_qa_metrics(["paris"], [["paris"]], confidences=[0.9])
    # BLEU/ROUGE may degrade to 0.0 without sacrebleu/rouge_score, but the keys
    # must always exist so report tables / DataFrame construction never KeyError.
    for key in (
        "exact_match",
        "anls",
        "token_f1",
        "precision",
        "recall",
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "avg_confidence",
        "n",
    ):
        assert key in out
    assert out["exact_match"] == 1.0
    assert out["anls"] == 1.0
    assert out["n"] == 1.0
    assert out["avg_confidence"] == pytest.approx(0.9)


def test_aggregate_qa_metrics_empty_inputs() -> None:
    """Empty inputs return a well-formed all-zero row (guards silent-empty runs)."""
    out = metrics.aggregate_qa_metrics([], [])
    assert out["n"] == 0.0
    assert out["exact_match"] == 0.0
