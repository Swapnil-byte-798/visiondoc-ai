"""Evaluation metrics for document-VQA and structured extraction.

Why this module looks the way it does
--------------------------------------
Document intelligence spans several answer shapes — free-form QA (DocVQA),
short exact fields (SROIE/CORD), and multi-key structured extraction — so no
single scalar tells the whole story. We therefore expose a small toolbox of
metrics and one :func:`aggregate_qa_metrics` roll-up that the evaluation
pipeline and reports consume.

Design constraints that shaped the code:

* **Graceful degradation over hard dependencies.** The heavy scoring libraries
  (``sacrebleu``, ``rouge_score``, ``scikit-learn``, ``Levenshtein``,
  ``nltk``) are *optional at runtime*. CI, the Docker slim image, or a laptop
  demo may not have all of them. Every one of them is imported **lazily inside
  the function that needs it** and, if missing, we log a debug line and return
  a neutral ``0.0`` (or a pure-Python fallback) instead of crashing an entire
  evaluation run over one unavailable metric.
* **SQuAD-style normalization** for exact-match/ANLS so scores are comparable
  with the wider VQA/reading-comprehension literature (lowercasing, punctuation
  and article stripping, whitespace collapse).
* **Stdlib-only core.** ``normalize_text``, ``exact_match``, ``anls`` and
  ``token_level_prf`` depend on nothing heavier than the standard library
  (ANLS uses ``Levenshtein`` when present but falls back to a pure-Python DP),
  which is what the unit tests assert against.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Articles stripped during SQuAD-style normalization. Compiled once at import
# time because normalization is called O(preds x refs) times per evaluation.
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}


# ---------------------------------------------------------------------------
# Text normalization & string-similarity primitives (stdlib only)
# ---------------------------------------------------------------------------
def normalize_text(s: str | None) -> str:
    """Normalize an answer string the way the SQuAD/DocVQA evaluators do.

    Steps: lowercase -> replace punctuation with spaces -> drop the articles
    ``a/an/the`` -> collapse runs of whitespace. This makes surface-form
    differences ("Total: $4.00" vs "total 4 00") irrelevant so that metrics
    reward *semantic* agreement rather than punctuation/casing luck.

    ``None`` and non-string inputs are coerced to ``""`` / ``str`` so callers
    never have to pre-sanitize model output (which can occasionally be ``None``
    on a generation failure).
    """
    if s is None:
        return ""
    # Punctuation -> space (not deletion) so "in-store" -> "in store", keeping
    # token boundaries that a pure-deletion pass would incorrectly merge.
    text = str(s).lower().translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    # Collapse whitespace last so article removal never leaves double spaces.
    return " ".join(text.split())


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings.

    Prefers the C-accelerated ``Levenshtein`` package (installed in production
    for fast ANLS over large test sets) but falls back to a classic two-row
    dynamic-programming implementation so the metric works in a stdlib-only
    environment. The two-row variant keeps memory at O(min(len)) rather than
    O(len_a * len_b), which matters for long OCR-heavy answers.
    """
    try:  # C extension — ~100x faster on real test sets.
        import Levenshtein  # type: ignore

        return int(Levenshtein.distance(a, b))
    except Exception:  # pragma: no cover - exercised only without the C ext.
        logger.debug("Levenshtein C extension unavailable; using pure-Python DP.")

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure the inner row is over the shorter string to bound memory.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _normalized_levenshtein(a: str, b: str) -> float:
    """Levenshtein distance scaled into ``[0, 1]`` by the longer length.

    ``0`` means identical, ``1`` means maximally different. This is the ``NL``
    term in the ANLS definition. Two empty strings are treated as identical.
    """
    if not a and not b:
        return 0.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return _levenshtein(a, b) / longest


# ---------------------------------------------------------------------------
# Per-sample QA metrics
# ---------------------------------------------------------------------------
def exact_match(pred: str, refs: list[str]) -> float:
    """1.0 if the normalized prediction equals *any* normalized reference.

    DocVQA/SQuAD answers come as a *set* of acceptable strings (annotator
    variants), so we credit a match against the best reference rather than a
    single canonical one.
    """
    if not refs:
        return 0.0
    pred_norm = normalize_text(pred)
    return 1.0 if any(pred_norm == normalize_text(r) for r in refs) else 0.0


def _anls_normalize(s: str) -> str:
    """Lowercase + whitespace-collapse ONLY — the official DocVQA ANLS convention.

    Deliberately does NOT strip articles/punctuation (unlike :func:`normalize_text`):
    the canonical ANLS lowercases and trims whitespace only, so reusing the
    SQuAD-style normalizer here would shrink edit distances and systematically
    *inflate* the reported ANLS.
    """
    return " ".join(str(s or "").lower().split())


def anls(pred: str, refs: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity — the standard DocVQA metric.

    For each reference we compute ``1 - NL(pred, ref)`` (a soft similarity that
    tolerates minor OCR/spelling slips) and keep the best. The ANLS convention
    then *zeroes* any score below ``threshold`` (default 0.5): a near-miss is
    partially credited, but a wrong answer earns nothing rather than leaking
    similarity points. We use the official lowercase+whitespace normalization
    (see :func:`_anls_normalize`), not the SQuAD-style one, so the score matches
    published DocVQA numbers.
    """
    if not refs:
        return 0.0
    pred_norm = _anls_normalize(pred)
    best = 0.0
    for ref in refs:
        similarity = 1.0 - _normalized_levenshtein(pred_norm, _anls_normalize(ref))
        if similarity > best:
            best = similarity
    # Threshold gate: below it the answer is considered wrong (score 0).
    return best if best >= threshold else 0.0


def token_level_prf(pred: str, ref: str) -> dict[str, float]:
    """Bag-of-tokens precision/recall/F1 over whitespace tokens.

    Uses multiset (``Counter``) overlap so repeated tokens are counted
    correctly — this is the SQuAD token-F1 formulation and rewards partial
    answers (e.g. predicting "New York" when the gold is "New York City").
    Both sides are SQuAD-normalized first for the same reasons as EM/ANLS.
    """
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()

    # Degenerate cases: if either side is empty, F1 is 1.0 only when *both*
    # are empty (they agree perfectly), otherwise 0.0 — matching SQuAD.
    if not pred_tokens or not ref_tokens:
        agree = float(pred_tokens == ref_tokens)
        return {"precision": agree, "recall": agree, "f1": agree}

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Corpus-level generation metrics (optional heavy deps)
# ---------------------------------------------------------------------------
def compute_bleu(preds: list[str], refs: list[list[str]]) -> float:
    """Corpus BLEU (0..100 scale) via ``sacrebleu``, ``nltk`` fallback, else 0.

    ``refs`` is ragged per-sample (each hypothesis may have several
    references). ``sacrebleu.corpus_bleu`` wants references *transposed* to
    ``[[ref_i for sample], ...]`` and pads uneven counts with empty strings,
    which we handle explicitly so a single missing annotator variant does not
    misalign the whole corpus.
    """
    if not preds:
        return 0.0

    # Preferred: sacrebleu (tokenization-agnostic, the MT-community standard).
    try:
        import sacrebleu  # type: ignore

        # Drop hypotheses that have no references, then transpose to one stream
        # per reference slot. We fill ragged slots by REPEATING a genuine
        # reference (modulo), never with empty strings: an empty-string ref is a
        # zero-length reference that corrupts sacrebleu's brevity penalty and
        # inflates corpus BLEU on short-answer data like DocVQA.
        paired = [(p, r) for p, r in zip(preds, refs) if r]
        if not paired:
            return 0.0
        preds2, refs2 = zip(*paired)
        max_refs = max(len(r) for r in refs2)
        transposed = [
            [refs2[i][j % len(refs2[i])] for i in range(len(preds2))]
            for j in range(max_refs)
        ]
        return float(sacrebleu.corpus_bleu(list(preds2), transposed).score)
    except Exception:
        logger.debug("sacrebleu unavailable; trying nltk BLEU fallback.")

    # Fallback: nltk sentence BLEU averaged over the corpus with smoothing so
    # short document answers do not collapse to 0 on missing higher n-grams.
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore

        smooth = SmoothingFunction().method1
        scores: list[float] = []
        for pred, ref_list in zip(preds, refs):
            if not ref_list:
                continue
            tokenized_refs = [r.split() for r in ref_list]
            scores.append(sentence_bleu(tokenized_refs, pred.split(), smoothing_function=smooth))
        # Return on the same 0..100 scale as sacrebleu for report consistency.
        return float(100.0 * sum(scores) / len(scores)) if scores else 0.0
    except Exception:
        logger.debug("nltk BLEU unavailable; returning 0.0 for BLEU.")
        return 0.0


def compute_rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    """ROUGE-1/2/L F-measures via ``rouge_score``; zeros if the dep is absent.

    ROUGE captures recall-oriented overlap that BLEU (precision-oriented)
    misses, which is informative for longer extractive answers. ``refs`` here
    is a single reference per prediction (the aggregator passes the primary
    annotator answer); if you have multiple, pick the canonical one upstream.
    """
    zero = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not preds:
        return zero
    try:
        from rouge_score import rouge_scorer  # type: ignore

        # use_stemmer=True so "invoices"/"invoice" are not spuriously distinct.
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n = 0
        for pred, ref in zip(preds, refs):
            scores = scorer.score(ref, pred)
            for key in totals:
                totals[key] += scores[key].fmeasure
            n += 1
        if n == 0:
            return zero
        return {key: totals[key] / n for key in totals}
    except Exception:
        logger.debug("rouge_score unavailable; returning 0.0 ROUGE.")
        return zero


def compute_classification_metrics(
    y_true: list, y_pred: list, labels: list | None = None
) -> dict[str, float]:
    """Accuracy + macro precision/recall/F1 for document-type classification.

    Prefers ``scikit-learn`` (handles label sets, zero-division, macro
    averaging correctly). Without it we fall back to a manual computation so at
    least accuracy and a hand-rolled macro-F1 are still reported rather than
    the whole classification row going blank.
    """
    empty = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not y_true or len(y_true) != len(y_pred):
        return empty

    try:
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            precision_recall_fscore_support,
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    except Exception:
        logger.debug("scikit-learn unavailable; using manual classification metrics.")

    # Manual macro-averaged fallback (pure stdlib).
    label_set = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    precisions, recalls, f1s = [], [], []
    for label in label_set:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == label and t == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == label and t != label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != label and t == label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    n_labels = len(label_set) or 1
    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / n_labels,
        "recall": sum(recalls) / n_labels,
        "f1": sum(f1s) / n_labels,
    }


# ---------------------------------------------------------------------------
# Structured field extraction
# ---------------------------------------------------------------------------
def _norm_field_value(value: object) -> str:
    """Case/whitespace-insensitive canonical form of a field value.

    Extraction gold and predictions differ in trivial ways ("$4.00" vs "4.00 ",
    "ACME" vs "acme"); we normalize both sides before comparing so the metric
    measures real extraction quality, not formatting noise.
    """
    return " ".join(str(value).strip().lower().split())


def structured_field_metrics(
    pred_fields: list[dict], gold_fields: list[dict]
) -> dict[str, float]:
    """Micro precision/recall/F1 over (key, value) pairs across documents.

    ``pred_fields`` and ``gold_fields`` are parallel lists — one dict of
    extracted fields per document. A predicted field counts as a true positive
    only when the (case-insensitive) key exists in gold **and** the normalized
    value matches. We aggregate counts across the whole corpus (micro-average)
    because per-document averaging would over-weight documents that happen to
    have few fields.
    """
    tp = 0
    n_pred = 0
    n_gold = 0

    for pred, gold in zip(pred_fields, gold_fields):
        pred = pred or {}
        gold = gold or {}
        # Normalize gold keys once per document for O(1) lookup.
        gold_norm = {str(k).strip().lower(): _norm_field_value(v) for k, v in gold.items()}
        n_pred += len(pred)
        n_gold += len(gold)
        for key, value in pred.items():
            norm_key = str(key).strip().lower()
            if norm_key in gold_norm and gold_norm[norm_key] == _norm_field_value(value):
                tp += 1

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"field_precision": precision, "field_recall": recall, "field_f1": f1}


# ---------------------------------------------------------------------------
# Aggregate roll-up consumed by evaluation.evaluate / reports
# ---------------------------------------------------------------------------
def aggregate_qa_metrics(
    preds: list[str],
    refs: list[list[str]],
    confidences: list[float] | None = None,
) -> dict[str, float]:
    """Roll every applicable QA metric into a single flat dict.

    For per-sample metrics (EM, ANLS, token P/R/F1) we score the prediction
    against the *best* reference and average across the dataset. Corpus metrics
    (BLEU) run once over the whole set. ROUGE uses each sample's primary
    reference. ``avg_confidence`` surfaces the model's self-reported certainty
    so reports can correlate confidence with correctness. ``n`` records the
    sample count for transparency (and to guard against silent empty runs).

    All keys are always present so downstream code (report tables, DataFrame
    construction) can rely on a stable schema even when a metric degraded to 0.
    """
    n = len(preds)
    keys = [
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
    ]
    if n == 0:
        return {k: 0.0 for k in keys}

    em_sum = 0.0
    anls_sum = 0.0
    f1_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    primary_refs: list[str] = []

    for pred, ref_list in zip(preds, refs):
        ref_list = ref_list or []
        em_sum += exact_match(pred, ref_list)
        anls_sum += anls(pred, ref_list)
        # Token P/R/F1: choose the reference that maximizes F1 for this sample,
        # so multi-annotator answers are scored fairly (best-match semantics).
        best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ref in ref_list:
            prf = token_level_prf(pred, ref)
            if prf["f1"] >= best["f1"]:
                best = prf
        f1_sum += best["f1"]
        prec_sum += best["precision"]
        rec_sum += best["recall"]
        # Primary reference for corpus ROUGE (first annotator answer).
        primary_refs.append(ref_list[0] if ref_list else "")

    bleu = compute_bleu(preds, refs)
    rouge = compute_rouge(preds, primary_refs)

    # avg_confidence defaults to 0.0 when the caller has no confidence signal
    # (e.g. beam search without probabilities) — never None, to keep the schema
    # numeric for report rendering / averaging.
    if confidences:
        valid = [c for c in confidences if c is not None]
        avg_conf = sum(valid) / len(valid) if valid else 0.0
    else:
        avg_conf = 0.0

    return {
        "exact_match": em_sum / n,
        "anls": anls_sum / n,
        "token_f1": f1_sum / n,
        "precision": prec_sum / n,
        "recall": rec_sum / n,
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "avg_confidence": avg_conf,
        "n": float(n),
    }


__all__ = [
    "normalize_text",
    "exact_match",
    "anls",
    "token_level_prf",
    "compute_bleu",
    "compute_rouge",
    "compute_classification_metrics",
    "structured_field_metrics",
    "aggregate_qa_metrics",
]
