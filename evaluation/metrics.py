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
* **Point estimates are not publishable on their own.** Every headline quality
  metric can be accompanied by a per-sample score vector
  (:func:`per_sample_qa_scores`, :func:`per_sample_field_scores`) and a
  deterministic percentile-bootstrap confidence interval
  (:func:`bootstrap_ci`, or ``with_ci=True`` on the aggregators). Report the
  interval and ``n`` with the number — on a 200-document eval set the interval
  is wide enough to swallow small baseline-vs-fine-tuned gaps, and hiding that
  would misrepresent the result.
"""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from functools import lru_cache

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
# Uncertainty quantification (percentile bootstrap)
# ---------------------------------------------------------------------------
# WHY a bootstrap at all
# ---------------------
# A point estimate like "ANLS = 0.41" is not publishable on its own: on an
# eval set of n=200 the sampling error is large enough that a baseline-vs-LoRA
# gap of a few points can be pure noise. Every quality number this module
# produces is therefore reportable with an interval.
#
# WHY the *percentile bootstrap* specifically
# -------------------------------------------
# * Our per-sample scores are bounded in [0, 1] and wildly non-normal — ANLS is
#   a spike at 0 (threshold-gated) plus a blob near 1, exact-match is literally
#   Bernoulli. A normal-approximation / t-interval assumes a shape the data
#   does not have and can produce bounds outside [0, 1].
# * The bootstrap makes no distributional assumption: it resamples *documents*
#   (the unit of independence) with replacement and reads the empirical spread
#   of the statistic. It also extends cleanly to non-mean statistics such as the
#   micro-averaged field F1, where the estimator is a ratio of summed counts and
#   there is no closed-form standard error worth trusting.
# * Percentile (rather than BCa) because it is dependency-free, deterministic,
#   easy to audit, and at n=200 the residual bias-correction is far smaller than
#   the interval width itself.
#
# HONEST CAVEAT that must travel with the numbers: at n=200 these intervals are
# WIDE (typically +/- 5-7 absolute points on a mid-range metric). They quantify
# sampling noise on THIS eval set only — they say nothing about variance across
# training seeds, and they do not license a claim that a gap smaller than the
# overlap of the two intervals is real.
_DEFAULT_N_BOOT = 2000


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    Hand-rolled (rather than ``numpy.percentile``) so the returned interval is
    byte-identical whether or not numpy happens to be installed in the
    environment that produced a published table. Matches numpy's default
    ``method="linear"`` so cross-checking against numpy is meaningful.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


@lru_cache(maxsize=2)
def _bootstrap_index_stream(
    n: int, n_boot: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Deterministic resampling indices, generated by the *stdlib* RNG.

    Index generation is deliberately kept in :mod:`random` even when numpy is
    available: numpy's Generator is a different bit stream, so letting it draw
    the indices would make a published CI depend on whether numpy happened to
    be installed. numpy is used only to vectorize the arithmetic afterwards
    (see :func:`bootstrap_ci`), which keeps the resamples identical across
    environments.

    Implementation notes:

    * One ``Random.choices`` call for all ``n_boot * n`` draws, then sliced into
      replicates. ``choices`` consumes exactly one ``random()`` per draw, so the
      flat-then-slice form yields the *same* stream as one call per replicate —
      it is ~3x faster, not a different sampler.
    * The population is materialized as a ``list``; indexing a ``range`` inside
      ``choices`` is measurably slower for the same result.
    * Results are memoized (``maxsize=2``) because the three QA metrics in one
      :func:`aggregate_qa_metrics` call share ``(n, n_boot, seed)`` and would
      otherwise re-draw identical indices three times. Rows are tuples so a
      cached stream cannot be mutated by a consumer. Peak memory is bounded at
      roughly ``2 * n_boot * n`` pointers (~6MB at the defaults), which matters
      on a 12.7GB Colab box.
    """
    rng = random.Random(seed)
    population = list(range(n))
    flat = rng.choices(population, k=n * n_boot)
    return tuple(tuple(flat[i * n : (i + 1) * n]) for i in range(n_boot))


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the MEAN of ``values``.

    Args:
        values: the per-sample score vector (one entry per evaluated document).
            Scores are expected in ``[0, 1]`` but nothing here requires it.
        confidence: nominal coverage, e.g. ``0.95`` for a 95% interval.
        n_boot: number of bootstrap replicates. 2000 is the usual floor for a
            stable 95% percentile interval; below ~1000 the endpoints
            themselves become noticeably noisy.
        seed: RNG seed. The same ``(values, confidence, n_boot, seed)`` always
            yields the same interval, so a published number can be regenerated.

    Returns:
        ``(ci_low, ci_high)``. Degenerate inputs return a zero-width interval
        rather than raising, because an evaluation run must never die on a
        reporting detail: an empty vector gives ``(0.0, 0.0)`` and a
        single-sample vector gives ``(v, v)`` (an honest "n=1 tells us nothing"
        signal that the caller reports alongside ``n``).

    Raises:
        ValueError: on a nonsensical ``confidence`` or ``n_boot``; those are
            programmer errors, not data conditions, and silently substituting a
            default would hide a wrong published interval.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot!r}")

    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (vals[0], vals[0])

    indices = _bootstrap_index_stream(n, n_boot, seed)

    means: list[float]
    try:  # Fast path: the arithmetic (not the sampling) is vectorized.
        import numpy as np  # type: ignore

        arr = np.asarray(vals, dtype=np.float64)
        means = arr[np.asarray(indices, dtype=np.int64)].mean(axis=1).tolist()
    except Exception:
        logger.debug("numpy unavailable; using pure-Python bootstrap means.")
        # Plain sum() over <=n bounded floats; the accumulation error (~1e-16)
        # is many orders below the interval width, so math.fsum would only buy
        # 2x slower replicates. The two paths can therefore differ in the last
        # float bits — never at any precision a report prints.
        means = [sum(map(vals.__getitem__, row)) / n for row in indices]

    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return (_percentile(means, alpha), _percentile(means, 1.0 - alpha))


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


def _field_counts_per_doc(
    pred_fields: list[dict], gold_fields: list[dict]
) -> list[tuple[int, int, int]]:
    """Per-document ``(true_positives, n_pred, n_gold)`` field counts.

    Factored out because two very different consumers need exactly these
    numbers: the micro-average point estimate (sum the columns once) and the
    bootstrap (resample *documents*, then sum the columns per replicate).
    Keeping one implementation guarantees the CI is an interval around the
    number we actually publish, not around a subtly different statistic.
    """
    counts: list[tuple[int, int, int]] = []
    for pred, gold in zip(pred_fields, gold_fields):
        pred = pred or {}
        gold = gold or {}
        # Normalize gold keys once per document for O(1) lookup.
        gold_norm = {str(k).strip().lower(): _norm_field_value(v) for k, v in gold.items()}
        tp = 0
        for key, value in pred.items():
            norm_key = str(key).strip().lower()
            if norm_key in gold_norm and gold_norm[norm_key] == _norm_field_value(value):
                tp += 1
        counts.append((tp, len(pred), len(gold)))
    return counts


def _micro_prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    """Micro precision/recall/F1 from summed counts (0.0 on empty denominators)."""
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def per_sample_field_scores(
    pred_fields: list[dict], gold_fields: list[dict]
) -> dict[str, list[float]]:
    """Per-DOCUMENT field precision/recall/F1 vectors (same order as inputs).

    Returns ``{"field_precision": [...], "field_recall": [...],
    "field_f1": [...]}`` with one entry per document, for error analysis and
    for plotting the score distribution behind a headline number.

    Degenerate documents follow this module's existing convention (see
    :func:`token_level_prf`): a document with *no* predicted and *no* gold
    fields scores 1.0 (the two sides agree perfectly); otherwise an empty
    denominator scores 0.0.

    NOTE — these macro-style per-document scores are deliberately **not** what
    :func:`structured_field_metrics` publishes or bootstraps. The headline
    number is micro-averaged over all (key, value) pairs so that receipts with
    many fields are not down-weighted to equal a receipt with two. Expect the
    mean of these vectors to differ from the micro figures; use them for
    inspection, not for the published table.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for tp, n_pred, n_gold in _field_counts_per_doc(pred_fields, gold_fields):
        if n_pred == 0 and n_gold == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        precision, recall, f1 = _micro_prf(tp, n_pred, n_gold)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {"field_precision": precisions, "field_recall": recalls, "field_f1": f1s}


def structured_field_metrics(
    pred_fields: list[dict],
    gold_fields: list[dict],
    with_ci: bool = False,
    ci_confidence: float = 0.95,
    ci_n_boot: int = _DEFAULT_N_BOOT,
    ci_seed: int = 0,
) -> dict[str, float]:
    """Micro precision/recall/F1 over (key, value) pairs across documents.

    ``pred_fields`` and ``gold_fields`` are parallel lists — one dict of
    extracted fields per document. A predicted field counts as a true positive
    only when the (case-insensitive) key exists in gold **and** the normalized
    value matches. We aggregate counts across the whole corpus (micro-average)
    because per-document averaging would over-weight documents that happen to
    have few fields.

    Args:
        pred_fields: one dict of predicted fields per document.
        gold_fields: the parallel gold dicts.
        with_ci: when True, additionally return ``field_precision_ci_low`` /
            ``_ci_high`` (and the same for recall and F1), ``n`` (number of
            documents) and ``ci_level``. Off by default so existing callers see
            byte-identical output and never pay the bootstrap cost.
        ci_confidence: nominal coverage of the interval.
        ci_n_boot: bootstrap replicates.
        ci_seed: RNG seed, so a published interval is reproducible.

    Why resample documents rather than fields: documents are the independent
    sampling unit. Fields within one receipt are correlated (a bad OCR crop
    ruins all of them at once), so resampling individual fields would
    understate the true uncertainty. Each replicate re-sums the ``(tp, n_pred,
    n_gold)`` counts of the drawn documents and recomputes the micro ratio, so
    the interval brackets exactly the estimator reported here.

    Caveat to publish alongside: on the n=200 CORD eval set these intervals are
    wide, and two arms whose intervals overlap have not been shown to differ.
    """
    counts = _field_counts_per_doc(pred_fields, gold_fields)
    precision, recall, f1 = _micro_prf(
        sum(c[0] for c in counts),
        sum(c[1] for c in counts),
        sum(c[2] for c in counts),
    )
    result = {"field_precision": precision, "field_recall": recall, "field_f1": f1}
    if not with_ci:
        return result

    lows, highs = _bootstrap_micro_prf_ci(
        counts, confidence=ci_confidence, n_boot=ci_n_boot, seed=ci_seed
    )
    for name in ("field_precision", "field_recall", "field_f1"):
        result[f"{name}_ci_low"] = lows[name]
        result[f"{name}_ci_high"] = highs[name]
    result["n"] = float(len(counts))
    result["ci_level"] = float(ci_confidence)
    return result


def _bootstrap_micro_prf_ci(
    counts: list[tuple[int, int, int]],
    confidence: float = 0.95,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Percentile bootstrap of the micro P/R/F1 ratio over document counts.

    Cannot reuse :func:`bootstrap_ci` because the statistic is a *ratio of
    sums*, not a mean of per-sample scores: the replicate value has to be
    recomputed from re-summed numerators and denominators. The resampling
    stream is shared with :func:`bootstrap_ci` (same stdlib RNG helper), so all
    three metrics are resampled consistently for a given seed.

    Returns ``(lows, highs)`` keyed by ``field_precision``/``field_recall``/
    ``field_f1``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot!r}")

    names = ("field_precision", "field_recall", "field_f1")
    n = len(counts)
    if n == 0:
        zero = {k: 0.0 for k in names}
        return zero, dict(zero)
    if n == 1:
        point = _micro_prf(*counts[0])
        single = {k: v for k, v in zip(names, point)}
        return single, dict(single)

    replicates: dict[str, list[float]] = {k: [] for k in names}
    for row in _bootstrap_index_stream(n, n_boot, seed):
        tp = n_pred = n_gold = 0
        for i in row:
            c = counts[i]
            tp += c[0]
            n_pred += c[1]
            n_gold += c[2]
        for name, value in zip(names, _micro_prf(tp, n_pred, n_gold)):
            replicates[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    for name, values in replicates.items():
        values.sort()
        lows[name] = _percentile(values, alpha)
        highs[name] = _percentile(values, 1.0 - alpha)
    return lows, highs


# ---------------------------------------------------------------------------
# Aggregate roll-up consumed by evaluation.evaluate / reports
# ---------------------------------------------------------------------------
def _per_sample_qa_vectors(
    preds: list[str], refs: list[list[str]]
) -> dict[str, list[float]]:
    """All five per-sample QA vectors: EM, ANLS, token F1/precision/recall.

    Single source of truth for per-sample scoring so the public vectors, the
    averages in :func:`aggregate_qa_metrics` and the bootstrap intervals can
    never drift apart (a CI computed over a *different* scoring pass than the
    point estimate would be indefensible). Token P/R/F1 use best-reference
    semantics: for each sample we keep the reference that maximizes F1, which
    is how multi-annotator DocVQA-style answers are scored.

    The vectors always have ``len(preds)`` entries: a prediction with no
    reference available scores 0.0 rather than being dropped, which preserves
    the historical denominator of :func:`aggregate_qa_metrics` and keeps the
    bootstrap vector aligned 1:1 with the predictions it describes.
    """
    em: list[float] = []
    anls_scores: list[float] = []
    token_f1: list[float] = []
    token_p: list[float] = []
    token_r: list[float] = []

    for i, pred in enumerate(preds):
        ref_list = refs[i] if i < len(refs) else []
        ref_list = ref_list or []
        em.append(exact_match(pred, ref_list))
        anls_scores.append(anls(pred, ref_list))
        best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ref in ref_list:
            prf = token_level_prf(pred, ref)
            if prf["f1"] >= best["f1"]:
                best = prf
        token_f1.append(best["f1"])
        token_p.append(best["precision"])
        token_r.append(best["recall"])

    return {
        "exact_match": em,
        "anls": anls_scores,
        "token_f1": token_f1,
        "precision": token_p,
        "recall": token_r,
    }


def per_sample_qa_scores(
    preds: list[str], refs: list[list[str]]
) -> dict[str, list[float]]:
    """Per-sample score vectors for exact-match, ANLS and token-F1.

    Returns ``{"exact_match": [...], "anls": [...], "token_f1": [...]}`` with
    one entry per prediction, **in input order**, so a vector can be zipped
    back onto sample ids for error analysis, paired significance testing
    between two arms, or fed straight into :func:`bootstrap_ci`.

    Exposing the vectors (not just the means) is what makes a published
    baseline-vs-LoRA comparison checkable: a reader can recompute the mean, the
    interval, and the per-sample deltas from the same numbers.
    """
    vectors = _per_sample_qa_vectors(preds, refs)
    return {k: vectors[k] for k in ("exact_match", "anls", "token_f1")}


def aggregate_qa_metrics(
    preds: list[str],
    refs: list[list[str]],
    confidences: list[float] | None = None,
    with_ci: bool = False,
    ci_confidence: float = 0.95,
    ci_n_boot: int = _DEFAULT_N_BOOT,
    ci_seed: int = 0,
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

    Args:
        preds: model predictions, one per sample.
        refs: acceptable references per sample (ragged inner lists).
        confidences: optional per-sample self-reported confidence.
        with_ci: when True, additionally emit ``exact_match_ci_low`` /
            ``_ci_high`` (and the same for ``anls`` and ``token_f1``) plus
            ``ci_level``. ``n`` is always present. Default False keeps the
            output identical for existing callers and skips the bootstrap cost.
        ci_confidence: nominal coverage of the intervals.
        ci_n_boot: bootstrap replicates.
        ci_seed: RNG seed, so a published interval can be regenerated exactly.

    The intervals are percentile bootstraps over the per-sample vectors (see
    :func:`bootstrap_ci`). Report them with ``n``: on a 200-sample eval set
    they are wide, and overlapping intervals between two arms mean the gap is
    not established.
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
        empty = {k: 0.0 for k in keys}
        if with_ci:
            for name in ("exact_match", "anls", "token_f1"):
                empty[f"{name}_ci_low"] = 0.0
                empty[f"{name}_ci_high"] = 0.0
            empty["ci_level"] = float(ci_confidence)
        return empty

    vectors = _per_sample_qa_vectors(preds, refs)
    # Primary reference for corpus ROUGE (first annotator answer). Built with
    # zip so a ragged/short ``refs`` truncates exactly as the scoring loop does.
    primary_refs: list[str] = [
        ref_list[0] if ref_list else "" for _, ref_list in zip(preds, refs)
    ]

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

    def _mean(values: list[float]) -> float:
        """Mean of a per-sample vector (vectors are always ``len(preds)`` long)."""
        return sum(values) / len(values) if values else 0.0

    result = {
        "exact_match": _mean(vectors["exact_match"]),
        "anls": _mean(vectors["anls"]),
        "token_f1": _mean(vectors["token_f1"]),
        "precision": _mean(vectors["precision"]),
        "recall": _mean(vectors["recall"]),
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "avg_confidence": avg_conf,
        "n": float(n),
    }
    if not with_ci:
        return result

    # One CI per headline quality metric. BLEU/ROUGE are corpus statistics
    # computed by third-party libraries with no per-sample decomposition
    # available here, so they are deliberately left without intervals rather
    # than given a fabricated one.
    for name in ("exact_match", "anls", "token_f1"):
        low, high = bootstrap_ci(
            vectors[name],
            confidence=ci_confidence,
            n_boot=ci_n_boot,
            seed=ci_seed,
        )
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
    result["ci_level"] = float(ci_confidence)
    return result


__all__ = [
    "normalize_text",
    "exact_match",
    "anls",
    "token_level_prf",
    "bootstrap_ci",
    "per_sample_qa_scores",
    "per_sample_field_scores",
    "compute_bleu",
    "compute_rouge",
    "compute_classification_metrics",
    "structured_field_metrics",
    "aggregate_qa_metrics",
]
