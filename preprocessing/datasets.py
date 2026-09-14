"""Dataset loaders that normalize every source into a common ``DocSample`` list.

Design rationale
----------------
The rest of the project never touches a raw Hugging Face row. Instead, every
supported corpus (DocVQA, CORD, FUNSD, SROIE) is funneled through a small
*loader* function that returns ``list[DocSample]`` — the single schema defined in
``preprocessing/schema.py``. This is what lets the model adapters, the trainer,
the evaluator, and the API stay completely dataset-agnostic: swapping the corpus
is a one-line YAML change (``data.dataset_name``) with no code edits anywhere
downstream.

Two cross-cutting concerns are handled here rather than in each loader:

* **Split carving.** Not every corpus ships a labelled validation *and* test
  split (DocVQA's test set is an unlabelled challenge set; FUNSD/SROIE ship only
  train+test). When a canonical split is missing we carve it out of the train
  pool *deterministically* (seeded permutation), so the same YAML always yields
  the exact same partition — reproducibility is non-negotiable for eval numbers.
* **Sample caps.** ``max_train_samples`` / ``max_eval_samples`` (and an explicit
  ``max_samples`` override) let a smoke run finish in minutes. They are pushed
  **down into the Hugging Face dataset object** (``Dataset.select``) *before* a
  single row is materialized. This is not a micro-optimization: the ``image``
  column is a lazily-decoded ``datasets.Image`` feature, so iterating rows
  decodes every page into a PIL object and holds it in RAM. Capping after the
  fact (``samples[:cap]``) still pays the full decode — for CORD's 800-row train
  split that is roughly 4.8 GB of PIL buffers, which alone overruns a free
  Colab runtime's ~12.7 GB of system RAM and is why capped training runs died
  before step 1. ``Dataset.select`` only rewrites an index map, so the rows we
  skip are never read off disk at all.

Heavy dependencies (``datasets``, ``PIL``) are imported lazily *inside* the
functions so that merely importing this module (e.g. to read ``load_samples``'s
signature) does not require the multi-hundred-MB ``datasets`` stack.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from utils.logging_utils import get_logger

from preprocessing.schema import DocSample

if TYPE_CHECKING:  # Import only for type hints; avoids a hard runtime dependency.
    import datasets as hfds

    from configs import ProjectConfig

logger = get_logger(__name__)

# Canonical split names used everywhere in the project. We accept a few common
# aliases (``val``/``dev``/``eval``) on input and normalize to these three.
_CANONICAL = ("train", "validation", "test")

# Synthesized prompts for the structured-extraction corpora. Keeping them as
# module constants means the trainer, evaluator, and any error-analysis notebook
# all agree on the *exact* prompt the model was conditioned on.
_CORD_QUESTION = "Extract the structured fields from this receipt as JSON."
_FUNSD_QUESTION = "Extract the key-value fields from this form as JSON."
_SROIE_QUESTION = (
    "Extract the key fields (company, date, address, total) "
    "from this receipt as JSON."
)


# ---------------------------------------------------------------------------
# Small defensive row/value helpers
# ---------------------------------------------------------------------------


def _first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-``None`` value among ``keys``.

    Hugging Face community mirrors of the same dataset frequently disagree on
    column names (``image`` vs ``img``, ``ner_tags`` vs ``labels``). Rather than
    hard-code one spelling and break on a mirror, every access goes through this
    tolerant lookup.
    """
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _canonical_split(split: str) -> str:
    """Normalize a user/config split name to one of :data:`_CANONICAL`."""
    s = str(split).strip().lower()
    if s in ("val", "valid", "dev", "eval", "validation"):
        return "validation"
    if s in ("train", "training"):
        return "train"
    if s in ("test", "testing"):
        return "test"
    # Unknown label: fail loudly rather than silently mis-route data.
    raise ValueError(f"Unrecognized split name: {split!r} (expected one of {_CANONICAL})")


def _image_size(row: dict[str, Any], image: Any) -> tuple[int | None, int | None]:
    """Best-effort (width, height) used to normalize absolute-pixel boxes."""
    # A live PIL image is the most reliable source of the true resolution.
    if image is not None and hasattr(image, "size"):
        try:
            w, h = image.size  # PIL uses (width, height)
            return int(w), int(h)
        except Exception:  # pragma: no cover - defensive against odd image types
            pass
    w = _first(row, "width", "image_width")
    h = _first(row, "height", "image_height")
    if w and h:
        return int(w), int(h)
    return None, None


def _normalize_box(
    box: Any, width: int | None, height: int | None
) -> tuple[float, float, float, float]:
    """Coerce any box representation into a normalized ``(x0,y0,x1,y1)`` in [0,1].

    Layout datasets are wildly inconsistent about box scale, so we detect it:

    * already-normalized coordinates (all <= 1.0) are passed through;
    * when we know the pixel resolution we divide by width/height;
    * otherwise we assume the LayoutLM-style ``0..1000`` convention that most HF
      FUNSD/SROIE mirrors adopt.

    The result is always clamped to [0, 1] so it is a valid ``NormBox`` — an
    out-of-range box would crash ``draw_boxes`` downstream.
    """
    try:
        x0, y0, x1, y1 = (float(c) for c in list(box)[:4])
    except Exception:
        # A malformed box should not abort the whole dataset load.
        return (0.0, 0.0, 0.0, 0.0)
    coords = [x0, y0, x1, y1]
    if max(coords) <= 1.0:
        norm = coords
    elif width and height:
        norm = [x0 / width, y0 / height, x1 / width, y1 / height]
    else:
        norm = [c / 1000.0 for c in coords]
    return tuple(min(1.0, max(0.0, c)) for c in norm)  # type: ignore[return-value]


def _flatten_fields(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested JSON structure into a flat ``{dotted.key: str}`` dict.

    The structured corpora (CORD especially) provide deeply nested ground truth
    (``{"menu": [{"nm": ..., "price": ...}], "total": {...}}``). A flat,
    string-valued dict is a much friendlier *generation target* for a VLM and is
    trivial to score field-by-field in the evaluator, so we flatten with dotted
    keys and stringify every leaf.
    """
    flat: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_fields(value, new_prefix))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            flat.update(_flatten_fields(value, new_prefix))
    else:
        flat[prefix or "value"] = "" if obj is None else str(obj)
    return flat


def _feature_label_names(ds: "hfds.Dataset", *columns: str) -> list[str]:
    """Extract the class-label names for a token-tag column, if the schema has them.

    HF stores NER tags as integer ``ClassLabel`` ids; the human-readable names
    (``B-QUESTION`` ...) live on the *feature*, not the row. We need them to
    decode per-token tags back into entity groups.
    """
    for col in columns:
        try:
            feat = ds.features[col]
            inner = getattr(feat, "feature", feat)  # Sequence(ClassLabel) -> ClassLabel
            names = getattr(inner, "names", None)
            if names:
                return list(names)
        except Exception:  # pragma: no cover - schema shapes vary across mirrors
            continue
    return []


# ---------------------------------------------------------------------------
# HF loading + per-dataset row mappers
# ---------------------------------------------------------------------------


# Known-good alternatives to suggest when a hub dataset lacks a split the config
# asks for. Keyed by lowercased ``dataset_id``. These are *hints* in an error
# message only — nothing here is ever loaded automatically, because silently
# swapping the corpus out from under a published number is exactly the kind of
# thing that makes a benchmark undefensible.
_SPLIT_SOURCE_HINTS: dict[str, str] = {
    "lmms-lab/docvqa": (
        "lmms-lab/DocVQA is an evaluation-only mirror (validation + unlabelled "
        "test). For a trainable DocVQA use 'HuggingFaceM4/DocumentVQA', or "
        "switch data.dataset_name/dataset_id to 'cord' / "
        "'naver-clova-ix/cord-v2', which ships train(800)/validation(100)/test(100)"
    ),
    "naver-clova-ix/cord-v2": (
        "cord-v2 ships exactly 'train', 'validation' and 'test' — check "
        "data.train_split / val_split / test_split for a typo"
    ),
}

# Split listings are a network round-trip; memoize per (dataset_id, subset) so a
# three-split ``build_splits`` does not re-query the Hub three times.
_SPLIT_NAME_CACHE: dict[tuple[str, str | None], list[str] | None] = {}


def _available_splits(config: "ProjectConfig") -> list[str] | None:
    """List the hub dataset's split names, or ``None`` if they can't be probed.

    Used purely to turn "split doesn't exist" into an actionable error. Probing
    can legitimately fail (offline runs against a warm cache, private repos), and
    a failed *probe* must never block a load that would otherwise work — so the
    caller treats ``None`` as "unknown, let ``load_dataset`` decide".
    """
    key = (str(config.data.dataset_id), config.data.dataset_subset)
    if key in _SPLIT_NAME_CACHE:
        return _SPLIT_NAME_CACHE[key]
    names: list[str] | None
    try:
        from datasets import get_dataset_split_names

        names = list(
            get_dataset_split_names(
                config.data.dataset_id, config.data.dataset_subset or None
            )
        )
    except Exception as exc:  # noqa: BLE001 - probing is best-effort by design
        logger.debug("Could not list splits for %s: %s", config.data.dataset_id, exc)
        names = None
    _SPLIT_NAME_CACHE[key] = names
    return names


def _missing_split_error(config: "ProjectConfig", hf_split: str, available: Any) -> ValueError:
    """Build the loud, actionable error raised when a split does not exist."""
    subset = f":{config.data.dataset_subset}" if config.data.dataset_subset else ""
    have = ", ".join(repr(s) for s in available) if available else "<could not list splits>"
    hint = _SPLIT_SOURCE_HINTS.get(
        str(config.data.dataset_id).strip().lower(),
        "pick a mirror that ships this split, or point the corresponding "
        "data.*_split at one of the available names",
    )
    return ValueError(
        f"Split {hf_split!r} does not exist in {config.data.dataset_id}{subset}. "
        f"Available splits: {have}. {hint}. "
        "Refusing to fall back to another split: a silent substitution would "
        "quietly change what the published numbers were measured on."
    )


def _hf_load(
    config: "ProjectConfig", hf_split: str, limit: int | None = None
) -> "hfds.Dataset":
    """Load a native HF split and apply ``limit`` *before* any row is decoded.

    ``limit`` is the whole point of this function's signature: the returned
    object is what every loader iterates, and iteration is what decodes images
    into RAM. Trimming here — via ``Dataset.select``, which only rewrites an
    index map and reads nothing — is the difference between a 64-sample smoke
    run touching 64 images and touching all 800.

    ``trust_remote_code`` is required by a few dataset *loading scripts*; newer
    ``datasets`` releases dropped the kwarg, so we try-with then fall back to a
    plain call to stay compatible across the pinned range.
    """
    from datasets import load_dataset

    # Fail loudly *before* downloading anything if the split cannot exist.
    available = _available_splits(config)
    if available is not None and hf_split not in available:
        raise _missing_split_error(config, hf_split, available)

    args: list[Any] = [config.data.dataset_id]
    if config.data.dataset_subset:
        args.append(config.data.dataset_subset)
    kwargs = dict(split=hf_split, cache_dir=config.data.cache_dir)
    logger.info(
        "Loading %s%s split=%s limit=%s", config.data.dataset_id,
        f":{config.data.dataset_subset}" if config.data.dataset_subset else "",
        hf_split, limit,
    )
    # Load WITHOUT trust_remote_code first: DocVQA/CORD/most mirrors are standard
    # Parquet datasets that need no script, and newer ``datasets`` releases log a
    # hard error for trust_remote_code even when it is unnecessary (it is accepted,
    # not a TypeError, so the old fallback never triggered). Only retry *with* the
    # flag if the dataset genuinely turns out to be script-based (older datasets).
    try:
        ds = load_dataset(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it's the script case
        msg = str(exc).lower()
        if "trust_remote_code" in msg or "loading script" in msg or "trust remote code" in msg:
            logger.info("Dataset appears script-based; retrying with trust_remote_code.")
            try:
                ds = load_dataset(*args, trust_remote_code=True, **kwargs)
            except TypeError:
                raise exc
        elif "unknown split" in msg or "bad split" in msg or "splitsnotfounderror" in msg:
            # ``load_dataset`` found the repo but not the split (the probe above
            # was unavailable). Re-raise with the same actionable message.
            raise _missing_split_error(config, hf_split, available) from exc
        else:
            raise
    return _truncate(ds, limit)


def _truncate(ds: "hfds.Dataset", limit: int | None) -> "hfds.Dataset":
    """Return the first ``limit`` rows of ``ds`` without materializing any row.

    Deterministic by construction: HF splits have a stable on-disk row order, so
    "the first N rows" is the same N rows on every machine and every rerun. We
    deliberately prefer a prefix over a random subsample — a seeded subsample
    would still have to enumerate the full index space and, more importantly,
    would make the cap interact with ``config.seed``, so two runs that differ
    only in cap size would share no samples at all.
    """
    if limit is None or limit < 0:
        return ds
    try:
        total = len(ds)
    except TypeError:  # pragma: no cover - iterable (streaming) datasets
        return ds
    if limit >= total:
        return ds
    return ds.select(range(limit))


def _load_docvqa(
    config: "ProjectConfig", hf_split: str, limit: int | None = None
) -> list[DocSample]:
    """Map DocVQA rows to QA ``DocSample``s.

    DocVQA is the canonical document-QA benchmark: a page image, a natural
    question, and (usually several) accepted answer strings. We keep *all*
    answers because ANLS scores against the best-matching reference.

    ``limit`` is honored by ``_hf_load`` before the row loop, so a capped call
    never decodes an image it is going to throw away.
    """
    ds = _hf_load(config, hf_split, limit)
    samples: list[DocSample] = []
    for i, row in enumerate(ds):
        image = _first(row, "image", "img")
        question = str(_first(row, "question", "query", default=""))
        # ``answers`` is a list on the labelled splits; the unlabelled test split
        # has none — we still emit the sample (answer becomes "").
        answers = _first(row, "answers", "answer", default=[])
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a) for a in (answers or [])]
        answer = answers[0] if answers else ""
        sample_id = str(
            _first(row, "questionId", "question_id", "qid", "id", default=f"docvqa-{hf_split}-{i}")
        )
        samples.append(
            DocSample(
                image=image,
                question=question,
                answer=answer,
                answers=answers,
                sample_id=sample_id,
                task="qa",
                metadata={"dataset": "docvqa", "split": hf_split,
                          "docId": _first(row, "docId", "doc_id", default="")},
            )
        )
    logger.info("DocVQA %s: %d samples", hf_split, len(samples))
    return samples


def _load_cord(
    config: "ProjectConfig", hf_split: str, limit: int | None = None
) -> list[DocSample]:
    """Map CORD-v2 receipt rows to extraction ``DocSample``s.

    CORD ground truth is a JSON string with a ``gt_parse`` tree of receipt
    fields. We flatten it into a dotted-key dict and serialize that as the target
    answer, turning receipt parsing into a JSON-generation task the VLM can learn.

    ``limit`` matters most here: CORD receipts are large scans, and the 800-row
    train split alone is several GB once decoded into PIL objects.
    """
    ds = _hf_load(config, hf_split, limit)
    samples: list[DocSample] = []
    for i, row in enumerate(ds):
        image = _first(row, "image", "img")
        gt_raw = _first(row, "ground_truth", "gt", "label", default="{}")
        try:
            gt = json.loads(gt_raw) if isinstance(gt_raw, str) else dict(gt_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            gt = {}
        # CORD nests the real fields under "gt_parse"; fall back to the root.
        parse = gt.get("gt_parse", gt) if isinstance(gt, dict) else {}
        fields = _flatten_fields(parse)
        answer = json.dumps(fields, ensure_ascii=False, sort_keys=True)
        samples.append(
            DocSample(
                image=image,
                question=_CORD_QUESTION,
                answer=answer,
                sample_id=str(_first(row, "id", "image_id", default=f"cord-{hf_split}-{i}")),
                task="extraction",
                metadata={"dataset": "cord", "split": hf_split, "fields": list(fields.keys())},
            )
        )
    logger.info("CORD %s: %d samples", hf_split, len(samples))
    return samples


def _entities_from_tokens(
    words: list[Any], tags: list[Any], boxes: list[Any], label_names: list[str],
    width: int | None, height: int | None,
) -> tuple[dict[str, str], list[tuple], list[str]]:
    """Group BIO/BIOES token tags into ``{entity: text}`` plus normalized boxes.

    FUNSD/SROIE mirrors that expose token-level ``ner_tags`` need the tokens
    re-assembled per entity type. We strip the ``B-``/``I-``/``S-`` prefix and
    concatenate the words sharing an entity label; boxes are normalized as we go
    so region-highlighting has something to draw.
    """
    groups: dict[str, list[str]] = {}
    norm_boxes: list[tuple] = []
    box_labels: list[str] = []
    for j, word in enumerate(words):
        tag = tags[j] if j < len(tags) else None
        if isinstance(tag, int) and 0 <= tag < len(label_names):
            name = label_names[tag]
        else:
            name = str(tag)
        entity = name.split("-", 1)[-1] if "-" in name else name
        is_outside = entity.upper() in ("O", "OTHER", "", "NONE")
        if not is_outside and str(word).strip():
            groups.setdefault(entity, []).append(str(word))
        if j < len(boxes) and boxes[j]:
            norm_boxes.append(_normalize_box(boxes[j], width, height))
            box_labels.append(entity)
    fields = {k: " ".join(v) for k, v in groups.items()}
    return fields, norm_boxes, box_labels


def _load_funsd(
    config: "ProjectConfig", hf_split: str, limit: int | None = None
) -> list[DocSample]:
    """Map FUNSD form rows to extraction ``DocSample``s (key-value + boxes).

    FUNSD ships in two common shapes on the Hub: (a) a nested ``form`` list of
    entities with ``linking`` that pairs questions to answers, and (b) a
    token-level ``words``/``bboxes``/``ner_tags`` layout. We support both and
    prefer the richer linked form when present.

    ``limit`` is applied by ``_hf_load`` before the row loop; ``ds.features``
    survives ``select`` untouched, so label-name lookup still works on the
    truncated view.
    """
    ds = _hf_load(config, hf_split, limit)
    label_names = _feature_label_names(ds, "ner_tags", "labels")
    samples: list[DocSample] = []
    for i, row in enumerate(ds):
        image = _first(row, "image", "img")
        width, height = _image_size(row, image)
        fields: dict[str, str] = {}
        boxes: list[tuple] = []
        box_labels: list[str] = []

        form = _first(row, "form", "annotations")
        if isinstance(form, list) and form:
            # (a) Linked-entity shape: pair each question with its linked answers.
            id_to_text: dict[Any, str] = {e.get("id"): e.get("text", "") for e in form}
            for ent in form:
                text = ent.get("text", "")
                label = ent.get("label", "entity")
                box = ent.get("box") or ent.get("bbox")
                if box:
                    boxes.append(_normalize_box(box, width, height))
                    box_labels.append(label)
                if label == "question" and text:
                    answers = []
                    for link in ent.get("linking", []) or []:
                        # ``linking`` pairs are [from_id, to_id]; grab the "other" end.
                        for other in link:
                            if other != ent.get("id") and other in id_to_text:
                                answers.append(id_to_text[other])
                    fields[text] = " ".join(a for a in answers if a)
        else:
            # (b) Token-level shape.
            words = _first(row, "words", "tokens", default=[]) or []
            tags = _first(row, "ner_tags", "labels", "tags", default=[]) or []
            token_boxes = _first(row, "bboxes", "boxes", "bbox", default=[]) or []
            fields, boxes, box_labels = _entities_from_tokens(
                words, tags, token_boxes, label_names, width, height
            )
            if not fields and words:
                # No usable tags: fall back to the full reading order as text.
                fields = {"text": " ".join(str(w) for w in words)}

        answer = json.dumps(fields, ensure_ascii=False, sort_keys=True)
        samples.append(
            DocSample(
                image=image,
                question=_FUNSD_QUESTION,
                answer=answer,
                boxes=boxes,
                box_labels=box_labels,
                sample_id=str(_first(row, "id", "file_name", "fname", default=f"funsd-{hf_split}-{i}")),
                task="extraction",
                metadata={"dataset": "funsd", "split": hf_split},
            )
        )
    logger.info("FUNSD %s: %d samples", hf_split, len(samples))
    return samples


def _load_sroie(
    config: "ProjectConfig", hf_split: str, limit: int | None = None
) -> list[DocSample]:
    """Map SROIE receipt rows to extraction ``DocSample``s.

    SROIE's task-2 target is four canonical fields (company, date, address,
    total). Mirrors provide them either as an explicit entity dict / JSON ground
    truth, as flat columns, or as token-level tags — we try each in turn.

    ``limit`` is honored by ``_hf_load`` before the row loop.
    """
    ds = _hf_load(config, hf_split, limit)
    label_names = _feature_label_names(ds, "ner_tags", "labels")
    samples: list[DocSample] = []
    for i, row in enumerate(ds):
        image = _first(row, "image", "img")
        width, height = _image_size(row, image)
        fields: dict[str, str] = {}
        boxes: list[tuple] = []
        box_labels: list[str] = []

        gt = _first(row, "ground_truth", "gt", "entities", "key_info", "objects")
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except json.JSONDecodeError:
                gt = None
        if isinstance(gt, dict):
            fields = _flatten_fields(gt.get("gt_parse", gt))
        else:
            # Flat standard columns are the most common SROIE mirror layout.
            for key in ("company", "date", "address", "total"):
                value = _first(row, key)
                if value:
                    fields[key] = str(value)
            if not fields:
                # Token-level fallback (words + ner_tags).
                words = _first(row, "words", "tokens", default=[]) or []
                tags = _first(row, "ner_tags", "labels", "tags", default=[]) or []
                token_boxes = _first(row, "bboxes", "boxes", "bbox", default=[]) or []
                fields, boxes, box_labels = _entities_from_tokens(
                    words, tags, token_boxes, label_names, width, height
                )

        answer = json.dumps(fields, ensure_ascii=False, sort_keys=True)
        samples.append(
            DocSample(
                image=image,
                question=_SROIE_QUESTION,
                answer=answer,
                boxes=boxes,
                box_labels=box_labels,
                sample_id=str(_first(row, "id", "file_name", "fname", default=f"sroie-{hf_split}-{i}")),
                task="extraction",
                metadata={"dataset": "sroie", "split": hf_split},
            )
        )
    logger.info("SROIE %s: %d samples", hf_split, len(samples))
    return samples


# ---------------------------------------------------------------------------
# Dataset registry: name -> (loader, which canonical splits are native)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DatasetSpec:
    """Per-corpus metadata driving loading and split carving.

    ``native`` records which canonical splits the corpus ships with usable
    labels. Any canonical split marked ``False`` is carved out of the train pool
    (see :func:`_partition_pool`).

    Every loader takes ``(config, hf_split, limit)`` — the third argument is the
    row cap, pushed down so it is applied before image decoding.
    """

    loader: Callable[["ProjectConfig", str, int | None], list[DocSample]]
    native: dict[str, bool]


_REGISTRY: dict[str, _DatasetSpec] = {
    # DocVQA test is an unlabelled challenge set -> carve our own test from train.
    "docvqa": _DatasetSpec(_load_docvqa, {"train": True, "validation": True, "test": False}),
    # CORD-v2 ships train/validation/test with labels.
    "cord": _DatasetSpec(_load_cord, {"train": True, "validation": True, "test": True}),
    # FUNSD ships only train/test -> carve validation from train.
    "funsd": _DatasetSpec(_load_funsd, {"train": True, "validation": False, "test": True}),
    # SROIE ships only train/test -> carve validation from train.
    "sroie": _DatasetSpec(_load_sroie, {"train": True, "validation": False, "test": True}),
}


def _spec_for(config: "ProjectConfig") -> _DatasetSpec:
    name = str(config.data.dataset_name).strip().lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset_name {name!r}. Supported: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def _hf_split_for(config: "ProjectConfig", canonical: str) -> str:
    """Map a canonical split to the concrete HF split name from the config."""
    return {
        "train": config.data.train_split,
        "validation": config.data.val_split,
        "test": config.data.test_split,
    }[canonical]


def _carved_targets(spec: _DatasetSpec) -> list[str]:
    """Canonical eval splits that must be carved from train (fixed order)."""
    return [s for s in ("validation", "test") if not spec.native.get(s, False)]


def _partition_pool(
    pool: list[DocSample], config: "ProjectConfig", carved: list[str]
) -> dict[str, list[DocSample]]:
    """Deterministically split the train ``pool`` into train + carved eval splits.

    A seeded permutation makes the partition a pure function of (data, seed,
    fractions), so re-running the pipeline — on any machine — reproduces the
    identical train/val/test membership. Reservations are disjoint and taken in a
    fixed (validation, then test) order.
    """
    indices = list(range(len(pool)))
    random.Random(config.seed).shuffle(indices)
    fractions = {"validation": config.data.val_fraction, "test": config.data.test_fraction}
    parts: dict[str, list[DocSample]] = {}
    cursor = 0
    for split in ("validation", "test"):  # fixed order for determinism
        if split not in carved:
            continue
        # At least one example so downstream code never sees an empty eval split.
        k = max(1, int(len(pool) * fractions[split])) if pool else 0
        chosen = indices[cursor:cursor + k]
        parts[split] = [pool[i] for i in chosen]
        cursor += k
    parts["train"] = [pool[i] for i in indices[cursor:]]
    return parts


def _resolve_cap(config: "ProjectConfig", canonical: str, max_samples: int | None) -> int | None:
    """Pick the effective sample cap: explicit arg wins over config defaults.

    Resolved *before* any loading happens (see :func:`load_samples`) so the cap
    can be pushed into ``Dataset.select``; resolving it afterwards is what made
    the old code decode the whole split.
    """
    if max_samples is not None:
        return max_samples
    if canonical == "train":
        return config.data.max_train_samples
    return config.data.max_eval_samples  # validation + test share the eval cap


def _pool_cap_for(
    config: "ProjectConfig", carved: list[str], caps: dict[str, int | None]
) -> int | None:
    """How many train-pool rows must be loaded to satisfy ``caps`` after carving.

    When a split is carved, a cap on the *carved* split cannot be pushed into
    ``Dataset.select`` directly — asking for 100 validation samples at a 10%
    validation fraction needs ~1000 pool rows. So we invert the fractions and
    load only the smallest prefix that can still fill every requested cap.

    Returns ``None`` (= load everything) if any requested split is uncapped, or
    if the fractions make the inversion meaningless.

    Trade-off worth stating plainly: a capped pool is a *different* pool, so the
    seeded partition over it selects different rows than an uncapped run would.
    That is unavoidable if we are to avoid decoding rows we discard, and it stays
    fully deterministic — the partition is a pure function of (pool prefix size,
    seed, fractions), all of which are recorded in the config. Any published
    number must therefore be produced with the caps it was measured under.
    """
    fractions = {
        "validation": float(config.data.val_fraction),
        "test": float(config.data.test_fraction),
    }
    # Whatever the carved splits do not reserve is what train is left with.
    train_fraction = 1.0 - sum(fractions[s] for s in carved if s in fractions)
    needed = 0
    for split, cap in caps.items():
        if cap is None or cap < 0:
            return None  # an uncapped split needs the whole pool
        if split == "train":
            frac = train_fraction
        elif split in carved:
            frac = fractions.get(split, 0.0)
        else:
            continue  # natively-available split: not served from the pool
        if frac <= 0.0:
            # The split cannot grow with the pool (fraction 0, or carving ate
            # 100% of train). Nothing to invert; other splits decide the size.
            continue
        # +1 row of slack absorbs the float rounding in ``int(len(pool) * frac)``.
        needed = max(needed, math.ceil(cap / frac) + 1)
    if needed <= 0:
        return None
    return needed


def _carve_splits(
    config: "ProjectConfig",
    spec: _DatasetSpec,
    carved: list[str],
    caps: dict[str, int | None],
) -> dict[str, list[DocSample]]:
    """Build the shared seeded partition of the train pool exactly once.

    ``caps`` maps canonical split -> effective cap for every split the caller
    actually wants; only those splits appear in the result. Loading the pool is
    by far the most expensive thing this module does (it decodes images), so the
    caller must never invoke this more than once per set of splits — that is why
    :func:`build_splits` calls it with all three at once instead of going through
    :func:`load_samples` three times.
    """
    pool_cap = _pool_cap_for(config, carved, caps)
    if pool_cap is not None:
        logger.info(
            "Carving %s from the train pool; loading only the first %d pool rows "
            "to satisfy caps %s (partition is deterministic for this prefix+seed).",
            carved, pool_cap, caps,
        )
    pool = spec.loader(config, _hf_split_for(config, "train"), pool_cap)
    parts = _partition_pool(pool, config, carved)

    out: dict[str, list[DocSample]] = {}
    for split, cap in caps.items():
        if split in parts:
            samples = parts[split]
        elif split in _CANONICAL:
            # Natively-labelled split (e.g. FUNSD's test): load it directly, with
            # its cap pushed down the same way.
            samples = spec.loader(config, _hf_split_for(config, split), cap)
        else:  # pragma: no cover - guarded by _canonical_split upstream
            continue
        if cap is not None and 0 <= cap < len(samples):
            samples = samples[:cap]
        out[split] = samples
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_samples(
    config: "ProjectConfig", split: str, max_samples: int | None = None
) -> list[DocSample]:
    """Load one canonical split of the configured dataset as ``DocSample``s.

    Dispatches on ``config.data.dataset_name``. If the requested split is not
    native to the corpus it is carved from the train pool deterministically. The
    result is capped by ``max_samples`` (explicit) or the relevant config cap.

    The cap is resolved *first* and pushed into the Hugging Face dataset object,
    so a ``max_samples=64`` call reads 64 images off disk rather than decoding
    the entire split and slicing the leftovers away.
    """
    canonical = _canonical_split(split)
    spec = _spec_for(config)
    carved = _carved_targets(spec)
    cap = _resolve_cap(config, canonical, max_samples)

    # A carved eval split — or the train split when carving reduces it — must be
    # produced from the shared, seeded partition to avoid cross-split leakage.
    if canonical in carved or (canonical == "train" and carved):
        samples = _carve_splits(config, spec, carved, {canonical: cap})[canonical]
    else:
        samples = spec.loader(config, _hf_split_for(config, canonical), cap)
        if cap is not None and 0 <= cap < len(samples):
            # Defensive only: ``_hf_load`` already truncated. This still matters
            # for a loader that emits more than one sample per row.
            samples = samples[:cap]
    logger.info("Split %s -> %d samples (cap=%s)", canonical, len(samples), cap)
    return samples


def build_splits(config: "ProjectConfig") -> dict[str, list[DocSample]]:
    """Load all three canonical splits into ``{"train","validation","test"}``.

    When the corpus needs splits carved, all three are served from a *single*
    partition of the train pool. Going through :func:`load_samples` three times
    would rebuild that pool three times — decoding every pool image three times
    over, which on a Colab runtime is the difference between finishing and being
    OOM-killed.
    """
    spec = _spec_for(config)
    carved = _carved_targets(spec)
    caps = {s: _resolve_cap(config, s, None) for s in _CANONICAL}
    if not carved:
        # Every split is native: each loads independently with its cap pushed down.
        return {split: load_samples(config, split) for split in _CANONICAL}
    splits = _carve_splits(config, spec, carved, caps)
    for name in _CANONICAL:
        logger.info("Split %s -> %d samples (cap=%s)", name, len(splits[name]), caps[name])
    return splits


def to_hf_dataset(samples: list[DocSample]) -> "hfds.Dataset":
    """Convert ``DocSample``s to a ``datasets.Dataset`` for caching/training.

    We build from the plain ``to_record()`` dicts and then *cast* the image
    column to the ``datasets.Image`` feature. That cast is what lets
    ``save_to_disk`` store images efficiently (and re-hydrate them as PIL on
    load) whether the samples currently hold PIL objects or path strings.
    """
    from datasets import Dataset
    from datasets import Image as HFImage

    records = [s.to_record() for s in samples]
    ds = Dataset.from_list(records)
    try:
        ds = ds.cast_column("image", HFImage())
    except Exception as exc:  # pragma: no cover - mixed/absent images degrade gracefully
        # Not fatal: the raw column is still usable, just less storage-efficient.
        logger.warning("Could not cast 'image' column to datasets.Image: %s", exc)
    return ds


__all__ = ["load_samples", "build_splits", "to_hf_dataset"]
