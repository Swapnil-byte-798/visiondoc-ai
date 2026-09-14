"""Preprocessing package: turn raw corpora into cached, model-ready splits.

Public API::

    from preprocessing import (
        load_samples, build_splits, to_hf_dataset,   # dataset loaders
        DocumentAugmenter, build_augmenter,           # augmentation
        DocumentDataset,                              # torch map-style dataset
        build_and_cache, load_cached,                 # cache lifecycle
        DocSample, Split,                             # shared schema
    )

Everything downstream (models, training, evaluation, inference) consumes the
unified :class:`DocSample` schema, so nothing outside this package needs to know
which corpus (DocVQA / CORD / FUNSD / SROIE) is in use.

Lazy re-exports (PEP 562)
-------------------------
``dataset``/``datasets``/``build_dataset`` transitively import torch and the
``datasets`` library, but ``preprocessing.schema`` is stdlib-only. Re-exporting
lazily means ``from preprocessing.schema import DocSample`` (used in tests and by
metrics) stays torch-free, while ``from preprocessing import DocumentDataset``
(used in training) still works and pulls torch only when actually reached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_LAZY_EXPORTS: dict[str, str] = {
    "load_samples": "preprocessing.datasets",
    "build_splits": "preprocessing.datasets",
    "to_hf_dataset": "preprocessing.datasets",
    "keep_images_encoded": "preprocessing.datasets",
    "DocumentAugmenter": "preprocessing.transforms",
    "build_augmenter": "preprocessing.transforms",
    "DocumentDataset": "preprocessing.dataset",
    "build_and_cache": "preprocessing.build_dataset",
    "load_cached": "preprocessing.cache",
    "save_splits": "preprocessing.cache",
    "cache_fingerprint": "preprocessing.cache",
    "DocSample": "preprocessing.schema",
    "Split": "preprocessing.schema",
    "NormBox": "preprocessing.schema",
}

if TYPE_CHECKING:  # real symbols for type checkers/IDEs, zero runtime cost.
    from preprocessing.build_dataset import build_and_cache
    from preprocessing.cache import cache_fingerprint, load_cached, save_splits
    from preprocessing.dataset import DocumentDataset
    from preprocessing.datasets import build_splits, load_samples, to_hf_dataset
    from preprocessing.schema import DocSample, NormBox, Split
    from preprocessing.transforms import DocumentAugmenter, build_augmenter


def __getattr__(name: str) -> Any:
    """Resolve a public export on first access (PEP 562 lazy import)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    attr = getattr(importlib.import_module(module_path), name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_EXPORTS})


__all__ = list(_LAZY_EXPORTS)
