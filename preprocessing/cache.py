"""On-disk caching of preprocessed dataset splits.

Design rationale
----------------
Downloading a corpus, mapping every row to a ``DocSample``, and carving splits is
non-trivial work we do not want to repeat on every training/eval launch. This
module persists the finished ``datasets.DatasetDict`` to disk keyed by a
*fingerprint* of exactly the config fields that change the produced data.

Why fingerprint instead of a fixed path? Two runs with different datasets, image
sizes, split fractions, seeds, or sample caps must NOT collide on the same cache
directory — silently training on a stale split is a nasty, hard-to-debug class of
bug. Hashing the data-affecting subset of the config guarantees a fresh cache the
moment any of those knobs changes, while identical configs reuse the cache.

``datasets`` is imported lazily so importing this module (e.g. to compute a
fingerprint for a log line) does not pull in the heavy library.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only imports; kept out of the runtime path.
    import datasets as hfds

    from configs import ProjectConfig

logger = get_logger(__name__)


def cache_fingerprint(config: "ProjectConfig") -> str:
    """Return a short, stable hash of the config fields that determine the data.

    Only fields that actually influence the produced splits are included, so
    unrelated changes (learning rate, model id, report backend) do not needlessly
    invalidate an expensive cache. Sample caps are included because they change
    the cached *contents*, and the seed/fractions because they change the carved
    partition membership.
    """
    data = config.data
    payload = {
        "dataset_name": data.dataset_name,
        "dataset_id": data.dataset_id,
        "dataset_subset": data.dataset_subset,
        # The canonical->native split mapping decides which rows land in each
        # split, so a change to it must invalidate the cache.
        "train_split": data.train_split,
        "val_split": data.val_split,
        "test_split": data.test_split,
        "image_size": data.image_size,
        "val_fraction": data.val_fraction,
        "test_fraction": data.test_fraction,
        "seed": config.seed,
        "max_train_samples": data.max_train_samples,
        "max_eval_samples": data.max_eval_samples,
    }
    # sort_keys makes the serialization canonical so the hash is order-independent.
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    # 16 hex chars (64 bits) is plenty to avoid collisions while keeping the
    # cache directory name short and readable.
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_dir_for(config: "ProjectConfig") -> Path:
    """Absolute-ish cache directory for this config: ``<cache_dir>/<fingerprint>``."""
    return Path(config.data.cache_dir) / cache_fingerprint(config)


def save_splits(splits_hf: "hfds.DatasetDict", config: "ProjectConfig") -> str:
    """Persist a ``DatasetDict`` to the fingerprinted cache directory.

    Returns the path so callers can log exactly where the artifact landed. The
    parent directory is created if missing.
    """
    path = cache_dir_for(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving preprocessed splits -> %s", path)
    # datasets writes an Arrow dataset + metadata; save_to_disk overwrites an
    # existing dir with the same fingerprint (same config => same data).
    splits_hf.save_to_disk(str(path))
    return str(path)


def load_cached(config: "ProjectConfig") -> "hfds.DatasetDict | None":
    """Load cached splits for this config, or ``None`` if not present/unreadable.

    We swallow load errors and return ``None`` (rather than raising) so a corrupt
    or partial cache simply triggers a clean rebuild instead of aborting the run.
    """
    path = cache_dir_for(config)
    if not path.exists():
        logger.info("No cache found at %s (will build).", path)
        return None
    try:
        from datasets import load_from_disk

        logger.info("Loading cached splits <- %s", path)
        return load_from_disk(str(path))
    except Exception as exc:  # pragma: no cover - corrupt cache -> rebuild
        logger.warning("Failed to load cache at %s (%s); will rebuild.", path, exc)
        return None


__all__ = ["cache_fingerprint", "cache_dir_for", "save_splits", "load_cached"]
