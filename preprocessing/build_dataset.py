"""CLI + library entry point that downloads, builds, and caches dataset splits.

This is the one command a user runs before training:

    python -m preprocessing.build_dataset --config configs/default.yaml

It resolves the configured corpus, maps every row to the unified ``DocSample``
schema, carves any missing splits deterministically, converts each split to a
``datasets.Dataset``, and persists the whole ``DatasetDict`` to a fingerprinted
cache so the (potentially slow) preprocessing happens exactly once.

``build_and_cache`` is exposed separately so the training pipeline and scripts can
trigger the same build programmatically without shelling out.

Heavy imports (``datasets``, ``utils.seed`` which pulls torch) are done lazily
inside functions so that importing this module — which the package ``__init__``
does — stays light.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from utils.logging_utils import get_logger

from preprocessing.cache import cache_dir_for, save_splits
from preprocessing.datasets import build_splits, to_hf_dataset

if TYPE_CHECKING:  # Type-only imports.
    import datasets as hfds

    from configs import ProjectConfig

logger = get_logger(__name__)


def build_and_cache(config: "ProjectConfig") -> "hfds.DatasetDict":
    """Build all splits, convert to a ``DatasetDict``, cache it, and return it.

    Returns the in-memory ``DatasetDict`` (not just the path) so a caller that
    just built the data can use it immediately without a round-trip to disk.
    """
    from datasets import DatasetDict

    splits = build_splits(config)
    logger.info(
        "Built splits: %s",
        {name: len(samples) for name, samples in splits.items()},
    )
    dataset_dict = DatasetDict(
        {name: to_hf_dataset(samples) for name, samples in splits.items()}
    )
    path = save_splits(dataset_dict, config)
    logger.info("Cached dataset -> %s", path)
    return dataset_dict


def _print_summary(config: "ProjectConfig", dataset_dict: "hfds.DatasetDict") -> None:
    """Log split sizes and one fully-materialized example for a sanity check."""
    logger.info("=" * 70)
    logger.info("Dataset build complete: %s", config.data.dataset_name)
    logger.info("Cache location: %s", cache_dir_for(config))
    for split_name, split_ds in dataset_dict.items():
        logger.info("  %-12s %8d examples", split_name, len(split_ds))

    # Show a single example (from whichever split exists) to make prompt/answer
    # formatting immediately visible in the console — the fastest way to catch a
    # broken row mapping before committing GPU hours to training.
    for split_name in ("train", "validation", "test"):
        if split_name in dataset_dict and len(dataset_dict[split_name]) > 0:
            example = dataset_dict[split_name][0]
            logger.info("-" * 70)
            logger.info("Example from '%s':", split_name)
            logger.info("  sample_id : %s", example.get("sample_id"))
            logger.info("  task      : %s", example.get("task"))
            logger.info("  question  : %s", example.get("question"))
            answer = str(example.get("answer", ""))
            logger.info("  answer    : %s", answer[:200] + ("..." if len(answer) > 200 else ""))
            logger.info("  #boxes    : %d", len(example.get("boxes") or []))
            break
    logger.info("=" * 70)


def main() -> None:
    """CLI: load the config, seed RNGs, build + cache the dataset, print a summary."""
    parser = argparse.ArgumentParser(description="Download, build, and cache VisionDoc datasets.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (defaults to $VISIONDOC_CONFIG or configs/default.yaml).",
    )
    args = parser.parse_args()

    # Imported here (not at module top) because utils.seed pulls in torch and
    # configs is only needed when actually running the CLI.
    from configs import load_config
    from utils.seed import set_seed

    config = load_config(args.config)
    # Seed so any deterministic carving/shuffling is reproducible across machines.
    set_seed(config.seed)

    dataset_dict = build_and_cache(config)
    _print_summary(config, dataset_dict)


if __name__ == "__main__":
    main()
