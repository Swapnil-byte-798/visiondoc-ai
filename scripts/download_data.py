"""CLI wrapper that downloads, builds, and caches the configured dataset.

This is the friendly, discoverable front door for data preparation. It is a
*thin* wrapper on purpose: all real work is delegated to
:func:`preprocessing.build_dataset.build_and_cache`, which is the same code path
the training pipeline uses. Duplicating the build logic here would risk the
"download" command and the "train" command diverging — so instead we call the
one canonical builder and only add CLI ergonomics (argument parsing, a printed
stats summary, and a non-zero exit code on failure for CI/shell scripting).

Usage::

    python -m scripts.download_data --config configs/default.yaml
    # or, after `pip install -e .`:
    visiondoc-download --config configs/default.yaml

Design notes:

* Heavy imports (``configs``, ``preprocessing``, ``utils.seed`` — the latter
  pulls in torch) are performed lazily inside :func:`main`. Import-time of this
  module must stay cheap because it is referenced by a ``[project.scripts]``
  console entry point, which package managers may import merely to validate.
* We seed RNGs before building so that any deterministic split-carving inside
  ``build_splits`` is reproducible across machines — matching the behaviour of
  ``preprocessing.build_dataset`` so the two entry points are interchangeable.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only imports; never executed at runtime.
    import datasets as hfds

    from configs import ProjectConfig

logger = get_logger(__name__)


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Exposed as its own function (rather than inlined into ``main``) so tests can
    exercise argument handling without triggering the heavy dataset build.
    """
    parser = argparse.ArgumentParser(
        prog="visiondoc-download",
        description=(
            "Download, build, and cache the dataset named in the VisionDoc "
            "config. Runs the same builder the training pipeline uses so the "
            "cache is guaranteed compatible."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a config YAML. Defaults to $VISIONDOC_CONFIG, then "
            "configs/default.yaml (resolved by configs.load_config)."
        ),
    )
    return parser.parse_args(argv)


def _print_stats(config: "ProjectConfig", dataset_dict: "hfds.DatasetDict") -> None:
    """Log per-split example counts and the on-disk cache location.

    We log through the shared logger rather than ``print`` so the output is
    timestamped and consistent with the rest of the pipeline; the summary is the
    signal a user or CI job checks to confirm the download succeeded.
    """
    # Imported here (not at module top) because ``preprocessing.cache`` reaches
    # into ``datasets`` internals; keep module import light.
    from preprocessing.cache import cache_dir_for

    total = sum(len(split_ds) for split_ds in dataset_dict.values())
    logger.info("=" * 70)
    logger.info("Dataset ready: %s", config.data.dataset_name)
    logger.info("Cache path   : %s", cache_dir_for(config))
    logger.info("-" * 70)
    # Iterate in canonical order when present so output is stable run-to-run,
    # then append any unexpected extra splits rather than silently dropping them.
    ordered = [s for s in ("train", "validation", "test") if s in dataset_dict]
    ordered += [s for s in dataset_dict if s not in ordered]
    for split_name in ordered:
        logger.info("  %-12s %8d examples", split_name, len(dataset_dict[split_name]))
    logger.info("-" * 70)
    logger.info("  %-12s %8d examples", "TOTAL", total)
    logger.info("=" * 70)


def main(argv: "list[str] | None" = None) -> int:
    """Entry point: load config, seed, build+cache the dataset, print stats.

    Returns a process exit code (0 = success) so shell scripts and CI can gate
    on it. Any failure is logged with a traceback and surfaced as exit code 1
    rather than a bare Python traceback, which reads better in ``scripts/*.sh``.
    """
    args = _parse_args(argv)

    # Lazy: configs is trivial, but utils.seed imports torch, and
    # preprocessing.build_dataset imports the (heavy) datasets stack. We only pay
    # that once we are actually going to build.
    from configs import load_config
    from preprocessing.build_dataset import build_and_cache
    from utils.seed import set_seed

    try:
        config = load_config(args.config)
    except Exception:  # noqa: BLE001 - top-level CLI guard: report cleanly, no traceback spew.
        logger.exception("Failed to load config (path=%s)", args.config)
        return 1

    # Reproducible deterministic split carving / shuffling across machines.
    set_seed(config.seed)

    try:
        dataset_dict = build_and_cache(config)
    except Exception:  # noqa: BLE001 - dataset download/build can fail many ways (network, disk, HF auth).
        logger.exception(
            "Dataset build failed for '%s' (id=%s). "
            "Check network access, HF_TOKEN for gated datasets, and disk space.",
            config.data.dataset_name,
            config.data.dataset_id,
        )
        return 1

    _print_stats(config, dataset_dict)
    return 0


if __name__ == "__main__":
    # Propagate the exit code so `python -m scripts.download_data` behaves like a
    # well-mannered CLI (usable in `set -e` shell scripts and CI pipelines).
    sys.exit(main())
