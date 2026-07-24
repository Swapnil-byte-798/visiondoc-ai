"""Reproducibility helpers.

Fixing every RNG (Python, NumPy, Torch, CUDA) is what makes an evaluation
number trustworthy and a bug reproducible. :func:`set_seed` is called at the
top of every entry point.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("visiondoc.seed")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed all RNGs used in the project.

    Parameters
    ----------
    seed:
        The integer seed applied to ``random``, ``numpy`` and ``torch``.
    deterministic:
        If ``True``, force cuDNN into deterministic mode. This guarantees
        bit-for-bit reproducibility but can slow training and disables some
        fast kernels, so it defaults to ``False`` (enable it for eval/debug).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Opt-in strict determinism; requires this env var for some CUDA kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - depends on torch build
            logger.debug("Deterministic algorithms not fully available on this build.")
    else:
        torch.backends.cudnn.benchmark = True

    logger.debug("Seeded RNGs with %d (deterministic=%s)", seed, deterministic)


__all__ = ["set_seed"]
