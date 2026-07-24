"""Research package: side-by-side model comparison and analysis utilities.

Public API::

    from research import compare_models
    result = compare_models(config, max_samples=200)   # base vs. LoRA fine-tune

The :mod:`research.compare` module keeps every heavy ML dependency (``models``,
``datasets``, ``pandas``, ``matplotlib``) behind function-local imports, so
importing this package does not build a model, load a dataset, or pull the
plotting stack until :func:`~research.compare.compare_models` / ``main`` is
actually called — only the lightweight shared logger is touched at import time.
"""

from __future__ import annotations

from research.compare import compare_models, main

__all__ = ["compare_models", "main"]
