"""Evaluation package: metrics, profiling, and report rendering.

Public API::

    from evaluation import (
        evaluate_model,        # run a model over a split -> result dict
        build_report,          # render that dict to self-contained HTML + JSON
        aggregate_qa_metrics,  # flat QA metric roll-up
        InferenceProfiler,     # latency / throughput / memory context manager
        ProfileResult,         # profiler summary dataclass
    )

Why the re-exports are *lazy*
-----------------------------
``evaluation.profiler`` and ``evaluation.evaluate`` transitively import torch
(via ``utils.device`` / ``models``), but ``evaluation.metrics`` is stdlib-only
by design so it can be unit-tested on a CPU CI runner with no torch installed.
Eagerly importing the heavy submodules here would make even
``import evaluation.metrics`` drag in torch and fail in that environment.

We therefore expose the names through a module-level ``__getattr__`` (PEP 562):
attribute access resolves the owning submodule *on demand*, so importing the
package — or importing ``evaluation.metrics`` — costs nothing heavy until a
caller actually reaches for the profiler/evaluation entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Map each public name to the submodule that defines it. Kept as data (not
# imports) so nothing heavy is pulled at package import time.
_LAZY_EXPORTS: dict[str, str] = {
    "evaluate_model": "evaluation.evaluate",
    "build_report": "evaluation.report",
    "results_to_dataframe": "evaluation.report",
    "aggregate_qa_metrics": "evaluation.metrics",
    "InferenceProfiler": "evaluation.profiler",
    "ProfileResult": "evaluation.profiler",
}

if TYPE_CHECKING:  # give type checkers/IDEs the real symbols without runtime cost.
    from evaluation.evaluate import evaluate_model
    from evaluation.metrics import aggregate_qa_metrics
    from evaluation.profiler import InferenceProfiler, ProfileResult
    from evaluation.report import build_report, results_to_dataframe


def __getattr__(name: str) -> Any:
    """Resolve a public export on first access (PEP 562 lazy import)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache so subsequent accesses skip __getattr__.
    return attr


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir(evaluation)`` for discoverability."""
    return sorted({*globals().keys(), *_LAZY_EXPORTS})


__all__ = [
    "evaluate_model",
    "build_report",
    "results_to_dataframe",
    "aggregate_qa_metrics",
    "InferenceProfiler",
    "ProfileResult",
]
