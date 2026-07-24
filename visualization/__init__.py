"""Visualization package for VisionDoc AI.

Re-exports the plotting API from :mod:`visualization.plots` so callers can write
``from visualization import plot_model_comparison`` without reaching into the
submodule. Importing this package pulls matplotlib (the Agg backend is selected in
``plots``), which is why heavy training/inference code imports these functions
lazily at their call sites rather than at module top level.
"""

from __future__ import annotations

from visualization.plots import (
    plot_attention_map,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_prediction_grid,
    plot_training_curves,
    set_style,
)

__all__ = [
    "set_style",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_confidence_distribution",
    "plot_attention_map",
    "plot_prediction_grid",
    "plot_model_comparison",
]
