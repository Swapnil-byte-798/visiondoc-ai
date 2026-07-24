"""Shared utilities: logging, device/precision, seeding, image I/O, OCR.

Lazy re-exports (PEP 562)
-------------------------
The convenience names below are resolved *on demand* via ``__getattr__`` rather
than eagerly imported. This matters because ``utils.device`` / ``utils.seed``
import ``torch`` at module top: if this package eagerly imported them, then a
lightweight ``from utils.logging_utils import get_logger`` (used by *every*
module, including the stdlib-only ``evaluation.metrics``) would transitively drag
in torch — breaking torch-free unit tests / CI. Deferring the imports keeps
submodule access as cheap as its own dependencies.

Public API (unchanged)::

    from utils import get_logger, set_seed, load_image, resolve_device, OCREngine
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Map each public name to the submodule that defines it (data, not imports).
_LAZY_EXPORTS: dict[str, str] = {
    "get_logger": "utils.logging_utils",
    "set_level": "utils.logging_utils",
    "set_seed": "utils.seed",
    "DeviceInfo": "utils.device",
    "resolve_device": "utils.device",
    "resolve_dtype": "utils.device",
    "gpu_memory_stats": "utils.device",
    "reset_peak_memory": "utils.device",
    "get_device_info": "utils.device",
    "log_device_summary": "utils.device",
    "load_image": "utils.image_utils",
    "resize_keep_aspect": "utils.image_utils",
    "draw_boxes": "utils.image_utils",
    "overlay_heatmap": "utils.image_utils",
    "pdf_to_images": "utils.image_utils",
    "pil_to_base64": "utils.image_utils",
    "base64_to_pil": "utils.image_utils",
    "OCREngine": "utils.ocr",
    "OCRWord": "utils.ocr",
}

if TYPE_CHECKING:  # real symbols for type checkers/IDEs, zero runtime cost.
    from utils.device import (
        DeviceInfo,
        get_device_info,
        gpu_memory_stats,
        log_device_summary,
        reset_peak_memory,
        resolve_device,
        resolve_dtype,
    )
    from utils.image_utils import (
        base64_to_pil,
        draw_boxes,
        load_image,
        overlay_heatmap,
        pdf_to_images,
        pil_to_base64,
        resize_keep_aspect,
    )
    from utils.logging_utils import get_logger, set_level
    from utils.ocr import OCREngine, OCRWord
    from utils.seed import set_seed


def __getattr__(name: str) -> Any:
    """Resolve a public export on first access (PEP 562 lazy import)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    attr = getattr(importlib.import_module(module_path), name)
    globals()[name] = attr  # cache so later accesses skip __getattr__.
    return attr


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_EXPORTS})


__all__ = list(_LAZY_EXPORTS)
