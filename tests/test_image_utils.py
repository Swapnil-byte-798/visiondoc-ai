"""Tests for the image I/O helpers (``utils/image_utils.py``).

These cover the three transforms every serving path (API upload, Streamlit
uploader, region highlighting) relies on: base64 transport, resolution capping,
and box drawing. They need Pillow + numpy (both present on the target image),
but importing ``utils.image_utils`` also runs ``utils/__init__`` and its
torch-backed device module — so we ``importorskip`` the module to skip cleanly
when torch is absent while running the real assertions when it is present.
"""

from __future__ import annotations

from typing import Any

import pytest

# numpy/Pillow are declared present on the CI image, but guard anyway so a
# minimal build degrades to a skip rather than a hard collection error.
pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

# Guarded: utils/__init__ -> utils.device -> `import torch`. Skip when torch is
# unavailable; the helpers under test are torch-free but the package init is not.
image_utils = pytest.importorskip("utils.image_utils")


def test_base64_roundtrip_preserves_size_and_mode(tiny_image: Any) -> None:
    """pil_to_base64 -> base64_to_pil returns an equivalent RGB image."""
    encoded = image_utils.pil_to_base64(tiny_image)
    assert isinstance(encoded, str) and encoded  # non-empty payload
    restored = image_utils.base64_to_pil(encoded)
    # PNG is lossless, so dimensions and colour mode survive the round-trip.
    assert restored.size == tiny_image.size
    assert restored.mode == "RGB"


def test_resize_keep_aspect_caps_long_side(tiny_image: Any) -> None:
    """A 64x64 image capped at 32 has its long side reduced to exactly 32."""
    out = image_utils.resize_keep_aspect(tiny_image, max_side=32)
    assert max(out.size) == 32
    # Square in, square out (aspect ratio preserved).
    assert out.size == (32, 32)


def test_resize_keep_aspect_preserves_aspect_ratio() -> None:
    """A non-square image keeps its ratio when downscaled."""
    # 100x50 capped at 50 -> scale 0.5 -> (50, 25); ratio 2:1 preserved.
    rect = Image.new("RGB", (100, 50), color=(0, 0, 0))
    out = image_utils.resize_keep_aspect(rect, max_side=50)
    assert out.size == (50, 25)


def test_resize_keep_aspect_never_upscales(tiny_image: Any) -> None:
    """Requesting a larger max_side must not enlarge a small image."""
    # Upscaling document scans would invent detail and waste VRAM/tokens, so the
    # helper is documented to only ever shrink — verify the no-op path.
    out = image_utils.resize_keep_aspect(tiny_image, max_side=512)
    assert out.size == tiny_image.size  # unchanged 64x64


def test_draw_boxes_returns_same_size_copy(tiny_image: Any) -> None:
    """draw_boxes returns a same-size PIL image and does not mutate the input."""
    boxes = [(8, 8, 40, 40)]  # absolute pixels within the 64x64 canvas
    out = image_utils.draw_boxes(tiny_image, boxes, labels=["field"])
    assert isinstance(out, Image.Image)
    assert out.size == tiny_image.size
    # Contract: draw_boxes copies (so callers keep an un-annotated original).
    assert out is not tiny_image


def test_load_image_passthrough_returns_rgb(tiny_image: Any) -> None:
    """load_image accepts an existing PIL image and normalizes it to RGB."""
    out = image_utils.load_image(tiny_image)
    assert out.size == tiny_image.size
    assert out.mode == "RGB"
