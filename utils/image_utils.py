"""Image & PDF I/O helpers shared by preprocessing, inference, API and app.

Everything that turns *some* user input (a path, raw bytes, a base64 string, a
NumPy array, a PDF page) into a normalized ``PIL.Image`` lives here, so the API
and Streamlit app never re-implement decoding. Region drawing and heatmap
overlay (used for "highlight important document regions" + attention viz) also
live here to keep visualization consistent everywhere.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A box is (x0, y0, x1, y1) in *absolute pixel* coordinates unless stated.
Box = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Decoding: anything -> PIL.Image (RGB)
# ---------------------------------------------------------------------------


def load_image(source: str | bytes | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load an image from many possible sources into an RGB ``PIL.Image``.

    Accepts a filesystem path, raw ``bytes``, a base64 data URI/string, a
    NumPy array (H, W, C), or an existing ``PIL.Image``. This is the single
    entry point used by the API upload handler and the Streamlit uploader.
    """
    if isinstance(source, Image.Image):
        return ensure_rgb(source)
    if isinstance(source, np.ndarray):
        return ensure_rgb(Image.fromarray(source.astype(np.uint8)))
    if isinstance(source, (str, Path)):
        s = str(source)
        # Base64 data-URI or bare base64 payload (used by the JSON API).
        if s.startswith("data:image") or _looks_like_base64(s):
            return base64_to_pil(s)
        return ensure_rgb(Image.open(s))
    if isinstance(source, bytes):
        return ensure_rgb(Image.open(io.BytesIO(source)))
    raise TypeError(f"Unsupported image source type: {type(source)!r}")


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB (documents may be grayscale, CMYK, or have alpha)."""
    return img if img.mode == "RGB" else img.convert("RGB")


def _looks_like_base64(s: str) -> bool:
    """Cheap heuristic: long, base64 alphabet, and not an existing file path."""
    if len(s) < 100 or Path(s[:256]).exists():
        return False
    sample = s.split(",", 1)[-1][:64]
    return all(c.isalnum() or c in "+/=\n\r" for c in sample)


# ---------------------------------------------------------------------------
# base64 <-> PIL (JSON-friendly transport for the API)
# ---------------------------------------------------------------------------


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode an image as a base64 string (no data-URI prefix)."""
    buf = io.BytesIO()
    ensure_rgb(img).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(data: str) -> Image.Image:
    """Decode a base64 string (with or without ``data:image/...;base64,`` prefix)."""
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return ensure_rgb(Image.open(io.BytesIO(raw)))


# ---------------------------------------------------------------------------
# Resizing / normalization (light — the model's own processor does the rest)
# ---------------------------------------------------------------------------


def resize_keep_aspect(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Downscale so the longest side == ``max_side``; never upscales.

    We only cap the resolution (to bound tokens/VRAM) and preserve aspect ratio
    so text is not distorted. The VLM processor handles final tiling/patching.
    """
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def to_normalized_array(img: Image.Image, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Return a CHW float32 array normalized to roughly [-1, 1].

    Provided for custom/debug pipelines and Donut-style preprocessing; the
    Hugging Face processors normally do this internally for us.
    """
    arr = np.asarray(ensure_rgb(img), dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


# ---------------------------------------------------------------------------
# Box helpers (for structured extraction + region highlighting)
# ---------------------------------------------------------------------------


def normalize_box(box: Box, width: int, height: int) -> Box:
    """Absolute pixel box -> [0, 1] fractions (used by FUNSD/DocVQA layouts)."""
    x0, y0, x1, y1 = box
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def denormalize_box(box: Box, width: int, height: int) -> Box:
    """[0, 1] fractional box -> absolute pixels for drawing."""
    x0, y0, x1, y1 = box
    return (x0 * width, y0 * height, x1 * width, y1 * height)


def draw_boxes(
    img: Image.Image,
    boxes: list[Box],
    labels: list[str] | None = None,
    color: str = "#FF3B30",
    width: int = 3,
    normalized: bool = False,
) -> Image.Image:
    """Return a copy of ``img`` with boxes (and optional labels) drawn.

    Used to visualize the "important document regions" a prediction attends to,
    and to draw extracted-field bounding boxes in the dashboard.
    """
    out = ensure_rgb(img).copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    for i, box in enumerate(boxes):
        b = denormalize_box(box, W, H) if normalized else box
        draw.rectangle(b, outline=color, width=width)
        if labels and i < len(labels):
            x0, y0 = b[0], max(0, b[1] - 12)
            draw.text((x0, y0), labels[i], fill=color, font=font)
    return out


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a 2D attention/saliency heatmap over the image (for attention viz).

    ``heatmap`` is any 2D array; it is min-max normalized, resized to the image,
    colorized (matplotlib 'jet' if available, else grayscale) and alpha-blended.
    """
    base = ensure_rgb(img)
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {hm.shape}")
    hm = (hm - hm.min()) / (np.ptp(hm) + 1e-8)  # np.ptp: ndarray.ptp() removed in NumPy 2.0
    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
    try:  # Nice perceptual colormap when matplotlib is present.
        import matplotlib.cm as cm

        colored = (cm.jet(np.asarray(hm_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)
        heat = Image.fromarray(colored)
    except Exception:  # pragma: no cover - matplotlib optional at runtime
        heat = hm_img.convert("RGB")
    return Image.blend(base, heat, alpha=alpha)


# ---------------------------------------------------------------------------
# PDF support (bonus): rasterize pages to images
# ---------------------------------------------------------------------------


def pdf_to_images(source: str | bytes | Path, dpi: int = 200, max_pages: int = 20) -> list[Image.Image]:
    """Rasterize a PDF into a list of RGB page images.

    Uses PyMuPDF (fitz) — fast and dependency-light. Imported lazily so the
    core install works without it. ``max_pages`` guards against giant PDFs.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "PDF support requires PyMuPDF. Install with `pip install PyMuPDF`."
        ) from exc

    if isinstance(source, bytes):
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        doc = fitz.open(str(source))

    zoom = dpi / 72.0  # 72 dpi is the PDF base resolution
    matrix = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    for page in doc[:max_pages]:
        pix = page.get_pixmap(matrix=matrix)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images


__all__ = [
    "Box",
    "load_image",
    "ensure_rgb",
    "pil_to_base64",
    "base64_to_pil",
    "resize_keep_aspect",
    "to_normalized_array",
    "normalize_box",
    "denormalize_box",
    "draw_boxes",
    "overlay_heatmap",
    "pdf_to_images",
]
