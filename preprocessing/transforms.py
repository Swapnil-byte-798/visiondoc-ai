"""Document-safe image augmentation.

Design rationale
----------------
Augmentation on *document* images is a minefield: the usual computer-vision
transforms (horizontal/vertical flips, large rotations, heavy crops, hue shifts)
destroy the very signal we care about — the text and its layout. A horizontally
flipped invoice is unreadable and would teach the model nonsense.

So this augmenter deliberately restricts itself to perturbations that a real
scanner or phone camera would plausibly introduce, and which leave the text
legible and its reading order intact:

* **±2° rotation** — pages are rarely fed perfectly straight; a tiny skew
  improves robustness. We keep it small so lines stay roughly horizontal and any
  normalized boxes remain approximately valid.
* **Brightness / contrast jitter** — models exposure differences between a bright
  scan and a dim phone photo.
* **Sharpness jitter + light Gaussian blur** — simulates focus variation and
  low-resolution scans.
* **Gaussian pixel noise** — sensor / compression noise on cheap scanners.
* **Occasional grayscale** — a large fraction of real documents are black-and-white
  faxes/scans; training on both teaches color invariance.

Explicitly **never**: horizontal or vertical flips, 90° rotations, or aggressive
color remapping — all of which would corrupt text semantics.

Only Pillow + NumPy are used (both hard dependencies already) so this module has
no torchvision requirement and stays importable in a minimal environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # Type-only import; no runtime dependency on configs here.
    from configs import ProjectConfig

logger = get_logger(__name__)


class DocumentAugmenter:
    """Callable that applies randomized, document-safe augmentations to a PIL image.

    A single instance holds its own seeded RNG so that, given a fixed seed and a
    fixed call order, the augmentation stream is reproducible — important when an
    eval or debug run needs to be repeatable.

    Note on DataLoader workers: when this runs inside a multi-worker
    ``DataLoader`` each worker forks a copy of the RNG. For strict per-epoch
    reproducibility set a ``worker_init_fn`` that re-seeds per worker; the
    document-safe transforms here are mild enough that correlated draws across
    workers do not meaningfully hurt training.
    """

    # Magnitudes are hard-coded (DataConfig only exposes the on/off ``augment``
    # flag) and kept intentionally conservative — see the module docstring for
    # why each range is safe for text-bearing images.
    _MAX_ROTATION_DEG = 2.0
    _BRIGHTNESS_RANGE = (0.85, 1.15)
    _CONTRAST_RANGE = (0.85, 1.15)
    _SHARPNESS_RANGE = (0.7, 1.6)
    _BLUR_MAX_RADIUS = 0.8
    _NOISE_MAX_SIGMA = 8.0  # in 0-255 pixel units

    # Per-op application probabilities. Not every image gets every transform, so
    # the model still sees plenty of clean examples.
    _P_ROTATE = 0.5
    _P_BRIGHTNESS = 0.5
    _P_CONTRAST = 0.5
    _P_SHARPNESS = 0.3
    _P_BLUR = 0.2
    _P_NOISE = 0.2
    _P_GRAYSCALE = 0.15

    def __init__(self, config: "ProjectConfig") -> None:
        # Seed from the project seed for reproducibility; we advance this RNG on
        # every call so successive samples differ while remaining deterministic.
        self._rng = np.random.RandomState(config.seed)
        self._config = config

    def _chance(self, prob: float) -> bool:
        """Return True with probability ``prob`` using the instance RNG."""
        return bool(self._rng.random_sample() < prob)

    def _uniform(self, low: float, high: float) -> float:
        return float(self._rng.uniform(low, high))

    def __call__(self, image: "Image.Image") -> "Image.Image":
        """Return an augmented RGB copy of ``image`` (input is never mutated)."""
        # Work on an RGB copy so enhancers/filters have a consistent 3-channel
        # input and the caller's original image is left untouched.
        img = image.convert("RGB") if image.mode != "RGB" else image.copy()

        if self._chance(self._P_ROTATE):
            angle = self._uniform(-self._MAX_ROTATION_DEG, self._MAX_ROTATION_DEG)
            # expand=False keeps the canvas size stable (so batched tensor shapes
            # stay predictable); white fill matches typical paper background.
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

        if self._chance(self._P_BRIGHTNESS):
            img = ImageEnhance.Brightness(img).enhance(self._uniform(*self._BRIGHTNESS_RANGE))

        if self._chance(self._P_CONTRAST):
            img = ImageEnhance.Contrast(img).enhance(self._uniform(*self._CONTRAST_RANGE))

        if self._chance(self._P_SHARPNESS):
            img = ImageEnhance.Sharpness(img).enhance(self._uniform(*self._SHARPNESS_RANGE))

        if self._chance(self._P_BLUR):
            img = img.filter(ImageFilter.GaussianBlur(radius=self._uniform(0.1, self._BLUR_MAX_RADIUS)))

        if self._chance(self._P_NOISE):
            img = self._add_gaussian_noise(img)

        if self._chance(self._P_GRAYSCALE):
            # Collapse to luminance then back to 3 channels: keeps tensor shape
            # while removing color — mimics a B/W scan/fax.
            img = img.convert("L").convert("RGB")

        return img

    def _add_gaussian_noise(self, img: "Image.Image") -> "Image.Image":
        """Add mild zero-mean Gaussian noise, clipped to valid pixel range."""
        sigma = self._uniform(1.0, self._NOISE_MAX_SIGMA)
        arr = np.asarray(img, dtype=np.float32)
        noise = self._rng.normal(loc=0.0, scale=sigma, size=arr.shape)
        noisy = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")


def build_augmenter(config: "ProjectConfig") -> DocumentAugmenter | None:
    """Return a :class:`DocumentAugmenter` when augmentation is enabled, else ``None``.

    Returning ``None`` (rather than an identity augmenter) lets callers cheaply
    skip the whole augmentation branch and makes eval/inference — which must be
    deterministic — trivially opt out.
    """
    if not config.data.augment:
        logger.info("Augmentation disabled (data.augment=false).")
        return None
    logger.info("Document augmentation enabled (safe transforms only).")
    return DocumentAugmenter(config)


__all__ = ["DocumentAugmenter", "build_augmenter"]
