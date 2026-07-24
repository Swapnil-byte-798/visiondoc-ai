"""``torch.utils.data.Dataset`` wrapper over a list of ``DocSample``s.

Design rationale
----------------
The trainer needs a map-style dataset whose ``__getitem__`` yields one plain
record dict per example; the model's ``collate_train`` then turns a *batch* of
those records into tensors. Keeping this class a thin adapter (records in, records
out) means all the model-specific tensorization lives in one place (the model
collator) and the dataset stays backbone-agnostic.

Two deliberate choices:

* **Lazy image loading.** Samples may hold a path string instead of a decoded
  image (large corpora should not sit fully decoded in RAM). We decode via
  ``utils.load_image`` only inside ``__getitem__`` — i.e. only for the indices a
  worker is currently fetching.
* **Augmentation at fetch time.** The augmenter is applied per ``__getitem__`` so
  every epoch sees freshly perturbed pixels, which is the whole point of online
  augmentation. Eval/inference pass ``augmenter=None`` for determinism.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.image_utils import load_image
from utils.logging_utils import get_logger

from preprocessing.schema import DocSample

if TYPE_CHECKING:  # For type hints only.
    from preprocessing.transforms import DocumentAugmenter

logger = get_logger(__name__)

# Inherit from torch's Dataset when torch is importable, else fall back to
# ``object``. torch.utils.data.Dataset is essentially a marker base (it adds no
# required behavior beyond __getitem__/__len__), so this fallback keeps the module
# importable in a torch-less context while behaving identically under training.
try:  # pragma: no cover - trivial import guard
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover - torch absent
    _TorchDataset = object  # type: ignore[assignment,misc]


class DocumentDataset(_TorchDataset):  # type: ignore[valid-type,misc]
    """Map-style dataset yielding augmented ``DocSample.to_record()`` dicts."""

    def __init__(
        self, samples: list[DocSample], augmenter: "DocumentAugmenter | None" = None
    ) -> None:
        """Wrap ``samples``; apply ``augmenter`` (if any) to each fetched image."""
        self.samples = samples
        self.augmenter = augmenter
        logger.info(
            "DocumentDataset: %d samples, augment=%s", len(samples), augmenter is not None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the record for ``index`` with its image decoded and augmented."""
        sample = self.samples[index]
        record = sample.to_record()

        image = sample.image
        if image is not None:
            # ``load_image`` accepts a path/bytes/ndarray/PIL and always returns an
            # RGB PIL image, so a single call handles every ``DocSample.image`` form.
            if not _is_pil(image):
                image = load_image(image)
            elif image.mode != "RGB":
                image = image.convert("RGB")
            if self.augmenter is not None:
                image = self.augmenter(image)
            record["image"] = image
        return record


def _is_pil(obj: Any) -> bool:
    """True if ``obj`` is a PIL image, without importing PIL at module top level.

    We avoid a top-level ``from PIL import Image`` so this module keeps its import
    surface minimal; the check is only needed at fetch time anyway.
    """
    try:
        from PIL.Image import Image as PILImage

        return isinstance(obj, PILImage)
    except Exception:  # pragma: no cover - PIL always present in practice
        # Fall back to a duck-typed check (PIL images expose .size and .mode).
        return hasattr(obj, "size") and hasattr(obj, "mode") and not isinstance(obj, (str, Path, bytes))


__all__ = ["DocumentDataset"]
