"""Regression guard: training must never decode a whole split into RAM.

Why this test exists
--------------------
Two published Colab runs died having consumed all ~12.7 GB of system RAM before
training reached step 1. The cause both times was the same shape of bug: some
code path iterated a ``datasets.Dataset`` whose ``image`` column still had a
``decode=True`` ``datasets.Image`` feature, so every row decoded a full-resolution
page into a PIL object and the entire split was held in a Python list. A CORD scan
is 3-28 MB decoded; 1000 rows is roughly 5-6 GB.

The first fix capped rows before decoding, which only helps when a cap is set --
and the published config deliberately trains on all 800 receipts. The second fix
disabled decoding on the *cached* splits, which only helps on a cache hit -- and a
first run has no cache, so it took the freshly-built branch and decoded anyway.

The invariant that actually matters is path-independent: **whatever
``_obtain_splits`` returns, no image may have been decoded to produce it.** This
test asserts that directly, with a fake ``datasets`` module that makes a decode
observable, so any future branch that forgets the guard fails here instead of in a
user's GPU session an hour into a run.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DECODES = {"n": 0}


class _FakeHFImage:
    """Stand-in for ``datasets.Image``; only the ``decode`` flag matters here."""

    def __init__(self, decode: bool = True) -> None:
        self.decode = decode


class _FakeSplit:
    """A Dataset-like split that counts how often iteration decodes an image."""

    def __init__(self, n: int, decode: bool = True) -> None:
        self.n = n
        self.decode = decode
        self.column_names = ["image", "question", "answer"]

    def __len__(self) -> int:
        return self.n

    def cast_column(self, column: str, feature: _FakeHFImage) -> "_FakeSplit":
        return _FakeSplit(self.n, decode=feature.decode)

    def __iter__(self):
        for i in range(self.n):
            if self.decode:
                # This is where a real datasets.Image(decode=True) would build a
                # full-resolution PIL object and pin it in RAM.
                DECODES["n"] += 1
                image = f"<decoded-pil-{i}>"
            else:
                image = {"bytes": b"\x89PNG-encoded-bytes", "path": None}
            yield {
                "image": image, "question": "q", "answer": "a", "answers": ["a"],
                "boxes": [], "box_labels": [], "sample_id": str(i),
                "task": "extraction", "metadata": {},
            }


def _splits() -> dict[str, _FakeSplit]:
    return {"train": _FakeSplit(800), "validation": _FakeSplit(100), "test": _FakeSplit(100)}


def _load_train_module():
    """Import training/train.py by path (the package __init__ needs transformers)."""
    spec = importlib.util.spec_from_file_location(
        "train_under_test", REPO_ROOT / "training" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_obtain_splits(cached, built):
    """Drive the real ``_obtain_splits`` with a stubbed preprocessing facade."""
    real_datasets_mod = importlib.import_module("preprocessing.datasets")
    keep = real_datasets_mod.keep_images_encoded  # the REAL guard under test
    schema_mod = sys.modules["preprocessing.schema"]

    facade = types.ModuleType("preprocessing")
    facade.keep_images_encoded = keep
    facade.load_cached = lambda _cfg: cached
    facade.build_and_cache = lambda _cfg: built
    facade.build_splits = lambda _cfg: {}

    saved = sys.modules.get("preprocessing")
    sys.modules["preprocessing"] = facade
    sys.modules["preprocessing.schema"] = schema_mod
    try:
        DECODES["n"] = 0
        return _load_train_module()._obtain_splits(object())
    finally:
        if saved is not None:
            sys.modules["preprocessing"] = saved


@pytest.fixture(autouse=True)
def _fake_datasets(monkeypatch):
    """Install a fake ``datasets`` module so a decode is observable."""
    module = types.ModuleType("datasets")
    module.Image = _FakeHFImage
    monkeypatch.setitem(sys.modules, "datasets", module)
    # Drop cached imports so the modules under test bind to the fake.
    for name in ("preprocessing.datasets", "preprocessing.schema"):
        sys.modules.pop(name, None)
    importlib.import_module("preprocessing.schema")
    yield


@pytest.mark.parametrize("branch", ["cache_hit", "fresh_build"])
def test_obtain_splits_never_decodes_a_whole_split(branch):
    """Neither resolution branch may decode images while materializing splits."""
    if branch == "cache_hit":
        splits = _run_obtain_splits(cached=_splits(), built=None)
    else:
        splits = _run_obtain_splits(cached=None, built=_splits())

    assert DECODES["n"] == 0, (
        f"{branch}: {DECODES['n']} images were decoded into RAM while building splits. "
        "Wrap the split in preprocessing.keep_images_encoded() before iterating it."
    )
    assert {k: len(v) for k, v in splits.items()} == {"train": 800, "validation": 100, "test": 100}
    # Images must still be the undecoded {"bytes", "path"} mapping that
    # utils.image_utils.load_image decodes one-at-a-time at collate time.
    assert isinstance(splits["train"][0].image, dict)


def test_fake_split_would_decode_without_the_guard():
    """Sanity-check the harness itself: unguarded iteration really does decode.

    Without this, a broken fake could make the test above pass vacuously.
    """
    DECODES["n"] = 0
    for split in _splits().values():
        for _ in split:
            pass
    assert DECODES["n"] == 1000
