"""Shared pytest configuration and fixtures for the VisionDoc AI test-suite.

Why this file exists
--------------------
The suite is designed to run on a *deliberately lightweight* CI image where
``numpy``/``Pillow``/``PyYAML``/``pydantic`` are installed but the heavy ML
stack (``torch``/``transformers``) may be **absent**. Two design choices follow
from that constraint:

* **Repo root on ``sys.path`` (here, at import time).** We prepend — not append
  — the repository root so the in-repo top-level packages (``configs``,
  ``utils``, ``preprocessing`` ...) win over any identically named packages that
  might exist in ``site-packages``. Doing it in ``conftest`` means the suite
  works from a fresh checkout that has not run ``pip install -e .`` yet, which
  is exactly the CI scenario we care about.
* **Fixtures skip (never error) on a missing optional dep.** ``tiny_image``
  needs Pillow and ``default_config`` needs PyYAML; both use
  ``pytest.importorskip`` so an install without one of those extras yields a
  clean *skip* rather than a red collection error.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Repo root = the parent of this ``tests/`` directory. Resolved to an absolute
# path so the suite is independent of pytest's current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # insert(0, ...) so first-party packages shadow any same-named installed one.
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tiny_image() -> Any:
    """A deterministic 64x64 RGB ``PIL.Image`` for image-pipeline tests.

    We paint a distinct inner rectangle rather than returning a flat fill so the
    image has real spatial structure: downscale/interpolation and base64
    round-trip assertions then exercise non-trivial pixel data instead of a
    uniform block that most codecs would collapse to almost nothing.

    Skips (rather than fails) when Pillow is unavailable, keeping the suite green
    on a minimal install.
    """
    pil_image = pytest.importorskip("PIL.Image")
    img = pil_image.new("RGB", (64, 64), color=(240, 240, 240))
    # A darker central block gives the image non-uniform content.
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (10, 90, 200))
    return img


@pytest.fixture
def default_config() -> Any:
    """The shipped default :class:`configs.ProjectConfig`, parsed from YAML.

    Loaded via an absolute path derived from ``REPO_ROOT`` (not a cwd-relative
    string) so the fixture is robust to wherever pytest is launched from.
    PyYAML is the config layer's only hard dependency, so we ``importorskip`` it
    to degrade to a skip on a build without it.
    """
    pytest.importorskip("yaml")
    # Imported lazily inside the fixture: the top-level import would run at
    # collection time and error out (rather than skip) if PyYAML were missing.
    from configs import load_config

    return load_config(str(REPO_ROOT / "configs" / "default.yaml"))
