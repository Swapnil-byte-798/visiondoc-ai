"""Operational entry-point scripts for VisionDoc AI.

This package intentionally holds only *thin* CLI wrappers (e.g.
:mod:`scripts.download_data`) whose real logic lives in the library packages
(``preprocessing``, ``training``, ``evaluation`` …). Keeping the heavy lifting in
importable library code — and the ``scripts`` here as one-screen dispatchers —
means the exact same behaviour is reachable three ways without duplication:

* as a console entry point declared in ``pyproject.toml``
  (``visiondoc-download = "scripts.download_data:main"``),
* as a module (``python -m scripts.download_data``), and
* from other Python code (``from preprocessing import build_and_cache``).

Making ``scripts`` a real package (rather than a loose folder of files) is what
lets the ``scripts.download_data:main`` entry point resolve after
``pip install -e .`` regardless of the current working directory.
"""

from __future__ import annotations

# NOTE: deliberately no eager imports here. The submodules pull in heavy,
# optional dependencies (torch via ``utils.seed``, ``datasets`` for the actual
# build). Importing the package should stay cheap so that tools which only need
# to *discover* the entry point (pip, ``python -m``) never pay that cost.

__all__ = ["download_data"]
