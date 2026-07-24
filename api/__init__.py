"""VisionDoc AI HTTP API package.

Exposes the FastAPI application object so it can be launched with the standard
``uvicorn api:app`` / ``uvicorn api.main:app`` invocations and imported by tests
that want to exercise the routes with ``fastapi.testclient.TestClient``.

Importing this package builds the app object (routes + middleware) but does
*not* load the model: :mod:`api.main` defers the config load and the
``DocumentPredictor`` construction to the first request (see ``get_predictor``),
so ``import api`` stays cheap and works without a GPU.
"""

from __future__ import annotations

from api.main import app, get_config, get_predictor

__all__ = ["app", "get_predictor", "get_config"]
