"""Project-wide logging setup.

A single :func:`get_logger` gives every module a consistently formatted logger.
When the optional ``rich`` dependency is present we use its colorized handler
(nice for the CLI and notebooks); otherwise we fall back to the stdlib format
so logging never becomes a hard dependency on ``rich``.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_DEFAULT_LEVEL = os.getenv("VISIONDOC_LOG_LEVEL", "INFO").upper()


def _configure_root() -> None:
    """Attach a single handler to the ``visiondoc`` root logger, once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("visiondoc")
    root.setLevel(_DEFAULT_LEVEL)
    root.propagate = False

    handler: logging.Handler
    try:  # Prefer rich for readable, colorized output when available.
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    except Exception:  # pragma: no cover - rich is optional
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str = "visiondoc") -> logging.Logger:
    """Return a namespaced logger under the ``visiondoc`` root.

    Example::

        from utils.logging_utils import get_logger
        logger = get_logger(__name__)
        logger.info("training started")
    """
    _configure_root()
    if name == "visiondoc" or name.startswith("visiondoc."):
        return logging.getLogger(name)
    # Namespace third-party module names (e.g. __name__ = "training.train")
    # under our root so they share formatting and level.
    return logging.getLogger(f"visiondoc.{name}")


def set_level(level: str | int) -> None:
    """Adjust the global log level at runtime (e.g. ``"DEBUG"``)."""
    _configure_root()
    logging.getLogger("visiondoc").setLevel(level)


__all__ = ["get_logger", "set_level"]
