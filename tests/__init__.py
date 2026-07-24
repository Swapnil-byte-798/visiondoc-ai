"""Test package marker for the VisionDoc AI suite.

This file makes ``tests`` an importable package so pytest imports the modules as
``tests.<name>`` (its default "prepend" import mode). We keep it intentionally
empty of logic: all shared setup (repo-root ``sys.path`` injection and fixtures)
lives in :mod:`tests.conftest`, which pytest loads automatically before any test
module is collected.
"""
