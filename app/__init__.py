"""Streamlit front-end package for VisionDoc AI.

This package holds the interactive dashboard (:mod:`app.streamlit_app`) — the
human-facing counterpart to the programmatic :mod:`api` service. It is kept as a
*separate* top-level package (rather than under ``inference`` or ``api``) for one
deliberate reason: the Streamlit runtime imports the script it is handed with a
fresh interpreter and re-executes it top-to-bottom on every widget interaction,
so its entry module must stay import-cheap and free of side effects at import
time. Placing it here keeps that constraint local and prevents the API package
from accidentally pulling Streamlit as a dependency.

There is intentionally no re-export here: the dashboard is launched via
``streamlit run app/streamlit_app.py`` (see ``scripts/run_app.sh``), never
imported by other code, so exposing its internals would only invite the heavy
Streamlit import at package-import time.
"""
