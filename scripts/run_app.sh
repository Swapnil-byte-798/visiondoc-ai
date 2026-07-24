#!/usr/bin/env bash
# ===========================================================================
# VisionDoc AI — Streamlit dashboard launcher
# ---------------------------------------------------------------------------
# Runs the interactive dashboard (app/streamlit_app.py). Used as the Docker
# "app" service entrypoint and for local demos.
#
# Env overrides:
#   APP_HOST   bind address (default: 0.0.0.0 — reachable from outside the
#              container)
#   APP_PORT   bind port    (default: 8501)
#   PYTHON     python interpreter (default: python3)
# Extra args are forwarded to streamlit after a `--` separator.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8501}"
PYTHON="${PYTHON:-python3}"

echo "[run_app] repo=${REPO_ROOT}"
echo "[run_app] http://${APP_HOST}:${APP_PORT}"

# `python -m streamlit` (not the bare `streamlit` binary) so we use the
# interpreter that has the project + deps installed. --server.headless=true
# stops Streamlit from trying to open a browser / prompt for an email inside a
# container. exec forwards SIGTERM so `docker stop` shuts it down cleanly.
exec "${PYTHON}" -m streamlit run app/streamlit_app.py \
    --server.address "${APP_HOST}" \
    --server.port "${APP_PORT}" \
    --server.headless true \
    "$@"
