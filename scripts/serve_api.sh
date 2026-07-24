#!/usr/bin/env bash
# ===========================================================================
# VisionDoc AI — FastAPI inference server launcher
# ---------------------------------------------------------------------------
# Serves api.main:app with uvicorn. Used as the Docker API service entrypoint
# and for local development.
#
# Env overrides:
#   API_HOST    bind address (default: 0.0.0.0 — required so it is reachable
#               from outside the container; localhost would be unreachable)
#   API_PORT    bind port    (default: 8000)
#   API_WORKERS uvicorn worker processes (default: 1 — a VLM holds large GPU
#               weights, so multiple workers would multiply VRAM use; scale out
#               with replicas/GPUs instead of workers)
#   PYTHON      python interpreter (default: python3)
# Extra args (e.g. --reload for dev) are forwarded to uvicorn.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
API_WORKERS="${API_WORKERS:-1}"
PYTHON="${PYTHON:-python3}"

echo "[serve_api] repo=${REPO_ROOT}"
echo "[serve_api] http://${API_HOST}:${API_PORT}  (workers=${API_WORKERS})"

# Invoke uvicorn via `python -m` rather than the bare `uvicorn` binary so we use
# the interpreter that has the project installed — avoids "module not found"
# when multiple Pythons/venvs are on PATH. exec forwards signals for clean
# shutdown (important for Docker stop).
exec "${PYTHON}" -m uvicorn api.main:app \
    --host "${API_HOST}" \
    --port "${API_PORT}" \
    --workers "${API_WORKERS}" \
    "$@"
