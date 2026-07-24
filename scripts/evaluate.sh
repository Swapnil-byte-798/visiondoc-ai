#!/usr/bin/env bash
# ===========================================================================
# VisionDoc AI — evaluation launcher
# ---------------------------------------------------------------------------
# Thin wrapper over `python -m evaluation.evaluate`. Forwards extra args so you
# can point it at a specific adapter / split, e.g.:
#   scripts/evaluate.sh --adapter outputs/run/adapter --split test --max-samples 200
#
# Env overrides:
#   CONFIG   config YAML (default: configs/default.yaml)
#   PYTHON   python interpreter (default: python3)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
PYTHON="${PYTHON:-python3}"

echo "[evaluate] repo=${REPO_ROOT}"
echo "[evaluate] config=${CONFIG}"

# exec: forward signals directly to the evaluation process (see train.sh).
exec "${PYTHON}" -m evaluation.evaluate --config "${CONFIG}" "$@"
