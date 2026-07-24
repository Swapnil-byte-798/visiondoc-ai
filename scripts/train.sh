#!/usr/bin/env bash
# ===========================================================================
# VisionDoc AI — LoRA fine-tuning launcher
# ---------------------------------------------------------------------------
# Thin wrapper over `python -m training.train`. Kept as a script (not just a
# Makefile target) so it works unchanged inside Docker, CI, and SLURM/torchrun
# jobs where a Makefile may not be present.
#
# Env overrides:
#   CONFIG   config YAML (default: configs/default.yaml)
#   PYTHON   python interpreter (default: python3)
# Extra args after the script are forwarded verbatim, e.g.:
#   scripts/train.sh --resume outputs/run/checkpoint-500
# ===========================================================================

# Strict mode: -e exit on error, -u error on unset vars, -o pipefail catch
# failures inside pipes. Non-negotiable for launch scripts so a failed training
# run never looks like a success in CI logs.
set -euo pipefail

# Resolve the repo root from this script's location so the command works no
# matter where it is invoked from (cwd-independence matters in Docker/CI).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
PYTHON="${PYTHON:-python3}"

echo "[train] repo=${REPO_ROOT}"
echo "[train] config=${CONFIG}"
echo "[train] python=$(command -v "${PYTHON}")"

# exec replaces the shell so signals (Ctrl-C, SIGTERM from Docker) reach the
# training process directly, allowing HF Trainer to checkpoint/clean up.
exec "${PYTHON}" -m training.train --config "${CONFIG}" "$@"
