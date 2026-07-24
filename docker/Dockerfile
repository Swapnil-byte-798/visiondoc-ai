# syntax=docker/dockerfile:1
# ===========================================================================
# VisionDoc AI — container image (CPU default)
# ---------------------------------------------------------------------------
# One image serves BOTH the FastAPI inference server (port 8000) and the
# Streamlit dashboard (port 8501); docker-compose picks which by overriding the
# command. Building a single image keeps the two services byte-for-byte
# identical (same weights cache, same code) and halves build time.
#
# GPU alternative (training / fast inference):
#   Swap the base image for an NVIDIA CUDA runtime that already ships the right
#   CUDA/cuDNN userspace, and install the matching CUDA build of torch:
#
#     FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#     RUN apt-get update && apt-get install -y python3.11 python3-pip \
#         && ln -sf /usr/bin/python3.11 /usr/bin/python
#     # then, instead of the plain `pip install -r requirements.txt` below:
#     RUN pip install torch torchvision --index-url \
#           https://download.pytorch.org/whl/cu124 \
#      && pip install -r requirements.txt
#
#   Run it with `--gpus all` (or the compose deploy block) and set
#   `device: cuda` in your config. bitsandbytes (4-bit QLoRA) only works on this
#   CUDA path — it is intentionally Linux-only in requirements.txt.
# ===========================================================================

# python:3.11-slim: Debian-based, small, and has a modern pip. We deliberately
# avoid `-alpine` because manylinux wheels (torch, opencv, pillow) are built for
# glibc; Alpine's musl would force slow source builds or fail outright.
FROM python:3.11-slim AS runtime

# --- Interpreter / pip behaviour ------------------------------------------
# PYTHONUNBUFFERED: stream logs immediately (no buffering) so `docker logs`
#   shows training/inference progress in real time.
# PYTHONDONTWRITEBYTECODE: skip .pyc files — pointless churn in an ephemeral
#   container layer.
# PIP_NO_CACHE_DIR: don't keep the wheel cache; it would bloat the image.
# HF_HOME: put the Hugging Face cache on a stable path we can mount as a volume
#   so multi-GB model weights survive container restarts (no re-download).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# --- System dependencies ---------------------------------------------------
# These are the non-Python libraries the wheels dlopen at runtime:
#   build-essential  : compiler toolchain for any sdist that lacks a wheel.
#   libgl1 / libglib2.0-0 : shared libs OpenCV (cv2) links against; missing
#                     libGL is the classic "ImportError: libGL.so.1" in slim.
#   poppler-utils    : `pdftoppm`/`pdftocairo` backing pdf2image (PDF -> image).
#   tesseract-ocr    : the OCR engine pytesseract shells out to (answer-region
#                     highlighting + VLM-abstain fallback).
#   curl             : used by the compose healthcheck to probe /health.
# Clean the apt lists in the SAME layer so they don't get baked into the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (cached layer) ------------------------------------
# Copy ONLY the requirements first. Docker caches this layer and reuses it on
# every rebuild where requirements.txt is unchanged, so editing source code does
# not trigger a full (very slow) torch/transformers reinstall.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# --- Application code ------------------------------------------------------
# Now copy the rest of the repo and install it editable so the absolute imports
# (`from models import ...`, `from configs import ...`) resolve from any cwd.
# `-e` + the pyproject `[project.scripts]` also puts `visiondoc-*` on PATH.
COPY . .
RUN pip install -e .

# Runtime dirs for artifacts; compose mounts host volumes over these so outputs
# and processed data persist outside the container lifecycle.
RUN mkdir -p /app/outputs /app/data /app/reports /app/.cache/huggingface

# 8000 = FastAPI (uvicorn), 8501 = Streamlit. Documented here; the actual
# published ports are set per-service in docker-compose.yml.
EXPOSE 8000 8501

# Default command runs the API. The Streamlit service overrides this in compose.
# Bind 0.0.0.0 so the server is reachable from outside the container.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
