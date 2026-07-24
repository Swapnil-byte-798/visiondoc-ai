# VisionDoc AI — Docker

Run the FastAPI inference server and the Streamlit dashboard in containers, from
a single image built once and reused by both services.

## Layout

| File | Purpose |
| --- | --- |
| `Dockerfile` | CPU image (`python:3.11-slim`) with OpenCV/poppler/tesseract system deps. Contains a commented **CUDA base** alternative for GPU training. |
| `docker-compose.yml` | Two services — `api` (:8000) and `app` (:8501) — sharing the image, an HF weights cache volume, and `outputs/`, `data/`, `reports/`. |
| `.dockerignore` | Keeps datasets, checkpoints, venvs, and secrets out of the build context. |

## Prerequisites

- Docker Engine 20.10+ with the Compose plugin (`docker compose …`).
- A `.env` file at the **repo root** (compose reads `../.env`):

  ```bash
  cp .env.example .env      # run from the repo root
  ```

## Quickstart

All commands are run from the **repo root**.

```bash
# Build the image and start both services (API + dashboard).
docker compose -f docker/docker-compose.yml up --build

#   API      -> http://localhost:8000   (Swagger UI at /docs)
#   Dashboard-> http://localhost:8501

# Health check (waits until the model singleton is built):
curl -fsS http://localhost:8000/health

# Run detached, then tail logs:
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml logs -f api

# Stop and remove containers (named volumes, incl. the HF cache, are kept):
docker compose -f docker/docker-compose.yml down
```

> The Makefile also wraps these: `make docker-build` and `make docker-up`.

### First request is slow — by design

The API/dashboard build the `DocumentPredictor` **lazily on first use**, which
downloads a multi-GB Vision-Language Model. That download is cached in the
`hf-cache` named volume, so subsequent container starts are fast. The API
healthcheck uses a 180s `start_period` to allow for this.

## Using a trained LoRA adapter

Point the services at an adapter directory produced by training (it lands under
`outputs/…/adapter`, which is mounted into the container):

```bash
# In .env at the repo root:
VISIONDOC_ADAPTER_PATH=outputs/<run-name>/adapter
```

Restart the stack; `/health` will then report `adapter_loaded: true`.

## Preparing data / training inside the container

The image installs the package editable, so the `visiondoc-*` entry points and
module launchers are available:

```bash
# One-off dataset download + cache:
docker compose -f docker/docker-compose.yml run --rm api \
  python -m scripts.download_data --config configs/default.yaml

# Fine-tune (CPU here; use the GPU image below for real training):
docker compose -f docker/docker-compose.yml run --rm api \
  bash scripts/train.sh
```

## GPU (training / fast inference)

The default image is **CPU-only**. For GPU:

1. Switch the base image to a CUDA runtime and install the CUDA build of torch —
   see the commented **"GPU alternative"** block at the top of `Dockerfile`.
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
   on the host.
3. Uncomment the `deploy.resources.reservations.devices` block on the `api`
   (and/or `app`) service in `docker-compose.yml`.
4. Set `device: cuda` in your config (or `VISIONDOC_DEVICE=cuda` in `.env`).

4-bit QLoRA (`bitsandbytes`) works **only** on this CUDA path — `bitsandbytes`
is intentionally Linux/CUDA-only in `requirements.txt`.

## Environment variables

Everything is driven by `../.env` (see `.env.example` for the full list). The
most relevant here:

| Variable | Meaning |
| --- | --- |
| `VISIONDOC_CONFIG` | Config YAML the services load (default `configs/default.yaml`). |
| `VISIONDOC_ADAPTER_PATH` | LoRA adapter dir to load on startup (optional). |
| `VISIONDOC_DEVICE` | `auto` \| `cuda` \| `mps` \| `cpu`. |
| `API_HOST` / `API_PORT` | API bind address/port. |
| `HF_TOKEN` | Only needed for gated models/datasets. |

## Troubleshooting

- **`ImportError: libGL.so.1`** — you are on a base image without `libgl1`; the
  provided `Dockerfile` already installs it. Rebuild with `--build`.
- **`.env` not found** — compose requires `../.env`; copy it from
  `.env.example` at the repo root.
- **Out of memory on CPU** — use the 3B model with a small `max_pixels`, or move
  to the GPU image; VLMs are memory-hungry.
