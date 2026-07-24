"""Device & precision resolution utilities.

Why this exists
---------------
Training and inference must run unchanged on three very different machines:

* a CUDA GPU box (the real training target),
* an Apple-Silicon laptop (MPS — used for smoke tests / the demo app),
* a CPU-only CI runner (syntax + logic tests).

Centralizing device/dtype selection here means no other module ever hard-codes
``"cuda"``. Everything asks :func:`resolve_device` / :func:`resolve_dtype`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger("visiondoc.device")


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of the compute environment, for logs and evaluation reports."""

    device: str  # "cuda" | "mps" | "cpu"
    device_name: str
    dtype: str
    n_gpus: int
    total_memory_gb: float


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available device.

    ``preference`` may be ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. An explicit preference
    that is unavailable falls back (with a warning) rather than crashing — a
    demo should still run on a laptop even if the config says ``cuda``.
    """
    preference = (preference or "auto").lower()

    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if preference == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if preference == "mps":
        if mps_ok:
            return torch.device("mps")
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if preference == "cpu":
        return torch.device("cpu")

    # auto
    if cuda_ok:
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device | None = None) -> torch.dtype:
    """Map a config dtype string to a ``torch.dtype``, respecting hardware.

    bfloat16 is only enabled on CUDA GPUs that support it; on MPS/CPU we fall
    back to float16/float32 to avoid silent numerical issues or hard errors.
    """
    name = (dtype or "float32").lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    chosen = mapping.get(name, torch.float32)

    if chosen == torch.bfloat16 and device is not None and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            logger.warning("bfloat16 unsupported on this GPU; using float16.")
            return torch.float16
    if device is not None and device.type in {"mps", "cpu"} and chosen == torch.bfloat16:
        # bf16 matmul is poorly supported off-CUDA; float32 is the safe choice.
        logger.info("bfloat16 not recommended on %s; using float32.", device.type)
        return torch.float32
    return chosen


def gpu_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """Return current/peak GPU memory (GB). Zeros on non-CUDA devices.

    Used by the evaluation profiler and the Streamlit dashboard's "GPU usage"
    panel. ``reset_peak_memory`` should be called before a measured section.
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
    idx = device.index if (device is not None and device.index is not None) else 0
    return {
        "allocated_gb": torch.cuda.memory_allocated(idx) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(idx) / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated(idx) / 1024**3,
    }


def reset_peak_memory(device: torch.device | None = None) -> None:
    """Reset CUDA peak-memory counters before a measured region (no-op off CUDA)."""
    if torch.cuda.is_available():
        idx = device.index if (device is not None and device.index is not None) else 0
        torch.cuda.reset_peak_memory_stats(idx)


def get_device_info(preference: str = "auto", dtype: str = "bfloat16") -> DeviceInfo:
    """Collect a human-readable summary of the compute environment."""
    device = resolve_device(preference)
    resolved_dtype = resolve_dtype(dtype, device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        n_gpus = torch.cuda.device_count()
    elif device.type == "mps":
        name, total, n_gpus = "Apple Silicon (MPS)", 0.0, 1
    else:
        name, total, n_gpus = "CPU", 0.0, 0
    return DeviceInfo(
        device=device.type,
        device_name=name,
        dtype=str(resolved_dtype).replace("torch.", ""),
        n_gpus=n_gpus,
        total_memory_gb=round(total, 2),
    )


def log_device_summary(preference: str = "auto", dtype: str = "bfloat16") -> DeviceInfo:
    """Log and return the device summary — call once at the start of any script."""
    info = get_device_info(preference, dtype)
    logger.info(
        "Compute: device=%s (%s) | dtype=%s | gpus=%d | mem=%.1f GB",
        info.device,
        info.device_name,
        info.dtype,
        info.n_gpus,
        info.total_memory_gb,
    )
    # Make thread counts deterministic-ish on CPU for reproducible smoke tests.
    if info.device == "cpu" and os.getenv("OMP_NUM_THREADS") is None:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    return info


__all__ = [
    "DeviceInfo",
    "resolve_device",
    "resolve_dtype",
    "gpu_memory_stats",
    "reset_peak_memory",
    "get_device_info",
    "log_device_summary",
]
