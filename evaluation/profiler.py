"""Latency / throughput / memory profiling for inference benchmarks.

Why this module exists
----------------------
A LoRA fine-tune is only "production ready" if we can quote what it costs to
serve: how long a document QA call takes, how many documents/second a worker
sustains, and how much GPU memory it pins. Those numbers feed the evaluation
report, the base-vs-finetuned comparison, and capacity planning for the API.

Design choices worth calling out:

* **Wall-clock over CUDA events.** We deliberately time end-to-end wall clock
  (``time.perf_counter``) rather than raw kernel time. The user-visible latency
  of the FastAPI ``/ask`` endpoint includes pre/post-processing and the
  Python/host round-trip, so wall-clock is the honest number for SLA purposes.
* **Warmup is mandatory.** The first call after model load pays lazy CUDA
  context / autotuner / cache costs that never recur in steady state. Reporting
  it would slander the model, so :func:`measure_latency` runs warmup passes
  whose timings are discarded, and quotes the *median* of the rest (robust to
  the occasional GC or scheduler hiccup that would skew a mean).
* **Torch is imported lazily.** The dataclass and percentile math are pure
  stdlib, so importing this module (e.g. in a metrics-only unit test) does not
  drag in torch. Only device-memory helpers touch it, and those already run in
  an environment where torch is present.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from utils.device import gpu_memory_stats, reset_peak_memory
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ProfileResult:
    """Immutable-ish summary of a profiled inference run.

    Kept as a plain dataclass (rather than a dict) so downstream code gets
    attribute access + a stable schema, while :meth:`to_dict` provides the flat
    form the JSON report and the Streamlit dashboard serialize.
    """

    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    throughput_samples_s: float
    peak_memory_gb: float
    n_samples: int
    device: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Flat, JSON-serializable view used by reports and the API/UI."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Percentile helper (stdlib only — numpy is not a guaranteed runtime dep here)
# ---------------------------------------------------------------------------
def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over ``values`` (``pct`` in 0..100).

    We roll our own instead of ``numpy.percentile`` so the profiler has zero
    heavy dependencies and behaves identically on the CPU CI runner and the GPU
    box. Matches numpy's default ('linear') interpolation for parity with any
    ad-hoc analysis a reader might do.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    # Interpolate between the two neighboring order statistics.
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _device_str(device: object) -> str:
    """Render a device (str, ``torch.device``, or None) as a short label."""
    if device is None:
        return "cpu"
    # torch.device has a ``.type``; strings pass through unchanged.
    return str(getattr(device, "type", device))


def _to_torch_device(device: object):
    """Best-effort coercion to a ``torch.device`` for the memory helpers.

    ``reset_peak_memory`` / ``gpu_memory_stats`` index into CUDA using
    ``device.index``, so a bare string like ``"cuda"`` must become a real
    ``torch.device``. Torch is imported lazily here (not at module top) and any
    failure degrades to ``None`` — the memory helpers already treat ``None`` and
    non-CUDA devices as a no-op returning zeros, so profiling never crashes on a
    CPU-only or torch-less environment.
    """
    if device is None:
        return None
    try:
        import torch  # local import: keep module import torch-free.

        if isinstance(device, torch.device):
            return device
        return torch.device(str(device))
    except Exception:  # pragma: no cover - torch missing / bad device string.
        logger.debug("Could not coerce %r to torch.device; memory stats disabled.", device)
        return None


# ---------------------------------------------------------------------------
# Context-manager profiler
# ---------------------------------------------------------------------------
class InferenceProfiler:
    """Accumulate per-sample latencies within a ``with`` block and summarize.

    Usage::

        with InferenceProfiler(model.device) as prof:
            for sample in samples:
                t0 = time.perf_counter()
                model.generate(...)
                prof.record((time.perf_counter() - t0) * 1000.0)
        print(prof.result().to_dict())

    Entering resets the CUDA peak-memory counter so ``peak_memory_gb`` reflects
    only work done inside the block; exiting snapshots the peak. Both are
    no-ops on CPU/MPS, so the same code path profiles cleanly on a laptop.
    """

    def __init__(self, device: object) -> None:
        # Keep a display label (for the report) and a torch.device (for the
        # memory counters) separately so we never require torch just to hold a
        # profiler instance.
        self.device_label: str = _device_str(device)
        self._torch_device = _to_torch_device(device)
        self._latencies: list[float] = []
        self._peak_memory_gb: float = 0.0

    def __enter__(self) -> "InferenceProfiler":
        # Zero the peak counter so memory attributed to prior work (model load,
        # earlier eval splits) does not contaminate this measurement window.
        reset_peak_memory(self._torch_device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Capture peak memory even if the block raised — partial numbers are
        # more useful than none when debugging an OOM. Return False so any
        # exception continues to propagate (we are a profiler, not a swallower).
        stats = gpu_memory_stats(self._torch_device)
        self._peak_memory_gb = float(stats.get("max_allocated_gb", 0.0))
        return False

    def record(self, latency_ms: float) -> None:
        """Record one sample's latency in milliseconds.

        Negative/NaN values (which can arise from a mis-ordered timer) are
        dropped with a debug note rather than poisoning the aggregate stats.
        """
        if latency_ms is None or math.isnan(latency_ms) or latency_ms < 0:
            logger.debug("Ignoring invalid latency sample: %r", latency_ms)
            return
        self._latencies.append(float(latency_ms))

    def result(self) -> ProfileResult:
        """Summarize the recorded latencies into a :class:`ProfileResult`.

        Throughput is derived from *total* measured time (sum of per-sample
        latencies), i.e. serial samples/second, which is the meaningful figure
        for a single-worker server. If nothing was recorded we return a
        well-formed zero result so callers/reports never see ``None``.
        """
        n = len(self._latencies)
        if n == 0:
            logger.debug("InferenceProfiler.result() called with no samples recorded.")
            return ProfileResult(
                latency_ms_mean=0.0,
                latency_ms_p50=0.0,
                latency_ms_p90=0.0,
                throughput_samples_s=0.0,
                peak_memory_gb=self._peak_memory_gb,
                n_samples=0,
                device=self.device_label,
            )

        mean_ms = statistics.fmean(self._latencies)
        total_seconds = sum(self._latencies) / 1000.0
        throughput = (n / total_seconds) if total_seconds > 0 else 0.0
        return ProfileResult(
            latency_ms_mean=mean_ms,
            latency_ms_p50=_percentile(self._latencies, 50.0),
            latency_ms_p90=_percentile(self._latencies, 90.0),
            throughput_samples_s=throughput,
            peak_memory_gb=self._peak_memory_gb,
            n_samples=n,
            device=self.device_label,
        )


# ---------------------------------------------------------------------------
# One-shot latency probe
# ---------------------------------------------------------------------------
def measure_latency(
    fn: Callable[..., object],
    *args: object,
    warmup: int = 1,
    repeats: int = 3,
    **kwargs: object,
) -> float:
    """Return the **median** wall-clock latency (ms) of calling ``fn``.

    ``warmup`` calls are executed and discarded to absorb one-time CUDA context
    / kernel-autotune / cache costs; ``repeats`` timed calls follow and their
    median is returned (median, not mean, to resist a single GC/scheduler
    outlier). ``*args``/``**kwargs`` are forwarded to ``fn`` unchanged, so this
    works for any callable — a ``model.generate`` closure, an API handler, etc.

    We do not call ``torch.cuda.synchronize`` here to avoid a hard torch import;
    callers benchmarking async CUDA kernels should have ``fn`` synchronize
    internally (``model.generate`` already blocks until decoding completes, so
    in practice the wall-clock is accurate for this project's usage).
    """
    warmup = max(0, warmup)
    repeats = max(1, repeats)

    for _ in range(warmup):
        fn(*args, **kwargs)

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        timings.append((time.perf_counter() - start) * 1000.0)

    return statistics.median(timings)


__all__ = [
    "ProfileResult",
    "InferenceProfiler",
    "measure_latency",
]
