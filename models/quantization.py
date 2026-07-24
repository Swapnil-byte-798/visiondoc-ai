"""Quantization helpers: QLoRA (train-time 4/8-bit) and dynamic INT8 (inference).

Two *distinct* quantization strategies live here because they solve two
different problems, and conflating them is a common source of confusion:

QLoRA — ``get_bnb_config`` (train time)
    ``bitsandbytes`` loads the *frozen* backbone weights in 4-bit ``nf4`` while
    LoRA adapters train in higher precision on top. This is how a 3B–7B VLM
    fits on a single consumer GPU: the huge base weights are stored 4-bit, and
    only the tiny low-rank matrices hold gradients/optimizer state. The 4-bit
    weights are *dequantized on the fly* for each matmul, so there is a small
    compute overhead in exchange for a massive VRAM saving. ``nf4`` (4-bit
    NormalFloat) plus *double quantization* (quantizing the quantization
    constants themselves) is the recipe from the QLoRA paper and is what we
    expose by default. **bitsandbytes kernels are CUDA-only** — on MPS/CPU this
    config is unusable, so the loader (``models/qwen_vl.py``) already ignores it
    off-CUDA. We still return a valid config object here so it can be built and
    inspected anywhere (e.g. unit tests on a CPU runner).

Post-training dynamic quantization — ``apply_dynamic_quantization`` (inference)
    A completely different mechanism aimed at **CPU** inference of an
    *already-trained* model. ``torch.ao.quantization.quantize_dynamic`` replaces
    ``nn.Linear`` weights with INT8 and quantizes activations *dynamically* at
    runtime (per-batch, computing the scale on the fly). No calibration dataset
    and no retraining are needed, which is why it is "dynamic". It shrinks the
    model ~4x and speeds up Linear-heavy models on CPU, at a small accuracy
    cost. It is orthogonal to QLoRA: QLoRA is for fitting training on a GPU;
    dynamic quant is for shipping a merged model cheaply on a CPU box.

These are intentionally the only two knobs exposed — they cover the realistic
laptop-GPU-train / CPU-serve lifecycle of this project without pulling in a full
calibration/static-quantization pipeline that the datasets here don't warrant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # import only for type hints; keeps the module importable
    import torch  # without torch/transformers installed (heavy, CUDA-oriented).
    import torch.nn as nn
    from transformers import BitsAndBytesConfig

logger = get_logger(__name__)


def _resolve_compute_dtype(compute_dtype: "str | torch.dtype") -> "torch.dtype":
    """Map a dtype *name* to a ``torch.dtype`` for the bnb compute path.

    We resolve here rather than reusing ``utils.device.resolve_dtype`` because
    that helper deliberately *downgrades* bf16 on non-CUDA devices — behaviour
    we do NOT want here: the bnb compute dtype describes the CUDA matmul dtype
    used after dequantization and must be honoured verbatim regardless of the
    host the config happens to be built on. A pre-resolved ``torch.dtype`` is
    passed through untouched so callers can override precisely.
    """
    import torch

    if isinstance(compute_dtype, torch.dtype):
        return compute_dtype

    name = str(compute_dtype).lower()
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
    if name not in mapping:
        raise ValueError(
            f"Unknown compute_dtype {compute_dtype!r}. "
            f"Expected one of {sorted(mapping)} or a torch.dtype."
        )
    return mapping[name]


def get_bnb_config(bits: int = 4, compute_dtype: "str | torch.dtype" = "bfloat16") -> "BitsAndBytesConfig":
    """Build a ``transformers.BitsAndBytesConfig`` for 4-bit QLoRA or 8-bit loading.

    Parameters
    ----------
    bits:
        ``4`` -> ``nf4`` double-quantized weights (the QLoRA default; best
        VRAM/quality trade-off). ``8`` -> LLM.int8() weight quantization (larger
        but numerically gentler, occasionally handy when 4-bit hurts a task).
    compute_dtype:
        The dtype matmuls run in *after* on-the-fly dequantization (4-bit only).
        ``bfloat16`` on Ampere+ is the standard choice; ``float16`` for older
        cards. Ignored for 8-bit.

    The import of ``BitsAndBytesConfig`` is lazy so this module imports fine in a
    torch-less environment. We only *warn* (not raise) when CUDA is absent: the
    config is still a valid, inspectable object, and the model loader is the
    layer that decides whether to actually apply it.
    """
    import torch
    from transformers import BitsAndBytesConfig

    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits!r}.")

    # bitsandbytes only ships CUDA kernels; surface this early so a user on a
    # laptop understands why a config built here won't actually quantize.
    if not torch.cuda.is_available():
        logger.warning(
            "get_bnb_config(%d-bit) built without CUDA available; bitsandbytes "
            "kernels are CUDA-only, so this config is inspectable but not usable "
            "for loading on this host.",
            bits,
        )

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            # NormalFloat-4 assumes ~normally-distributed weights and packs them
            # more informatively than plain int4 — the QLoRA paper's win.
            bnb_4bit_quant_type="nf4",
            # Double quantization also quantizes the per-block scale constants,
            # saving a further ~0.4 bits/param for negligible quality loss.
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_compute_dtype(compute_dtype),
        )

    # 8-bit: LLM.int8() with default outlier handling. No compute-dtype knob —
    # int8 matmul + fp16 accumulation is handled internally by bitsandbytes.
    return BitsAndBytesConfig(load_in_8bit=True)


def apply_dynamic_quantization(
    torch_module: "nn.Module",
    dtype: "torch.dtype | None" = None,
    layer_types: "Iterable[type[nn.Module]] | None" = None,
    inplace: bool = False,
) -> "nn.Module":
    """Post-training dynamic INT8 quantization of ``nn.Linear`` layers for CPU.

    Wraps ``torch.ao.quantization.quantize_dynamic``. Weights of the targeted
    layer types are converted to INT8 up front; activations are quantized
    *dynamically* at inference time (their scale computed per forward pass),
    which needs neither a calibration set nor retraining. This is the right tool
    for serving a *merged* (adapter-folded) model on a CPU-only box: it roughly
    quarters the Linear weight footprint and speeds up matmul-bound inference.

    Parameters
    ----------
    torch_module:
        The module to quantize. It should live on **CPU** — INT8 dynamic-quant
        kernels are CPU-only; a CUDA module triggers a warning below.
    dtype:
        Quantized weight dtype. Defaults to ``torch.qint8`` (the only widely
        supported dynamic dtype for Linear). ``torch.float16`` is accepted by
        the API for a weight-only fp16 path on some builds.
    layer_types:
        Module classes to quantize. Defaults to ``{nn.Linear}`` — the layers
        that dominate a transformer's FLOPs and memory; we deliberately leave
        conv/embedding/norm layers untouched as dynamic quant doesn't help them.
    inplace:
        ``False`` (default) returns a quantized *copy*, leaving the caller's
        model untouched — safer when the original is still needed (e.g. for a
        quality comparison). ``True`` mutates in place to avoid doubling memory.

    Returns
    -------
    The dynamically quantized module (a new object unless ``inplace=True``).
    """
    import torch
    import torch.nn as nn

    # Prefer the modern ``torch.ao`` namespace; fall back to the legacy alias so
    # this keeps working on older torch builds where ``ao`` isn't present.
    try:
        quantize_dynamic = torch.ao.quantization.quantize_dynamic
    except AttributeError:  # pragma: no cover - depends on installed torch
        quantize_dynamic = torch.quantization.quantize_dynamic

    q_dtype = dtype if dtype is not None else torch.qint8
    targets = set(layer_types) if layer_types is not None else {nn.Linear}

    # Dynamic-quant INT8 ops only run on CPU; warn (don't move silently) so the
    # caller stays in control of device placement of a potentially large model.
    try:
        param_device = next(torch_module.parameters()).device
        if param_device.type != "cpu":
            logger.warning(
                "apply_dynamic_quantization: module is on %s but INT8 dynamic "
                "quantization runs on CPU — move it to CPU before serving.",
                param_device.type,
            )
    except StopIteration:  # module has no parameters; nothing to warn about
        pass

    quantized = quantize_dynamic(torch_module, targets, dtype=q_dtype, inplace=inplace)
    logger.info(
        "Applied dynamic %s quantization to layer types %s (inplace=%s).",
        str(q_dtype).replace("torch.", ""),
        sorted(t.__name__ for t in targets),
        inplace,
    )
    return quantized


__all__ = ["get_bnb_config", "apply_dynamic_quantization"]
