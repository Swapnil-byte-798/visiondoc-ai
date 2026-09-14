"""LoRA / parameter accounting helpers.

The core LoRA wrapping lives on :meth:`models.base.VisionDocModel.apply_lora`;
this module holds the small, reusable accounting utilities that the training
logs, the evaluation report, and the research comparison table all need — kept
separate so they can be unit-tested without loading a model.

**Why this module is fussy about quantized parameters.** Under QLoRA the
backbone is loaded with bitsandbytes in 4-bit. A ``bitsandbytes.nn.Params4bit``
tensor does not store one element per weight: it stores the weights *packed two
per ``uint8`` byte*. ``Tensor.numel()`` therefore reports the number of storage
elements, not the number of model parameters — roughly **half** the truth for a
4-bit backbone. Reporting that raw number would publish a 3.75B-parameter model
as "2034M params", which is simply wrong, and it would also inflate the
"trainable %" headline that the LoRA story rests on. :func:`count_parameters`
un-packs the count and states in ``counting_note`` exactly how it did so, so the
README/report can be honest about the provenance of the number.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from configs.config import LoRAConfig

# bitsandbytes parameter classes we know how to un-pack. Detection is by class
# *name* (not isinstance) on purpose: importing bitsandbytes here would drag a
# CUDA-only dependency into metrics-only callers, and the name is stable across
# bnb releases.
_QUANTIZED_PARAM_CLASSES = {"Params4bit", "Int8Params"}

#: Bits per stored weight for each known bitsandbytes parameter class.
_CLASS_BITS = {"Params4bit": 4, "Int8Params": 8}


def _is_quantized_param(p: Any) -> bool:
    """True when ``p`` is a bitsandbytes quantized parameter.

    Two independent signals, because either can be missing depending on the bnb
    version and on whether the tensor has actually been quantized yet:
    the class name, and the presence of a ``quant_state`` payload.
    """
    if type(p).__name__ in _QUANTIZED_PARAM_CLASSES:
        return True
    return getattr(p, "quant_state", None) is not None


def _quant_bits(p: Any) -> int:
    """Bits used per stored weight for a bitsandbytes parameter (4 or 8)."""
    bits = _CLASS_BITS.get(type(p).__name__)
    if bits is not None:
        return bits
    # Fall back to the quant_state's declared type ("nf4" / "fp4" / "int8").
    quant_type = getattr(getattr(p, "quant_state", None), "quant_type", None)
    if isinstance(quant_type, str):
        if "4" in quant_type:
            return 4
        if "8" in quant_type:
            return 8
    return 4  # bnb's only *packed* format is 4-bit; assume the packed case.


def true_numel(p: Any) -> tuple[int, str]:
    """Return ``(true_element_count, method)`` for one parameter tensor.

    ``method`` is one of ``"plain"``, ``"quant_state.shape"`` or ``"unpack"`` and
    exists so :func:`count_parameters` can describe its own provenance.

    Resolution order for a quantized tensor:

    1. ``p.quant_state.shape`` — bitsandbytes records the *original* (pre-pack)
       shape there. This is exact, so we prefer it whenever it is present.
    2. Unpacking arithmetic — ``numel * (storage_bits // quant_bits)``. For the
       usual ``uint8`` storage at 4 bits that is the documented ``x2``; deriving
       it from ``element_size()`` rather than hard-coding 2 keeps it correct if
       bnb is configured with a wider ``quant_storage`` dtype (e.g. ``int32``).
    """
    if not _is_quantized_param(p):
        return int(p.numel()), "plain"

    shape = getattr(getattr(p, "quant_state", None), "shape", None)
    if shape is not None:
        try:
            exact = int(math.prod(int(d) for d in shape))
        except (TypeError, ValueError):
            exact = 0
        if exact > 0:
            return exact, "quant_state.shape"

    packed = int(p.numel())
    bits = _quant_bits(p)
    try:
        storage_bits = int(p.element_size()) * 8
    except Exception:  # pragma: no cover - meta/fake tensors
        storage_bits = 8
    per_storage_element = max(1, storage_bits // max(bits, 1))
    return packed * per_storage_element, "unpack"


def _iter_parameters(module: Any) -> Iterable[Any]:
    """Yield the module's unique parameters (``nn.Module.parameters`` de-dupes)."""
    return module.parameters()


def count_parameters(model: Any) -> dict[str, Any]:
    """Return trainable/total parameter counts, the trainable %, and provenance.

    Works on a bare ``nn.Module``, a PEFT-wrapped model, or a
    :class:`~models.base.VisionDocModel` (via its ``.model`` attribute).

    ``total`` is the **true** number of model parameters: bitsandbytes 4-bit
    tensors are un-packed (see :func:`true_numel`) instead of being counted by
    their packed ``uint8`` storage, which would under-report a 4-bit backbone by
    ~2x. ``trainable`` is counted with plain ``numel()`` — LoRA adapters (and
    anything in ``modules_to_save``) are never quantized, so there is nothing to
    un-pack, and using the raw count keeps this figure identical to what the
    optimizer actually sees.

    Returned keys ``trainable``/``total``/``trainable_pct``/``trainable_millions``/
    ``total_millions`` are the historical contract and keep their meaning. The
    added keys are purely additive:

    ``quantized_tensors``
        How many parameter tensors were detected as bitsandbytes-quantized.
    ``packed_storage_elements``
        What ``numel()`` would have returned for those tensors (the wrong number).
    ``quantized_true_elements``
        What they actually hold once un-packed.
    ``counting_note``
        Plain-English description of exactly how the count was produced, meant to
        be pasted into the README/report rather than paraphrased.
    """
    module = getattr(model, "model", model)

    total = 0
    trainable = 0
    quantized_tensors = 0
    packed_storage_elements = 0
    quantized_true_elements = 0
    methods: dict[str, int] = {}

    for p in _iter_parameters(module):
        n_true, method = true_numel(p)
        total += n_true
        if method != "plain":
            quantized_tensors += 1
            packed_storage_elements += int(p.numel())
            quantized_true_elements += n_true
            methods[method] = methods.get(method, 0) + 1
        if p.requires_grad:
            # Deliberately the raw count: trainable params are the fp16/fp32 LoRA
            # matrices, never quantized, so numel() is already the truth.
            trainable += int(p.numel())

    if quantized_tensors == 0:
        note = (
            "All parameters counted directly with Tensor.numel(); no bitsandbytes "
            "quantized parameters were present (model loaded unquantized)."
        )
    else:
        under = quantized_true_elements - packed_storage_elements
        raw_total = total - under
        pct = 100.0 * under / max(total, 1)
        by_method = ", ".join(
            f"{count} via {name}" for name, count in sorted(methods.items())
        )
        note = (
            f"{quantized_tensors} bitsandbytes quantized parameter tensors were "
            f"counted by their ORIGINAL (pre-packing) element count ({by_method}); "
            f"4-bit weights are stored packed in uint8, so Tensor.numel() reports "
            f"storage elements rather than parameters. Raw numel() would have "
            f"reported {raw_total:,} total parameters instead of {total:,} — it "
            f"omits {under:,} parameters, {pct:.1f}% of the true total. Trainable "
            f"params are un-quantized LoRA matrices and are counted with plain numel()."
        )

    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
        "trainable_millions": trainable / 1e6,
        "total_millions": total / 1e6,
        "quantized_tensors": quantized_tensors,
        "packed_storage_elements": packed_storage_elements,
        "quantized_true_elements": quantized_true_elements,
        "counting_note": note,
    }


def format_parameter_summary(model: Any) -> str:
    """One-line human-readable parameter summary for logs/reports."""
    c = count_parameters(model)
    line = (
        f"trainable={c['trainable']:,} ({c['trainable_pct']:.3f}%) | "
        f"total={c['total']:,} | trainable={c['trainable_millions']:.2f}M"
    )
    if c["quantized_tensors"]:
        # Say it out loud in the log: a reader comparing this against
        # `print_trainable_parameters()` from PEFT will otherwise see two
        # different totals and have no way to know which one is right.
        line += (
            f" | total un-packed from {c['quantized_tensors']} bnb 4-bit tensors "
            f"(raw numel would say "
            f"{(c['total'] - c['quantized_true_elements'] + c['packed_storage_elements']) / 1e6:.0f}M)"
        )
    return line


def peft_config_from(
    cfg: LoRAConfig, target_modules: list[str] | str | None = None
) -> Any:
    """Build a ``peft.LoraConfig`` from our typed :class:`LoRAConfig`.

    Provided for callers (e.g. research scripts) that want to construct adapters
    directly without going through a :class:`VisionDocModel`.

    ``target_modules`` accepts a ``str`` as well as a list because PEFT treats a
    string as a **regex** matched with ``re.fullmatch`` against each module's
    fully-qualified name — which is how
    :attr:`models.qwen_vl.Qwen25VLModel.default_lora_target_modules` expresses
    "language decoder only, never the vision tower".
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=getattr(TaskType, cfg.task_type, TaskType.CAUSAL_LM),
        target_modules=target_modules or cfg.target_modules,
        modules_to_save=cfg.modules_to_save,
    )


__all__ = [
    "count_parameters",
    "format_parameter_summary",
    "peft_config_from",
    "true_numel",
]
