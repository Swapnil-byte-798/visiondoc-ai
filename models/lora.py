"""LoRA / parameter accounting helpers.

The core LoRA wrapping lives on :meth:`models.base.VisionDocModel.apply_lora`;
this module holds the small, reusable accounting utilities that the training
logs, the evaluation report, and the research comparison table all need — kept
separate so they can be unit-tested without loading a model.
"""

from __future__ import annotations

from typing import Any

from configs.config import LoRAConfig


def count_parameters(model: Any) -> dict[str, float]:
    """Return trainable/total parameter counts and the trainable percentage.

    Works on a bare ``nn.Module``, a PEFT-wrapped model, or a
    :class:`~models.base.VisionDocModel` (via its ``.model`` attribute).
    """
    module = getattr(model, "model", model)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
        "trainable_millions": trainable / 1e6,
        "total_millions": total / 1e6,
    }


def format_parameter_summary(model: Any) -> str:
    """One-line human-readable parameter summary for logs/reports."""
    c = count_parameters(model)
    return (
        f"trainable={c['trainable']:,} ({c['trainable_pct']:.3f}%) | "
        f"total={c['total']:,} | trainable={c['trainable_millions']:.2f}M"
    )


def peft_config_from(cfg: LoRAConfig, target_modules: list[str] | None = None) -> Any:
    """Build a ``peft.LoraConfig`` from our typed :class:`LoRAConfig`.

    Provided for callers (e.g. research scripts) that want to construct adapters
    directly without going through a :class:`VisionDocModel`.
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


__all__ = ["count_parameters", "format_parameter_summary", "peft_config_from"]
