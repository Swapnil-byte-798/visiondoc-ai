"""Model package: backbone-agnostic interface, registry, and implementations.

Public API::

    from models import build_model, VisionDocModel, GenerationOutput
    model = build_model(config)          # selects backbone from config.model.model_type
    model.apply_lora()                   # wrap with LoRA, freeze the rest
    out = model.generate(image, "What is the total?")
"""

from models.base import GenerationOutput, VisionDocModel, build_model, register_model
from models.lora import count_parameters, format_parameter_summary, peft_config_from

__all__ = [
    "VisionDocModel",
    "GenerationOutput",
    "build_model",
    "register_model",
    "count_parameters",
    "format_parameter_summary",
    "peft_config_from",
]
