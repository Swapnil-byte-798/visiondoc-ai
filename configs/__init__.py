"""Configuration package for VisionDoc AI.

Re-exports the typed config schema so callers can simply do::

    from configs import load_config, ProjectConfig
"""

from configs.config import (
    DataConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    load_config,
)

__all__ = [
    "ModelConfig",
    "LoRAConfig",
    "DataConfig",
    "TrainingConfig",
    "InferenceConfig",
    "ProjectConfig",
    "load_config",
]
