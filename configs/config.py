"""Typed configuration schema for VisionDoc AI.

Design rationale
----------------
Every script in the project (download, preprocess, train, evaluate, serve) is
driven by a *single* immutable config object loaded from a YAML file. We use
frozen ``dataclasses`` rather than raw dicts so that:

* misspelled keys fail loudly at load time (not three hours into training),
* IDEs/type-checkers can autocomplete and verify field access,
* defaults live in one place and are self-documenting.

A handful of high-churn fields (device, adapter path, config path) can be
overridden by environment variables so the same YAML works unchanged across a
laptop, a CI runner, and a multi-GPU box. See :func:`ProjectConfig.from_yaml`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Which backbone to load and how.

    ``model_type`` selects the adapter implementation in ``models/`` via a
    registry, so swapping Qwen2.5-VL for Donut/Florence-2 is a one-line YAML
    change with no code edits.
    """

    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_type: str = "qwen2_5_vl"  # qwen2_5_vl | donut | florence2
    # Processor usually shares the model_id; kept separate for models that
    # split the tokenizer/processor onto a different hub repo.
    processor_id: str | None = None
    torch_dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    # "flash_attention_2" on supported CUDA GPUs, else "sdpa" (portable) or
    # "eager". We default to sdpa because it works on CUDA, MPS, and CPU.
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    # 4-bit (QLoRA) drastically cuts VRAM; requires bitsandbytes (CUDA only).
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    # Qwen2.5-VL dynamic-resolution bounds (in pixels). Capping max_pixels is
    # the single most effective lever for controlling VRAM on document images.
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28


@dataclass(frozen=True)
class LoRAConfig:
    """PEFT/LoRA hyper-parameters.

    ``target_modules`` defaults to the attention + MLP projections of the LLM
    decoder. Leaving it ``None`` in YAML falls back to the backbone adapter's
    ``default_lora_target_modules`` (PEFT does not scan linear layers unless you
    pass the special ``"all-linear"``). We deliberately do NOT target the vision
    encoder by default — fine-tuning only the language side is cheaper and
    usually sufficient for document QA.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"  # none | all | lora_only
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] | None = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    # Extra modules to train in full precision alongside the adapters
    # (e.g. a task head). Empty by default for pure adapter training.
    modules_to_save: list[str] | None = None


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection and preprocessing knobs.

    ``dataset_name`` chooses the loader in ``preprocessing/datasets.py``;
    ``dataset_id`` is the concrete Hugging Face Hub repo it pulls from.
    """

    dataset_name: str = "docvqa"  # docvqa | cord | funsd | sroie
    dataset_id: str = "lmms-lab/DocVQA"
    dataset_subset: str | None = "DocVQA"  # HF config name, if any
    # Longest side the image is resized to before the model's own processor
    # runs. Keep generous — document text needs resolution to stay legible.
    image_size: int = 1024
    max_seq_length: int = 1024
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    # Fractions used only when a dataset lacks native splits (we carve our own).
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    cache_dir: str = "data/cache"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    num_workers: int = 4
    augment: bool = True
    # Cap samples for fast smoke runs; None = use the full split.
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """HF ``Trainer`` arguments + early-stopping controls.

    Field names mirror ``transformers.TrainingArguments`` where possible so the
    trainer can forward them with minimal translation.
    """

    output_dir: str = "outputs/qwen2_5vl-docvqa-lora"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # effective batch = 8 on a single GPU
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    # Precision: prefer bf16 on Ampere+; the trainer downgrades gracefully.
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True  # trade compute for a big VRAM win
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_anls"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    resume_from_checkpoint: str | None = None
    report_to: str = "wandb"  # wandb | tensorboard | none
    run_name: str = "qwen2_5vl-docvqa-lora"
    dataloader_num_workers: int = 4
    seed: int = 42


@dataclass(frozen=True)
class InferenceConfig:
    """Generation + confidence settings shared by the API, app, and evaluator."""

    adapter_path: str | None = None  # None => base model only
    max_new_tokens: int = 128
    do_sample: bool = False  # greedy is deterministic + best for extraction
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 1
    batch_size: int = 4
    return_attention: bool = False  # attention rollout is expensive; opt-in
    # How to derive a confidence score from generation:
    #   "seq_prob" -> exp(mean token log-prob);  "min_prob" -> min token prob.
    confidence_method: str = "seq_prob"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level configuration aggregating every sub-config."""

    name: str = "visiondoc-ai"
    seed: int = 42
    device: str = "auto"  # auto | cuda | mps | cpu
    output_root: str = "outputs"
    report_dir: str = "reports"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # -- Serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict (YAML/JSON friendly)."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist the *resolved* config next to a run's artifacts.

        Saving the exact config used for a run is essential for reproducibility
        and for the evaluation/serving code to know how the model was trained.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    # -- Construction -------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, apply_env: bool = True) -> ProjectConfig:
        """Load and validate a config from a YAML file.

        Unknown keys raise ``ValueError`` so typos never silently no-op. After
        construction, selected environment variables (``VISIONDOC_DEVICE``,
        ``VISIONDOC_MODEL_ID``, ``VISIONDOC_ADAPTER_PATH``) override the file so
        the same YAML is portable across machines.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = _from_dict(cls, raw)
        if apply_env:
            cfg = cfg._with_env_overrides()
        return cfg

    def _with_env_overrides(self) -> ProjectConfig:
        """Return a copy with a few env-driven overrides applied."""
        device = os.getenv("VISIONDOC_DEVICE")
        model_id = os.getenv("VISIONDOC_MODEL_ID")
        adapter = os.getenv("VISIONDOC_ADAPTER_PATH")

        model = self.model
        if model_id:
            model = _replace(model, model_id=model_id)
        inference = self.inference
        if adapter:
            inference = _replace(inference, adapter_path=adapter)
        return _replace(
            self,
            device=device or self.device,
            model=model,
            inference=inference,
        )


# ---------------------------------------------------------------------------
# Dataclass <-> dict helpers (strict; reject unknown keys)
# ---------------------------------------------------------------------------


def _replace(obj: Any, **changes: Any) -> Any:
    """``dataclasses.replace`` re-export kept local to avoid an extra import."""
    from dataclasses import replace

    return replace(obj, **changes)


def _resolve_field_types(cls: type) -> dict[str, Any]:
    """Return field-name -> *actual* type for a dataclass.

    ``from __future__ import annotations`` (used in this module) stores every
    annotation as a STRING, so ``dataclasses.Field.type`` is e.g. ``"ModelConfig"``
    rather than the class. ``get_type_hints`` evaluates those strings back into
    real objects in the module's namespace, which is what lets us detect nested
    dataclass fields below. We fall back to the raw ``.type`` if resolution fails.
    """
    try:
        return get_type_hints(cls)
    except Exception:  # pragma: no cover - only on exotic forward refs
        return {f.name: f.type for f in fields(cls)}


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a dict, strictly.

    Raises ``ValueError`` on any key that is not a field of the target
    dataclass — this is what turns a silent typo into an immediate, actionable
    error at config-load time.
    """
    if not is_dataclass(cls):
        return data
    field_map = {f.name: f for f in fields(cls)}
    hints = _resolve_field_types(cls)
    unknown = set(data) - set(field_map)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(field_map)}"
        )
    kwargs: dict[str, Any] = {}
    for name in field_map:
        if name not in data:
            continue
        value = data[name]
        # Recurse into nested dataclass fields (e.g. model:, lora:, ...). We use
        # the *resolved* type (not Field.type, which is a string under
        # `from __future__ import annotations`) so nested sections become real
        # dataclasses instead of leaking through as raw dicts.
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Convenience loader used by CLIs.

    Resolution order: explicit ``path`` arg > ``VISIONDOC_CONFIG`` env var >
    ``configs/default.yaml``.
    """
    resolved = path or os.getenv("VISIONDOC_CONFIG") or "configs/default.yaml"
    return ProjectConfig.from_yaml(resolved)


__all__ = [
    "ModelConfig",
    "LoRAConfig",
    "DataConfig",
    "TrainingConfig",
    "InferenceConfig",
    "ProjectConfig",
    "load_config",
]
