"""Tests for the typed configuration layer (``configs/config.py``).

These are pure-Python + PyYAML tests (no torch), so they run on the lightest CI
image. They lock down the two invariants the rest of the pipeline depends on:

1. the shipped ``configs/default.yaml`` parses into the expected typed values
   (the backbone and dataset switches everything else dispatches on);
2. unknown keys are rejected *loudly* with ``ValueError`` — a silent typo in a
   hand-edited YAML is the classic way to waste a multi-hour training run, so we
   verify the strict-loader contract holds at both the top level and nested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# PyYAML is the only hard dependency of the config layer. Guard the import so a
# build without it skips cleanly instead of erroring at collection time; the
# ``configs`` import below transitively pulls yaml, hence the ordering.
pytest.importorskip("yaml")

from configs import ProjectConfig, load_config  # noqa: E402  (must follow importorskip)


def test_default_config_core_switches(default_config: Any) -> None:
    """default.yaml selects the Qwen2.5-VL backbone and the DocVQA dataset."""
    cfg = default_config
    # model_type drives the model registry; dataset_name drives the loader
    # dispatch. If either regresses, the whole pipeline silently builds the
    # wrong thing, so these are the highest-value assertions in the file.
    assert cfg.model.model_type == "qwen2_5_vl"
    assert cfg.data.dataset_name == "docvqa"


def test_default_config_is_strongly_typed(default_config: Any) -> None:
    """Nested YAML maps become real typed sub-configs, not raw dicts."""
    cfg = default_config
    assert isinstance(cfg, ProjectConfig)
    # A dict would not expose attribute access — this confirms _from_dict
    # actually recursed into the nested dataclasses.
    assert isinstance(cfg.seed, int)
    assert isinstance(cfg.model.trust_remote_code, bool)
    # Enumerated field stays within its documented domain.
    assert cfg.inference.confidence_method in {"seq_prob", "min_prob"}


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    """A misspelled *top-level* key must raise, not silently no-op."""
    bad = tmp_path / "bad_top.yaml"
    # `not_a_real_key` is not a field of ProjectConfig.
    bad.write_text("name: demo\nnot_a_real_key: 123\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ProjectConfig.from_yaml(bad)


def test_unknown_nested_key_rejected(tmp_path: Path) -> None:
    """A misspelled key inside a sub-config (``model:``) must also raise."""
    bad = tmp_path / "bad_nested.yaml"
    # `bogus_field` is not a field of ModelConfig; strict recursion must catch it.
    bad.write_text("model:\n  bogus_field: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ProjectConfig.from_yaml(bad)


def test_partial_valid_config_fills_defaults(tmp_path: Path) -> None:
    """A partial-but-valid file loads, overriding only what it names."""
    good = tmp_path / "good.yaml"
    good.write_text("seed: 7\nmodel:\n  model_type: donut\n", encoding="utf-8")
    cfg = ProjectConfig.from_yaml(good)
    assert cfg.seed == 7
    assert cfg.model.model_type == "donut"
    # Fields the file never mentioned keep their dataclass defaults, proving
    # the loader merges rather than requiring an exhaustive file.
    assert cfg.data.dataset_name == "docvqa"


def test_load_config_accepts_explicit_path(tmp_path: Path) -> None:
    """load_config() honours an explicit path argument over the env/default."""
    cfg_file = tmp_path / "explicit.yaml"
    cfg_file.write_text("name: explicit-run\nseed: 99\n", encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg.name == "explicit-run"
    assert cfg.seed == 99
