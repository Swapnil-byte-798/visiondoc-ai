"""Backbone-agnostic model interface + registry.

Everything downstream (training, inference, evaluation, API, app, research)
talks to a :class:`VisionDocModel`, never to Qwen or Donut directly. Swapping
backbones is therefore a config change (``model.model_type``) that selects a
different registered subclass — no call-site edits.

A :class:`VisionDocModel` owns three responsibilities:

1. **Loading** the pretrained weights + processor onto the right device/dtype.
2. **Adaptation** — wrapping the backbone with LoRA and freezing the rest.
3. **Tensorization** — turning a :class:`DocSample` into a supervised training
   batch (``collate_train``) and turning ``(image, question)`` into inference
   inputs, plus decoding generations back to text with a confidence score.

Backbone-specific logic (chat templates, label masking, seq2seq vs causal)
lives in the subclass; the generation loop, confidence math, LoRA plumbing and
adapter I/O are shared here.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image

from configs.config import ProjectConfig
from preprocessing.schema import DocSample
from utils.device import resolve_device, resolve_dtype
from utils.logging_utils import get_logger

logger = get_logger("models.base")

# Registry mapping ``model_type`` string -> concrete subclass.
_MODEL_REGISTRY: dict[str, type["VisionDocModel"]] = {}


def register_model(name: str) -> Callable[[type["VisionDocModel"]], type["VisionDocModel"]]:
    """Class decorator registering a backbone under a ``model_type`` key."""

    def _wrap(cls: type["VisionDocModel"]) -> type["VisionDocModel"]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model type {name!r} already registered.")
        _MODEL_REGISTRY[name] = cls
        cls.model_type = name
        return cls

    return _wrap


def build_model(config: ProjectConfig, load: bool = True) -> "VisionDocModel":
    """Instantiate the model selected by ``config.model.model_type``.

    Importing the implementations here (lazily) both registers them and avoids
    a circular import at module load time.
    """
    import models.qwen_vl  # noqa: F401  (registers qwen2_5_vl)

    try:
        import models.donut  # noqa: F401  (registers donut)
    except Exception:  # pragma: no cover - donut deps optional
        logger.debug("Donut backbone not available; skipping its registration.")

    model_type = config.model.model_type
    if model_type not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type={model_type!r}. Registered: {sorted(_MODEL_REGISTRY)}"
        )
    model = _MODEL_REGISTRY[model_type](config)
    if load:
        model.load()
    return model


@dataclass
class GenerationOutput:
    """Result of a single generation, enriched with a confidence score.

    ``confidence`` in [0, 1] is derived from the per-token probabilities of the
    generated sequence (see :meth:`VisionDocModel._sequence_confidence`), giving
    the API/app a calibrated-ish signal without a separate scoring head.
    """

    text: str
    confidence: float
    prompt: str = ""
    token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    attentions: Any = None  # raw attentions if requested (best-effort)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "num_tokens": len(self.token_ids),
        }


class VisionDocModel(ABC):
    """Abstract backbone wrapper. Subclasses implement the tensorization hooks."""

    #: set by :func:`register_model`
    model_type: str = "base"

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.model.torch_dtype, self.device)
        self.model: Any = None  # populated by load()
        self.processor: Any = None
        self._lora_applied = False

    # -- Abstract hooks (backbone-specific) ---------------------------------

    @abstractmethod
    def load(self) -> "VisionDocModel":
        """Load weights + processor onto ``self.device`` and set eval mode."""

    @abstractmethod
    def collate_train(self, batch: Sequence[DocSample | dict]) -> dict[str, torch.Tensor]:
        """Turn a list of samples into a padded supervised batch.

        Must return a dict with at least ``input_ids``, ``attention_mask`` and
        ``labels`` (prompt tokens masked to -100), plus any vision tensors the
        backbone's ``forward`` expects (e.g. ``pixel_values``). This callable is
        passed straight to the HF ``Trainer`` as its ``data_collator``.
        """

    @abstractmethod
    def prepare_inference_inputs(
        self, images: Sequence[Image.Image], questions: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        """Build model-ready inputs for generation (moved to ``self.device``)."""

    @abstractmethod
    def decode(self, generated_ids: torch.Tensor, prompt_width: int) -> list[str]:
        """Decode generated token ids to strings.

        ``prompt_width`` is the padded prompt length. Causal-LM backbones (Qwen)
        must slice it off — ``generate`` returns ``prompt + completion``; seq2seq
        backbones (Donut) ignore it — ``generate`` returns only the completion.
        """

    @property
    @abstractmethod
    def default_lora_target_modules(self) -> list[str] | None:
        """Fallback LoRA targets if the config leaves them unset."""

    # -- Shared: LoRA / PEFT -------------------------------------------------

    def apply_lora(self) -> "VisionDocModel":
        """Wrap the backbone with LoRA adapters and freeze the base weights.

        Only the injected low-rank matrices (and any ``modules_to_save``) remain
        trainable, cutting the optimizer state + gradient memory by orders of
        magnitude versus full fine-tuning while keeping the frozen backbone's
        pretrained knowledge intact.
        """
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if self.model is None:
            raise RuntimeError("Call load() before apply_lora().")
        if self._lora_applied:
            logger.warning("LoRA already applied; skipping.")
            return self

        lcfg = self.config.lora
        targets = lcfg.target_modules or self.default_lora_target_modules
        use_ckpt = self.config.training.gradient_checkpointing

        # QLoRA stability: when the base was actually loaded in 4/8-bit,
        # prepare_model_for_kbit_training upcasts LayerNorm/RMSNorm + the output
        # head to fp32 (the numerical guard that keeps fp16 QLoRA loss from going
        # NaN on a T4) and wires up gradient checkpointing + input grads. We gate
        # on the *real* k-bit flags (not just the config), since quantization is
        # skipped on non-CUDA devices even when load_in_4bit is set.
        is_kbit = getattr(self.model, "is_loaded_in_4bit", False) or getattr(
            self.model, "is_loaded_in_8bit", False
        )
        if is_kbit:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=use_ckpt
            )
        elif use_ckpt and hasattr(self.model, "enable_input_require_grads"):
            # Non-quantized path: gradient checkpointing still needs inputs to
            # require grad; enabling it on the input embeddings is the standard recipe.
            self.model.enable_input_require_grads()

        peft_config = LoraConfig(
            r=lcfg.r,
            lora_alpha=lcfg.lora_alpha,
            lora_dropout=lcfg.lora_dropout,
            bias=lcfg.bias,
            task_type=getattr(TaskType, lcfg.task_type, TaskType.CAUSAL_LM),
            target_modules=targets,
            modules_to_save=lcfg.modules_to_save,
        )
        self.model = get_peft_model(self.model, peft_config)
        self._lora_applied = True
        trainable, total = self.trainable_parameters()
        logger.info(
            "LoRA applied: %s trainable / %s total params (%.3f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )
        return self

    def trainable_parameters(self) -> tuple[int, int]:
        """Return ``(trainable, total)`` parameter counts."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def save_adapter(self, path: str | Path) -> None:
        """Persist the LoRA adapter (+ processor) to ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        if self.processor is not None:
            self.processor.save_pretrained(str(path))
        logger.info("Saved adapter to %s", path)

    def load_adapter(self, path: str | Path) -> "VisionDocModel":
        """Attach a previously trained LoRA adapter onto the loaded base model.

        ``path`` may be a local directory (produced by :meth:`save_adapter`) or a
        Hugging Face Hub repo id (``namespace/name``). We validate a local path up
        front: PEFT otherwise misreads a missing local path as a Hub repo id and
        raises a confusing ``HFValidationError`` — the usual cause is running
        evaluation/inference before training has finished writing the adapter.
        """
        import re

        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("Call load() before load_adapter().")

        p = Path(path)
        if not p.exists():
            # A bare "namespace/name" is a plausible Hub id — let PEFT try it.
            is_hub_id = bool(re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", str(path)))
            if not is_hub_id:
                raise FileNotFoundError(
                    f"LoRA adapter not found at '{path}'. Training writes the adapter to "
                    f"<output_dir>/adapter only after it finishes — run training to "
                    f"completion first, or pass a valid local path / Hub repo id. "
                    f"(To evaluate the untuned base model, omit the adapter entirely.)"
                )
        elif not (p / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"'{path}' exists but contains no adapter_config.json, so it is not a "
                f"saved LoRA adapter. Point at the directory that holds adapter_config.json "
                f"(e.g. <output_dir>/adapter or a checkpoint-*/ folder)."
            )

        self.model = PeftModel.from_pretrained(self.model, str(path))
        self.model.to(self.device)
        self.model.eval()
        self._lora_applied = True
        logger.info("Loaded adapter from %s", path)
        return self

    def merge_and_unload(self) -> "VisionDocModel":
        """Fold LoRA weights into the base model for faster, adapter-free inference."""
        if self._lora_applied and hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            self._lora_applied = False
            logger.info("Merged LoRA adapter into base weights.")
        return self

    # -- Shared: generation + confidence ------------------------------------

    @torch.no_grad()
    def generate(
        self,
        images: Image.Image | Sequence[Image.Image],
        questions: str | Sequence[str],
        max_new_tokens: int | None = None,
        return_attention: bool | None = None,
        **gen_kwargs: Any,
    ) -> GenerationOutput | list[GenerationOutput]:
        """Generate answer(s) with per-sequence confidence.

        Accepts a single ``(image, question)`` or equal-length batches. Returns
        a single :class:`GenerationOutput` for scalar input, else a list.
        """
        single = isinstance(questions, str)
        imgs = [images] if isinstance(images, Image.Image) else list(images)
        qs = [questions] if single else list(questions)
        if len(imgs) != len(qs):
            raise ValueError(f"images ({len(imgs)}) and questions ({len(qs)}) length mismatch")

        icfg = self.config.inference
        want_attn = icfg.return_attention if return_attention is None else return_attention

        inputs = self.prepare_inference_inputs(imgs, qs)
        prompt_width = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0

        gen_config: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens or icfg.max_new_tokens,
            do_sample=icfg.do_sample,
            num_beams=icfg.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if icfg.do_sample:
            gen_config.update(temperature=icfg.temperature, top_p=icfg.top_p)
        if want_attn:
            gen_config["output_attentions"] = True
        gen_config.update(gen_kwargs)

        out = self.model.generate(**inputs, **gen_config)
        sequences = out.sequences
        texts = self.decode(sequences, prompt_width)

        # Per-token log-probs of the *chosen* tokens, from generate() scores.
        # beam_indices (present only under beam search) maps each returned token
        # to the flattened beam row that produced it, so we gather correctly.
        eos_id = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        logprobs_per_example = _token_logprobs_from_scores(
            out.scores, sequences, eos_id, getattr(out, "beam_indices", None)
        )
        gen_start = sequences.shape[1] - len(out.scores or ())

        results: list[GenerationOutput] = []
        for i, text in enumerate(texts):
            lps = logprobs_per_example[i]
            results.append(
                GenerationOutput(
                    text=text.strip(),
                    confidence=self._sequence_confidence(lps),
                    prompt=qs[i],
                    token_ids=sequences[i, gen_start:].tolist(),
                    token_logprobs=lps,
                    attentions=(_slice_attentions(out.attentions, i) if want_attn else None),
                )
            )
        return results[0] if single else results

    def _sequence_confidence(self, token_logprobs: list[float]) -> float:
        """Map generated-token log-probs to a [0, 1] confidence score.

        ``seq_prob`` = exp(mean log p) — the geometric-mean token probability;
        ``min_prob`` = the least-confident token's probability (a conservative
        floor, useful for flagging risky extractions).
        """
        if not token_logprobs:
            return 0.0
        method = self.config.inference.confidence_method
        if method == "min_prob":
            return float(math.exp(min(token_logprobs)))
        mean_lp = sum(token_logprobs) / len(token_logprobs)
        return float(math.exp(mean_lp))

    # -- Convenience --------------------------------------------------------

    def to_eval(self) -> "VisionDocModel":
        """Put the model in eval mode (no dropout) for inference/evaluation."""
        if self.model is not None:
            self.model.eval()
        return self


# ---------------------------------------------------------------------------
# Free functions used by generate() (kept module-level for unit-testing)
# ---------------------------------------------------------------------------


def _slice_attentions(attn: Any, i: int) -> Any:
    """Index the batch dim of a (deeply nested) attentions structure for example i.

    ``generate(output_attentions=True)`` returns nested tuples of tensors shaped
    ``(batch, heads, q, k)``. Assigning the whole batch object to every
    per-example result (the naive approach) is wrong; we recurse and slice the
    batch dim, keeping it size-1 so downstream reduction still sees a 4-D shape.
    """
    if attn is None:
        return None
    if isinstance(attn, (list, tuple)):
        return type(attn)(_slice_attentions(a, i) for a in attn)
    try:
        return attn[i : i + 1]
    except Exception:
        return attn


def _token_logprobs_from_scores(
    scores: tuple[torch.Tensor, ...],
    sequences: torch.Tensor,
    eos_token_id: int | None = None,
    beam_indices: torch.Tensor | None = None,
) -> list[list[float]]:
    """Recover the log-prob of each generated token from ``generate`` scores.

    ``scores`` is a tuple of length ``num_generated_steps``; ``scores[t]`` has
    shape ``(num_rows, vocab)`` — ``num_rows`` is the batch for greedy/sampling
    but ``batch * num_beams`` for beam search. Generated tokens always occupy the
    trailing ``len(scores)`` positions of ``sequences``, regardless of left/right
    prompt padding. Per-sequence accumulation stops at the first EOS so padding
    tokens on early-finished sequences are not counted.

    Under beam search, ``beam_indices[b, t]`` identifies which flattened beam row
    produced example ``b``'s returned token at step ``t`` (``-1`` once finished),
    so we gather from that row rather than assuming row == example.
    """
    batch = sequences.shape[0]
    if not scores:
        return [[] for _ in range(batch)]
    gen_start = sequences.shape[1] - len(scores)
    out: list[list[float]] = [[] for _ in range(batch)]
    finished = [False] * batch
    for t, step_scores in enumerate(scores):
        logprobs = torch.log_softmax(step_scores.float(), dim=-1)
        chosen = sequences[:, gen_start + t]
        for b in range(batch):
            if finished[b]:
                continue
            if beam_indices is not None:
                row = int(beam_indices[b, t])
                if row < 0:  # this example already ended; remaining are pads
                    finished[b] = True
                    continue
            else:
                row = b
            tok = int(chosen[b])
            out[b].append(float(logprobs[row, tok]))
            if eos_token_id is not None and tok == eos_token_id:
                finished[b] = True
    return out


__all__ = [
    "VisionDocModel",
    "GenerationOutput",
    "register_model",
    "build_model",
]
