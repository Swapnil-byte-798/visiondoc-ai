"""Qwen2.5-VL backbone adapter (the default, preferred VLM).

Implements the :class:`VisionDocModel` tensorization hooks for
``Qwen/Qwen2.5-VL-*-Instruct``:

* prompts are built with the chat template + ``qwen_vl_utils.process_vision_info``
  (which applies Qwen's smart-resize within the configured pixel budget),
* training examples mask the prompt tokens (loss only on the answer),
* generation uses **left** padding (required so appended tokens line up across a
  batch) while training uses **right** padding,
* LoRA targets are resolved to fully-qualified **language-decoder** module names
  so the vision tower really is frozen (see
  :data:`LANGUAGE_ONLY_TARGET_REGEX` and :meth:`Qwen25VLModel.apply_lora`).

Qwen2.5-VL is a *causal* LM, so ``generate`` returns ``prompt + completion`` and
:meth:`decode` slices the prompt off.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

import torch
from PIL import Image

from models.base import VisionDocModel, register_model
from preprocessing.schema import DocSample
from utils.image_utils import load_image
from utils.logging_utils import get_logger

logger = get_logger("models.qwen_vl")

# Instruction prefix nudges the model toward terse, extraction-friendly answers.
_SYSTEM_PROMPT = (
    "You are a document understanding assistant. Answer the question about the "
    "document image concisely, using only information visible in the document."
)

# --- LoRA targeting: making "the ViT stays frozen" actually true -------------
#
# Qwen2.5-VL's module tree (verified against the HF implementation of
# ``Qwen2_5_VLForConditionalGeneration``):
#
#   language decoder : model.layers.{i}.self_attn.{q,k,v,o}_proj
#                      model.layers.{i}.mlp.{gate,up,down}_proj
#                      (transformers >= 4.52 nests this one level deeper as
#                       model.language_model.layers.{i}. ...)
#   vision tower     : visual.blocks.{i}.attn.{qkv,proj}
#                      visual.blocks.{i}.mlp.{gate,up,down}_proj   <-- collision
#                      visual.merger.mlp.{0,2}
#                      (again, model.visual.blocks... on newer transformers)
#
# PEFT matches a *bare* target name by suffix (``key == t or key.endswith("."+t)``).
# So the historical target list ``[..., "gate_proj", "up_proj", "down_proj"]`` also
# matched ``visual.blocks.*.mlp.*_proj`` — i.e. the ViT MLPs were being adapted
# and the "vision encoder is frozen" claim in the docs was FALSE.
#
# A regex string is passed to PEFT instead, which PEFT matches with
# ``re.fullmatch`` against the fully-qualified module name. Two things make it
# vision-safe: it requires a ``layers.<int>.`` segment (the vision tower uses
# ``blocks.<int>.``) and it additionally refuses any name containing a
# ``visual``/``vision_tower``/``vision_model`` path segment.
LANGUAGE_ONLY_TARGET_REGEX = (
    r"^(?!.*(?:^|\.)(?:visual|vision_tower|vision_model)\.)"
    r"(?:.*\.)?layers\.\d+\."
    r"(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)$"
)

#: Path segments that mark a module as belonging to the (frozen) vision tower.
_VISION_PATH_SEGMENTS = frozenset({"visual", "vision_tower", "vision_model"})


def _is_vision_module_name(name: str) -> bool:
    """True when a fully-qualified module name sits inside the vision tower."""
    return any(seg in _VISION_PATH_SEGMENTS for seg in name.split("."))


def _is_adaptable_linear(module: Any) -> bool:
    """True for LoRA-injectable linear layers (``nn.Linear``, bnb ``Linear4bit``, ...).

    Duck-typed on ``in_features``/``out_features`` rather than ``isinstance`` so
    quantized bitsandbytes layers count without importing bitsandbytes here.
    """
    return hasattr(module, "in_features") and hasattr(module, "out_features")


def language_only_linear_targets(model: Any) -> list[str]:
    """Enumerate the *actual* language-decoder linear layers of a loaded model.

    Returns fully-qualified module names, which PEFT matches exactly. Resolving
    against the real module tree (rather than trusting a hand-written list) is
    what makes the "vision tower frozen" claim checkable: anything under
    ``visual.*`` is filtered out by construction, and the returned list is what
    gets recorded in the run config.
    """
    pattern = re.compile(LANGUAGE_ONLY_TARGET_REGEX)
    names: list[str] = []
    for name, module in model.named_modules():
        if not name or _is_vision_module_name(name) or not _is_adaptable_linear(module):
            continue
        if pattern.fullmatch(name):
            names.append(name)
    return names


def vision_modules_matched_by(model: Any, targets: Sequence[str]) -> list[str]:
    """Return vision-tower modules that ``targets`` would (wrongly) adapt.

    Replicates PEFT's bare-name matching rule (exact name or dotted suffix) so
    the check reflects what PEFT would really do, not what we hope it does. An
    empty result means the target list is already vision-safe on this model.
    """
    hits: list[str] = []
    for name, module in model.named_modules():
        if not name or not _is_vision_module_name(name) or not _is_adaptable_linear(module):
            continue
        for t in targets:
            if name == t or name.endswith("." + t):
                hits.append(name)
                break
    return hits


@register_model("qwen2_5_vl")
class Qwen25VLModel(VisionDocModel):
    """Adapter for Qwen2.5-VL instruction-tuned vision-language models."""

    @property
    def default_lora_target_modules(self) -> list[str] | str | None:
        """Language-decoder attention + MLP projections **only** — the ViT stays frozen.

        Two forms are returned, both language-only (see the module-level notes on
        Qwen2.5-VL's naming):

        * once the backbone is loaded, the *resolved* list of fully-qualified
          module names (``model.layers.7.mlp.gate_proj``, ...). Enumerating the
          real module tree means the "no vision modules" property is verified,
          not asserted, and the exact adapted layers end up in the run config.
        * before loading (or if enumeration finds nothing, e.g. an unexpected
          transformers layout), :data:`LANGUAGE_ONLY_TARGET_REGEX`. PEFT accepts
          a string target as a regex and matches it with ``re.fullmatch``.

        The return type widens the base-class annotation (``list[str] | None``)
        by allowing ``str``; PEFT accepts both, and the regex fallback is the
        only way to stay vision-safe without a loaded model to inspect.
        """
        if self.model is not None:
            resolved = language_only_linear_targets(self.model)
            if resolved:
                return resolved
            logger.warning(
                "No language-decoder linear layers matched %s on this Qwen2.5-VL "
                "build; falling back to the regex target (PEFT will re-match it).",
                LANGUAGE_ONLY_TARGET_REGEX,
            )
        return LANGUAGE_ONLY_TARGET_REGEX

    # -- LoRA ---------------------------------------------------------------

    def apply_lora(self) -> "Qwen25VLModel":
        """Apply LoRA, first rejecting target names that would leak into the ViT.

        ``config.lora.target_modules`` historically shipped as bare leaf names
        (``gate_proj``/``up_proj``/``down_proj``), and PEFT resolves bare names by
        dotted-suffix match — which silently also hits ``visual.blocks.*.mlp.*``.
        Every experiment run with that config therefore fine-tuned the vision
        tower while the report claimed it was frozen.

        Rather than trusting the config, we ask the *loaded model* whether the
        configured targets match any vision module. Only if they do (i.e. the
        claim would be false on this concrete build) do we substitute the
        resolved language-only targets, and we say so at WARNING level. A config
        that deliberately and unambiguously targets vision modules — qualified
        names under ``visual.*`` — is left untouched, because then the user meant
        it and no false claim is being made on their behalf.
        """
        if self.model is None:
            raise RuntimeError("Call load() before apply_lora().")

        configured = self.config.lora.target_modules
        if isinstance(configured, (list, tuple)) and configured:
            leaked = vision_modules_matched_by(self.model, list(configured))
            if leaked:
                safe = self.default_lora_target_modules
                logger.warning(
                    "config.lora.target_modules=%s would also adapt %d vision-tower "
                    "modules (e.g. %s), contradicting the 'ViT frozen / language-side "
                    "only' claim. Overriding with %s language-only targets. Set "
                    "target_modules: null in the YAML config to silence this.",
                    list(configured),
                    len(leaked),
                    ", ".join(leaked[:3]),
                    len(safe) if isinstance(safe, list) else "regex-matched",
                )
                # ProjectConfig/LoRAConfig are frozen dataclasses; rebuild rather
                # than mutate so ``model.config`` reports what actually ran.
                self.config = replace(
                    self.config, lora=replace(self.config.lora, target_modules=safe)
                )

        super().apply_lora()
        return self

    # -- Loading ------------------------------------------------------------

    def load(self) -> "Qwen25VLModel":
        """Load the Qwen2.5-VL weights + processor.

        Design choices:
        * ``AutoProcessor`` with ``min_pixels``/``max_pixels`` bounds Qwen's
          dynamic-resolution tiling — the main VRAM lever for document scans.
        * On CUDA we let 4-bit quantization (QLoRA) kick in when configured; on
          MPS/CPU we load in the resolved dtype and place the whole model on one
          device (no ``device_map='auto'`` sharding).
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        mcfg = self.config.model
        processor_id = mcfg.processor_id or mcfg.model_id
        self.processor = AutoProcessor.from_pretrained(
            processor_id,
            min_pixels=mcfg.min_pixels,
            max_pixels=mcfg.max_pixels,
            trust_remote_code=mcfg.trust_remote_code,
        )

        load_kwargs: dict[str, Any] = dict(
            torch_dtype=self.dtype,
            attn_implementation=mcfg.attn_implementation,
            trust_remote_code=mcfg.trust_remote_code,
        )
        quant = self._maybe_quantization_config()
        if quant is not None:
            load_kwargs["quantization_config"] = quant
            load_kwargs["device_map"] = {"": 0}  # single-GPU placement for bnb

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mcfg.model_id, **load_kwargs
        )
        if quant is None:
            self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded Qwen2.5-VL '%s' on %s (%s)", mcfg.model_id, self.device, self.dtype)
        return self

    def _maybe_quantization_config(self) -> Any | None:
        """Build a bitsandbytes 4/8-bit config when requested (CUDA only)."""
        mcfg = self.config.model
        if not (mcfg.load_in_4bit or mcfg.load_in_8bit):
            return None
        if self.device.type != "cuda":
            logger.warning("Quantization requested but device is %s; ignoring.", self.device.type)
            return None
        from transformers import BitsAndBytesConfig

        if mcfg.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    # -- Prompt construction ------------------------------------------------

    def _messages(self, image: Image.Image, question: str, answer: str | None = None) -> list[dict]:
        """Build a Qwen chat-format message list for one example."""
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    # -- Training collation -------------------------------------------------

    def collate_train(self, batch: Sequence[DocSample | dict]) -> dict[str, torch.Tensor]:
        """Collate a batch into ``input_ids``/``attention_mask``/``labels`` + vision tensors.

        Labels mask everything up to (and including) the generation prompt so the
        cross-entropy loss is computed *only* over the assistant's answer tokens.
        Right padding keeps the prompt prefix aligned across the batch, which is
        what makes the per-sample prompt-length masking correct.
        """
        from qwen_vl_utils import process_vision_info

        samples = [s if isinstance(s, DocSample) else DocSample.from_record(s) for s in batch]
        self.processor.tokenizer.padding_side = "right"

        full_texts: list[str] = []
        all_images: list[Any] = []
        prompt_lens: list[int] = []
        for s in samples:
            img = load_image(s.image)
            full_msgs = self._messages(img, s.question, s.answer)
            prompt_msgs = self._messages(img, s.question)

            full_texts.append(
                self.processor.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(full_msgs)
            all_images.extend(image_inputs)

            # Prompt length *including* expanded image tokens (identical image =>
            # identical expansion in both prompt-only and full tokenizations).
            prompt_enc = self.processor(
                text=[prompt_text], images=image_inputs, return_tensors="pt"
            )
            prompt_lens.append(int(prompt_enc["input_ids"].shape[1]))

        enc = self.processor(
            text=full_texts, images=all_images, padding=True, return_tensors="pt"
        )
        labels = enc["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100  # supervise only the answer span
        enc["labels"] = labels
        return dict(enc)

    # -- Inference ----------------------------------------------------------

    def prepare_inference_inputs(
        self, images: Sequence[Image.Image], questions: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        """Build generation inputs (left-padded) on the model device."""
        from qwen_vl_utils import process_vision_info

        self.processor.tokenizer.padding_side = "left"
        texts: list[str] = []
        all_images: list[Any] = []
        for img, q in zip(images, questions):
            msgs = self._messages(load_image(img), q)
            texts.append(
                self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )
            image_inputs, _ = process_vision_info(msgs)
            all_images.extend(image_inputs)

        enc = self.processor(text=texts, images=all_images, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def decode(self, generated_ids: torch.Tensor, prompt_width: int) -> list[str]:
        """Strip the prompt prefix and decode the completion tokens to text."""
        trimmed = generated_ids[:, prompt_width:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


__all__ = [
    "Qwen25VLModel",
    "LANGUAGE_ONLY_TARGET_REGEX",
    "language_only_linear_targets",
    "vision_modules_matched_by",
]
