"""Qwen2.5-VL backbone adapter (the default, preferred VLM).

Implements the :class:`VisionDocModel` tensorization hooks for
``Qwen/Qwen2.5-VL-*-Instruct``:

* prompts are built with the chat template + ``qwen_vl_utils.process_vision_info``
  (which applies Qwen's smart-resize within the configured pixel budget),
* training examples mask the prompt tokens (loss only on the answer),
* generation uses **left** padding (required so appended tokens line up across a
  batch) while training uses **right** padding.

Qwen2.5-VL is a *causal* LM, so ``generate`` returns ``prompt + completion`` and
:meth:`decode` slices the prompt off.
"""

from __future__ import annotations

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


@register_model("qwen2_5_vl")
class Qwen25VLModel(VisionDocModel):
    """Adapter for Qwen2.5-VL instruction-tuned vision-language models."""

    @property
    def default_lora_target_modules(self) -> list[str] | None:
        # Language-decoder attention + MLP projections. We leave the ViT frozen.
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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


__all__ = ["Qwen25VLModel"]
