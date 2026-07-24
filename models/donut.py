"""Donut backbone adapter (lightweight, OCR-free alternative).

Donut (``naver-clova-ix/donut-base``) is a *Vision-Encoder-Decoder* — a Swin
image encoder + a BART-style text decoder — so its tensorization differs from a
causal VLM like Qwen:

* there is no image *token*: pixels enter through the encoder as ``pixel_values``;
* generation is seq2seq — ``generate`` returns only the decoder output, seeded by
  a task/question prompt supplied as ``decoder_input_ids``;
* Donut uses task-specific special tokens (``<s_docvqa>``, ``<s_cord-v2>``, ...).

This adapter exists mainly to demonstrate that the whole training/eval/serving
pipeline is backbone-agnostic: selecting it is a one-line config change.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from PIL import Image

from models.base import VisionDocModel, register_model
from preprocessing.schema import DocSample
from utils.image_utils import load_image
from utils.logging_utils import get_logger

logger = get_logger("models.donut")


@register_model("donut")
class DonutModel(VisionDocModel):
    """Adapter for Donut (VisionEncoderDecoder) document models."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Task token chosen from the dataset so the decoder is prompted correctly.
        name = config.data.dataset_name
        self.task_start = {
            "docvqa": "<s_docvqa>",
            "cord": "<s_cord-v2>",
            "sroie": "<s_sroie>",
            "funsd": "<s_funsd>",
        }.get(name, "<s_doc>")
        self.task = "qa" if name == "docvqa" else "extraction"
        self._decoder_prompt_width = 0  # set per inference batch, used by decode()

    @property
    def default_lora_target_modules(self) -> list[str] | None:
        # None => let PEFT auto-detect linear layers in the BART-style decoder.
        return None

    # -- Loading ------------------------------------------------------------

    def load(self) -> "DonutModel":
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        mcfg = self.config.model
        self.processor = DonutProcessor.from_pretrained(mcfg.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(mcfg.model_id, torch_dtype=self.dtype)

        # Register task + QA structural tokens and grow the embedding table.
        special = [self.task_start, "<s_question>", "</s_question>", "<s_answer>", "</s_answer>"]
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": special})
        self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))

        # Seed generation with the task-start token as the decoder BOS.
        self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.task_start
        )
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.max_length = self.config.data.max_seq_length

        # Constrain the image processor to the configured resolution.
        self.processor.image_processor.size = {
            "height": self.config.data.image_size,
            "width": self.config.data.image_size,
        }
        self.model.to(self.device).eval()
        logger.info("Loaded Donut '%s' on %s (task=%s)", mcfg.model_id, self.device, self.task)
        return self

    # -- Prompt / target formatting ----------------------------------------

    def _prompt_prefix(self, question: str) -> str:
        """Decoder prompt that conditions generation (no answer)."""
        if self.task == "qa":
            return f"{self.task_start}<s_question>{question}</s_question><s_answer>"
        return self.task_start  # extraction: model emits the structured sequence

    def _target_text(self, question: str, answer: str) -> str:
        """Full decoder sequence used for supervised training."""
        eos = self.processor.tokenizer.eos_token
        if self.task == "qa":
            return f"{self._prompt_prefix(question)}{answer}</s_answer>{eos}"
        return f"{self.task_start}{answer}{eos}"

    # -- Training collation -------------------------------------------------

    def collate_train(self, batch: Sequence[DocSample | dict]) -> dict[str, torch.Tensor]:
        """Build ``pixel_values`` + ``labels`` for teacher-forced seq2seq training.

        The prompt prefix (task/question tokens) is masked to -100 so loss is
        computed only on the answer/structured output.
        """
        samples = [s if isinstance(s, DocSample) else DocSample.from_record(s) for s in batch]
        images = [load_image(s.image) for s in samples]
        pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(self.dtype)

        tok = self.processor.tokenizer
        max_len = self.config.data.max_seq_length
        targets = [self._target_text(s.question, s.answer) for s in samples]
        enc = tok(
            targets,
            add_special_tokens=False,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        # Mask the prompt-prefix tokens for each example.
        for i, s in enumerate(samples):
            prefix_len = len(tok(self._prompt_prefix(s.question), add_special_tokens=False)["input_ids"])
            labels[i, :prefix_len] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    # -- Inference ----------------------------------------------------------

    def prepare_inference_inputs(
        self, images: Sequence[Image.Image], questions: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        """Return ``pixel_values`` + ``decoder_input_ids`` (the task/question prompt)."""
        tok = self.processor.tokenizer
        pil = [load_image(im) for im in images]
        pixel_values = self.processor(pil, return_tensors="pt").pixel_values.to(self.dtype)

        prompts = [self._prompt_prefix(q) for q in questions]
        # Uniform prompt lengths make prefix-stripping trivial; pad on the right.
        tok.padding_side = "right"
        dec = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
        self._decoder_prompt_width = int(dec["input_ids"].shape[1])
        return {
            "pixel_values": pixel_values.to(self.device),
            "decoder_input_ids": dec["input_ids"].to(self.device),
        }

    def decode(self, generated_ids: torch.Tensor, prompt_width: int) -> list[str]:
        """Strip the decoder prompt prefix and clean Donut's structural tokens."""
        trimmed = generated_ids[:, self._decoder_prompt_width :]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=False)
        cleaned = []
        for t in texts:
            for tokstr in [
                self.processor.tokenizer.eos_token,
                self.processor.tokenizer.pad_token,
                "</s_answer>",
                "<s_answer>",
                self.task_start,
            ]:
                if tokstr:
                    t = t.replace(tokstr, "")
            cleaned.append(t.strip())
        return cleaned

    @torch.no_grad()
    def generate(self, images, questions, **kwargs):  # type: ignore[override]
        """Generate one example at a time (correct batching for seq2seq).

        Donut conditions generation on a task/question prompt supplied as
        ``decoder_input_ids``, and those prompts differ in length across a batch.
        A *right*-padded decoder prefix makes ``generate`` emit the first token
        from a PAD position, and *left* padding shifts BART's absolute positions
        — neither is safe. We therefore loop per sample (each call is batch-1, so
        no padding is involved), trading throughput for correctness on this
        non-default backbone. The base implementation handles confidence/decoding.
        """
        single = isinstance(questions, str)
        imgs = [images] if isinstance(images, Image.Image) else list(images)
        qs = [questions] if single else list(questions)
        outs = [super(DonutModel, self).generate(im, q, **kwargs) for im, q in zip(imgs, qs)]
        return outs[0] if single else outs


__all__ = ["DonutModel"]
