"""Best-effort ONNX export for the language side of a Vision-Language Model.

Honest scope statement (read this before trusting the output)
-------------------------------------------------------------
**End-to-end ONNX export of a multimodal VLM like Qwen2.5-VL is not officially
supported and is genuinely hard.** The obstacles are real, not incidental:

* the vision tower uses dynamic-resolution tiling whose tensor shapes depend on
  the input image, so a single static ONNX graph cannot represent it;
* Qwen2.5-VL relies on custom ops (``qwen_vl_utils`` vision preprocessing, M-RoPE
  positional encoding, ``rotary`` embeddings) that have no clean ONNX lowering;
* generation is an autoregressive Python loop with a KV-cache — ``torch.onnx``
  exports a single forward, not the loop;
* ``transformers``/``optimum`` do not register an ONNX export config for the
  ``Qwen2_5_VL`` architecture at time of writing.

Rather than fake a full export (or emit a broken graph and call it success),
this module does two honest things, in order of preference:

1. **Try ``optimum``'s in-memory exporter** (``onnx_export_from_model``). If the
   installed ``optimum`` *does* support the architecture, you get a real,
   complete export and we return that directory. For most VLMs this raises
   "unsupported architecture" — which we catch and report, not swallow.

2. **Fall back to a representative sub-graph**: the model's **language head**
   (the final ``lm_head`` ``Linear`` that maps decoder hidden states to
   vocabulary logits). This is a genuinely useful, verifiable artifact — it is
   the piece you would fuse/quantize for a custom serving stack, and it exports
   cleanly with static per-token semantics and dynamic batch/sequence axes.

The function never reports success it didn't achieve: if neither path produces a
file, it raises ``RuntimeError`` with actionable guidance instead of returning a
path to a non-existent or invalid model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # hints only — module must import without torch/optimum present
    import torch.nn as nn

    from models.base import VisionDocModel

logger = get_logger(__name__)


def export_language_head_to_onnx(
    model: "VisionDocModel",
    out_path: str | Path,
    opset: int = 17,
) -> str:
    """Export (a real, best-effort ONNX artifact for) the model's language side.

    Attempts a full ``optimum`` export first, then falls back to exporting the
    representative ``lm_head`` sub-graph. See the module docstring for why a full
    multimodal export is not generally attainable.

    Parameters
    ----------
    model:
        A *loaded* :class:`~models.base.VisionDocModel` (call ``model.load()``
        first; a LoRA-wrapped or merged model is fine — we unwrap it).
    out_path:
        Destination ``.onnx`` file for the fallback head export. The optimum
        path writes a sibling directory (it emits multiple files).
    opset:
        ONNX opset version. 17 is a safe modern default (LayerNorm et al.).

    Returns
    -------
    Path (as ``str``) to the produced ONNX file or export directory.

    Raises
    ------
    RuntimeError
        If neither export path succeeds — with guidance on what to install/check.
        We deliberately never return a path to a file we failed to write.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Preferred: let optimum try a real, complete export of the whole model.
    try:
        produced = _export_with_optimum(model, out_path, opset)
        logger.info("optimum produced a full ONNX export at %s", produced)
        return produced
    except Exception as exc:  # unsupported arch / optimum missing / trace failure
        # This is the *expected* branch for VLMs; log clearly and fall through.
        logger.warning(
            "Full-model ONNX export via optimum is unavailable for this model "
            "(%s: %s). Falling back to a representative language-head export.",
            type(exc).__name__,
            exc,
        )

    # 2) Fallback: export the language head sub-graph — real and verifiable.
    try:
        produced = _export_language_head(model, out_path, opset)
        logger.info("Exported representative language head to %s", produced)
        return produced
    except Exception as exc:
        # Honest failure: no artifact was written, so say so with a fix-it hint.
        raise RuntimeError(
            f"ONNX export failed ({type(exc).__name__}: {exc}). Checklist: "
            "(1) `pip install onnx onnxruntime` (and optionally `optimum`); "
            "(2) ensure the model is loaded via model.load(); "
            "(3) confirm the backbone exposes an `lm_head` linear. "
            "Note that full multimodal VLM export is not officially supported — "
            "see this module's docstring."
        ) from exc


# ---------------------------------------------------------------------------
# Internal export paths
# ---------------------------------------------------------------------------


def _export_with_optimum(model: "VisionDocModel", out_path: Path, opset: int) -> str:
    """Attempt a full export using ``optimum.exporters.onnx``.

    Uses the *in-memory* entrypoint (``onnx_export_from_model``) so we export the
    exact loaded weights (including any merged LoRA) rather than re-downloading
    from the Hub. Raises for architectures optimum can't handle — which the
    caller treats as a signal to fall back, not as a hard error.
    """
    # Import lazily; optimum is an optional dep and older versions lack this API.
    from optimum.exporters.onnx import onnx_export_from_model

    hf_model = _unwrap_hf_model(model)
    # optimum writes several files (model.onnx, config.json, ...) into a dir.
    output_dir = out_path.parent / f"{out_path.stem}_optimum"
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_export_from_model(
        hf_model,
        output=str(output_dir),
        opset=opset,
        # do_validation left at its default; if optimum can build the config it
        # will also validate numerics, giving us a truthful success signal.
    )
    return str(output_dir)


def _export_language_head(model: "VisionDocModel", out_path: Path, opset: int) -> str:
    """Trace and export the ``lm_head`` linear (hidden_states -> vocab logits).

    Why the LM head specifically: it is the smallest self-contained, statically
    shaped piece of the language decoder (a plain ``Linear``), so it traces
    cleanly with no custom ops, while still being the component a custom serving
    stack most often wants standalone (to fuse/quantize the vocab projection).
    """
    import copy

    import torch

    hf_model = _unwrap_hf_model(model)
    lm_head = _find_lm_head(hf_model)

    # weight is (vocab_size, hidden_size); prefer the declared attrs when present.
    hidden_size = int(getattr(lm_head, "in_features", None) or lm_head.weight.shape[1])
    vocab_size = int(getattr(lm_head, "out_features", None) or lm_head.weight.shape[0])

    # Deepcopy + cast to fp32 on CPU: we must NOT mutate the live model, and the
    # ONNX exporter is most reliable on fp32/CPU (bf16/fp16 export has spotty op
    # coverage). This is a best-effort artifact, so fp32 is the honest choice.
    head = copy.deepcopy(lm_head).float().cpu().eval()

    # Representative input: one short sequence of hidden vectors. Batch and
    # sequence are marked dynamic so the exported graph accepts any shape.
    dummy_hidden = torch.zeros(1, 8, hidden_size, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            head,
            dummy_hidden,
            str(out_path),
            input_names=["hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    logger.info(
        "Language-head ONNX graph: hidden_size=%d -> vocab_size=%d (fp32, opset %d). "
        "Note: this is a representative sub-graph, NOT the full multimodal model.",
        hidden_size,
        vocab_size,
        opset,
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------


def _unwrap_hf_model(model: "VisionDocModel"):
    """Return the underlying HF ``nn.Module`` from a :class:`VisionDocModel`.

    Peels back two layers of wrapping: the :class:`VisionDocModel` facade
    (``.model``) and, if present, the PEFT wrapper (``get_base_model()``) so we
    trace the real transformer rather than the adapter shell. A merged or bare
    model passes through unchanged.
    """
    hf_model = getattr(model, "model", model)
    if hf_model is None:
        raise RuntimeError(
            "model.model is None — call model.load() before exporting to ONNX."
        )
    # PeftModel exposes get_base_model(); unwrap so lm_head is directly reachable.
    if hasattr(hf_model, "get_base_model"):
        try:
            hf_model = hf_model.get_base_model()
        except Exception:  # pragma: no cover - defensive; fall back to wrapper
            logger.debug("get_base_model() failed; using the wrapped model as-is.")
    return hf_model


def _find_lm_head(hf_model) -> "nn.Module":
    """Locate the language-model head (vocabulary projection) on the backbone.

    Tries the conventional ``.lm_head`` attribute first, then searches
    ``named_modules`` for a leaf named ``lm_head`` — robust to nesting introduced
    by PEFT or by ``*ForConditionalGeneration`` wrappers. Raises with guidance if
    none is found so the caller can surface a precise, actionable error.
    """
    import torch.nn as nn

    head = getattr(hf_model, "lm_head", None)
    if isinstance(head, nn.Module) and hasattr(head, "weight"):
        return head

    for name, module in hf_model.named_modules():
        # Match the leaf name so nested paths (e.g. "model.lm_head") still hit.
        if name.rsplit(".", 1)[-1] == "lm_head" and hasattr(module, "weight"):
            return module

    raise RuntimeError(
        "Could not locate an `lm_head` linear on the model. This backbone may "
        "tie/rename its output projection; export the appropriate module "
        "manually or use the optimum path if the architecture is supported."
    )


__all__ = ["export_language_head_to_onnx"]
