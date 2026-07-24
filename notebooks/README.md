# Notebooks

Three narrative notebooks that walk the **VisionDoc AI** pipeline end to end,
each calling the exact same public APIs the CLI, FastAPI service, and Streamlit
app use (no parallel re-implementations).

- **`01_dataset_exploration.ipynb`** — loads the project config, pulls a handful
  of DocVQA validation samples through the `DocSample` schema
  (`preprocessing.load_samples`), visualises a few document pages with their
  questions and gold answers, prints the split sizes, and sketches the
  answer-length distribution. Also explains what DocVQA is and why ANLS is its
  metric.
- **`02_inference_demo.ipynb`** — builds a `DocumentPredictor.from_config`,
  answers a question on a real page (answer + confidence + latency), extracts
  structured fields as JSON (`invoice`/`receipt`-style schema via
  `predictor.extract`), and highlights the OCR-grounded region the answer came
  from (`predictor.highlight_regions`).
- **`03_evaluation_and_comparison.ipynb`** — scores the base backbone vs. a LoRA
  fine-tune on a small validation slice with `evaluation.evaluate_model`, runs
  `research.compare_models` for a test-split head-to-head, and renders the
  comparison table plus bar chart, confidence distribution, and (when a training
  run exists) loss curves via `visualization.plots`.

## Requirements

These notebooks run the real model, so they need the **full project
dependencies** (`torch`, `transformers`, `peft`, `datasets`, `Pillow`,
`matplotlib`, and the metric libs) — install with `pip install -e .` from the
repo root. The first run of notebook 01/03 downloads DocVQA into the configured
cache, and highlighting in notebook 02 additionally needs Tesseract +
`pytesseract` (it degrades gracefully to "no regions" without it).

Notebooks **02** and **03** use a trained LoRA adapter when
`config.inference.adapter_path` is set and exists; otherwise they **fall back to
the base model (zero-shot)** — everything still runs, the answers are just
un-tuned and the fine-tuned comparison column is empty. Train one first with
`python -m training.train` to populate it. Keep the `max_samples` / `MAX_SAMPLES`
caps small: VLM generation is the bottleneck.

Each notebook starts with a bootstrap cell that puts the repo root on
`sys.path`, so they work whether launched from `notebooks/` or the repo root.
