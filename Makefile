# ===========================================================================
# VisionDoc AI — developer convenience targets
# `make help` lists everything. Targets are thin wrappers over the CLI so the
# same commands work in Docker, CI, and locally.
# ===========================================================================
.DEFAULT_GOAL := help
PYTHON ?= python
CONFIG ?= configs/default.yaml

# The *published* experiment: CORD-v2 (train 800 / val 100 / test 100), scored on
# validation+test pooled into one fixed 200-document set. Kept separate from
# CONFIG so `make results` always means the benchmark run that backs the README,
# while `make train CONFIG=...` stays free for experiments.
BENCH_CONFIG ?= configs/benchmark_cord.yaml
EVAL_SPLIT ?= validation+test
# The published n. Passed explicitly because compare.py falls back to
# data.max_eval_samples (100), which governs the cheap *in-training* eval loop —
# leaving it off would silently publish a 100-sample result labelled n=100 while
# the benchmark claims 200.
EVAL_N ?= 200
ADAPTER ?= outputs/qwen2_5vl-cord-lora-benchmark/adapter

.PHONY: help install install-dev download preprocess train evaluate compare \
        results readme readme-check api app lint format typecheck test smoke \
        docker-build docker-up clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies + package (editable)
	$(PYTHON) -m pip install -r requirements.txt && $(PYTHON) -m pip install -e .

install-dev:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt && $(PYTHON) -m pip install -e .

download:  ## Download + split the dataset named in the config
	$(PYTHON) -m scripts.download_data --config $(CONFIG)

preprocess:  ## Build + cache the processed dataset
	$(PYTHON) -m preprocessing.build_dataset --config $(CONFIG)

train:  ## Fine-tune with LoRA
	$(PYTHON) -m training.train --config $(CONFIG)

evaluate:  ## Evaluate a checkpoint and write a report
	$(PYTHON) -m evaluation.evaluate --config $(CONFIG)

compare:  ## Base model vs LoRA fine-tuned research comparison
	$(PYTHON) -m research.compare --config $(CONFIG)

# --- The published-numbers pipeline ---------------------------------------
# results -> reports/results.json (the canonical artifact, needs a GPU)
# readme  -> rewrites the <!-- EVAL --> block in README.md from that file
# readme-check -> the CI drift gate; fails if the committed block disagrees
results:  ## Run the published base-vs-LoRA comparison -> reports/results.json
	$(PYTHON) -m research.compare --config $(BENCH_CONFIG) --adapter $(ADAPTER) \
	  --split $(EVAL_SPLIT) --max-samples $(EVAL_N) --results-json reports/results.json

readme:  ## Regenerate the README results block from reports/results.json
	$(PYTHON) scripts/update_readme.py --write

readme-check:  ## Fail if the committed README block is stale (CI gate)
	$(PYTHON) scripts/update_readme.py --check

api:  ## Serve the FastAPI inference server
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run app/streamlit_app.py

lint:  ## Ruff lint
	ruff check .

format:  ## Black + ruff autofix
	black . && ruff check --fix .

typecheck:  ## Mypy static type check
	mypy .

test:  ## Run the test suite
	pytest

smoke:  ## Syntax-compile every module (no heavy deps required)
	$(PYTHON) -m compileall -q configs preprocessing models training evaluation \
	  inference research visualization api app utils scripts tests

docker-build:  ## Build API + app images
	docker compose -f docker/docker-compose.yml build

docker-up:  ## Run API + app via docker compose
	docker compose -f docker/docker-compose.yml up

# reports/results.json is deliberately NOT deleted: it is the committed artifact
# the README table is generated from, so wiping it would make `make readme`
# rewrite the block to "no published run" and turn a `make clean` into a silent
# deletion of the project's published numbers.
clean:  ## Remove caches and generated artifacts (keeps reports/results.json)
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ \
	  data/cache/* reports/*.html
	@[ -d reports ] && find reports -maxdepth 1 -name '*.json' \
	  ! -name 'results.json' -delete || true
