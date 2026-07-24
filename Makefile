# ===========================================================================
# VisionDoc AI — developer convenience targets
# `make help` lists everything. Targets are thin wrappers over the CLI so the
# same commands work in Docker, CI, and locally.
# ===========================================================================
.DEFAULT_GOAL := help
PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: help install install-dev download preprocess train evaluate compare \
        api app lint format typecheck test smoke docker-build docker-up clean

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

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ \
	  data/cache/* reports/*.html reports/*.json
