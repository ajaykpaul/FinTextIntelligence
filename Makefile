# `make check` is what CI runs and what should pass before a commit.

PYTHON ?= python3
PYTHONPATH := src
export PYTHONPATH

.DEFAULT_GOAL := help
.PHONY: help install install-dev install-train test test-cov lint format typecheck \
        bench demo ui-install ui-dev ui-build serve serve-asgi replay canary \
        review review-precision \
        licence-log check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (the runtime needs only the standard library)
	$(PYTHON) -m pip install -e .

install-dev: ## Install with test and lint tooling
	$(PYTHON) -m pip install -e '.[dev,serve]'

install-train: ## Install the training stack (torch, transformers, datasets)
	$(PYTHON) -m pip install -e '.[dev,serve,onnx,train]'

test: ## Run the tests
	$(PYTHON) -m pytest tests

test-cov: ## Run the tests with coverage
	$(PYTHON) -m pytest tests --cov=fintext --cov-report=term-missing

lint: ## Lint with ruff
	$(PYTHON) -m ruff check src tests

format: ## Format with ruff
	$(PYTHON) -m ruff format src tests

typecheck: ## Type-check with mypy
	$(PYTHON) -m mypy src/fintext

bench: ## Measure latency and throughput; fails if an SLO is missed
	$(PYTHON) -m fintext.cli bench --gate --output reports/latency.json

demo: ## Score a built-in set of examples
	$(PYTHON) -m fintext.cli demo

ui-install: ## Install the React UI's dependencies
	cd web && npm install

ui-dev: ## Run the UI with hot reload; proxies the API to a running `make serve`
	cd web && npm run dev

ui-build: ## Build the UI into the Python package (commit the result)
	cd web && npm run build

serve: ## Run the service on the standard-library HTTP server
	$(PYTHON) -m fintext.cli serve

serve-asgi: ## Run the service on uvicorn (needs the serve extra)
	$(PYTHON) -m fintext.cli serve --asgi

replay: ## Score the provenance replay set (PRD 8.2)
	$(PYTHON) -m fintext.cli replay data/samples/provenance_replay.jsonl

canary: ## Run the weekly adversarial canary (PRD 8.4)
	$(PYTHON) -m fintext.cli canary

review: ## Draw this week's human-review sample, then use the /ui review tab (PRD 8.4)
	$(PYTHON) -m fintext.cli review sample --days 7 --size 50

review-precision: ## Measure review precision and feed it to the acceptance gate
	$(PYTHON) -m fintext.cli review precision --metrics reports/measured_metrics.json

licence-log: ## Regenerate the licence log for legal review (PRD 7)
	$(PYTHON) -m fintext.cli catalogue --licence-log docs/LICENCE_LOG.txt

check: lint typecheck test bench ## Everything CI runs

clean: ## Remove build and test artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
