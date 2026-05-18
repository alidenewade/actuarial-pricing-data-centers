# dcrisk Makefile — paper build + simulation orchestration
# All targets are self-contained; no implicit dependencies.

SHELL := /bin/bash
CONDA_ENV ?= dcrisk

PAPER_DIR := paper
PAPER_SRC := $(PAPER_DIR)/dc_paper.tex
PAPER_OUT := $(PAPER_DIR)/build/dc_paper.pdf

NB_DIR    := notebooks
NOTEBOOKS := $(wildcard $(NB_DIR)/*.ipynb)

.PHONY: help paper paper-clean sim app test lint format clean check-env

help:
	@echo "dcrisk — make targets:"
	@echo "  paper        Build paper/dc_paper.pdf via latexmk into paper/build/"
	@echo "  paper-clean  Remove paper/build/ aux files (keep PDF)"
	@echo "  sim          Execute all notebooks in-place via papermill"
	@echo "  app          Launch the Streamlit dashboard"
	@echo "  test         Run pytest"
	@echo "  lint         Run ruff + mypy"
	@echo "  format       Run black + isort"
	@echo "  clean        Remove __pycache__, .pytest_cache, latex aux, paper/build/"

# ---- paper ----------------------------------------------------------------

paper: $(PAPER_OUT)

$(PAPER_OUT): $(wildcard $(PAPER_DIR)/*.tex)
	@mkdir -p $(PAPER_DIR)/build
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -shell-escape -outdir=build dc_paper.tex

paper-clean:
	cd $(PAPER_DIR) && latexmk -c -outdir=build || true

# ---- simulation -----------------------------------------------------------

sim: check-env
	@if ! command -v papermill >/dev/null 2>&1; then \
	  echo "papermill not found in PATH — activate the $(CONDA_ENV) env first."; exit 1; \
	fi
	@for nb in $(NOTEBOOKS); do \
	  echo "==> executing $$nb"; \
	  papermill --kernel $(CONDA_ENV) "$$nb" "$$nb" || exit 1; \
	done

# ---- dashboard ------------------------------------------------------------

app: check-env
	streamlit run src/dcrisk/dashboards/streamlit_app.py

# ---- tests / quality -----------------------------------------------------

test: check-env
	pytest -q

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	isort src tests

# ---- cleanup --------------------------------------------------------------

clean: paper-clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	rm -rf $(PAPER_DIR)/build/*.aux $(PAPER_DIR)/build/*.log $(PAPER_DIR)/build/*.out

# ---- helper ---------------------------------------------------------------

check-env:
	@if [ "$$CONDA_DEFAULT_ENV" != "$(CONDA_ENV)" ]; then \
	  echo "WARN: active conda env is '$$CONDA_DEFAULT_ENV', expected '$(CONDA_ENV)'."; \
	  echo "      Run: conda activate $(CONDA_ENV)"; \
	fi
