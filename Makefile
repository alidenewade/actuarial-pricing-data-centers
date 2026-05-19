# dcrisk Makefile — paper + simulation + GPU orchestration on adu-00.
# All targets are self-contained; no implicit dependencies.

SHELL := /bin/bash
CONDA_ENV ?= dcrisk-gpu

PAPER_DIR := paper
PAPER_SRC := $(PAPER_DIR)/dc_paper.tex
PAPER_OUT := $(PAPER_DIR)/build/dc_paper.pdf
WEAR_SRC  := $(PAPER_DIR)/wearables.tex
WEAR_OUT  := $(PAPER_DIR)/build/wearables.pdf

NB_DIR        := notebooks
NOTEBOOKS     := $(wildcard $(NB_DIR)/*.ipynb)
NOTEBOOKS_GPU := $(NB_DIR)/04b_sde_simulation_gpu.ipynb $(NB_DIR)/08_oep_full_simulation.ipynb

# Streamlit binds to the Tailscale IPv4 so the dashboard stays inside the tailnet.
TAILSCALE_IP := $(shell tailscale ip --4 2>/dev/null | head -n1)
STREAMLIT_PORT ?= 8501

.PHONY: help paper wearables papers paper-clean sim sim-gpu app app-tunnel test test-gpu lint format clean bench check-env

help:
	@echo "dcrisk — make targets (env: $(CONDA_ENV)):"
	@echo "  paper        Build paper/build/dc_paper.pdf via latexmk"
	@echo "  wearables    Build paper/build/wearables.pdf (Series Paper 2) via latexmk"
	@echo "  papers       Build both papers"
	@echo "  paper-clean  Remove latex aux files from paper/build (keep PDFs)"
	@echo "  sim          Execute all notebooks in-place via papermill"
	@echo "  sim-gpu      Execute only the GPU notebooks (04b, 08)"
	@echo "  app          Launch Streamlit bound to the Tailscale IP, port $(STREAMLIT_PORT)"
	@echo "  app-tunnel   Print the URL to open the dashboard from another tailnet node"
	@echo "  test         Run pytest (CPU smoke tests)"
	@echo "  test-gpu     Run only tests/test_gpu_smoke.py (JAX/Torch/CuPy on RTX 5090)"
	@echo "  bench        CPU vs GPU benchmark of the (V, T, C) SDE integrator"
	@echo "  lint         ruff + mypy"
	@echo "  format       black + isort"
	@echo "  clean        Remove __pycache__, .pytest_cache, latex aux, paper/build/"

# ---- papers ---------------------------------------------------------------

paper: $(PAPER_OUT)

$(PAPER_OUT): $(wildcard $(PAPER_DIR)/*.tex)
	@mkdir -p $(PAPER_DIR)/build
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -file-line-error \
	   -synctex=1 -shell-escape -outdir=build dc_paper.tex

wearables: $(WEAR_OUT)

$(WEAR_OUT): $(WEAR_SRC) $(PAPER_DIR)/wearables_body.tex
	@mkdir -p $(PAPER_DIR)/build
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -file-line-error \
	   -synctex=1 -shell-escape -outdir=build wearables.tex

papers: paper wearables

paper-clean:
	cd $(PAPER_DIR) && latexmk -c -outdir=build || true

# ---- simulation -----------------------------------------------------------

sim: check-env
	@if ! command -v papermill >/dev/null 2>&1; then \
	  echo "papermill not found — activate the $(CONDA_ENV) env first."; exit 1; \
	fi
	@for nb in $(NOTEBOOKS); do \
	  echo "==> executing $$nb"; \
	  papermill --kernel $(CONDA_ENV) "$$nb" "$$nb" || exit 1; \
	done

sim-gpu: check-env
	@if ! command -v papermill >/dev/null 2>&1; then \
	  echo "papermill not found — activate the $(CONDA_ENV) env first."; exit 1; \
	fi
	@for nb in $(NOTEBOOKS_GPU); do \
	  echo "==> executing (GPU) $$nb"; \
	  papermill --kernel $(CONDA_ENV) "$$nb" "$$nb" || exit 1; \
	done

# ---- dashboard ------------------------------------------------------------

app: check-env
	@if [ -z "$(TAILSCALE_IP)" ]; then \
	  echo "ERROR: tailscale ip --4 returned empty. Is tailscaled running?"; exit 1; \
	fi
	@echo "Streamlit -> http://$(TAILSCALE_IP):$(STREAMLIT_PORT)"
	streamlit run src/dcrisk/dashboards/streamlit_app.py \
	  --server.address $(TAILSCALE_IP) \
	  --server.port    $(STREAMLIT_PORT) \
	  --server.headless true \
	  --browser.gatherUsageStats false

app-tunnel:
	@if [ -z "$(TAILSCALE_IP)" ]; then \
	  echo "tailscaled not active — no URL to print."; exit 1; \
	fi
	@echo "Open from any tailnet node:"
	@echo "    http://$(TAILSCALE_IP):$(STREAMLIT_PORT)"

# ---- tests / quality -----------------------------------------------------

test: check-env
	pytest -q --ignore=tests/test_gpu_smoke.py

test-gpu: check-env
	pytest tests/test_gpu_smoke.py -v

# ---- benchmark -----------------------------------------------------------

bench: check-env
	@python -c "import time, numpy as np; \
from dcrisk.sde.ptcyber import simulate_ptcyber; \
from dcrisk.sde.ptcyber_gpu import simulate_ptcyber_gpu; \
_ = simulate_ptcyber_gpu(T_max=0.5, dt=1/60.0, n_paths=64); \
t0=time.perf_counter(); simulate_ptcyber(T_max=24.0, dt=1/60.0, n_paths=10_000, rng=np.random.default_rng(0)); print(f'CPU (1e4 paths, 24h): {(time.perf_counter()-t0)*1000:.1f} ms'); \
t0=time.perf_counter(); simulate_ptcyber_gpu(T_max=24.0, dt=1/60.0, n_paths=10_000); print(f'GPU (1e4 paths, 24h): {(time.perf_counter()-t0)*1000:.1f} ms')"

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
	# NEVER touch envs — `clean` is for build artefacts only.

# ---- helper ---------------------------------------------------------------

check-env:
	@if [ "$$CONDA_DEFAULT_ENV" != "$(CONDA_ENV)" ]; then \
	  echo "WARN: active conda env is '$$CONDA_DEFAULT_ENV', expected '$(CONDA_ENV)'."; \
	  echo "      Run: conda activate $(CONDA_ENV)"; \
	fi
