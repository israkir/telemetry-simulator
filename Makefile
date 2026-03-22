.PHONY: help venv venv-force check-venv install install-dev clean clean-venv format lint typecheck check test
.PHONY: run-validate list-scenarios
.PHONY: jaeger-up jaeger-down
# Python and venv paths
# Prefer 'python' (user's default in PATH), fallback to 'python3'. Override with PYTHON=... if needed.
PYTHON ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
VENV := .venv
VENV_BIN := $(VENV)/bin
VENV_ACTIVATE := $(VENV_BIN)/activate
VENV_PYTHON := $(VENV_BIN)/python

# CLI program name (from project.scripts in pyproject.toml; use variable to avoid hardcoding)
CLI_NAME ?= otelsim

# Basic ANSI colors (can be disabled with COLOR=0)
COLOR ?= 1
ifeq ($(COLOR),0)
	COLOR_RESET  :=
	COLOR_BOLD   :=
	COLOR_BLUE   :=
	COLOR_GREEN  :=
	COLOR_YELLOW :=
	COLOR_RED    :=
else
	COLOR_RESET  := \033[0m
	COLOR_BOLD   := \033[1m
	COLOR_BLUE   := \033[34m
	COLOR_GREEN  := \033[32m
	COLOR_YELLOW := \033[33m
	COLOR_RED    := \033[31m
endif

# Check if venv exists and is usable
check-venv:
	@if [ ! -f "$(VENV_PYTHON)" ]; then \
		echo "❌ Error: Virtual environment not found at $(VENV)/"; \
		echo ""; \
		echo "Please run 'make venv' first to create the virtual environment."; \
		echo "Example:"; \
		echo "  make venv"; \
		echo "  make install"; \
		exit 1; \
	fi
	@$(VENV_PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || { \
		echo "❌ Error: This project requires Python >=3.11. The venv at $(VENV)/ was created with an older Python."; \
		echo ""; \
		echo "Recreate the venv with a supported Python, e.g.:"; \
		echo "  make clean-venv"; \
		echo "  PYTHON=python3.14 make venv"; \
		echo "  make install-dev"; \
		exit 1; \
	}
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Warning: Virtual environment is not activated in your shell."; \
		echo "   Commands will still use the venv's Python, but consider activating it:"; \
		echo "   source $(VENV_ACTIVATE)"; \
		echo ""; \
	fi

# Default target
help:
	@printf '$(COLOR_BOLD)Telemetry Simulator$(COLOR_RESET)\n'
	@printf 'Schema-driven OTEL telemetry simulation for LLM observability\n\n'
	@printf 'Setup:\n'
	@printf '  $(COLOR_GREEN)make venv$(COLOR_RESET)                  - Create virtual environment (if not exists)\n'
	@printf '  $(COLOR_GREEN)make venv-force$(COLOR_RESET)            - Force recreate virtual environment\n'
	@printf '  $(COLOR_GREEN)make install$(COLOR_RESET)               - Install the package in development mode\n'
	@printf '  $(COLOR_GREEN)make install-dev$(COLOR_RESET)           - Install with development dependencies\n\n'
	@printf 'Running the simulator:\n'
	@printf '  $(COLOR_GREEN)make run-validate$(COLOR_RESET)          - Same as: $(CLI_NAME) validate --show-schema --show-spans --show-metrics\n'
	@printf '  $(COLOR_GREEN)make list-scenarios$(COLOR_RESET)        - Same as: $(CLI_NAME) list\n\n'
	@printf 'Code quality and tests:\n'
	@printf '  $(COLOR_GREEN)make test$(COLOR_RESET)                  - Run tests (pytest)\n'
	@printf '  $(COLOR_GREEN)make format$(COLOR_RESET)                - Format code with black\n'
	@printf '  $(COLOR_GREEN)make lint$(COLOR_RESET)                  - Lint code with ruff\n'
	@printf '  $(COLOR_GREEN)make typecheck$(COLOR_RESET)             - Type check with mypy\n'
	@printf '  $(COLOR_GREEN)make check$(COLOR_RESET)                 - Run format, lint, and typecheck\n\n'
	@printf 'Cleanup:\n'
	@printf '  $(COLOR_GREEN)make clean$(COLOR_RESET)                 - Clean generated cache files\n'
	@printf '  $(COLOR_GREEN)make clean-venv$(COLOR_RESET)            - Remove virtual environment\n\n'
	@printf 'Live trace visualization (Docker or Podman):\n'
	@printf '  $(COLOR_GREEN)make jaeger-up$(COLOR_RESET)             - Start Jaeger (OTLP + UI) for live traces\n'
	@printf '  $(COLOR_GREEN)make jaeger-down$(COLOR_RESET)           - Stop and remove Jaeger container\n'
	@printf '  (then run simulator with CLI and open http://localhost:16686/search)\n\n'
	@printf 'Example workflows:\n'
	@printf '  $(COLOR_GREEN)make venv$(COLOR_RESET) && $(COLOR_GREEN)make install$(COLOR_RESET)\n'
	@printf '  $(CLI_NAME) run --count 10\n'
	@printf '  make jaeger-up && $(CLI_NAME) run\n'

# Virtual environment
venv:
	@if [ -f "$(VENV_ACTIVATE)" ]; then \
		echo "✓ Virtual environment already exists at $(VENV)/"; \
		echo ""; \
		echo "To activate, run:"; \
		echo "  source $(VENV_ACTIVATE)"; \
		echo ""; \
		echo "To deactivate, run:"; \
		echo "  deactivate"; \
	else \
		echo "Creating virtual environment with $$($(PYTHON) -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '$(PYTHON)')..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "✓ Virtual environment created (Python: $$($(VENV_BIN)/python --version 2>/dev/null))."; \
		echo ""; \
		echo "To activate, run:"; \
		echo "  source $(VENV_ACTIVATE)"; \
		echo ""; \
		echo "To deactivate, run:"; \
		echo "  deactivate"; \
	fi

# Force recreate virtual environment
venv-force: clean-venv venv
	@echo "Virtual environment recreated successfully"

# Install package
install: check-venv
	@echo "📦 Installing simulator..."
	$(VENV_PYTHON) -m pip install -e .
	@echo "✅ Installation complete"
	@echo ""
	@echo "Run with CLI: $(CLI_NAME) run"

# Install with development dependencies
install-dev: check-venv
	@echo "📦 Installing simulator with dev dependencies..."
	$(VENV_PYTHON) -m pip install -e ".[dev]"
	@echo "✅ Installation complete"
	@echo ""
	@echo "Run with CLI: $(CLI_NAME) run"

# =============================================================================
# SIMULATOR COMMANDS
# =============================================================================

# Validate schema
run-validate: check-venv
	$(VENV_PYTHON) -m simulator.cli validate \
		--show-schema \
		--show-spans \
		--show-metrics

# List available scenarios
list-scenarios: check-venv
	$(VENV_PYTHON) -m simulator.cli list

# =============================================================================
# LIVE TRACE VISUALIZATION (Jaeger)
# =============================================================================
# Start Jaeger all-in-one with OTLP receiver. Simulator sends to localhost:4318.
# Open http://localhost:16686/search to view traces.
# Prefer Podman if available, otherwise Docker.

JAEGER_IMAGE ?= jaegertracing/jaeger:2.15.0
JAEGER_NAME ?= jaeger
# Use CONTAINER_CMD=podman or CONTAINER_CMD=docker to force; otherwise prefer podman
CONTAINER_CMD ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null || echo "")

jaeger-up:
	@if [ -z "$(CONTAINER_CMD)" ]; then \
		echo "✗ Error: Neither podman nor docker found. Install one to run Jaeger."; \
		exit 1; \
	fi
	@_out=$$($(CONTAINER_CMD) ps -q -f name=^/$(JAEGER_NAME)$$ 2>&1) || true; \
	if echo "$$_out" | grep -q "proxy already running"; then \
		echo "✗ Podman proxy error: machine may be in a bad state."; \
		echo "  Try: podman machine stop && podman machine start"; \
		echo "  Or use Docker: CONTAINER_CMD=docker make jaeger-up"; \
		exit 1; \
	fi
	@if $(CONTAINER_CMD) ps -q -f name=^/$(JAEGER_NAME)$$ 2>/dev/null | grep -q .; then \
		echo "✓ Jaeger is already running (container: $(JAEGER_NAME))"; \
		echo "  UI: http://localhost:16686/search  OTLP: http://localhost:4318"; \
	else \
		echo "Starting Jaeger with $(CONTAINER_CMD) (OTLP :4317/:4318, UI :16686)..."; \
		$(CONTAINER_CMD) rm -f $(JAEGER_NAME) 2>/dev/null || true; \
		_run_out=$$($(CONTAINER_CMD) run -d --name $(JAEGER_NAME) \
			-p 16686:16686 \
			-p 4317:4317 \
			-p 4318:4318 \
			-e JAEGER_LISTEN_HOST=0.0.0.0 \
			$(JAEGER_IMAGE) 2>&1) || true; \
		if $(CONTAINER_CMD) ps -q -f name=^/$(JAEGER_NAME)$$ 2>/dev/null | grep -q .; then \
			echo "✓ Jaeger started. UI: http://localhost:16686/search  OTLP: http://localhost:4318"; \
			echo "  Run: $(CLI_NAME) run  then open the UI."; \
		else \
			echo "✗ Failed to start Jaeger."; \
			if [ -n "$$_run_out" ]; then echo "  $$_run_out"; fi; \
			if echo "$$_run_out" | grep -q "proxy already running"; then \
				echo "  Podman proxy: try 'podman machine stop && podman machine start' or CONTAINER_CMD=docker make jaeger-up"; \
			elif echo "$$_run_out" | grep -q "already in use"; then \
				echo "  Remove existing container: make jaeger-down"; \
			fi; \
			exit 1; \
		fi; \
	fi

jaeger-down:
	@_c=$$(command -v podman 2>/dev/null || command -v docker 2>/dev/null); \
	if [ -n "$$_c" ]; then $$_c stop $(JAEGER_NAME) 2>/dev/null || true; $$_c rm $(JAEGER_NAME) 2>/dev/null || true; fi
	@echo "Jaeger stopped and removed."

# =============================================================================
# CODE QUALITY
# =============================================================================

# Format code
format: check-venv
	@echo "🎨 Formatting code with black..."
	$(VENV_PYTHON) -m black src/
	@echo "✅ Code formatted"

# Lint code
lint: check-venv
	@echo "🔍 Linting code with ruff..."
	$(VENV_PYTHON) -m ruff check src/ --fix
	@echo "✅ Linting complete"

# Type check
typecheck: check-venv
	@echo "🔍 Type checking with mypy..."
	$(VENV_PYTHON) -m mypy src/simulator/
	@echo "✅ Type checking complete"

# Run format, lint, and type checks
check: format lint typecheck
	@echo "✅ All checks passed"

# Run tests (uses testpaths from pyproject.toml)
test: check-venv
	@echo "🧪 Running tests..."
	$(VENV_PYTHON) -m pytest -v
	@echo "✅ Tests complete"

# =============================================================================
# CLEANUP
# =============================================================================

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf src/simulator/__pycache__/
	rm -rf src/simulator/**/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf tests/**/__pycache__/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	rm -rf .coverage .coverage.*
	rm -rf htmlcov/
	rm -rf build/ dist/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	@echo "✅ Cleanup complete"

# Remove virtual environment
clean-venv:
	@echo "🗑️  Removing virtual environment..."
	rm -rf $(VENV)/
	@echo "✅ Virtual environment removed"

# Clean everything
clean-all: clean clean-venv
	@echo "✅ All cleanup complete"

