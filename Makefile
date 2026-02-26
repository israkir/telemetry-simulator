.PHONY: help venv venv-force check-venv install install-dev clean clean-venv format lint typecheck check test
.PHONY: run run-validate list-scenarios
.PHONY: jaeger-up jaeger-down

# Python and venv paths
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
VENV_ACTIVATE := $(VENV_BIN)/activate
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

# Default OTLP endpoint (override as needed)
OTLP_ENDPOINT ?= http://localhost:4318

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
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Warning: Virtual environment is not activated in your shell."; \
		echo "   Commands will still use the venv's Python, but consider activating it:"; \
		echo "   source $(VENV_ACTIVATE)"; \
		echo ""; \
	fi

# Default target
help:
	@echo "Telemetry Simulator"
	@echo "Schema-driven OTEL telemetry simulation for LLM observability"
	@echo ""
	@echo "Setup:"
	@echo "  make venv                  - Create virtual environment (if not exists)"
	@echo "  make venv-force            - Force recreate virtual environment"
	@echo "  make install               - Install the package in development mode"
	@echo "  make install-dev           - Install with development dependencies"
	@echo ""
	@echo "Running the simulator:"
	@echo "  make run                   - Run mixed workload (requires SEMCONV)"
	@echo "  make run-validate          - Validate schema and show summary"
	@echo "  make list-scenarios        - List available scenarios"
	@echo ""
	@echo "Code quality and tests:"
	@echo "  make test                  - Run tests (pytest)"
	@echo "  make format                - Format code with black"
	@echo "  make lint                  - Lint code with ruff"
	@echo "  make typecheck             - Type check with mypy"
	@echo "  make check                 - Run format, lint, and typecheck"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                 - Clean generated cache files"
	@echo "  make clean-venv            - Remove virtual environment"
	@echo ""
	@echo "Environment variables:"
	@echo "  SEMCONV                            - Path to semantic-conventions YAML (required)"
	@echo "  OTLP_ENDPOINT                      - OTLP collector endpoint (default: http://localhost:4318)"
	@echo "  VENDOR                             - Attribute prefix for spans/metrics (default: vendor)"
	@echo "  SCENARIOS_DIR                      - Folder with scenario YAML files (default: built-in sample definitions)"
	@echo ""
	@echo "Live trace visualization (Docker or Podman):"
	@echo "  make jaeger-up             - Start Jaeger (OTLP + UI) for live traces"
	@echo "  make jaeger-down           - Stop and remove Jaeger container"
	@echo "  (then: make run and open http://localhost:16686)"
	@echo ""
	@echo "Example workflows:"
	@echo "  make venv && make install && make run"
	@echo "  SEMCONV=/path/to/otel-semantic-conventions.yaml make run"
	@echo "  make jaeger-up && make run"

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
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "✓ Virtual environment created."; \
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
	@echo "📦 Installing telemetry-simulator..."
	$(VENV_PIP) install -e .
	@echo "✅ Installation complete"
	@echo ""
	@echo "You can now run: make run"

# Install with development dependencies
install-dev: check-venv
	@echo "📦 Installing telemetry-simulator with dev dependencies..."
	$(VENV_PIP) install -e ".[dev]"
	@echo "✅ Installation complete"
	@echo ""
	@echo "You can now run: make run"

# =============================================================================
# SIMULATOR COMMANDS
# =============================================================================

# Run mixed workload (SCENARIOS_DIR for custom definitions folder). Requires SEMCONV.
run: check-venv
	@if [ -z "$(SEMCONV)" ]; then \
		echo "✗ SEMCONV is required for make run."; \
		echo "  Example:"; \
		echo "    SEMCONV=/path/to/otel-semantic-conventions.yaml make run"; \
		exit 1; \
	fi
	@echo "🚀 Starting telemetry simulator..."
	@echo "   Endpoint: $(OTLP_ENDPOINT)"
	@echo "   Schema:   $(SEMCONV)"
	@echo ""
	@echo "💡 Press Ctrl+C to stop"
	@echo ""
	$(VENV_PYTHON) -m simulator.cli --endpoint $(OTLP_ENDPOINT) run \
		--semconv $(SEMCONV) \
		$(if $(SCENARIOS_DIR),--scenarios-dir $(SCENARIOS_DIR),)

# Validate schema
run-validate: check-venv
	$(VENV_PYTHON) -m simulator.cli validate \
		$(if $(SEMCONV),--semconv $(SEMCONV),) \
		--show-schema \
		--show-spans \
		--show-metrics

# List available scenarios (use SCENARIOS_DIR for custom definitions folder)
list-scenarios: check-venv
	$(VENV_PYTHON) -m simulator.cli list $(if $(SCENARIOS_DIR),--scenarios-dir $(SCENARIOS_DIR),)

# =============================================================================
# LIVE TRACE VISUALIZATION (Jaeger)
# =============================================================================
# Start Jaeger all-in-one with OTLP receiver. Simulator sends to localhost:4318.
# Open http://localhost:16686 to view traces. See docs/live-trace-visualization.md.
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
		echo "  UI: http://localhost:16686  OTLP: http://localhost:4318"; \
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
			echo "✓ Jaeger started. UI: http://localhost:16686  OTLP: http://localhost:4318"; \
			echo "  Run: make run then open the UI."; \
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
	rm -rf *.pyc
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
