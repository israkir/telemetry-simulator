# telemetry-simulator

Schema-driven OpenTelemetry telemetry simulator for LLM observability.

## Quick start

### 1) Requirements

- Python 3.11+

### 2) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 3) Run

```bash
# Show available commands
otelsim --help

# List available scenarios
otelsim list

# Run one scenario by name
otelsim scenario --name single_tool_call

# Run mixed workload continuously (Ctrl+C to stop)
otelsim run --interval 500
```

## Common options

- `--endpoint`: OTLP endpoint (default: `http://localhost:4318`)
- `--service-name`: service name to emit (default: `otelsim`)
- `--no-metrics`: disable metrics
- `--no-logs`: disable logs
- `--output-file <path>`: write output to a file instead of OTLP

Examples:

```bash
# Send to a custom OTLP endpoint
otelsim run --endpoint http://localhost:4318 --count 50

# Write generated traces to a local file
otelsim scenario --name single_tool_call --output-file traces.jsonl
```

## Validate configuration

```bash
otelsim validate --show-schema
```

## Project structure

- `src/simulator/`: simulator source code
- `resource/config/`: tenant and tool payload config
- `resource/scenarios/`: scenario definitions
