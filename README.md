# Telemetry Simulator

Schema-driven OpenTelemetry telemetry simulator for LLM observability. Generates realistic traces, metrics, and logs aligned with configurable semantic conventions.

## Overview

The telemetry simulator produces OTEL-compliant telemetry for testing and validating observability pipelines:

- **Schema-Driven**: Reads your semantic-conventions YAML (path required) to ensure attribute compliance
- **Full Trace Hierarchies**: Generates canonical span types (e.g. a2a.orchestrate, planner, task.execute, llm.call, mcp.tool.execute, response.compose) with proper parent-child relationships and `{prefix}.span.class` per type
- **Multi-Signal**: Emits correlated traces, metrics, and logs
- **Scenario-Based**: YAML-defined scenarios for reproducible testing
- **OTEL Compliant**: Validates against OTEL GenAI semantic conventions
- **Vendor-Agnostic**: Attribute namespace is configurable via `TELEMETRY_SIMULATOR_ATTR_PREFIX` so any project can use its own convention (e.g. `vendor`, `acme`)

## Architecture

```
src/simulator/
├── schemas/           # Schema parser and attribute generator
├── generators/        # Trace, metric, and log generators
├── scenarios/         # YAML scenario loader and runner
├── validators/        # OTEL payload validator
├── exporters/         # OTLP, file, and console exporters
└── cli.py             # Command-line interface
```

## Quick Start

### Prerequisites

- Python 3.11+
- **Schema path**: Set `TELEMETRY_SIMULATOR_SCHEMA_PATH` or pass `--schema-path` (before or after the subcommand, e.g. `telemetry-simulator scenario --name foo --schema-path /path/to/conventions.yaml`).
- OpenTelemetry Collector running (port 4318) or use `--output-file` to export to file
- `TENANT_UUID` set in environment (required for local run; single ID or comma-separated for multiple). In container, `TELEMETRY_TENANT_UUIDS` / `TELEMETRY_TENANT_WEIGHTS` are used and default to multi-tenant.

### Setup

```bash
# From the telemetry-simulator project directory
make venv
make install
```

### Run Mixed Workload

```bash
# Default: 100 traces, 500ms interval
make run

# Fast mode: 200ms interval
make run-fast

# Custom count
COUNT=500 make run
```

### Run Specific Scenarios

The simulator ships with **sample scenario definitions** in `src/simulator/scenarios/definitions/`. You can use these as-is or point to your own folder with `--scenarios-dir` (or `SCENARIOS_DIR` with make).

```bash
# List available scenarios (uses sample definitions by default)
make list-scenarios

# Run a sample scenario
SCENARIO=successful_agent_turn make run-scenario

# Use a custom folder of scenario YAML files
telemetry-simulator scenario --name my_scenario --scenarios-dir /path/to/my/definitions
make run-scenario SCENARIO=my_scenario SCENARIOS_DIR=/path/to/my/definitions
```

**Note:** Tenants come from `TENANT_UUID` (required): one ID or comma-separated (e.g. `TENANT_UUID=tenant-a,tenant-b,tenant-c`). Optionally set `TENANT_WEIGHTS=0.5,0.3,0.2` for distribution; if omitted with multiple tenants, a realistic skewed distribution is used. The default `make run` uses a mixed workload (varied trace patterns); for a single reproducible pattern, use a YAML scenario.

### Live trace visualization

To view traces in a browser while running the simulator on the host:

1. Start Jaeger: `make jaeger-up`
2. Run the simulator: `make run` or `SCENARIO=successful_agent_turn make run-scenario`
3. Open **http://localhost:16686** and select service `telemetry-simulator`
4. Stop Jaeger when done: `make jaeger-down`

See [docs/live-trace-visualization.md](docs/live-trace-visualization.md) for details.

## Span Types

The simulator emits spans using the **configurable vendor prefix** (`TELEMETRY_SIMULATOR_ATTR_PREFIX`, default `vendor`). Span names are `{prefix}.a2a.orchestrate`, `{prefix}.planner`, etc. Use the same prefix in scenario YAML for `type` (e.g. `vendor.a2a.orchestrate`).

| Span Type (emitted) | Kind | Description |
|---------------------|------|-------------|
| `{prefix}.a2a.orchestrate` | SERVER | Root span for user interaction turn |
| `{prefix}.planner` | INTERNAL | Planning/tool-selection step |
| `{prefix}.task.execute` | INTERNAL | Per sub-task execution (with `{prefix}.task.id`, `{prefix}.task.type`) |
| `{prefix}.llm.call` | CLIENT | LLM model inference (OTEL GenAI) |
| `{prefix}.mcp.tool.execute` | CLIENT | MCP tool invocation (parent); child `{prefix}.mcp.tool.execute.attempt` per attempt |
| `{prefix}.response.compose` | INTERNAL | Response composition (with `{prefix}.response.format`, `{prefix}.step.outcome`) |
| `rag.retrieve` | INTERNAL | RAG/vector retrieval |
| `a2a.call` | CLIENT | Agent-to-agent delegation |
| `cp.request` | SERVER | Control-plane entry point |

Each vendor-prefixed span sets `{prefix}.span.class` (e.g. `a2a.orchestrate`, `planner`, `task.execute`, `llm.call`, `mcp.tool.execute`, `response.compose`) for convention alignment.

Set `TELEMETRY_SIMULATOR_ATTR_PREFIX` to your vendor name (e.g. `gentoro`, `acme`) so span names and vendor attributes use your namespace.

### Trace Hierarchy Example

```
{prefix}.a2a.orchestrate (root, SERVER)
├── {prefix}.planner (INTERNAL)
│   └── {prefix}.llm.call (CLIENT) - tool selection
├── {prefix}.task.execute (INTERNAL) - optional
├── {prefix}.mcp.tool.execute (CLIENT)
│   └── {prefix}.mcp.tool.execute.attempt (CLIENT) [per attempt]
├── {prefix}.llm.call (CLIENT) - final response
└── {prefix}.response.compose (INTERNAL)
```

## YAML Scenarios

**Sample definitions** are bundled in `src/simulator/scenarios/definitions/` (e.g. `successful_agent_turn.yaml`, `tool_retry.yaml`). A reference scenario `example_scenario.yaml` documents all YAML options; it is excluded from `list` and mixed workload when using the built-in samples but can be run with `--name example_scenario`. You can run samples as-is, add your own YAML there, or use a **custom definitions folder** via `--scenarios-dir` (see CLI Reference). Example structure:

```yaml
name: my_scenario
description: Custom test scenario

# Tenant(s) and weights come from env at runtime:
#   TENANT_UUID=id1,id2,id3
#   TENANT_WEIGHTS=0.5,0.3,0.2  (optional; realistic default if omitted)

repeat_count: 100
interval_ms: 500

emit_metrics: true
emit_logs: true

root:
  type: vendor.a2a.orchestrate
  latency:
    mean_ms: 1500
    variance: 0.3
  error:
    rate: 0.02
  
  children:
    - type: vendor.planner
      latency:
        mean_ms: 300
    
    - type: vendor.mcp.tool.execute
      latency:
        mean_ms: 200
      error:
        rate: 0.05
      attributes:
        gen_ai.tool.name: my_tool
    
    - type: vendor.llm.call
      latency:
        mean_ms: 600
      attributes:
        gen_ai.operation.name: chat
```

## CLI Reference

```bash
# Run mixed workload
telemetry-simulator run --count 100 --interval 500

# Run YAML scenario (sample or custom definitions folder)
telemetry-simulator scenario --name successful_agent_turn --count 50
telemetry-simulator scenario --name my_scenario --scenarios-dir /path/to/definitions --count 50

# Schema path can appear before or after the subcommand
telemetry-simulator scenario --name successful_agent_turn --schema-path /path/to/semantic-conventions.yaml --count 10

# List scenarios (from sample or custom folder)
telemetry-simulator list
telemetry-simulator list --scenarios-dir /path/to/definitions

# Validate schema and show span/metric definitions
telemetry-simulator validate --show-schema --show-spans --show-metrics

# Show each generated trace (trace_id, tenant_id, span names)
telemetry-simulator run --count 10 --show-spans
telemetry-simulator run --show-all-attributes

# Print complete span content (name, trace_id, span_id, kind, status, all attributes) for every span
telemetry-simulator run --count 5 --show-full-spans
```

### Configuration (environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEMETRY_SIMULATOR_ATTR_PREFIX` | `vendor` | Vendor prefix for span names and attributes (e.g. `vendor` → `vendor.a2a.orchestrate`, `vendor.session.id`). Set to your vendor name (e.g. `gentoro`, `acme`) so no vendor is hardcoded. |
| `TELEMETRY_SIMULATOR_VENDOR_NAME` | *(capitalized prefix)* | Display name used in validation messages. |
| `TELEMETRY_SIMULATOR_SCHEMA_PATH` | *(required)* | Full path to your semantic-conventions YAML file. Must be set by the client (or pass `--schema-path`). |
| `TELEMETRY_SIMULATOR_SERVICE_NAME` | `telemetry-simulator` | Resource attribute `service.name`. |
| `TELEMETRY_SIMULATOR_SERVICE_VERSION` | `1.0.0` | Resource attribute `service.version`. |
| `TELEMETRY_SIMULATOR_MODULE` | `data-plane` | Resource attribute `{prefix}.module` (e.g. control-plane). |
| `TELEMETRY_SIMULATOR_COMPONENT` | `simulator` | Resource attribute `{prefix}.component` (e.g. gateway-ext). |
| `TELEMETRY_SIMULATOR_OTEL_SOURCE` | `internal` | Resource attribute `{prefix}.otel.source`; allowed: `internal`, `propagated`. |
| `TELEMETRY_SIMULATOR_RESOURCE_SCHEMA_URL` | *(Gentoro schema URL)* | OTEL resource `schemaUrl` for ingestion. |
| `SERVICE_INSTANCE_ID` | — | Optional resource attribute `service.instance.id` (e.g. pod/container ID). |
| `DEPLOYMENT_ENVIRONMENT` | — | Optional resource attribute `deployment.environment.name` (e.g. prod, staging). Fallback: `TELEMETRY_SIMULATOR_DEPLOYMENT_ENVIRONMENT` (default: development). |

Scenario YAML attribute overrides should use the same prefix as `TELEMETRY_SIMULATOR_ATTR_PREFIX` (e.g. `vendor.turn.status.code` when prefix is `vendor`). Bundled scenarios use the default `vendor.*` namespace.

All spans/metrics/logs are emitted with resource attributes per the OTEL resource spec: required `service.name`, `service.version`, `{prefix}.module`, `{prefix}.component`, `{prefix}.tenant.id`, `{prefix}.otel.source`, plus optional `service.instance.id` and `deployment.environment.name`. The resource `schemaUrl` is set for schema-aware ingestion.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:4318` | OTLP HTTP endpoint |
| `--schema-path` | *(required)* | Path to your semantic-conventions YAML (or set `TELEMETRY_SIMULATOR_SCHEMA_PATH`) |
| `--count` | `100` | Number of traces |
| `--interval` | `500` | Interval in ms |
| `--tenants` | From `TENANT_UUID` env | Tenant IDs (comma-separated in env); distribution from `TENANT_WEIGHTS` or realistic default |
| `--output-file` | None | Export to file instead of OTLP |
| `--no-metrics` | False | Disable metric generation |
| `--no-logs` | False | Disable log generation |
| `--show-spans` | False | Print each trace (trace_id, tenant_id, span names) |
| `--show-all-attributes` | False | Same as --show-spans; used by default in container |
| `--show-full-spans` | False | Print full span content (name, ids, kind, status, all attributes) for every span |
| `--scenarios-dir` | *(built-in samples)* | Folder with scenario YAML files (for `run`, `scenario`, `list`) |

## Schema Validation

The simulator validates all telemetry against the schema file you provide (via `TELEMETRY_SIMULATOR_SCHEMA_PATH` or `--schema-path`):

```bash
# Show schema summary
make run-validate

# Programmatic validation
from simulator.validators import OtelValidator

validator = OtelValidator()
result = validator.validate_span("vendor.mcp.tool.execute", attributes)  # use your prefix
print(result)  # Shows errors and warnings
```

### Validation Checks

- Required attributes present
- Attribute types match schema
- Enum values valid
- OTEL GenAI convention compliance
- Vendor-specific conventions (attribute prefix from config)

## Metrics

The simulator emits canonical metrics from the schema:

| Metric | Type | Description |
|--------|------|-------------|
| `{prefix}.turn.count` | Counter | Agent turn count |
| `{prefix}.turn.duration_ms` | Histogram | Turn duration |
| `{prefix}.tool.count` | Counter | MCP tool calls |
| `{prefix}.tool.latency_ms` | Histogram | Tool latency |
| `{prefix}.llm.count` | Counter | LLM inference calls |
| `{prefix}.llm.tokens.input` | Counter | Input tokens |
| `{prefix}.llm.tokens.output` | Counter | Output tokens |
| `{prefix}.rag.count` | Counter | RAG retrievals |
| `{prefix}.a2a.count` | Counter | A2A calls |
| `{prefix}.cp.request.count` | Counter | CP requests |

*(`{prefix}` is set by `TELEMETRY_SIMULATOR_ATTR_PREFIX`, default `vendor`.)*

## Logs

Correlated log records are emitted for each operation:

- Turn start/end events
- Tool call results
- LLM inference completion
- RAG retrieval results
- A2A call results
- Control-plane decisions
- Safety events

Logs include `trace_id` and `span_id` for correlation.

## Development

```bash
# Install dev dependencies
make install-dev

# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Run all checks (format, lint, typecheck)
make check
```

## Pipeline Integration

The simulator can run **as a container** (with your own Docker setup) or **locally** (with venv). When running in a container you can use **multi-tenant** via `TELEMETRY_TENANT_UUIDS` / `TELEMETRY_TENANT_WEIGHTS` and `--show-full-spans` to log full span content.

### Local run (venv)

```bash
export TENANT_UUID=dev-tenant-1
make venv && make install
make run
```

## Troubleshooting

### Connection Refused

Ensure an OTLP receiver is running on the configured endpoint (default `http://localhost:4318`), or use `--output-file` to export to file instead.

### TENANT_UUID or TELEMETRY_TENANT_UUIDS required

If you see "TENANT_UUID or TELEMETRY_TENANT_UUIDS must be set", export tenant IDs before running:
```bash
export TENANT_UUID=dev-tenant-1
make run
```

### Span output truncated in container logs

Container log drivers often limit line length (e.g. 16KB). The simulator keeps progress lines compact: trace_id is shown as 16 chars and span names are limited so lines are not cut off. Full trace_id and all span data are still sent to OTLP.

### Schema Not Found

Set the schema path via environment or `--schema-path` (before or after the subcommand):
```bash
export TELEMETRY_SIMULATOR_SCHEMA_PATH=/path/to/your-semantic-conventions.yaml
telemetry-simulator run

# Or pass per run (any of these work):
telemetry-simulator run --schema-path /path/to/your-schema.yaml
telemetry-simulator scenario --name successful_agent_turn --schema-path /path/to/your-schema.yaml
```

### Virtual Environment Issues

Recreate venv:
```bash
make venv-force
make install
```

## See Also

- [Generating Telemetry](docs/generating-telemetry.md) – Guide and happy path example
- [TROUBLESHOOTING](TROUBLESHOOTING.md)

When using this simulator inside another repo, provide your schema YAML path (`TELEMETRY_SIMULATOR_SCHEMA_PATH` or `--schema-path`) and set `TELEMETRY_SIMULATOR_ATTR_PREFIX` to your project’s attribute namespace.
