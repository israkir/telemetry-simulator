# Telemetry Simulator

Schema-driven OpenTelemetry telemetry simulator for LLM observability. Generates traces, metrics, and logs aligned with configurable semantic conventions.

## Overview

The telemetry simulator produces OTEL-compliant telemetry for testing and validating observability pipelines:

- **Schema-Driven**: Reads your semantic-conventions YAML (path required); the schema defines which attributes exist per span type (types, allowed values, defaults). Scenario YAML overrides or supplies distribution-based values.
- **Scenario + Randomness + SemConv**: Combines scenario-driven content (conversation samples, failure modes like 4xx or wrong division) with controlled randomness (latency distributions, which scenario/sample runs). All enum-like values (e.g. `error.type`, `step.outcome`) are chosen only from semantic-convention allowed values so traces stay valid and queryable. See [Generating Telemetry](docs/generating-telemetry.md#realism-randomness-and-semantic-conventions) (scenario content, randomness, SemConv).
- **Full Trace Hierarchies**: Generates canonical span types (e.g. a2a.orchestrate, planner, task.execute, llm.call, mcp.tool.execute, response.compose) with proper parent-child relationships and `{prefix}.span.class` per type
- **Multi-Signal**: Emits correlated traces, metrics, and logs
- **Scenario-Based**: YAML-defined scenarios for reproducible testing
- **OTEL Compliant**: Validates against OTEL GenAI semantic conventions
- **Vendor-Agnostic**: Attribute namespace is configurable via `VENDOR` so any project can use its own convention (e.g. `vendor`, `acme`)

## Architecture

```
src/simulator/
├── schemas/           # Schema parser and attribute generator
├── generators/        # Trace, metric, and log generators
├── scenarios/         # YAML scenario loader and runner
├── validators/        # OTEL payload validator
├── exporters/         # OTLP and file exporters
└── cli.py             # Command-line interface
```

## Quick Start

### Prerequisites

- Python 3.11+
- **Schema path**: Set `SEMCONV` or pass `--semconv` (before or after the subcommand, e.g. `otelsim scenario --vendor=your_vendor --name foo --semconv /path/to/conventions.yaml`).
- OpenTelemetry Collector running (port 4318) or use `--output-file` to export to file
- **Tenant ID** is read from `resource/config/config.yaml` (`tenants`; scenarios reference `context.tenant` by key, e.g. `toro`); no env vars required for tenants.

### Setup

```bash
# From the otelsim project directory
make venv
make install
```

### Run Mixed Workload

```bash
# Run mixed workload (uses simulator defaults for count/interval)
otelsim run --vendor=your_vendor --semconv /path/to/semconv.yaml

# Quick run
otelsim run --vendor=your_vendor --count 100 --interval 50
```

### Run Specific Scenarios

The simulator ships with **sample scenario definitions** in `resource/scenarios/definitions/`. You can use these as-is or point to your own folder with `--scenarios-dir` (or `SCENARIOS_DIR` with make).

```bash
# List available scenarios (uses sample definitions by default)
make list-scenarios

# Run a sample scenario
otelsim scenario --vendor=your_vendor --name new_claim_phone

# Use a custom folder of scenario YAML files
otelsim scenario --vendor=your_vendor --name my_scenario --scenarios-dir /path/to/my/definitions --semconv /path/to/semconv.yaml
```

**Note:** Tenant IDs come from `resource/config/config.yaml` (`tenants`; scenario sets `context.tenant` by key, e.g. `toro`). Use `otelsim run --vendor=your_vendor` for a mixed workload (varied trace patterns); use `otelsim scenario --vendor=your_vendor --name <name>` for a single reproducible pattern.

### Live trace visualization

To view traces in a browser while running the simulator on the host:

1. Start Jaeger: `make jaeger-up`
2. Run the simulator: `otelsim run --vendor=your_vendor --semconv /path/to/semconv.yaml`
3. Open **http://localhost:16686** and select service `otelsim`
4. Stop Jaeger when done: `make jaeger-down`

See [docs/live-trace-visualization.md](docs/live-trace-visualization.md) for details.

## Span Types

The simulator emits spans using the **configurable vendor prefix** (`VENDOR`, default `vendor`). Span names are `{prefix}.a2a.orchestrate`, `{prefix}.planner`, etc. Use the same prefix in scenario YAML for `type` (e.g. `vendor.a2a.orchestrate`).

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

Set `VENDOR` to your vendor name (e.g. `acme`) so span names and vendor attributes use your namespace.

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

Scenarios are YAML files in `resource/scenarios/definitions/` (or a custom folder via `--scenarios-dir`). Bundled samples include data-plane flows (e.g. `new_claim_phone`, `new_claim_phone_mcp_tool_retry_then_success`) and control-plane outcomes (e.g. `request_blocked_by_policy`, `request_allowed_audit_flagged`). Use `make list-scenarios` or `otelsim list` to see names.

For **YAML structure**, **adding new scenarios**, and **tags/workflow/mcp_retry**, see [Scenario YAML Reference](docs/scenario-yaml-reference.md). For config (tenants, workflows, latency, MCP retry templates), see [Generating Telemetry](docs/generating-telemetry.md#configuration-reference-what-exists).

## CLI Reference

```bash
# Run mixed workload (all scenarios)
otelsim run --vendor=your_vendor --count 100 --interval 500

# Run only scenarios with a given tag (e.g. control-plane or data-plane)
otelsim run --vendor=your_vendor --count 50 --tags=control-plane
otelsim run --vendor=your_vendor --count 50 --tags=data-plane,multi-turn
otelsim run --vendor=your_vendor --each-once                       # each scenario once
otelsim run --vendor=your_vendor --tags=control-plane --each-once  # each tagged scenario once

# Run YAML scenario (sample or custom definitions folder)
otelsim scenario --vendor=your_vendor --name new_claim_phone
otelsim scenario --vendor=your_vendor --name my_scenario --scenarios-dir /path/to/definitions

# List scenarios (from sample or custom folder)
otelsim list
otelsim list --scenarios-dir /path/to/definitions

# Validate schema and show span/metric definitions
otelsim validate --show-schema --show-spans --show-metrics

# Show each generated trace (trace_id, tenant_id, span names)
otelsim run --vendor=your_vendor --count 10 --show-spans
otelsim run --vendor=your_vendor --show-all-attributes

# Print complete span content (name, trace_id, span_id, kind, status, all attributes) for every span
otelsim run --vendor=your_vendor --count 5 --show-full-spans
```

### Configuration (environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `VENDOR` | `vendor` | Vendor prefix for span names and attributes (e.g. `vendor` → `vendor.a2a.orchestrate`, `vendor.session.id`). Set to your vendor name (e.g. `acme`) so no specific vendor is hardcoded. |
| `TELEMETRY_SIMULATOR_VENDOR_NAME` | *(capitalized prefix)* | Display name used in validation messages. |
| `SEMCONV` | *(optional)* | Full path to your semantic-conventions YAML. Default: `resource/scenarios/conventions/semconv.yaml` (or pass `--semconv`). |

Resource attributes (`service.name`, `service.version`, `service.instance.id`, `deployment.environment.name`, `{prefix}.module`, `{prefix}.component`, `{prefix}.otel.source`) and resource `schemaUrl` are read from `resource/config/resource.yaml`. Configure them there; env vars are not used for these.

The schema defines which attributes exist and their types; scenario `attributes` override those values or supply distribution-based values. Use the same prefix as `VENDOR` (e.g. `vendor.turn.status.code` when prefix is `vendor`). Bundled scenarios use the default `vendor.*` namespace.

All spans/metrics/logs are emitted with resource attributes per the OTEL resource spec: required `service.name`, `service.version`, `{prefix}.module`, `{prefix}.component`, `{prefix}.tenant.id`, `{prefix}.otel.source`, plus optional `service.instance.id` and `deployment.environment.name`. The resource `schemaUrl` is set for schema-aware ingestion.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:4318` | OTLP HTTP endpoint |
| `--semconv` | *(optional)* | Path to your semantic-conventions YAML (default: resource/scenarios/conventions/semconv.yaml; or set `SEMCONV`) |
| `--count` | `100` | Number of traces |
| `--interval` | `500` | Interval in ms |
| `--output-file` | None | Export to file instead of OTLP |
| `--no-metrics` | False | Disable metric generation |
| `--no-logs` | False | Disable log generation |
| `--show-spans` | False | Print each trace (trace_id, tenant_id, span names) |
| `--show-all-attributes` | False | Same as --show-spans; used by default in container |
| `--show-full-spans` | False | Print full span content (name, ids, kind, status, all attributes) for every span |
| `--scenarios-dir` | *(built-in samples)* | Folder with scenario YAML files (for `run`, `scenario`, `list`) |
| `--tags` | None | Comma-separated tags; only run scenarios that have at least one of these tags (e.g. `--tags=control-plane`, `--tags=data-plane,multi-turn`) |
| `--each-once` | False | Run each (tagged) scenario exactly once instead of `--count` random picks; combine with `--tags` to run only tagged scenarios once |

## Schema Validation

The simulator validates all telemetry against the schema file you provide (via `SEMCONV`, `--semconv`, or the default `resource/scenarios/conventions/semconv.yaml`):

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

*(`{prefix}` is set by `VENDOR`, default `vendor`.)*

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

The simulator can run **as a container** (with your own Docker setup) or **locally** (with venv). Tenant IDs are taken from `config/config.yaml`; use `--show-full-spans` to log full span content.

### Local run (venv)

```bash
make venv && make install
otelsim run --vendor=your_vendor --semconv /path/to/semconv.yaml
```

## Troubleshooting

### Connection Refused

Ensure an OTLP receiver is running on the configured endpoint (default `http://localhost:4318`), or use `--output-file` to export to file instead.

### Span output truncated in container logs

Container log drivers often limit line length (e.g. 16KB). The simulator keeps progress lines compact: trace_id is shown as 16 chars and span names are limited so lines are not cut off. Full trace_id and all span data are still sent to OTLP.

### Schema Not Found

Set the schema path via environment or `--semconv` (before or after the subcommand):
```bash
export SEMCONV=/path/to/your-semconv.yaml
otelsim run --vendor=your_vendor

# Or pass per run (any of these work):
otelsim run --vendor=your_vendor --semconv /path/to/your-schema.yaml
otelsim scenario --vendor=your_vendor --name new_claim_phone --semconv /path/to/your-schema.yaml
```

### Virtual Environment Issues

Recreate venv:
```bash
make venv-force
make install
```

## See Also

- [Generating Telemetry](docs/generating-telemetry.md) – Guide, scenario/randomness/SemConv, and scenario examples

When using this simulator inside another repo, provide your schema YAML path (`SEMCONV`, `--semconv`, or default `resource/scenarios/conventions/semconv.yaml`) and set `VENDOR` to your project’s attribute namespace.
