# Telemetry Simulator

Schema-driven OpenTelemetry telemetry simulator for LLM observability. Generates realistic traces, metrics, and logs aligned with configurable semantic conventions.

## Overview

The telemetry simulator produces OTEL-compliant telemetry for testing and validating observability pipelines:

- **Schema-Driven**: Reads your semantic-conventions YAML (path required); the schema defines which attributes exist per span type (types, allowed values, defaults). Scenario YAML overrides or supplies distribution-based values.
- **Realism + Randomness + SemConv**: Combines realistic scenario content (conversation samples, failure modes like 4xx or wrong division) with controlled randomness (latency distributions, which scenario/sample runs). All enum-like values (e.g. `error.type`, `step.outcome`) are chosen only from semantic-convention allowed values so traces stay valid and queryable. See [Generating Telemetry](docs/generating-telemetry.md#realism-randomness-and-semantic-conventions).
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
├── exporters/         # OTLP, file, and console exporters
└── cli.py             # Command-line interface
```

## Quick Start

### Prerequisites

- Python 3.11+
- **Schema path**: Set `SEMCONV` or pass `--semconv` (before or after the subcommand, e.g. `otelsim scenario --name foo --semconv /path/to/conventions.yaml`).
- OpenTelemetry Collector running (port 4318) or use `--output-file` to export to file
- **Tenant ID** is read from `src/simulator/scenarios/config/config.yaml` (`tenant.id`); no env vars required for tenants.

### Setup

```bash
# From the otelsim project directory
make venv
make install
```

### Run Mixed Workload

```bash
# Run mixed workload (uses simulator defaults for count/interval)
otelsim run --semconv /path/to/semconv.yaml

# Quick run with Gentoro vendor prefix
otelsim run --vendor=gentoro --count 100 --interval 50
```

### Run Specific Scenarios

The simulator ships with **sample scenario definitions** in `src/simulator/scenarios/definitions/`. You can use these as-is or point to your own folder with `--scenarios-dir` (or `SCENARIOS_DIR` with make).

```bash
# List available scenarios (uses sample definitions by default)
make list-scenarios

# Run a sample scenario
otelsim scenario --name new_claim_phone

# Use a custom folder of scenario YAML files
otelsim scenario --name my_scenario --scenarios-dir /path/to/my/definitions --semconv /path/to/semconv.yaml
```

**Note:** Tenant IDs come from `config/config.yaml` (`tenant.id`) or from the scenario YAML `context.tenant_uuid`. Use `otelsim run` for a mixed workload (varied trace patterns); use `otelsim scenario --name <name>` for a single reproducible pattern.

### Live trace visualization

To view traces in a browser while running the simulator on the host:

1. Start Jaeger: `make jaeger-up`
2. Run the simulator: `otelsim run --semconv /path/to/semconv.yaml`
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

**Sample definitions** are bundled in `src/simulator/scenarios/definitions/` (e.g. `new_claim_phone.yaml`, `request_blocked_by_policy.yaml`, `request_error_policy_runtime.yaml`, `new_claim_phone_multi_turn.yaml`). A reference scenario `example_scenario.yaml` documents all YAML options; it is excluded from `list` and mixed workload when using the built-in samples but can be run with `--name example_scenario`. You can run samples as-is, add your own YAML there, or use a **custom definitions folder** via `--scenarios-dir` (see CLI Reference). Example structure:

```yaml
name: my_scenario
description: Custom test scenario

# Tenant: use tenant.id from config/config.yaml, or set context.tenant_uuid in scenario YAML.

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

### Adding new scenarios (no code changes)

You can add new scenarios by **adding YAML files only** (no code changes):

1. **Data-plane (a2a.orchestrate) scenarios** – Use **key-based context**: `context.tenant`, `context.agent`, `context.mcp_server`. Define data-plane in the scenario YAML with `data_plane.workflow` (key into config `workflow_templates` for step list), optional `data_plane.simulation_goal`, and optional `data_plane.control_plane_template`. Config supplies only key → UUID and workflow step names.
2. **Control-plane-only (e.g. blocked/error) scenarios** – Set `control_plane.request_outcome` and `control_plane.block_reason`, or `control_plane.template`. Use **minimal context**: `context.tenant` and `context.agent` only (no `mcp_server`, `workflow`, `correct_flow`, or `error_pattern`). Only the incoming request-validation trace is emitted; no data-plane or response-validation. To use a template but override the policy-span exception (e.g. for a variant), set `control_plane.policy_exception: { type: "...", message: "..." }` in the scenario YAML; the template’s default exception is overridden by these values.
3. **New control-plane outcome/template** – Add entries under `control_plane.request_validation_templates` and, if needed, `control_plane.trace_flow` in `config/config.yaml`. Attribute values should follow `scenarios/conventions/semconv.yaml`.
4. **New data-plane workflow** – Add a workflow to `realistic_scenarios.workflow_templates` in `config/config.yaml` (workflow name → list of steps); then in a scenario YAML set `data_plane.workflow`, optional `data_plane.simulation_goal`, and optional `data_plane.control_plane_template`.
5. **Tags** – In scenario YAML, set `tags: [control-plane]`, `tags: [data-plane, happy-path]`, etc. Then run a subset of scenarios with `otelsim run --tags=control-plane` or `--tags=data-plane,multi-turn` (scenarios that have *at least one* of the given tags are included). Use `--each-once` to run each (tagged) scenario exactly once instead of `--count` random picks.

The simulator resolves tenant/agent/MCP IDs from config; data-plane behavior is defined per scenario (workflow, simulation_goal, control_plane_template), and control-plane behavior from config templates.

## CLI Reference

```bash
# Run mixed workload (all scenarios)
otelsim run --count 100 --interval 500

# Run only scenarios with a given tag (e.g. control-plane or data-plane)
otelsim run --count 50 --tags=control-plane
otelsim run --count 50 --tags=data-plane,multi-turn
otelsim run --each-once                       # each scenario once
otelsim run --tags=control-plane --each-once  # each tagged scenario once

# Run YAML scenario (sample or custom definitions folder)
otelsim scenario --name new_claim_phone
otelsim scenario --name my_scenario --scenarios-dir /path/to/definitions

# List scenarios (from sample or custom folder)
otelsim list
otelsim list --scenarios-dir /path/to/definitions

# Validate schema and show span/metric definitions
otelsim validate --show-schema --show-spans --show-metrics

# Show each generated trace (trace_id, tenant_id, span names)
otelsim run --count 10 --show-spans
otelsim run --show-all-attributes

# Print complete span content (name, trace_id, span_id, kind, status, all attributes) for every span
otelsim run --count 5 --show-full-spans
```

### Configuration (environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `VENDOR` | `vendor` | Vendor prefix for span names and attributes (e.g. `vendor` → `vendor.a2a.orchestrate`, `vendor.session.id`). Set to your vendor name (e.g. `acme`) so no specific vendor is hardcoded. |
| `TELEMETRY_SIMULATOR_VENDOR_NAME` | *(capitalized prefix)* | Display name used in validation messages. |
| `SEMCONV` | *(optional)* | Full path to your semantic-conventions YAML. Default: `scenarios/conventions/semconv.yaml` (or pass `--semconv`). |

Resource attributes (`service.name`, `service.version`, `service.instance.id`, `deployment.environment.name`, `{prefix}.module`, `{prefix}.component`, `{prefix}.otel.source`) and resource `schemaUrl` are read from `src/simulator/scenarios/config/resource.yaml`. Configure them there; env vars are not used for these.

The schema defines which attributes exist and their types; scenario `attributes` override those values or supply distribution-based values. Use the same prefix as `VENDOR` (e.g. `vendor.turn.status.code` when prefix is `vendor`). Bundled scenarios use the default `vendor.*` namespace.

All spans/metrics/logs are emitted with resource attributes per the OTEL resource spec: required `service.name`, `service.version`, `{prefix}.module`, `{prefix}.component`, `{prefix}.tenant.id`, `{prefix}.otel.source`, plus optional `service.instance.id` and `deployment.environment.name`. The resource `schemaUrl` is set for schema-aware ingestion.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:4318` | OTLP HTTP endpoint |
| `--semconv` | *(optional)* | Path to your semantic-conventions YAML (default: scenarios/conventions/semconv.yaml; or set `SEMCONV`) |
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

The simulator validates all telemetry against the schema file you provide (via `SEMCONV`, `--semconv`, or the default `scenarios/conventions/semconv.yaml`):

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
otelsim run --semconv /path/to/semconv.yaml
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
otelsim run

# Or pass per run (any of these work):
otelsim run --semconv /path/to/your-schema.yaml
otelsim scenario --name new_claim_phone --semconv /path/to/your-schema.yaml
```

### Virtual Environment Issues

Recreate venv:
```bash
make venv-force
make install
```

## See Also

- [Generating Telemetry](docs/generating-telemetry.md) – Guide, realism/randomness/SemConv, and scenario examples

When using this simulator inside another repo, provide your schema YAML path (`SEMCONV`, `--semconv`, or default `scenarios/conventions/semconv.yaml`) and set `VENDOR` to your project’s attribute namespace.
