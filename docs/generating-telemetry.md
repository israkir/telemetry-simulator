# Generating Telemetry Data

This guide describes how to generate realistic OpenTelemetry data with the telemetry simulator. Use this for pipeline testing, dashboard validation, and load testing.

## Overview

The simulator produces OTEL-compliant **traces**, **metrics**, and **logs** with configurable semantic conventions (attribute prefix set via `TELEMETRY_SIMULATOR_ATTR_PREFIX`). Data is defined by **YAML scenarios**: each scenario describes a trace hierarchy with realistic latency distributions, error propagation, and probabilistic behavior.

- **Traces**: A tree of spans using the configured vendor prefix (e.g. `{prefix}.a2a.orchestrate` → `{prefix}.planner`, `{prefix}.mcp.tool.execute`, `{prefix}.llm.call`)
- **Metrics**: Correlated with spans (e.g. turn duration, tool call counts)
- **Logs**: Emitted for span events where configured

Tenant IDs and weights come from environment variables (`TENANT_UUID` or `TELEMETRY_TENANT_UUIDS` / `TELEMETRY_TENANT_WEIGHTS`). No hardcoded tenant IDs.

---

## Quick Start

### Prerequisites

- Python 3.11+
- **Schema path**: Set `TELEMETRY_SIMULATOR_SCHEMA_PATH` or pass `--schema-path` (before or after the subcommand)
- OTLP endpoint (e.g. data-plane collector on port 4318)
- `TENANT_UUID` set (or `TELEMETRY_TENANT_UUIDS` in container)

### Run a Scenario

By default the simulator uses **sample scenario definitions** bundled in `src/simulator/scenarios/definitions/`. You can also pass a custom folder with `--scenarios-dir`.

```bash
# From the telemetry-simulator project directory
make venv && make install

export TENANT_UUID=dev-tenant-1

# Run sample scenarios (built-in definitions)
telemetry-simulator scenario --name successful_agent_turn --count 100
telemetry-simulator scenario --name tool_retry --count 50

# With schema path (option can appear before or after the subcommand)
telemetry-simulator scenario --name successful_agent_turn --count 5 --schema-path /path/to/otel-semantic-conventions.yaml

# Use your own scenario folder
telemetry-simulator scenario --name my_scenario --scenarios-dir /path/to/my/definitions --count 100

# Show full span output
telemetry-simulator scenario --name tool_retry --count 10 --show-full-spans
```

### Mixed Workload

Run all scenarios in random mix:

```bash
telemetry-simulator run --count 500 --interval 200
```

---

## Example Scenarios

### Happy Path

Models a successful agent turn: user asks something, agent plans, calls a tool, returns an answer.

| Step | Span type (emitted) | Role |
|------|---------------------|------|
| 1    | `{prefix}.a2a.orchestrate` | Root; one user turn |
| 2    | `{prefix}.planner` | Planning / tool selection |
| 3    | `{prefix}.llm.call` | LLM call for tool selection |
| 4    | `{prefix}.mcp.tool.execute` + `{prefix}.mcp.tool.execute.attempt` | MCP tool call |
| 5    | `{prefix}.llm.call` | LLM call for final answer |
| 6    | `{prefix}.response.compose` | Response composition (direct child of root) |

```bash
telemetry-simulator scenario --name successful_agent_turn --count 100
```

### Tool Retry

Models a tool call that fails initially and succeeds after retry, with:
- Log-normal latency distributions
- Forced initial failure with exponential backoff
- Error propagation between spans
- Retry-specific attributes (`retry.attempt`, `error.type`)

```bash
telemetry-simulator scenario --name tool_retry --count 50 --show-full-spans
```

---

## Scenario Configuration

Scenarios support realistic generation features:

| Feature | Description |
|---------|-------------|
| **Log-normal latencies** | Right-skewed distributions matching real-world behavior |
| **Probabilistic spans** | Not all traces have all span types |
| **Count distributions** | Variable number of tool calls (e.g. Poisson) |
| **Error propagation** | Correlated failures through hierarchies |
| **Retry sequences** | Multi-attempt operations with backoff |
| **Attribute distributions** | Sample values from distributions |

### Example: Latency Distribution

```yaml
latency:
  distribution: log_normal
  median_ms: 200
  sigma: 0.8    # 0.5=tight, 0.8=moderate, 1.2=heavy tail
```

### Example: Probabilistic Span

```yaml
children:
  - type: rag.retrieve
    probability: 0.3    # Only 30% of traces include RAG
```

### Example: Retry Behavior

```yaml
retry:
  enabled: true
  max_attempts: 3
  force_initial_failure: true
  success_rate_per_attempt: [0.0, 0.75, 0.92]
```

See the [Scenario Configuration Reference](./statistical-scenarios.md) for complete documentation.

---

## Scenario File Structure

Scenarios are YAML files. The simulator ships with **sample definitions** in `src/simulator/scenarios/definitions/` (e.g. `successful_agent_turn.yaml`, `tool_retry.yaml`). A reference file `example_scenario.yaml` documents all configuration options; it is excluded from `list` and from mixed workload when using the built-in samples, but you can run it explicitly with `--name example_scenario`. You can add your own YAML in the sample folder or use a **custom folder** via `--scenarios-dir` (or `SCENARIOS_DIR` with make). Example:

```yaml
name: my_scenario
description: Description of what this scenario tests

tags:
  - baseline
  - success

repeat_count: 100      # Number of traces to generate
interval_ms: 500       # Delay between traces

emit_metrics: true
emit_logs: true

root:
  type: vendor.a2a.orchestrate
  latency:
    distribution: log_normal
    median_ms: 1500
    sigma: 0.4
  error:
    rate: 0.02
  attributes:
    vendor.turn.status.code: SUCCESS

  children:
    - type: vendor.planner
      # ... nested spans
```

### Supported Span Types

Use `type` in scenario YAML with the same prefix as `TELEMETRY_SIMULATOR_ATTR_PREFIX` (default `vendor`):

| Span type | Description |
|-----------|-------------|
| `{prefix}.a2a.orchestrate` | Root span for user turn |
| `{prefix}.planner` | Planning / tool selection |
| `{prefix}.task.execute` | Per sub-task (with `{prefix}.task.id`, `{prefix}.task.type`) |
| `{prefix}.llm.call` | LLM inference call |
| `{prefix}.mcp.tool.execute` | MCP tool (parent; child `{prefix}.mcp.tool.execute.attempt` per attempt) |
| `{prefix}.response.compose` | Response composition (with `{prefix}.response.format`, `{prefix}.step.outcome`) |
| `rag.retrieve` | RAG retrieval |
| `a2a.call` | Agent-to-agent call |
| `cp.request` | Control plane request |

Sample scenario YAML uses the default `vendor.*` namespace; the loader normalizes these to `TELEMETRY_SIMULATOR_ATTR_PREFIX` (e.g. `gentoro`) at runtime.

---

## Running Scenarios

### CLI Commands

Global options (`--schema-path`, `--endpoint`, `--service-name`) can appear before or after the subcommand.

```bash
# Run specific scenario
telemetry-simulator scenario --name successful_agent_turn --count 100

# With schema path (required unless TELEMETRY_SIMULATOR_SCHEMA_PATH is set)
telemetry-simulator scenario --name successful_agent_turn --count 50 --schema-path /path/to/semantic-conventions.yaml

# Show spans as they're generated
telemetry-simulator scenario --name tool_retry --count 10 --show-spans

# Show full span details (all attributes)
telemetry-simulator scenario --name tool_retry --count 5 --show-full-spans

# Run mixed workload (all scenarios from default or custom folder)
telemetry-simulator run --count 500 --interval 200
telemetry-simulator run --count 500 --scenarios-dir /path/to/definitions
```

### Container

```bash
# From repo root
./tools/dev/dev deps-up

# Run in container
podman run --rm \
  -e TELEMETRY_TENANT_UUIDS=dev-tenant-1 \
  -e OTLP_HTTP_ENDPOINT=http://data-plane:4318 \
  your-image telemetry-simulator scenario --name successful_agent_turn --count 200
```

### Make Targets

```bash
make run-scenario SCENARIO=successful_agent_turn
make run              # Mixed workload
```

---

## Adding New Scenarios

1. Create a YAML file either in the **sample definitions** folder (`src/simulator/scenarios/definitions/`) or in your own folder (then use `--scenarios-dir`). Example: `with_rag.yaml`

2. Define the scenario structure:
   ```yaml
   name: with_rag
   description: Agent turn with RAG retrieval
   
   repeat_count: 100
   interval_ms: 500
   
   root:
     type: vendor.a2a.orchestrate
     # ... your span hierarchy
   ```

3. Run it (use `--scenarios-dir` if your file is not in the sample definitions folder):
   ```bash
   telemetry-simulator scenario --name with_rag --count 50
   telemetry-simulator scenario --name with_rag --scenarios-dir /path/to/definitions --count 50
   ```

4. It will automatically be included in mixed workloads when you use the same folder (default or `--scenarios-dir`).

---

## What You Get

- **Traces**: One trace per repeat, with the defined hierarchy. Each span has realistic timing, status, and attributes.
- **Metrics**: Turn duration, tool call counts, LLM token usage, etc.
- **Logs**: Log records tied to spans where logging is enabled.

Data is sent to the configured OTLP endpoint (e.g. data-plane collector), which can export to Kafka, ClickHouse, or other backends.

---

## Live trace visualization

To view traces in a browser, run Jaeger and point the simulator at it (no data-plane container needed):

```bash
make jaeger-up
make run
# Open http://localhost:16686, select service "telemetry-simulator"
make jaeger-down   # when done
```

See [Live Trace Visualization](./live-trace-visualization.md) for the full flow and options.

---

## See Also

- [Scenario Configuration Reference](./statistical-scenarios.md) - Complete YAML reference with distributions, error propagation, retries
- [Live Trace Visualization](./live-trace-visualization.md) - View traces in Jaeger UI
- [README](../README.md) - Quick start and CLI reference
- [TROUBLESHOOTING](../TROUBLESHOOTING.md) - Common issues
