# Scenario YAML Reference

This document describes how to define and extend scenarios with YAML. For config (tenants, workflows, latency, MCP retry templates), see [Generating Telemetry](generating-telemetry.md#configuration-reference-what-exists). For running scenarios and mixed workloads, see [README](../README.md#quick-start).

## Sample definitions

Scenarios are YAML files. The simulator bundles **sample definitions** in `resource/scenarios/definitions/`. For the full list by category (happy path, multi-turn, retries, higher-latency, 4xx, agent-confusion, control-plane), see [resource/scenarios/definitions/README.md](../resource/scenarios/definitions/README.md).

Representative samples: **Data-plane** — `new_claim_phone.yaml`, `new_claim_phone_multi_turn.yaml`, `new_claim_phone_mcp_tool_retry_then_success.yaml`, `cancel_claim_appliances.yaml`, `*_higher_latency*.yaml`, `*_tool_4xx_invalid_params.yaml`, `agent_confusion_*.yaml`. **Control-plane** — `request_blocked_by_policy.yaml`, `request_blocked_invalid_payload.yaml`, `request_allowed_audit_flagged.yaml`, `request_error_policy_runtime.yaml`, etc. **Reference** — `_EXAMPLE_SCENARIO_.yaml` documents all YAML options; excluded from `list` and mixed workload but can be run with `--name _EXAMPLE_SCENARIO_`.

You can run these as-is, add your own YAML in that folder, or use a **custom definitions folder** via `--scenarios-dir` (see [README – CLI Reference](../README.md#cli-reference)).

## How bundled scenarios look

Bundled scenarios use **data_plane** (workflow from config) or **control_plane** (template or outcome), plus **context** (tenant/agent keys resolved from `resource/config/config.yaml`). No hand-written `root` tree.

### Data-plane (workflow from config)

Hierarchy (steps, latencies) comes from `resource/config/config.yaml` → `scenarios.workflow_templates` and `latency_profiles`. Scenario references the workflow key and optional overrides; span latencies use `data_plane.latency_profile` (default `happy_path`):

```yaml
name: new_claim_phone
description: Happy path for Phone MCP (planner → tool → response).

tags:
  - data-plane
  - happy-path
workload_weight: 10.0

mcp_server: phone
repeat_count: 5
interval_ms: 100
emit_metrics: false
emit_logs: false

data_plane:
  workflow: new_claim
  goal: happy_path
  control_plane_template: allowed

expected:
  mcp_server: phone
  tools:
    - new_claim

conversation:
  samples:
    - user_input: "I dropped my phone, how do I start a claim?"
      llm_response: "I've started a claim (PH-8842). Describe what happened..."

context:
  tenant: toro
  agent: toro-customer-assistant-001
  mcp_server: phone
```

### Control-plane only (no data-plane)

Only the request-validation trace is emitted. Minimal context (tenant + agent):

```yaml
name: request_blocked_by_policy
description: Control-plane blocks the request; no a2a.orchestrate trace.

tags:
  - control-plane
workload_weight: 0.4

repeat_count: 2
interval_ms: 50
emit_metrics: false
emit_logs: false

control_plane:
  request_outcome: blocked
  block_reason: request_policy

context:
  tenant: toro
  agent: toro-customer-assistant-001
```

## Explicit root (alternative)

If you need a **custom span tree** instead of a config-driven workflow, you can define it with a `root` block (no `data_plane.workflow`). The loader parses `root.type`, `root.latency`, `root.error`, `root.children`, and per-node `attributes`. None of the bundled scenarios use this.

```yaml
name: my_custom_scenario
description: Custom span tree (no workflow from config).

tags:
  - data-plane
workload_weight: 1.0

context:
  tenant: toro
  agent: toro-customer-assistant-001

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

Use `type` with the same prefix as `VENDOR` / `--vendor` (default `vendor`). Supported span types include `{prefix}.a2a.orchestrate`, `{prefix}.planner`, `{prefix}.task.execute`, `{prefix}.llm.call`, `{prefix}.mcp.tool.execute`, `{prefix}.response.compose`, `rag.retrieve`, `a2a.call`, `cp.request`. See [Generating Telemetry – Scenario File Structure](generating-telemetry.md#scenario-file-structure) for the full table.

## Adding new scenarios (no code changes)

You can add new scenarios by **adding YAML files only**:

1. **Data-plane (a2a.orchestrate) scenarios** – Use **key-based context**: `context.tenant`, `context.agent`, `context.mcp_server`. Define data-plane with `data_plane.workflow` (key into config `scenarios.workflow_templates`), optional `data_plane.goal`, optional `data_plane.control_plane_template`, optional `data_plane.latency_profile` (default `happy_path`), and optional `data_plane.mcp_retry` (template name or inline `attempts`). Config supplies key → UUID and workflow step names; span latencies come from config **`latency_profiles.<profile>`** (see [Generating Telemetry – Configuration reference](generating-telemetry.md#configuration-reference-what-exists)).

2. **Control-plane-only (e.g. blocked/error) scenarios** – Set `control_plane.request_outcome` and `control_plane.block_reason`, or `control_plane.template`. Use **minimal context**: `context.tenant` and `context.agent` only (no `mcp_server`, `workflow`, or `error_pattern`). Only the incoming request-validation trace is emitted. To override the policy-span exception (e.g. for a variant), set `control_plane.policy_exception: { type: "...", message: "..." }` in the scenario YAML.

3. **New control-plane outcome/template** – Add entries under `control_plane.request_validation_templates` and, if needed, `control_plane.trace_flow` in `resource/config/config.yaml`. Attribute values should follow `resource/scenarios/conventions/semconv.yaml`.

4. **New data-plane workflow** – Add a workflow to `scenarios.workflow_templates` in `resource/config/config.yaml` (workflow name → list of steps); then in a scenario YAML set `data_plane.workflow`, optional `data_plane.goal`, and optional `data_plane.control_plane_template`.

5. **4xx invalid-params (correct tool, wrong/missing params)** – To simulate agent calling the correct tool with incorrect or missing parameters (e.g. wrong date format, claim ID with/without dash) → 4xx: set `data_plane.goal: 4xx_invalid_arguments`, use the same `data_plane.workflow` as the happy-path variant. Under `scenario_overrides` you can set `exception_type` and `exception_message` so the failing MCP attempt span records a concrete exception event instead of a generic "Error occurred" (e.g. `exception_type: "InvalidClaimIdFormat"`, `exception_message: "Claim ID must include division prefix (e.g. HA-7733). Received malformed or missing claim_id."`). Optional `step_index_for_4xx` fixes which MCP step gets the error. Copy `new_claim_electronics_tool_4xx_invalid_params.yaml` or `claim_status_appliances_tool_4xx_invalid_params.yaml` and change name, mcp_server (phone | electronics | appliances), workflow, description, and conversation samples. No code changes.

6. **Tags** – In scenario YAML, set `tags: [control-plane]`, `tags: [data-plane, happy-path]`, `tags: [data-plane, 4xx, invalid-params]`, etc. Run a subset with `otelsim run --vendor=your_vendor --tags=control-plane` or `otelsim run --vendor=your_vendor --tags=data-plane,multi-turn`. Use `--each-once` to run each (tagged) scenario exactly once instead of `--count` random picks.

The simulator resolves tenant/agent/MCP IDs from config; data-plane behavior is defined per scenario (workflow, goal, control_plane_template, mcp_retry), and control-plane behavior from config templates.

## See also

- [Generating Telemetry](generating-telemetry.md) – Config reference, workload weights, and scenario examples
- [README – Quick Start](../README.md#quick-start) – Run commands and CLI reference
- [Telemetry Trace Schema](telemetry-trace-schema.md) – Span and attribute schema
