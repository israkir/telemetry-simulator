# Telemetry Data Trace: Schema and Examples

This document describes the structure of the telemetry trace data in `traces.jsonl`: span schema, trace hierarchy, attribute semantics, and examples. It is aligned with the **Gentoro LLM Observability Semantic Conventions** defined in `src/simulator/scenarios/conventions/semconv.yaml` (schema version **1.0.0**).

---

## Conventions source and alignment

| Item | Source of truth | Notes |
|------|-----------------|--------|
| Span names, kinds, parent rules | `semconv.yaml` → `span_names` | Canonical. |
| Required / recommended attributes | `semconv.yaml` → span specs, `a2a_orchestration_model`, `llm_call_model`, `tools_recommend_model`, `mcp_tool_execution_model` | MUST = required, SHOULD = recommended. |
| Enum values | `semconv.yaml` → `enums` under each model | Normative where specified. |
| Status and outcome rules | `semconv.yaml` → `status_schema`, `failure_rules`, `error_recording` | Root/task/compose status and outcome interaction. |

**Simulator / `traces.jsonl` notes** (differences to be aware of when validating data against semconv):

- **Span class attribute**: Semconv defines **`gentoro.span.class`** as the canonical attribute for span identity (e.g. `a2a.orchestrate`, `planner`, `task.execute`). When running with Gentoro semantics, spans SHOULD carry `gentoro.span.class` in line with the schema.
- **`deployment.environment.name`**: Semconv allowed values are `development`, `staging`, `production`. The simulator may use `prod` as shorthand; map to `production` for analytics if needed.
- **`gentoro.component`**: Semconv allowed values are `orchestrator`, `planning`, `retrieval`, `llm`, `mcp_client`, `tool_recommender`. The simulator may use `gateway-ext`; treat as an orchestrator/gateway component for grouping.

---

## 1. Trace Data Format

- **Format**: JSON Lines (`.jsonl`) — one JSON object per line, each representing a single **span**.
- **Source**: OpenTelemetry-style spans emitted by the telemetry-simulator (Python, SDK 1.39.1).
- **Content**: Agent-to-Agent (A2A) orchestration traces for a customer-assistant flow (e.g. claims, tool use, LLM calls).

Each line is a complete span. Spans are grouped by `trace_id`; within a trace, the hierarchy is given by `parent_span_id` (root spans have `parent_span_id: null`).

---

## 2. Span Schema (Top-Level)

Every span in `traces.jsonl` has the following top-level fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Span name (e.g. `gentoro.a2a.orchestrate`, `gentoro.llm.call`). |
| `trace_id` | string | Yes | W3C trace ID; same for all spans in one trace (32 hex chars). |
| `span_id` | string | Yes | Unique ID for this span (16 hex chars). |
| `parent_span_id` | string \| null | Yes | Parent span ID; `null` for the root span. |
| `start_time` | integer | Yes | Start time in **nanoseconds** (Unix epoch). |
| `end_time` | integer | Yes | End time in **nanoseconds** (Unix epoch). |
| `status` | object | Yes | Span status (see below). |
| `attributes` | object | Yes | Key-value span attributes (see Attributes section). |
| `kind` | string | Yes | Span kind: `SERVER`, `CLIENT`, or `INTERNAL`. |
| `resource` | object | Yes | Resource attributes (service, SDK, deployment; see below). |

### 2.1 Status Object

| Field | Type | Description |
|-------|------|-------------|
| `status_code` | string | One of: `UNSET`, `OK`, `ERROR`. |
| `description` | string \| null | Optional human-readable description (often `null`). |

### 2.2 Resource Object

Resource attributes identify the emitting service and environment (OTEL resource). All spans in this file share the same resource shape. Semconv defines first-class and metadata categories; below are the keys present in `traces.jsonl`:

| Key | Type | Semconv | Example / Description |
|-----|------|---------|------------------------|
| `telemetry.sdk.language` | string | — | `"python"` |
| `telemetry.sdk.name` | string | — | `"opentelemetry"` |
| `telemetry.sdk.version` | string | — | `"1.39.1"` |
| `service.name` | string | required | `"telemetry-simulator"` |
| `service.version` | string | required | `"1.0.0"` |
| `service.instance.id` | string | optional | `"pod-7f9c6d4b8d-xyz12"` |
| `deployment.environment.name` | string | required | Canonical: `development` \| `staging` \| `production`. Simulator may use `"prod"`. |
| `gentoro.component` | string | optional | Canonical: `orchestrator` \| `planning` \| `retrieval` \| `llm` \| `mcp_client` \| `tool_recommender`. Simulator may use `"gateway-ext"`. |
| `gentoro.otel.source` | string | — | `"propagated"` |
| `gentoro.module` | string | optional | `"data-plane"` |
| `gentoro.tenant.id` | string | required | UUID, e.g. `"9cafa427-504f-4bb7-a09f-ec1f5524facf"` |

---

## 3. Trace Hierarchy and Span Types

A single logical request is represented by **three separate traces**, each with its own `trace_id` but sharing the same `gentoro.session.id` (and typically the same `gentoro.request.id`):

- **Incoming control-plane validation trace**: root `gentoro.request.validation` (SERVER) + child validation spans (payload, policy, augmentation).
- **Data-plane orchestration trace**: root `gentoro.a2a.orchestrate` (SERVER) + planner/tasks/LLM/tools/compose.
- **Outgoing control-plane validation trace** (when a response is produced): root `gentoro.response.validation` (SERVER) + child policy span.

Within the Data plane portion, a typical hierarchy is:

```
gentoro.a2a.orchestrate (root, SERVER)
├── gentoro.planner (INTERNAL)
├── gentoro.task.execute (INTERNAL) [type: llm_call]
│   └── gentoro.llm.call (CLIENT)
├── gentoro.task.execute (INTERNAL) [type: tool_recommendation]
│   └── gentoro.tools.recommend (INTERNAL)
├── gentoro.task.execute (INTERNAL) [type: tool_execution]
│   └── gentoro.mcp.tool.execute (CLIENT)
│       └── gentoro.mcp.tool.execute.attempt (CLIENT) [one or more]
└── gentoro.response.compose (INTERNAL)
```

### 3.1 Span Types Summary (semconv-aligned)

| Span Name | Kind | Parent (semconv) | Required identity |
|-----------|------|------------------|-------------------|
| `gentoro.request.validation` | SERVER | — (root) | `gentoro.span.class` = `request.validation` |
| `gentoro.validation.payload` | INTERNAL | `gentoro.request.validation` | `gentoro.span.class` = `validation.payload` |
| `gentoro.validation.policy` | INTERNAL | `gentoro.request.validation` or `gentoro.response.validation` | `gentoro.span.class` = `validation.policy` |
| `gentoro.augmentation` | INTERNAL | `gentoro.request.validation` | `gentoro.span.class` = `augmentation` |
| `gentoro.a2a.orchestrate` | SERVER | — (root) | `gentoro.span.class` = `a2a.orchestrate` |
| `gentoro.planner` | INTERNAL | `gentoro.a2a.orchestrate` | `gentoro.span.class` = `planner` |
| `gentoro.task.execute` | INTERNAL | `gentoro.planner` or `gentoro.a2a.orchestrate` | `gentoro.span.class` = `task.execute` |
| `gentoro.llm.call` | CLIENT | `gentoro.task.execute` (type=llm_call) | `gentoro.span.class` = `llm.call` |
| `gentoro.tools.recommend` | INTERNAL | `gentoro.task.execute` (type=tool_recommendation) | `gentoro.span.class` = `tools.recommend` |
| `gentoro.mcp.tool.execute` | CLIENT | `gentoro.llm.call` or `gentoro.task.execute` (type=tool_execution) | `gentoro.span.class` = `mcp.tool.execute` |
| `gentoro.mcp.tool.execute.attempt` | CLIENT | `gentoro.mcp.tool.execute` | `gentoro.span.class` = `mcp.tool.execute.attempt` |
| `gentoro.response.compose` | INTERNAL | `gentoro.a2a.orchestrate` | `gentoro.span.class` = `response.compose` |
| `gentoro.response.validation` | SERVER | — (root) | `gentoro.span.class` = `response.validation` |

**Status and outcome rules (semconv):**

- **Root** (`gentoro.a2a.orchestrate`): `status.code` = `UNSET` when `gentoro.a2a.outcome` is `success` or `partial`; `status.code` = `ERROR` when `gentoro.a2a.outcome` = `error`. Blocked outcomes are handled in control-plane.
- **Task / planner / compose / LLM / tools / MCP**: On failure, span MUST set `status.code` = `ERROR`, set outcome to `fail` where applicable, and SHOULD set `error.type`; MUST emit exception event when an exception exists (exception.type, optionally exception.message, exception.stacktrace).
- **Root outcome**: All tasks succeed → `gentoro.a2a.outcome` = `success`. ≥1 task fails but valid response → `partial`. Task failure prevents valid response → `error`.

---

## 4. Attributes Schema

Attributes are span-specific and aligned with semconv. **REQUIRED** = MUST be present; **RECOMMENDED** = SHOULD when applicable. The canonical span identity attribute is **`gentoro.span.class`**.

### 4.1 Common / Session (multiple span types)

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.session.id` | string | REQUIRED (first-class) | Session identifier; A2A contextId or CP-generated. |
| `gen_ai.conversation.id` | string | RECOMMENDED | Conversation ID (OTEL); SHOULD equal session when same conversation. |
| `gentoro.span.class` | string | REQUIRED (per span type) | Canonical span class for orchestration, LLM, tools, MCP, and control-plane validation spans (e.g. `a2a.orchestrate`, `planner`, `task.execute`, `llm.call`, `tools.recommend`, `mcp.tool.execute`, `mcp.tool.execute.attempt`, `response.compose`, `request.validation`, `validation.payload`, `validation.policy`, `augmentation`, `response.validation`). |
| `gentoro.step.outcome` | string | REQUIRED (where applicable) | Data-plane spans use `success` \| `fail` \| `skipped`; control-plane validation spans use `pass` \| `fail` \| `block` \| `skip` per the control-plane validation model. |
| `gentoro.error.category` | string | OPTIONAL (on failure) | Enum: `validation` \| `policy` \| `runtime`; low-cardinality category describing the error source for failed steps. |

### 4.2 gentoro.a2a.orchestrate

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `a2a.orchestrate`. |
| `gentoro.a2a.agent.target.id` | string | REQUIRED | Target agent identifier. |
| `gentoro.a2a.outcome` | string | REQUIRED | Enum: `success` \| `partial` \| `error`. |
| `gentoro.enduser.pseudo.id` | string | REQUIRED | Pseudonymized user ID. |
| `gentoro.session.id` | string | REQUIRED | Session ID. |
| `gen_ai.conversation.id` | string | RECOMMENDED | Conversation ID. |
| `enduser.id` | string | REQUIRED (common) | End-user ID (e.g. `usr_sha256:...`). |
| `gentoro.tenant.id` | string | REQUIRED (common) | Tenant UUID. |
| `gentoro.redaction.applied` | string | REQUIRED (common) | Enum: `none` \| `basic` \| `strict`. |

### 4.3 gentoro.planner

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `planner`. |
| `gentoro.step.outcome` | string | REQUIRED | Enum: `success` \| `fail`. |
| `gentoro.planner.output.task.count` | number | REQUIRED | Number of tasks produced. |
| `gentoro.planner.output.entity.count` | number | REQUIRED | Number of entities. |
| `gentoro.planner.strategy` | string | RECOMMENDED | Enum: `rules` \| `model` \| `hybrid`. |
| `gentoro.planner.output.format` | string | RECOMMENDED | Enum: `task_list` \| `task_graph` \| `task_tree`. |
| `gentoro.planner.fallback.used` | boolean | RECOMMENDED | Whether fallback was used. |
| `error.type` | string | SHOULD (if failed) | Low-cardinality error identifier. |

### 4.4 gentoro.task.execute

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `task.execute`. |
| `gentoro.task.id` | string | REQUIRED | Stable task identifier from planner. |
| `gentoro.task.type` | string | REQUIRED | Enum: `context_augmentation` \| `llm_call` \| `tool_recommendation` \| `tool_execution` \| `other`. |
| `gentoro.step.outcome` | string | REQUIRED | Enum: `success` \| `fail` \| `skipped`. |
| `gentoro.task.retry.count` | number | RECOMMENDED | Number of retries. |
| `gentoro.task.fallback.used` | boolean | RECOMMENDED | Whether fallback was used. |
| `error.type` | string | SHOULD (if failed) | Low-cardinality error classification. |

### 4.5 gentoro.llm.call

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `llm.call`. |
| `gentoro.step.outcome` | string | REQUIRED | Enum: `success` \| `fail` \| `partial`. |
| `gen_ai.system` | string | REQUIRED | Provider (e.g. openai, gcp.vertex_ai, aws.bedrock). |
| `gen_ai.request.model` | string | REQUIRED | Model identifier. |
| `gen_ai.request.type` | string | RECOMMENDED | Enum: `completion` \| `chat` \| `tool_call`. |
| `gen_ai.response.finish_reason` | string | RECOMMENDED | Enum: `stop` \| `length` \| `tool_call` \| `error`. |
| `gen_ai.usage.input_tokens` | number | RECOMMENDED | Input token count. |
| `gen_ai.usage.output_tokens` | number | RECOMMENDED | Output token count. |
| `gentoro.llm.turn.count` | number | RECOMMENDED | Number of turns. |
| `gentoro.llm.tool.request.count` | number | RECOMMENDED | Tool requests in this call. |
| `gentoro.llm.tool.execution.count` | number | RECOMMENDED | Tool executions. |
| `gentoro.llm.streaming` | boolean | RECOMMENDED | Whether response was streamed. |
| `gen_ai.tool.name` / `gen_ai.tool.call.id` | string | RECOMMENDED (when tools used) | Tool identity. |
| `error.type` | string | SHOULD (if failed) | Low-cardinality error (e.g. model_timeout, provider_unavailable). |
| `gen_ai.input.messages` | array / object | OPTIONAL (opt-in) | User/input messages for this LLM call, aligned with OTEL GenAI `gen_ai.input.messages`. Only present when `gentoro.llm.content.capture.enabled=true`. |
| `gen_ai.output.messages` | array / object | OPTIONAL (opt-in) | LLM output messages for this call, aligned with OTEL GenAI `gen_ai.output.messages`. Only present when `gentoro.llm.content.capture.enabled=true`. |
| `gentoro.gen_ai.input.redacted` | array / object | OPTIONAL (opt-in) | Redacted variant of `gen_ai.input.messages` when redaction is enabled. |
| `gentoro.gen_ai.output.redacted` | array / object | OPTIONAL (opt-in) | Redacted variant of `gen_ai.output.messages` when redaction is enabled. |
| `gentoro.llm.content.capture.enabled` | boolean | OPTIONAL (control) | When true, simulator MAY emit `gen_ai.input.messages` and `gen_ai.output.messages`. Default: `false`. |
| `gentoro.llm.content.redaction.enabled` | boolean | OPTIONAL (control) | When true (and content capture enabled), simulator SHOULD emit redacted variants (`gentoro.gen_ai.input.redacted`, `gentoro.gen_ai.output.redacted`). Default: `false`. |

Message content (user input and LLM output) is now **opt-in** on trace spans via the GenAI message attributes above. By default (`gentoro.llm.content.capture.enabled=false`) no raw content is stored on spans and only aggregate usage metrics are present; when enabled, content is captured per OTEL GenAI conventions and SHOULD be redacted via the `gentoro.gen_ai.*.redacted` attributes for sensitive data.

### 4.6 gentoro.tools.recommend

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `tools.recommend`. |
| `gentoro.step.outcome` | string | REQUIRED | Enum: `success` \| `fail` \| `skipped`. |
| `gentoro.mcp.tools.available.count` | number | REQUIRED | MCP tools available. |
| `gentoro.mcp.tools.selected.count` | number | REQUIRED | Tools selected. |
| `gentoro.mcp.selection.strategy` | string | RECOMMENDED | Enum: `capability_match` \| `semantic_match` \| `policy_filtered` \| `hybrid` \| `default`. |
| `gentoro.mcp.selection.constraints` | string | RECOMMENDED | Enum: `policy` \| `permissions` \| `environment` \| `latency_budget` \| `none`. |
| `gentoro.mcp.selection.latency.ms` | number | RECOMMENDED | Selection latency in ms. |
| `gentoro.mcp.selection.fallback.used` | boolean | RECOMMENDED | Whether fallback was used. |
| `gentoro.tools.recommended.source` | string | — | Enum: `static` \| `learned` \| `hybrid`. |
| `error.type` | string | SHOULD (if failed) | e.g. no_tools_available, policy_violation. |

**Diagnostic event on this span**:

- **Event name**: `gentoro.agent.tool_selection` (Gentoro schema).
- **Event attributes**:
  - `gentoro.agent.tool_selection.input.raw` — raw user input message text considered during tool selection (for this simulator, the first `user` text message; may be truncated by downstream processors if needed).
  - `gentoro.agent.tool_selection.tool.plan` — JSON-encoded `tool_plan` structure describing the selected tools (bounded; simulator uses a single-entry plan with the full user input as both `trigger_summary` and `trigger_quote`, and `missing_info: null`).

### 4.7 gentoro.mcp.tool.execute and gentoro.mcp.tool.execute.attempt

**Parent span (gentoro.mcp.tool.execute):**

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `mcp.tool.execute`. |
| `gentoro.step.outcome` | string | REQUIRED | Enum: `success` \| `fail` \| `skipped`. |
| `gentoro.mcp.server.uuid` | string | REQUIRED | MCP server UUID. |
| `gentoro.mcp.tool.uuid` | string | REQUIRED | MCP tool UUID. |
| `gentoro.mcp.tool.call.id` | string | REQUIRED | Tool call ID. |
| `gentoro.retry.count` | number | RECOMMENDED | Number of retries. |
| `gentoro.retry.policy` | string | RECOMMENDED | Enum: `none` \| `fixed` \| `exponential` \| `exponential_jitter` \| `circuit_breaker`. |
| `gen_ai.tool.name` / `gen_ai.tool.call.id` | string | RECOMMENDED | Tool identity. |
| `error.type` | string | SHOULD (if failed) | Suggested: `timeout` \| `unavailable` \| `invalid_arguments` \| `tool_error` \| `protocol_error`. |

**Attempt span (gentoro.mcp.tool.execute.attempt):**

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `mcp.tool.execute.attempt`. |
| `gentoro.mcp.tool.call.id` | string | REQUIRED | Same as parent. |
| `gentoro.mcp.attempt.index` | number | REQUIRED | Attempt sequence (1..N). |
| `gentoro.mcp.attempt.outcome` | string | REQUIRED | Enum: `success` \| `fail`. |
| `gentoro.retry.reason` | string | RECOMMENDED | Enum: `timeout` \| `unavailable` \| `rate_limited` \| `transient_error` \| `unknown`. |
| `error.type` | string | SHOULD (if failed) | Low-cardinality classifier. |

### 4.8 gentoro.response.compose

| Attribute | Type | Semconv | Description |
|-----------|------|---------|-------------|
| `gentoro.span.class` | string | REQUIRED | Value: `response.compose`. |
| `gentoro.response.format` | string | REQUIRED | Enum: `a2a_json` \| `a2a_stream`. |
| `gentoro.step.outcome` | string | REQUIRED | Enum: `success` \| `fail`. |
| `error.type` | string | SHOULD (if failed) | e.g. serialization_error. |


---

## 5. Examples

### 5.1 Root span (gentoro.a2a.orchestrate)

```json
{
  "name": "gentoro.a2a.orchestrate",
  "trace_id": "8bb330b9fdb966f1bc33c8dc9f45f002",
  "span_id": "50cb5832df180f3c",
  "parent_span_id": null,
  "start_time": 1772121706604840000,
  "end_time": 1772121706638719000,
  "status": { "status_code": "UNSET", "description": null },
  "attributes": {
    "gen_ai.conversation.id": "sess_toro_ed111e616866",
    "gentoro.session.id": "sess_toro_ed111e616866",
    "enduser.id": "usr_sha256:0e5b1315ddcd8f7a",
    "gentoro.tenant.id": "9cafa427-504f-4bb7-a09f-ec1f5524facf",
    "gentoro.redaction.applied": "none",
    "gentoro.a2a.outcome": "success",
    "gentoro.enduser.pseudo.id": "usr_sha256:0e5b1315ddcd8f7a",
    "gentoro.a2a.agent.target.id": "toro-customer-assistant-001"
  },
  "kind": "SERVER",
  "resource": {
    "telemetry.sdk.language": "python",
    "telemetry.sdk.name": "opentelemetry",
    "telemetry.sdk.version": "1.39.1",
    "service.name": "telemetry-simulator",
    "service.version": "1.0.0",
    "service.instance.id": "pod-7f9c6d4b8d-xyz12",
    "deployment.environment.name": "prod",
    "gentoro.component": "gateway-ext",
    "gentoro.otel.source": "propagated",
    "gentoro.module": "data-plane",
    "gentoro.tenant.id": "9cafa427-504f-4bb7-a09f-ec1f5524facf"
  }
}
```

### 5.2 Planner span (gentoro.planner)

```json
{
  "name": "gentoro.planner",
  "trace_id": "2f3d726e0569f74581c0291152b73eda",
  "span_id": "303723eab4ad2aa4",
  "parent_span_id": "de3fe848ebe01a84",
  "start_time": 1772121725894768000,
  "end_time": 1772121725901853000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.planner.output.task.count": 6,
    "gentoro.planner.output.entity.count": 0,
    "gentoro.planner.strategy": "model",
    "gentoro.planner.output.format": "task_graph",
    "gentoro.planner.fallback.used": false,
    "gentoro.step.outcome": "success",
    "gentoro.session.id": "sess_toro_61ae8beca0c1",
    "gen_ai.conversation.id": "sess_toro_61ae8beca0c1"
  },
  "kind": "INTERNAL",
  "resource": { "...": "..." }
}
```

### 5.3 LLM call span (gentoro.llm.call)

```json
{
  "name": "gentoro.llm.call",
  "trace_id": "2f3d726e0569f74581c0291152b73eda",
  "span_id": "216f759e14607f07",
  "parent_span_id": "f171e82028301d10",
  "start_time": 1772121725902275000,
  "end_time": 1772121725908129000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.llm.turn.count": 1,
    "gentoro.llm.tool.request.count": 2,
    "gentoro.llm.tool.execution.count": 2,
    "gentoro.llm.streaming": true,
    "gen_ai.system": "gcp.vertex_ai",
    "gen_ai.request.model": "claude-3-opus",
    "gen_ai.request.type": "chat",
    "gen_ai.response.finish_reason": "error",
    "gen_ai.usage.input_tokens": 109,
    "gen_ai.usage.output_tokens": 1393,
    "gen_ai.tool.name": "name_ksthls",
    "gen_ai.tool.call.id": "call_mszuSIzqtI65i1wAUOE8w5H4",
    "gentoro.step.outcome": "success",
    "gentoro.session.id": "sess_toro_61ae8beca0c1",
    "gen_ai.conversation.id": "sess_toro_61ae8beca0c1"
  },
  "kind": "CLIENT",
  "resource": { "...": "..." }
}
```

### 5.4 Task execute (tool_execution) and MCP tool execute

```json
{
  "name": "gentoro.task.execute",
  "trace_id": "2f3d726e0569f74581c0291152b73eda",
  "span_id": "f223cf299ddedf52",
  "parent_span_id": "de3fe848ebe01a84",
  "start_time": 1772121725913218000,
  "end_time": 1772121725916202000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.task.id": "task_5",
    "gentoro.step.outcome": "success",
    "gentoro.task.type": "tool_execution",
    "gentoro.task.retry.count": 1,
    "gentoro.task.fallback.used": true,
    "gentoro.session.id": "sess_toro_61ae8beca0c1",
    "gen_ai.conversation.id": "sess_toro_61ae8beca0c1"
  },
  "kind": "INTERNAL",
  "resource": { "...": "..." }
}
```

```json
{
  "name": "gentoro.mcp.tool.execute",
  "trace_id": "2f3d726e0569f74581c0291152b73eda",
  "span_id": "36e91e960b97362a",
  "parent_span_id": "f223cf299ddedf52",
  "start_time": 1772121725913291000,
  "end_time": 1772121725915223000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.span.class": "task.execute",
    "gentoro.step.outcome": "success",
    "gentoro.mcp.server.uuid": "11111111-1111-4eec-8001-000000000001",
    "gentoro.mcp.tool.uuid": "11111111-1111-4eec-8001-000000000011",
    "gentoro.mcp.tool.call.id": "mcp_call_5fd4eed6ea73",
    "gentoro.retry.count": 1,
    "gentoro.retry.policy": "none",
    "gen_ai.tool.name": "new_claim",
    "gen_ai.tool.call.id": "call_mszuSIzqtI65i1wAUOE8w5H4",
    "error.type": "unavailable",
    "gentoro.session.id": "sess_toro_61ae8beca0c1",
    "gen_ai.conversation.id": "sess_toro_61ae8beca0c1"
  },
  "kind": "CLIENT",
  "resource": { "...": "..." }
}
```

```json
{
  "name": "gentoro.mcp.tool.execute.attempt",
  "trace_id": "2f3d726e0569f74581c0291152b73eda",
  "span_id": "ea527143dd8921f8",
  "parent_span_id": "36e91e960b97362a",
  "start_time": 1772121725913342000,
  "end_time": 1772121725915077000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.mcp.tool.call.id": "mcp_call_5fd4eed6ea73",
    "gentoro.mcp.attempt.index": 1,
    "gentoro.mcp.attempt.outcome": "fail",
    "error.type": "invalid_arguments",
    "gentoro.retry.reason": "rate_limited",
    "gentoro.session.id": "sess_toro_61ae8beca0c1",
    "gen_ai.conversation.id": "sess_toro_61ae8beca0c1"
  },
  "kind": "CLIENT",
  "resource": { "...": "..." }
}
```

### 5.5 Response compose

```json
{
  "name": "gentoro.response.compose",
  "trace_id": "2f3d726e0569f74581c0291152b73eda",
  "span_id": "28824bfdc151af6e",
  "parent_span_id": "de3fe848ebe01a84",
  "start_time": 1772121725916261000,
  "end_time": 1772121725917025000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.span.class": "response.compose",
    "gentoro.response.format": "a2a_json",
    "gentoro.step.outcome": "success",
    "gentoro.session.id": "sess_toro_61ae8beca0c1",
    "gen_ai.conversation.id": "sess_toro_61ae8beca0c1"
  },
  "kind": "INTERNAL",
  "resource": { "...": "..." }
}
```

**Error recording (semconv):** When a span is in error (`status.code` = `ERROR`), instrumentation SHOULD set `error.type` to a low-cardinality identifier. If an exception is available, record an exception event with `exception.type`, optionally `exception.message` and `exception.stacktrace`. See `error_recording` in semconv.yaml.

---

## 6. Time and IDs

- **Timestamps**: `start_time` and `end_time` are in **nanoseconds** since Unix epoch. Duration in ms = `(end_time - start_time) / 1_000_000`.
- **Trace ID**: 32-character hex string (W3C Trace Context).
- **Span ID**: 16-character hex string. Unique within the trace.
- **Parent**: Root span has `parent_span_id: null`; all other spans reference their parent’s `span_id`.

---

This schema is aligned with **`src/simulator/scenarios/conventions/semconv.yaml`** (schema version 1.0.0). The examples and attribute tables reflect both the normative conventions and representative `traces.jsonl` output (including simulator deviations noted at the top). Use this document for validation, dashboards, and downstream trace processors.
