# Telemetry Data Trace: Schema and Examples

This document describes the structure of telemetry trace data produced by the simulator: span schema, trace hierarchy, attribute semantics, and examples. When exporting to file (e.g. `otelsim run --output-file traces.jsonl`), each line is one span in JSON Lines format. The schema is aligned with the **Gentoro LLM Observability Semantic Conventions** in `src/simulator/scenarios/conventions/semconv.yaml` (schema version **1.0.0**).

**Vendor prefix:** Span and attribute names use the configurable vendor prefix (`VENDOR` env or `--vendor`, default `vendor`). The bundled config and semconv use the same structure; examples below use the `gentoro.*` prefix as in the convention. Replace with your prefix (e.g. `vendor.*`) when validating or querying.

---

## Conventions source and alignment

| Item | Source of truth | Notes |
|------|-----------------|--------|
| Span names, kinds, parent rules | `semconv.yaml` → `span_names` | Canonical. |
| Required / recommended attributes | `semconv.yaml` → span specs, `a2a_orchestration_model`, `llm_call_model`, `tools_recommend_model`, `mcp_tool_execution_model` | MUST = required, SHOULD = recommended. |
| Enum values | `semconv.yaml` → `enums` under each model | Normative where specified. |
| Status and outcome rules | `semconv.yaml` → `status_schema`, `failure_rules`, `error_recording` | Root/task/compose status and outcome interaction. |

**Simulator notes** (differences to be aware of when validating data against semconv):

- **Span class attribute**: Semconv defines **`gentoro.span.class`** as the canonical attribute for span identity (e.g. `a2a.orchestrate`, `planner`, `task.execute`). When running with Gentoro semantics, spans SHOULD carry `gentoro.span.class` in line with the schema.
- **`deployment.environment.name`**: Semconv allowed values are `development`, `staging`, `production`. The simulator may use `prod` as shorthand; map to `production` for analytics if needed.
- **`gentoro.component`**: Semconv allowed values are `orchestrator`, `planning`, `retrieval`, `llm`, `mcp_client`, `tool_recommender`. The simulator may use `gateway-ext`; treat as an orchestrator/gateway component for grouping.

---

## 1. Trace Data Format

- **Format**: JSON Lines (`.jsonl`) when using `--output-file` — one JSON object per line, each representing a single **span**. When sending to OTLP, the same span structure is emitted.
- **Source**: OpenTelemetry spans emitted by the simulator (OTEL SDK).
- **Content**: Agent-to-Agent (A2A) orchestration traces (e.g. claims, tool use, LLM calls).

Each line is a complete span. Spans are grouped by `trace_id`; within a trace, the hierarchy is given by `parent_span_id` (root spans have `parent_span_id: null`).

---

## 2. Span Schema (Top-Level)

Every span has the following top-level fields:

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

Resource attributes identify the emitting service and environment (OTEL resource). All spans share the same resource shape. Semconv defines first-class and metadata categories; below are the keys present in simulator output (prefix may be `gentoro` or your `VENDOR`):

| Key | Type | Semconv | Example / Description |
|-----|------|---------|------------------------|
| `telemetry.sdk.language` | string | — | `"python"` |
| `telemetry.sdk.name` | string | — | `"opentelemetry"` |
| `telemetry.sdk.version` | string | — | OTEL SDK version (from dependency) |
| `service.name` | string | required | CLI default or config (e.g. `"otelsim"`) |
| `service.version` | string | required | Package version (e.g. `"1.0.0"`) |
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

In the simulator, **session and conversation IDs** are generated as follows: format templates are defined in `config/config.yaml` under `id_formats` (e.g. `session_id: "sess_toro_{hex:12}"`). The `ScenarioIdGenerator` produces a new value per logical request (per iteration in single-scenario run, per step in mixed workload). Multi-turn scenarios reuse the same session_id for all turns in one iteration. So: one session_id per logical session; different iterations/steps yield different sessions. See [Generating Telemetry – Trace, session, and conversation IDs](./generating-telemetry.md#trace-session-and-conversation-ids) for details.

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
  "trace_id": "9173ed749bd04c19ab07d76dfa0d2978",
  "span_id": "43c7fbfd55ffe765",
  "parent_span_id": null,
  "start_time": 1772195175426134000,
  "end_time": 1772195175459874000,
  "status": { "status_code": "UNSET", "description": null },
  "attributes": {
    "gen_ai.conversation.id": "sess_toro_d862fb6b415b",
    "gentoro.session.id": "sess_toro_d862fb6b415b",
    "enduser.id": "usr_sha256:024bf78e76b0e929",
    "gentoro.tenant.id": "9cafa427-504f-4bb7-a09f-ec1f5524facf",
    "gentoro.redaction.applied": "none",
    "gentoro.a2a.outcome": "success",
    "gentoro.enduser.pseudo.id": "usr_sha256:024bf78e76b0e929",
    "gentoro.a2a.agent.target.id": "toro-customer-assistant-001"
  },
  "kind": "SERVER",
  "resource": {
    "telemetry.sdk.language": "python",
    "telemetry.sdk.name": "opentelemetry",
    "telemetry.sdk.version": "1.39.1",
    "service.name": "otelsim",
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
  "trace_id": "9173ed749bd04c19ab07d76dfa0d2978",
  "span_id": "9fd0574476096695",
  "parent_span_id": "43c7fbfd55ffe765",
  "start_time": 1772195175426191000,
  "end_time": 1772195175429025000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.planner.output.task.count": 6,
    "gentoro.planner.output.entity.count": 0,
    "gentoro.planner.strategy": "model",
    "gentoro.planner.output.format": "task_list",
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
  "trace_id": "9173ed749bd04c19ab07d76dfa0d2978",
  "span_id": "112b0e702a6791d0",
  "parent_span_id": "6b6c96e32584034c",
  "start_time": 1772195175429251000,
  "end_time": 1772195175435900000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.llm.turn.count": 4,
    "gentoro.llm.tool.request.count": 0,
    "gentoro.llm.tool.execution.count": 2,
    "gentoro.llm.streaming": false,
    "gen_ai.system": "mistral_ai",
    "gen_ai.request.model": "gpt-4.1-mini",
    "gen_ai.request.type": "tool_call",
    "gen_ai.response.finish_reason": "length",
    "gen_ai.usage.input_tokens": 1860,
    "gen_ai.usage.output_tokens": 1226,
    "gen_ai.tool.name": "name_rmeoev",
    "gen_ai.tool.call.id": "call_mszuSIzqtI65i1wAUOE8w5H4",
    "gentoro.llm.content.capture.enabled": false,
    "gentoro.llm.content.redaction.enabled": false,
    "gen_ai.input.messages": "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Start a claim for my phone. My address is <ADDRESS> and policy number is <POLICY_NUMBER>.\"}]}]",
    "gen_ai.output.messages": "[{\"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"I've started a claim for your phone at <ADDRESS> under policy <POLICY_NUMBER>. Your claim ID is PH-9900; I'll now find appointment options for you.\"}]}]",
    "gentoro.gen_ai.input.redacted": "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"What is the status of my claim?\"}]}]",
    "gentoro.gen_ai.output.redacted": "[{\"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"Your claim <CLAIM_ID> is in review. Expected completion within 5 business days.\"}]}]",
    "gentoro.step.outcome": "success",
    "gentoro.session.id": "sess_toro_d862fb6b415b",
    "gen_ai.conversation.id": "sess_toro_d862fb6b415b"
  },
  "kind": "CLIENT",
  "resource": { "...": "..." }
}
```

### 5.4 Task execute (tool_execution) and MCP tool execute

```json
{
  "name": "gentoro.task.execute",
  "trace_id": "9173ed749bd04c19ab07d76dfa0d2978",
  "span_id": "6b6c96e32584034c",
  "parent_span_id": "43c7fbfd55ffe765",
  "start_time": 1772195175429105000,
  "end_time": 1772195175437001000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.task.id": "task_5",
    "gentoro.step.outcome": "success",
    "gentoro.task.type": "llm_call",
    "gentoro.task.retry.count": 2,
    "gentoro.task.fallback.used": true,
    "gentoro.session.id": "sess_toro_d862fb6b415b",
    "gen_ai.conversation.id": "sess_toro_d862fb6b415b"
  },
  "kind": "INTERNAL",
  "resource": { "...": "..." }
}
```

```json
{
  "name": "gentoro.mcp.tool.execute",
  "trace_id": "9173ed749bd04c19ab07d76dfa0d2978",
  "span_id": "8c3126d9ffa90a69",
  "parent_span_id": "ce77bfbd263fc11b",
  "start_time": 1772195175438827000,
  "end_time": 1772195175441279000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.span.class": "mcp.tool.execute",
    "gentoro.step.outcome": "success",
    "gentoro.mcp.server.uuid": "11111111-1111-4eec-8001-000000000001",
    "gentoro.mcp.tool.uuid": "11111111-1111-4eec-8001-000000000011",
    "gentoro.mcp.tool.call.id": "mcp_call_082d8443825a",
    "gentoro.retry.count": 2,
    "gentoro.retry.policy": "none",
    "gen_ai.tool.name": "new_claim",
    "gen_ai.tool.call.id": "call_mszuSIzqtI65i1wAUOE8w5H4",
    "error.type": "timeout",
    "gentoro.session.id": "sess_toro_d862fb6b415b",
    "gen_ai.conversation.id": "sess_toro_d862fb6b415b"
  },
  "kind": "CLIENT",
  "resource": { "...": "..." }
}
```

```json
{
  "name": "gentoro.mcp.tool.execute.attempt",
  "trace_id": "9173ed749bd04c19ab07d76dfa0d2978",
  "span_id": "9f8345df07741d10",
  "parent_span_id": "8c3126d9ffa90a69",
  "start_time": 1772195175438856000,
  "end_time": 1772195175441142000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.mcp.tool.call.id": "mcp_call_082d8443825a",
    "gentoro.mcp.attempt.index": 1,
    "gentoro.mcp.attempt.outcome": "success",
    "error.type": "invalid_arguments",
    "gentoro.retry.reason": "rate_limited",
    "gentoro.session.id": "sess_toro_d862fb6b415b",
    "gen_ai.conversation.id": "sess_toro_d862fb6b415b"
  },
  "kind": "CLIENT",
  "resource": { "...": "..." }
}
```

### 5.5 Response compose

```json
{
  "name": "gentoro.response.compose",
  "trace_id": "9173ed749bd04c19ab07d76dfa0d2978",
  "span_id": "d809f6a565a5b176",
  "parent_span_id": "43c7fbfd55ffe765",
  "start_time": 1772195175442137000,
  "end_time": 1772195175442948000,
  "status": { "status_code": "OK", "description": null },
  "attributes": {
    "gentoro.span.class": "response.compose",
    "gentoro.response.format": "a2a_json",
    "gentoro.step.outcome": "success",
    "gentoro.session.id": "sess_toro_d862fb6b415b",
    "gen_ai.conversation.id": "sess_toro_d862fb6b415b"
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

This schema is aligned with **`src/simulator/scenarios/conventions/semconv.yaml`** (schema version 1.0.0). The examples and attribute tables reflect the normative conventions and representative simulator output (including simulator deviations noted at the top). Use this document for validation, dashboards, and downstream trace processors. For concrete control-plane shapes, see sample scenarios such as `request_allowed_audit_flagged` (allow with audit), `request_blocked_rate_limited` (rate-limited before policy), and `request_blocked_invalid_payload_multi` (multiple validation error events on the payload span).
