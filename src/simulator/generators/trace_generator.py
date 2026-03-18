"""
Generate complete trace hierarchies aligned with vendor-prefixed OTEL semantic conventions.

Multi-turn semantics:
  - Each request reaching the control-plane is one incoming request: trace_id is generated
    at CP and shared (e.g. cp.incoming_trace_id, cp.outgoing_trace_id).
  - A new request in the same session: again incoming, different trace_id, same session_id
    (caller passes same context.session_id for all turns in that session).

Example tree:
  {prefix}.a2a.orchestrate (SERVER)
  ├── {prefix}.planner (INTERNAL)
  │   └── {prefix}.llm.call (CLIENT)
  ├── {prefix}.task.execute (INTERNAL)  # optional
  ├── {prefix}.mcp.tool.execute (CLIENT)
  │   └── {prefix}.mcp.tool.execute.attempt (CLIENT)
  ├── {prefix}.llm.call (CLIENT)
  └── {prefix}.response.compose (INTERNAL)
"""

import json
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    Tracer,
)

from .. import __version__ as _OTELSIM_VERSION
from ..config import (
    DATA_PLANE_COMPONENT_VALUES,
    SEMCONV_ERROR_TYPE_VALUES,
    resource_schema_url,
)
from ..config import attr as config_attr
from ..config import resource_attributes as config_resource_attributes
from ..config import span_name as config_span_name
from ..defaults import get_default_tenant_ids
from ..schemas.attribute_generator import AttributeGenerator, GenerationContext
from ..schemas.schema_parser import SchemaParser, TelemetrySchema


class SpanType(Enum):
    """Canonical span types; vendor-prefixed names use config.span_name(suffix)."""

    A2A_ORCHESTRATE = "a2a.orchestrate"
    PLANNER = "planner"
    TASK_EXECUTE = "task.execute"
    LLM_CALL = "llm.call"
    TOOLS_RECOMMEND = "tools.recommend"
    MCP_TOOL_EXECUTE = "mcp.tool.execute"
    MCP_TOOL_EXECUTE_ATTEMPT = "mcp.tool.execute.attempt"
    LLM_TOOL_RESPONSE_BRIDGE = "llm.tool.response.bridge"
    RESPONSE_COMPOSE = "response.compose"
    RAG_RETRIEVE = "rag.retrieve"
    A2A_CALL = "a2a.call"
    PAYLOAD_VALIDATION = "payload.validation"
    POLICY_VALIDATION = "policy.validation"
    AUGMENTATION_VALIDATION = "augmentation.validation"
    REQUEST_VALIDATION = "request.validation"
    RESPONSE_VALIDATION = "response.validation"
    CP_REQUEST = "cp.request"


# Minimum gap between span end and next span start (ns) so the waterfall view shows clearly.
_MIN_SPAN_GAP_NS = 15 * 1_000_000  # 15 ms

# Span types that use vendor prefix for emitted name; others use enum value as-is.
_VENDOR_SPAN_SUFFIXES = {
    SpanType.A2A_ORCHESTRATE,
    SpanType.PLANNER,
    SpanType.TASK_EXECUTE,
    SpanType.LLM_CALL,
    SpanType.TOOLS_RECOMMEND,
    SpanType.MCP_TOOL_EXECUTE,
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT,
    SpanType.LLM_TOOL_RESPONSE_BRIDGE,
    SpanType.RESPONSE_COMPOSE,
    SpanType.PAYLOAD_VALIDATION,
    SpanType.POLICY_VALIDATION,
    SpanType.AUGMENTATION_VALIDATION,
    SpanType.REQUEST_VALIDATION,
    SpanType.RESPONSE_VALIDATION,
}


def get_span_name(span_type: SpanType) -> str:
    """Return the emitted span name (vendor-prefixed or plain) for this span type."""
    if span_type in _VENDOR_SPAN_SUFFIXES:
        return config_span_name(span_type.value)
    return span_type.value


def _sanitize_scope_name(name: str) -> str:
    """Sanitize scenario name for use in otel.scope.name (alphanumeric, underscore, dot, hyphen)."""
    if not name or not isinstance(name, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name.strip())[:128]


def _context_from_trace_and_span(trace_id_hex: str, span_id_hex: str) -> Any:
    """Build an OpenTelemetry context with the given trace_id and span_id as the current span (for use as parent)."""
    span_context = SpanContext(
        trace_id=int(trace_id_hex, 16),
        span_id=int(span_id_hex, 16),
        is_remote=False,
        trace_flags=TraceFlags(0x01),
    )
    return trace.set_span_in_context(NonRecordingSpan(span_context))


# Exception type and message derived from error.type when not set on config (e.g. failed MCP attempts).
# Aligned with semconv error.type allowed_values; used for exception event on any failing span.
_EXCEPTION_FROM_ERROR_TYPE: dict[str, tuple[str, str]] = {
    "timeout": ("TimeoutError", "Call exceeded latency budget"),
    "unavailable": ("UnavailableError", "Server or tool unreachable"),
    "invalid_arguments": ("InvalidArgumentsError", "Invalid or missing parameters"),
    "tool_error": ("ToolError", "Tool returned failure"),
    "protocol_error": ("ProtocolError", "Protocol or serialization error"),
}


def _exception_from_error_type(error_type: str, _span_type: SpanType) -> tuple[str, str]:
    """Return (exception_type, exception_message) for a given error.type (SemConv-aligned)."""
    return _EXCEPTION_FROM_ERROR_TYPE.get(error_type, ("RuntimeError", "Error occurred"))


def _record_span_error(
    span: Any,
    span_type: SpanType,
    overrides: dict[str, Any],
    message: str = "Error occurred",
    error_type_override: str | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
) -> None:
    """Set status=ERROR, error.type (low-cardinality), and exception event (SemConv-aligned).

    When a span is in error, instrumentation SHOULD set error.type and record an exception
    event with exception.type, exception.message, exception.stacktrace (OpenTelemetry).
    error.type must be one of SEMCONV_ERROR_TYPE_VALUES from conventions/semconv.yaml.
    If exception_type/exception_message are provided (e.g. from control-plane template), use them for the exception event.
    Otherwise derive from error.type so failed MCP attempts and other spans get SemConv-aligned exception events.
    """
    raw = (
        error_type_override
        or overrides.get("error.type")
        or overrides.get(config_attr("error.type"))
        or _ERROR_TYPE_BY_SPAN_TYPE.get(span_type, "unavailable")
    )
    error_type = (
        raw
        if raw in SEMCONV_ERROR_TYPE_VALUES
        else _ERROR_TYPE_BY_SPAN_TYPE.get(span_type, "unavailable")
    )
    span.set_attribute("error.type", error_type)
    # Low-cardinality error category (validation | policy | runtime) for control-/data-plane errors.
    raw_category = (
        overrides.get("gentoro.error.category")
        or overrides.get(config_attr("error.category"))
        or _ERROR_CATEGORY_BY_SPAN_TYPE.get(span_type)
    )
    if raw_category:
        span.set_attribute(config_attr("error.category"), raw_category)
    span.set_status(Status(StatusCode.ERROR, message))
    # Exception event: use provided type/message or derive from error.type (e.g. timeout → TimeoutError).
    if not exception_type or not exception_message:
        derived = _exception_from_error_type(error_type, span_type)
        exc_type = exception_type or derived[0]
        exc_msg = exception_message or derived[1]
    else:
        exc_type = exception_type
        exc_msg = exception_message
    msg = exc_msg or message
    if exc_type:
        exc_cls = type(exc_type, (RuntimeError,), {})
        exc = exc_cls(msg)
    else:
        exc = RuntimeError(msg)
    if hasattr(span, "record_exception"):
        span.record_exception(exc)


def _response_compose_exception_defaults(
    config: "SpanConfig",
    overrides: dict[str, Any],
    attr_fn: Any,
    error_type_override: str | None,
) -> tuple[str | None, str | None]:
    """Default exception type and message for response.compose when in error.
    When config has exception_type/exception_message (e.g. from scenario), use those.
    Otherwise derive from error.type so the exception event is SemConv-aligned.
    """
    if getattr(config, "exception_type", None) and getattr(config, "exception_message", None):
        return (
            config.exception_type,
            config.exception_message,
        )
    raw = (
        error_type_override
        or overrides.get("error.type")
        or overrides.get(attr_fn("error.type"))
        or "protocol_error"
    )
    # Map error.type to exception class name and message (OTEL exception event).
    _RESPONSE_COMPOSE_EXCEPTION_BY_ERROR_TYPE: dict[str, tuple[str, str]] = {
        "protocol_error": ("JsonSerializationError", "Failed to serialize A2A response payload"),
        "unavailable": ("ResponseCompositionError", "Response composition failed"),
        "tool_error": ("ResponseCompositionError", "Response composition failed"),
        "timeout": ("TimeoutError", "Response composition timed out"),
        "invalid_arguments": ("ValidationError", "Invalid response structure"),
    }
    exc_type, exc_msg = _RESPONSE_COMPOSE_EXCEPTION_BY_ERROR_TYPE.get(
        raw, ("ResponseCompositionError", "Response composition failed")
    )
    return exc_type, exc_msg


def _set_higher_latency_condition_attributes(
    span: Any,
    context: GenerationContext,
    attr_fn: Any,
) -> None:
    """Set span attributes from context.higher_latency_condition so telemetry captures the condition."""
    condition = getattr(context, "higher_latency_condition", None) if context else None
    if not condition or not isinstance(condition, dict):
        return
    keys = list(condition.keys())
    if not keys:
        return
    span.set_attribute(attr_fn("higher_latency.condition.keys"), ",".join(keys))
    # Structured condition keys with known shape
    if "peak_hours" in condition and isinstance(condition["peak_hours"], dict):
        ph = condition["peak_hours"]
        if isinstance(ph.get("timezone"), str):
            span.set_attribute(attr_fn("higher_latency.peak_hours.timezone"), ph["timezone"])
        if isinstance(ph.get("start_hour"), (int, float)):
            span.set_attribute(
                attr_fn("higher_latency.peak_hours.start_hour"), int(ph["start_hour"])
            )
        if isinstance(ph.get("end_hour"), (int, float)):
            span.set_attribute(attr_fn("higher_latency.peak_hours.end_hour"), int(ph["end_hour"]))
        if isinstance(ph.get("weekdays"), list):
            span.set_attribute(
                attr_fn("higher_latency.peak_hours.weekdays"), json.dumps(ph["weekdays"])
            )
    if "post_long_weekend" in condition and isinstance(condition["post_long_weekend"], dict):
        plw = condition["post_long_weekend"]
        if isinstance(plw.get("days_after"), (int, float)):
            span.set_attribute(
                attr_fn("higher_latency.post_long_weekend.days_after"), int(plw["days_after"])
            )
    # Arbitrary key-value conditions (claim_status_output, zip_code, etc.)
    for key in keys:
        if key in ("peak_hours", "post_long_weekend"):
            continue
        val = condition[key]
        if val is None:
            continue
        safe_key = key.replace(".", "_").replace(" ", "_").lower()
        attr_name = attr_fn(f"higher_latency.condition.{safe_key}")
        if isinstance(val, (str, int, float, bool)):
            span.set_attribute(attr_name, str(val) if not isinstance(val, bool) else val)
        else:
            span.set_attribute(attr_name, json.dumps(val))


SPAN_KIND_MAP = {
    SpanType.A2A_ORCHESTRATE: SpanKind.SERVER,
    SpanType.PLANNER: SpanKind.INTERNAL,
    SpanType.TASK_EXECUTE: SpanKind.INTERNAL,
    SpanType.LLM_CALL: SpanKind.CLIENT,
    SpanType.TOOLS_RECOMMEND: SpanKind.INTERNAL,
    SpanType.MCP_TOOL_EXECUTE: SpanKind.CLIENT,
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: SpanKind.CLIENT,
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: SpanKind.INTERNAL,
    SpanType.RESPONSE_COMPOSE: SpanKind.INTERNAL,
    SpanType.RAG_RETRIEVE: SpanKind.INTERNAL,
    SpanType.A2A_CALL: SpanKind.CLIENT,
    SpanType.PAYLOAD_VALIDATION: SpanKind.INTERNAL,
    SpanType.POLICY_VALIDATION: SpanKind.INTERNAL,
    SpanType.AUGMENTATION_VALIDATION: SpanKind.INTERNAL,
    SpanType.REQUEST_VALIDATION: SpanKind.SERVER,
    SpanType.RESPONSE_VALIDATION: SpanKind.SERVER,
    SpanType.CP_REQUEST: SpanKind.SERVER,
}

# Low-cardinality error.type per span type when status.code=ERROR.
# Values are from conventions/semconv.yaml (timeout, unavailable, invalid_arguments, tool_error, protocol_error).
_ERROR_TYPE_BY_SPAN_TYPE = {
    SpanType.A2A_ORCHESTRATE: "unavailable",
    SpanType.PLANNER: "unavailable",
    SpanType.TASK_EXECUTE: "tool_error",
    SpanType.LLM_CALL: "unavailable",
    SpanType.TOOLS_RECOMMEND: "protocol_error",
    SpanType.MCP_TOOL_EXECUTE: "tool_error",
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: "tool_error",
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: "protocol_error",
    SpanType.RESPONSE_COMPOSE: "protocol_error",
    SpanType.RAG_RETRIEVE: "unavailable",
    SpanType.A2A_CALL: "unavailable",
    SpanType.PAYLOAD_VALIDATION: "invalid_arguments",
    SpanType.POLICY_VALIDATION: "invalid_arguments",
    SpanType.AUGMENTATION_VALIDATION: "unavailable",
    SpanType.REQUEST_VALIDATION: "invalid_arguments",
    SpanType.RESPONSE_VALIDATION: "invalid_arguments",
    SpanType.CP_REQUEST: "invalid_arguments",
}

# Error category (validation | policy | runtime) for classification when spans fail.
_ERROR_CATEGORY_BY_SPAN_TYPE: dict[SpanType, str] = {
    # Control-plane validation
    SpanType.REQUEST_VALIDATION: "validation",
    SpanType.PAYLOAD_VALIDATION: "validation",
    SpanType.POLICY_VALIDATION: "policy",
    SpanType.AUGMENTATION_VALIDATION: "runtime",
    SpanType.RESPONSE_VALIDATION: "policy",
    # Data-plane orchestration and RAG/LLM/tooling
    SpanType.A2A_ORCHESTRATE: "runtime",
    SpanType.PLANNER: "runtime",
    SpanType.TASK_EXECUTE: "runtime",
    SpanType.LLM_CALL: "runtime",
    SpanType.TOOLS_RECOMMEND: "runtime",
    SpanType.MCP_TOOL_EXECUTE: "runtime",
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: "runtime",
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: "runtime",
    SpanType.RESPONSE_COMPOSE: "runtime",
    SpanType.RAG_RETRIEVE: "runtime",
    SpanType.A2A_CALL: "runtime",
    SpanType.CP_REQUEST: "validation",
}

# Single source of truth: span type -> span.class value (prefix applied at runtime via config_attr).
# All vendor-prefixed and control-plane spans that have a span class are listed here.
# NOTE: These values are aligned with the Gentoro management-plane pipeline expectations
# (see GentoroSpanDerivedStorageMapper in gentoro-enterprise), so control-plane validation
# spans use payload.validation / policy.validation / augmentation.validation.
SPAN_CLASS_BY_TYPE: dict[SpanType, str] = {
    SpanType.A2A_ORCHESTRATE: "a2a.orchestrate",
    SpanType.PLANNER: "planner",
    SpanType.TASK_EXECUTE: "task.execute",
    SpanType.LLM_CALL: "llm.call",
    SpanType.TOOLS_RECOMMEND: "tools.recommend",
    SpanType.MCP_TOOL_EXECUTE: "mcp.tool.execute",
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: "mcp.tool.execute.attempt",
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: "llm.tool.response.bridge",
    SpanType.RESPONSE_COMPOSE: "response.compose",
    SpanType.REQUEST_VALIDATION: "request.validation",
    SpanType.RESPONSE_VALIDATION: "response.validation",
    SpanType.PAYLOAD_VALIDATION: "payload.validation",
    SpanType.POLICY_VALIDATION: "policy.validation",
    SpanType.AUGMENTATION_VALIDATION: "augmentation.validation",
}

# Convention attributes per span type (step.outcome, response.format, etc.). No span.class here.
CONVENTION_ATTRIBUTES: dict[SpanType, dict[str, Any]] = {
    SpanType.A2A_ORCHESTRATE: {},
    SpanType.PLANNER: {},
    SpanType.TASK_EXECUTE: {},
    SpanType.LLM_CALL: {},
    SpanType.TOOLS_RECOMMEND: {},  # step.outcome from hierarchy attribute_overrides (loader)
    SpanType.MCP_TOOL_EXECUTE: {},
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: {},
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: {},
    SpanType.RESPONSE_COMPOSE: {},  # response.format, step.outcome from hierarchy attribute_overrides (loader)
    SpanType.PAYLOAD_VALIDATION: {},
    SpanType.POLICY_VALIDATION: {},
    SpanType.AUGMENTATION_VALIDATION: {},
    SpanType.REQUEST_VALIDATION: {},
    SpanType.RESPONSE_VALIDATION: {},
}

# Data-plane: resource attribute prefix.component per span type (semconv gentoro.component).
# Control-plane span types are not in this map; they use the default resource from config.
_DATA_PLANE_COMPONENT_BY_SPAN_TYPE: dict[SpanType, str] = {
    SpanType.A2A_ORCHESTRATE: "orchestrator",
    SpanType.PLANNER: "planning",
    SpanType.TASK_EXECUTE: "orchestrator",
    SpanType.RAG_RETRIEVE: "retrieval",
    SpanType.LLM_CALL: "llm",
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: "llm",
    SpanType.TOOLS_RECOMMEND: "tool_recommender",
    SpanType.MCP_TOOL_EXECUTE: "mcp_client",
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: "mcp_client",
    SpanType.RESPONSE_COMPOSE: "orchestrator",
}


def _get_component_for_span_type(span_type: SpanType) -> str | None:
    """Return data-plane component for resource attribute, or None for control-plane (use config default)."""
    return _DATA_PLANE_COMPONENT_BY_SPAN_TYPE.get(span_type)


@dataclass
class SpanConfig:
    """Configuration for generating a single span."""

    span_type: SpanType
    latency_mean_ms: float = 100.0
    latency_variance: float = 0.3
    error_rate: float = 0.02
    attribute_overrides: dict[str, Any] = field(default_factory=dict)
    # Optional: when span is in error, record exception event with this type/message (e.g. from control-plane template).
    exception_type: str | None = None
    exception_message: str | None = None
    # Optional: for VALIDATION_PAYLOAD fail, emit gentoro.validation.error events (list of {path, rule, code}).
    validation_errors: list[dict[str, Any]] | None = None
    # Optional: latency profile name (e.g. happy_path, higher_latency). Used to allow 5% spikes only for
    # higher_latency and retry contexts so happy_path stays below them.
    latency_profile: str | None = None
    # Optional: explicit heavy-tail configuration per span, derived from latency_profiles.<profile>.spans.<span>.spike.
    # When set, this overrides the generic 5% spike logic below.
    spike_probability: float | None = None
    spike_min_multiplier: float | None = None
    spike_max_multiplier: float | None = None


@dataclass
class TraceHierarchy:
    """Defines the structure of a complete trace with child spans."""

    root_config: SpanConfig
    children: list["TraceHierarchy"] = field(default_factory=list)

    def span_names(self) -> list[str]:
        """Return emitted span names in tree order (root then children)."""
        names = [get_span_name(self.root_config.span_type)]
        for child in self.children:
            names.extend(child.span_names())
        return names


def _subtree_has_configured_failure(
    hierarchy: TraceHierarchy,
) -> tuple[bool, str | None, SpanConfig | None]:
    """Return (True, error_type, failing_config) if this subtree or any descendant has logical failure.

    Logical failure is per node: step.outcome=fail or (for attempts) mcp.attempt.outcome=fail.
    We recurse so task.execute wrapping MCP is detected (task root=success, MCP root=fail).
    For MCP_TOOL_EXECUTE we do not recurse into attempts: the parent's step.outcome is the
    logical call outcome (retry-then-success has parent step.outcome=success).
    failing_config is the SpanConfig of the node that has the failure (for exception_type/exception_message).
    """
    cfg = hierarchy.root_config
    overrides = cfg.attribute_overrides or {}
    step_outcome = (overrides.get(config_attr("step.outcome")) or "").strip().lower()
    attempt_outcome = (overrides.get(config_attr("mcp.attempt.outcome")) or "").strip().lower()
    root_has_fail = cfg.error_rate >= 1.0 or step_outcome == "fail" or attempt_outcome == "fail"
    if root_has_fail:
        error_type = overrides.get("error.type") or overrides.get(config_attr("error.type"))
        if isinstance(error_type, str):
            return True, error_type, cfg
        for child in hierarchy.children:
            child_fail, child_err, child_cfg = _subtree_has_configured_failure(child)
            if child_fail and child_err:
                return True, child_err, child_cfg or cfg
        return True, _ERROR_TYPE_BY_SPAN_TYPE.get(cfg.span_type), cfg
    # Root success: recurse so task->mcp (task success, mcp fail) is detected; do not recurse into MCP attempts.
    if cfg.span_type == SpanType.MCP_TOOL_EXECUTE:
        return False, None, None
    for child in hierarchy.children:
        found, child_err, failing_cfg = _subtree_has_configured_failure(child)
        if found:
            return True, child_err or "tool_error", failing_cfg
    return False, None, None


class SpanBuilder:
    """Build individual spans with proper attributes and timing."""

    def __init__(
        self,
        schema: TelemetrySchema,
        attr_generator: AttributeGenerator,
        tracer: Tracer,
    ):
        """Initialize span builder."""
        self.schema = schema
        self.attr_generator = attr_generator
        self.tracer = tracer

    def generate_latency(self, config: SpanConfig) -> float:
        """Generate latency based on config.

        Base latency uses a lognormal distribution (right-skewed, more realistic for latency)
        parameterized by latency_mean_ms and latency_variance (treated as coefficient of
        variation, roughly matching the old gaussian multiplier behavior).

        On top of the base distribution, we optionally apply heavy-tail behavior:

        - If SpanConfig has explicit spike_probability / spike_*_multiplier (from config latency_profiles),
          we use those values.
        - Otherwise we apply:
          - a small-probability heavy-tail multiplier (rare slow outliers)
          - plus an optional moderate spike (more common than the heavy-tail)

        We keep the "happy_path should not randomly exceed higher_latency" property by
        suppressing non-explicit spikes when latency_profile == "happy_path" and mean_ms < 2000.
        """
        mean_ms = max(0.0, float(config.latency_mean_ms))
        cv = max(0.0, float(config.latency_variance))
        profile = (config.latency_profile or "").strip().lower()

        # Lognormal parameters derived from mean and coefficient of variation:
        # If X ~ LogNormal(mu, sigma^2), then:
        #   E[X] = exp(mu + sigma^2 / 2)
        #   CV^2 = exp(sigma^2) - 1
        # => sigma^2 = ln(1 + CV^2), mu = ln(mean) - sigma^2/2
        if mean_ms <= 0.0:
            latency = 0.0
        else:
            # Cap cv to avoid extreme params when variance is misconfigured.
            cv = min(cv, 3.0)
            sigma2 = math.log(1.0 + (cv * cv))
            sigma = math.sqrt(max(0.0, sigma2))
            mu = math.log(mean_ms) - (sigma2 / 2.0)
            latency = random.lognormvariate(mu, sigma)

        # Determine spike parameters: explicit per-span config wins, otherwise fallback heuristic.
        explicit_prob = config.spike_probability if config.spike_probability is not None else 0.0
        explicit_min = config.spike_min_multiplier if config.spike_min_multiplier is not None else 1.0
        explicit_max = config.spike_max_multiplier if config.spike_max_multiplier is not None else 1.0

        if explicit_prob > 0.0 and explicit_max > 1.0 and explicit_max >= explicit_min:
            if random.random() < explicit_prob:
                latency *= random.uniform(explicit_min, explicit_max)
        else:
            allow_spike = profile != "happy_path" or mean_ms >= 2000.0
            if allow_spike:
                # Rare heavy-tail outliers (e.g. backend stalls, queueing).
                # random.paretovariate(alpha) returns >= 1; we convert to a multiplier >= 1.
                # Use a modest alpha so outliers exist but are rare and bounded-ish.
                if random.random() < 0.01:
                    tail = random.paretovariate(3.0)  # ~1..inf, heavy tail
                    latency *= min(12.0, tail)  # keep within a sane range for dashboards
                # More common moderate spikes (cold path, cache miss).
                elif random.random() < 0.04:
                    latency *= random.uniform(1.8, 3.5)

        # Avoid 0/negative and keep a small floor so spans are visible.
        return max(15.0, float(latency))

    def should_error(self, config: SpanConfig) -> bool:
        """Determine if span should have error status."""
        return random.random() < config.error_rate

    def get_attributes(
        self,
        span_type: SpanType,
        context: GenerationContext,
        overrides: dict[str, Any],
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Get attributes for the span type from schema, context, and convention defaults."""
        tid = trace_id
        if span_type == SpanType.CP_REQUEST and tid is None:
            tid = uuid.uuid4().hex
        full_span_name = get_span_name(span_type)
        attrs = self.attr_generator.generate_attributes_for_span(
            full_span_name,
            context,
            overrides,
            trace_id=tid,
        )
        # span.class from single source of truth (span type -> value); prefix applied here.
        if span_type in SPAN_CLASS_BY_TYPE:
            attrs[config_attr("span.class")] = SPAN_CLASS_BY_TYPE[span_type]
        # Convention attributes (step.outcome, response.format, etc.).
        convention = CONVENTION_ATTRIBUTES.get(span_type)
        if convention:
            for k, v in convention.items():
                attrs.setdefault(k, v)
        # vendor.request.validation: tenant id, request id, enduser pseudo id. vendor.response.validation: tenant id only (no enduser).
        if span_type in (SpanType.REQUEST_VALIDATION, SpanType.RESPONSE_VALIDATION):
            attrs[config_attr("tenant.id")] = context.tenant_id
        if span_type == SpanType.REQUEST_VALIDATION:
            attrs[config_attr("request.id")] = context.request_id
            attrs[config_attr("enduser.pseudo.id")] = context.user_id or ""
            # Include target agent identity on incoming validation so control-plane
            # spans can be correlated to the same agent as the data-plane root.
            agent_target = overrides.get(config_attr("a2a.agent.target.id"))
            if isinstance(agent_target, str) and agent_target.strip():
                attrs[config_attr("a2a.agent.target.id")] = agent_target.strip()
        # vendor.a2a.orchestrate gets enduser.pseudo.id from schema (a2a_orchestrate_attributes)
        # vendor.a2a.orchestrate root: only root-level attrs (no response.format, no route).
        # vendor.response.compose: only compose attrs (no a2a.outcome, a2a.agent.target.id).
        if span_type == SpanType.A2A_ORCHESTRATE:
            attrs.pop(config_attr("response.format"), None)
            attrs.pop(config_attr("route"), None)
        elif span_type == SpanType.RESPONSE_COMPOSE:
            attrs.pop(config_attr("a2a.outcome"), None)
            attrs.pop(config_attr("a2a.agent.target.id"), None)
        # When step.outcome or mcp.attempt.outcome is success, do not set error attributes (semantically correct).
        step_outcome = overrides.get(config_attr("step.outcome"))
        attempt_outcome = overrides.get(config_attr("mcp.attempt.outcome"))
        success = (isinstance(step_outcome, str) and step_outcome.strip().lower() == "success") or (
            isinstance(attempt_outcome, str) and attempt_outcome.strip().lower() == "success"
        )
        if success:
            attrs.pop("error.type", None)
            attrs.pop(config_attr("error.type"), None)
            attrs.pop(config_attr("error.category"), None)
        # task.execute: success implies no retries and no fallback used (semantic consistency).
        if span_type == SpanType.TASK_EXECUTE:
            step_outcome_val = attrs.get(config_attr("step.outcome")) or overrides.get(
                config_attr("step.outcome")
            )
            if isinstance(step_outcome_val, str) and step_outcome_val.strip().lower() == "success":
                attrs[config_attr("task.retry.count")] = 0
                attrs[config_attr("task.fallback.used")] = False
        # When mcp.tool.execute has retry.count 0 there is only one attempt (index 1); do not set retry.reason.
        # retry.reason is only for retries (attempt index > 1).
        if span_type == SpanType.MCP_TOOL_EXECUTE_ATTEMPT:
            attempt_index = overrides.get(config_attr("mcp.attempt.index"))
            if attempt_index == 1:
                attrs.pop(config_attr("retry.reason"), None)
        # Higher-latency condition is set only on a2a.orchestrate by _set_higher_latency_condition_attributes.
        # Strip schema-driven higher_latency.* from all root span types; real values set only on a2a.orchestrate.
        if span_type in (
            SpanType.CP_REQUEST,
            SpanType.REQUEST_VALIDATION,
            SpanType.A2A_ORCHESTRATE,
            SpanType.RESPONSE_VALIDATION,
        ):
            for key in list(attrs.keys()):
                if "higher_latency" in key:
                    attrs.pop(key, None)
        # gen_ai.tool.call.arguments and gen_ai.tool.call.result only on attempt spans, not parent.
        if span_type == SpanType.MCP_TOOL_EXECUTE:
            attrs.pop("gen_ai.tool.call.arguments", None)
            attrs.pop("gen_ai.tool.call.result", None)
        return attrs


class _PrintSpanProcessor:
    """SpanProcessor that prints full span content (name, attributes, etc.) to stdout."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        trace_id = format(span.context.trace_id, "032x")
        span_id = format(span.context.span_id, "016x")
        parent_id = format(span.parent.span_id, "016x") if span.parent else ""
        kind = span.kind.name if span.kind else "INTERNAL"
        status = span.status.status_code.name if span.status else "UNSET"
        print(
            f"   span name={span.name} trace_id={trace_id} span_id={span_id} parent_id={parent_id} kind={kind} status={status}"
        )
        if span.attributes:
            for k, v in sorted(span.attributes.items()):
                val_str = str(v)  # handles str, int, float, bool, sequence
                print(f"      {k}={val_str}")
        print()

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _RefCountingSpanExporter(SpanExporter):
    """Wraps a SpanExporter so shutdown() is only forwarded once after all refs have called shutdown.
    Used when the same logical exporter is shared by multiple BatchSpanProcessors (one per provider).
    """

    def __init__(self, exporter: SpanExporter, shared_ref_count: list[int]):
        self._exporter = exporter
        self._ref_count = shared_ref_count
        self._lock: Any = __import__("threading").Lock()

    def export(self, spans: Any) -> SpanExportResult:
        return self._exporter.export(spans)

    def shutdown(self) -> None:
        with self._lock:
            self._ref_count[0] -= 1
            if self._ref_count[0] <= 0:
                self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


class TraceGenerator:
    """Generate complete traces with hierarchical spans."""

    def __init__(
        self,
        exporter: SpanExporter,
        schema_path: str | None = None,
        service_name: str = "otelsim",
        service_version: str = "1.0.0",
        show_full_spans: bool = False,
    ):
        """Initialize trace generator with exporter."""
        parser = SchemaParser(schema_path)
        self.schema = parser.parse()
        self.attr_generator = AttributeGenerator(self.schema)

        tenant_id = get_default_tenant_ids()[0]
        # Tracers keyed by (component, scenario_name_key); scope name = otelsim.{scenario_name} when set
        self._tracers: dict[tuple[str | None, str], Tracer] = {}
        self._provider_by_component: dict[str | None, TracerProvider] = {}
        self._providers: list[TracerProvider] = []

        def make_resource(component: str | None) -> Resource:
            attrs = dict(config_resource_attributes(tenant_id, component=component))
            attrs["service.name"] = service_name
            attrs["service.version"] = service_version
            return Resource.create(attrs, schema_url=resource_schema_url())

        # Share one exporter across N BatchSpanProcessors; only shut it down after all have shut down.
        num_providers = 1 + len(DATA_PLANE_COMPONENT_VALUES)
        ref_count: list[int] = [num_providers]

        # Default provider for control-plane (no component override)
        default_provider = TracerProvider(resource=make_resource(None))
        if show_full_spans:
            default_provider.add_span_processor(_PrintSpanProcessor())  # type: ignore[arg-type]
        default_provider.add_span_processor(
            BatchSpanProcessor(_RefCountingSpanExporter(exporter, ref_count))
        )
        self._providers.append(default_provider)
        self._provider_by_component[None] = default_provider
        # Instrumentation scope: name = module, version = otelsim package version so otel.scope.version is populated.
        self._tracers[(None, "")] = default_provider.get_tracer(__name__, _OTELSIM_VERSION)
        trace.set_tracer_provider(default_provider)

        # One provider per data-plane component so resource attr prefix.component is correct
        for comp in DATA_PLANE_COMPONENT_VALUES:
            prov = TracerProvider(resource=make_resource(comp))
            prov.add_span_processor(
                BatchSpanProcessor(_RefCountingSpanExporter(exporter, ref_count))
            )
            self._providers.append(prov)
            self._provider_by_component[comp] = prov
            self._tracers[(comp, "")] = prov.get_tracer(__name__, _OTELSIM_VERSION)

        self.tracer = self._tracers[(None, "")]
        self.span_builder = SpanBuilder(self.schema, self.attr_generator, self.tracer)

    def _get_tracer(self, component: str | None, scenario_name: str | None = None) -> Tracer:
        """Return tracer for the given component and optional scenario; scope name = otelsim.{scenario_name} when set."""
        scenario_key = _sanitize_scope_name(scenario_name) if scenario_name else ""
        key = (component, scenario_key)
        if key not in self._tracers:
            provider = self._provider_by_component.get(component, self._provider_by_component[None])
            scope_name = f"otelsim.{scenario_key}" if scenario_key else __name__
            # Use otelsim package version for instrumentation scope version so otel.scope.version is non-empty.
            self._tracers[key] = provider.get_tracer(scope_name, _OTELSIM_VERSION)
        return self._tracers[key]

    def generate_trace(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext | None = None,
        span_callback: Any | None = None,
    ) -> str:
        """Generate a complete trace from hierarchy definition.

        Multi-turn semantics: each call models one request reaching the control-plane
        (incoming request). A new trace_id is generated for each call. The caller must
        pass the same context.session_id for all requests in the same session so that
        spans carry the same session id across turns.
        """
        if context is None:
            context = GenerationContext.create()

        planned_latencies_ms, root_start_ns = self._plan_trace_timing(hierarchy)
        trace_id, _, _, _ = self._generate_span_recursive(
            hierarchy,
            context,
            None,
            None,
            span_callback=span_callback,
            planned_latencies_ms=planned_latencies_ms,
            logical_start_ns=root_start_ns,
        )
        return trace_id

    def generate_unified_request_trace(
        self,
        incoming_hierarchy: TraceHierarchy,
        data_plane_hierarchy: TraceHierarchy,
        outgoing_hierarchy: TraceHierarchy,
        context: GenerationContext,
        span_callback: Any | None = None,
    ) -> str:
        """Generate one trace for one request: incoming at control-plane -> data-plane -> response validation.

        One request reaching control-plane is modeled as one incoming trace: trace_id is
        generated when the request hits the control-plane and shared (e.g. cp.incoming_trace_id,
        cp.outgoing_trace_id). For multi-turn: each new request to control-plane must be a
        separate call, yielding a different trace_id; use the same context.session_id for
        all calls that belong to the same session.

        Builds a single combined hierarchy so parent spans (request.validation, a2a.orchestrate)
        are ended only after all logical children have been generated. That keeps parent duration
        >= sum of children and avoids clock-skew warnings (child not contained in parent).
        Uses a single TracerProvider for the whole trace so all spans export in one batch.
        """
        # Unified tree: request.validation -> [payload, policy, augmentation, a2a.orchestrate -> [planner, tasks..., response_compose, response.validation -> [policy.validation...]]]
        # So request.validation's end_time is computed from last_child_end_ns including a2a.orchestrate (and thus response.validation).
        data_plane_with_outgoing = TraceHierarchy(
            root_config=data_plane_hierarchy.root_config,
            children=list(data_plane_hierarchy.children) + [outgoing_hierarchy],
        )
        unified_hierarchy = TraceHierarchy(
            root_config=incoming_hierarchy.root_config,
            children=list(incoming_hierarchy.children) + [data_plane_with_outgoing],
        )
        planned_latencies_ms, root_start_ns = self._plan_trace_timing(unified_hierarchy)
        trace_id, _, _, _ = self._generate_span_recursive(
            unified_hierarchy,
            context,
            None,
            None,
            span_callback=span_callback,
            use_single_tracer=True,
            planned_latencies_ms=planned_latencies_ms,
            logical_start_ns=root_start_ns,
        )
        return trace_id

    def _plan_trace_timing(
        self,
        hierarchy: TraceHierarchy,
    ) -> tuple[dict[int, float], int]:
        """Plan simulated latencies and an anchored root start time.

        We generate one simulated latency per SpanConfig (per trace emission) and compute
        the total duration of the resulting sequential hierarchy (with _MIN_SPAN_GAP_NS gaps).
        We then shift the whole trace timeline into the past before emitting spans, so
        explicit start/end timestamps are never in the future.
        """

        planned_latencies_ms: dict[int, float] = {}

        def plan_node(node: TraceHierarchy, start_ns: int) -> int:
            cfg = node.root_config
            latency_ms = self.span_builder.generate_latency(cfg)
            planned_latencies_ms[id(cfg)] = latency_ms

            last_child_end_ns = start_ns
            next_child_start_ns = start_ns + _MIN_SPAN_GAP_NS
            for child in node.children:
                child_end_ns = plan_node(child, next_child_start_ns)
                if child_end_ns > last_child_end_ns:
                    last_child_end_ns = child_end_ns
                next_child_start_ns = child_end_ns + _MIN_SPAN_GAP_NS

            span_end_from_latency_ns = start_ns + int(latency_ms * 1_000_000)
            return max(span_end_from_latency_ns, last_child_end_ns)

        # Use a synthetic 0-based clock for planning.
        end_ns = plan_node(hierarchy, 0)
        total_duration_ns = max(0, end_ns)

        # Anchor in the past so span timestamps are never "in the future" relative to wall clock.
        # Some backends drop/ignore future-timestamp spans, which looks like missing spans.
        root_start_ns = time.time_ns() - total_duration_ns - _MIN_SPAN_GAP_NS
        return planned_latencies_ms, root_start_ns

    def _generate_span_recursive(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext,
        parent_span: Any,
        parent_context: Any | None = None,
        compose_accumulated_failure: str | None = None,
        use_single_tracer: bool = False,
        logical_start_ns: int | None = None,
        planned_latencies_ms: dict[int, float] | None = None,
        span_callback: Any | None = None,
    ) -> tuple[str, str, int, float]:
        """Recursively generate spans in the hierarchy. Returns (trace_id_hex, span_id_hex, end_time_ns, tool_latency_ms).
        end_time_ns is the span's logical end timestamp so parents can ensure end >= last child end.
        tool_latency_ms is the gentoro.tool.latency_ms value for this span (non-zero only for MCP tool execute/attempt).
        compose_accumulated_failure: when set, response.compose accumulates this error (step.outcome=fail, status, exception).
        use_single_tracer: when True, use the default tracer for all spans so they export in one batch (avoids parent-not-in-trace).
        logical_start_ns: when set, use as span start time (epoch ns) so gaps and parent-encompassing children are preserved in the waterfall.
        """
        config = hierarchy.root_config
        # When at A2A root, detect if any child (including response_compose) is configured to fail;
        # pass so response.compose accumulates step.outcome, status, error.type, and exception.
        # We also keep the failing child's config so a2a.orchestrate can record the same exception_type/exception_message.
        child_compose_failure = compose_accumulated_failure
        failing_child_config: SpanConfig | None = None
        if config.span_type == SpanType.A2A_ORCHESTRATE:
            for ch in hierarchy.children:
                has_fail, err_type, fail_cfg = _subtree_has_configured_failure(ch)
                if has_fail:
                    child_compose_failure = (err_type or "tool_error").strip()
                    if child_compose_failure not in SEMCONV_ERROR_TYPE_VALUES:
                        child_compose_failure = "tool_error"
                    failing_child_config = fail_cfg
                    break

        span_name = get_span_name(config.span_type)
        span_kind = SPAN_KIND_MAP.get(config.span_type, SpanKind.INTERNAL)

        if planned_latencies_ms is not None:
            latency_ms = planned_latencies_ms.get(id(config), None)
        else:
            latency_ms = None
        if latency_ms is None:
            latency_ms = self.span_builder.generate_latency(config)

        overrides = config.attribute_overrides or {}
        component = _get_component_for_span_type(config.span_type)
        scenario_name = getattr(context, "scenario_name", None) if context else None
        scenario_key = _sanitize_scope_name(scenario_name) if scenario_name else ""
        scope_name_for_span = f"otelsim.{scenario_key}" if scenario_key else __name__

        # When generating a unified request trace (control-plane + data-plane in one logical trace),
        # we still want data-plane spans to carry the correct resource.gentoro.module="data-plane"
        # and component (orchestrator, llm, etc.) so downstream pipelines can distinguish modules
        # purely from resource attributes (as in the Gentoro reference generator).
        #
        # To achieve this, we continue to use a dedicated TracerProvider per data-plane component
        # even when use_single_tracer=True; providers share the same exporter, and context propagation
        # keeps trace_id/span_id consistent across control-plane and data-plane spans.
        if use_single_tracer and component is None:
            tracer_component = None
        else:
            tracer_component = component

        tracer = self._get_tracer(tracer_component, scenario_name)
        current_trace_id = ""
        current_span_id = ""
        span_attrs = self.span_builder.get_attributes(
            config.span_type,
            context,
            overrides,
        )
        kwargs: dict[str, Any] = {
            "kind": span_kind,
            "attributes": span_attrs,
        }
        if parent_context is not None:
            kwargs["context"] = parent_context
        # Use logical start time so root encompasses children and there is a minimal gap between spans for the waterfall.
        start_time_ns = logical_start_ns if logical_start_ns is not None else time.time_ns()
        kwargs["start_time"] = start_time_ns
        # end_on_exit=False so we can call span.end(end_time=...) once; avoids "Calling end() on an ended span" warning.
        with tracer.start_as_current_span(span_name, **kwargs, end_on_exit=False) as span:
            current_trace_id = format(span.get_span_context().trace_id, "032x")
            current_span_id = format(span.get_span_context().span_id, "016x")
            # Single source of truth for span source: mirror otel.scope.name (e.g. otelsim.<scenario_name>).
            # This allows downstream pipelines to attribute spans to the logical scenario file.
            span.set_attribute(config_attr("span.source"), scope_name_for_span)
            if span_callback is not None:
                try:
                    span_callback(config, current_trace_id, current_span_id)
                except Exception:
                    pass
            # When using a single tracer (unified trace), resource has no per-span component; set as span attribute for filtering.
            if use_single_tracer and component is not None:
                span.set_attribute(config_attr("component"), component)

            # Record simulated latency/duration as span attribute so telemetry reflects config/modifier.
            # MCP_TOOL_EXECUTE parent gets tool.latency_ms set after children (sum of attempt latencies).
            latency_attr_ms = round(latency_ms)
            tool_latency_ms_return = 0.0
            if config.span_type == SpanType.MCP_TOOL_EXECUTE_ATTEMPT:
                span.set_attribute(config_attr("tool.latency_ms"), latency_attr_ms)
                tool_latency_ms_return = float(latency_attr_ms)
                # OpenTelemetry GenAI: gen_ai.tool.call.arguments and gen_ai.tool.call.result
                # on tool attempts when scenario-defined arguments/results are available.
                # This applies to both successful and failed attempts.
                try:
                    tool_name_attr = None
                    if isinstance(span_attrs, dict):
                        tool_name_attr = span_attrs.get("gen_ai.tool.name")
                    tool_name = (
                        tool_name_attr.strip()
                        if isinstance(tool_name_attr, str) and tool_name_attr.strip()
                        else "unknown.tool"
                    )
                    simple_key = tool_name.split(".")[-1]

                    # Arguments: prefer scenario-defined tool_call_arguments; fall back to single entry.
                    ctx_tool_args = getattr(context, "tool_call_arguments", None)
                    args_obj: dict[str, Any] | None = None
                    if isinstance(ctx_tool_args, dict) and ctx_tool_args:
                        val = ctx_tool_args.get(tool_name)
                        if not isinstance(val, dict):
                            val = ctx_tool_args.get(simple_key)
                        if not isinstance(val, dict) and len(ctx_tool_args) == 1:
                            only_val = next(iter(ctx_tool_args.values()))
                            if isinstance(only_val, dict):
                                val = only_val
                        if isinstance(val, dict):
                            args_obj = val
                    if args_obj is not None:
                        args_json = json.dumps(args_obj)
                        if len(args_json) > 512:
                            args_json = args_json[:509] + "..."
                        span.set_attribute("gen_ai.tool.call.arguments", args_json)

                    # Results: prefer scenario-defined tool_call_results; same resolution strategy as arguments.
                    ctx_tool_results = getattr(context, "tool_call_results", None)
                    result_obj: dict[str, Any] | None = None
                    if isinstance(ctx_tool_results, dict) and ctx_tool_results:
                        val = ctx_tool_results.get(tool_name)
                        if not isinstance(val, dict):
                            val = ctx_tool_results.get(simple_key)
                        if not isinstance(val, dict) and len(ctx_tool_results) == 1:
                            only_val = next(iter(ctx_tool_results.values()))
                            if isinstance(only_val, dict):
                                val = only_val
                        if isinstance(val, dict):
                            result_obj = val
                    # Only emit gen_ai.tool.call.result when scenario-defined results are available;
                    # do not synthesize a fallback payload from call arguments.
                    if result_obj is not None:
                        result_json = json.dumps(result_obj)
                        if len(result_json) > 512:
                            result_json = result_json[:509] + "..."
                        span.set_attribute("gen_ai.tool.call.result", result_json)
                except Exception:
                    # Best-effort only; failures here must not break trace generation.
                    pass
            elif config.span_type == SpanType.MCP_TOOL_EXECUTE:
                pass  # parent: set tool.latency_ms after children
            elif config.span_type == SpanType.A2A_ORCHESTRATE:
                span.set_attribute(config_attr("orchestration.duration_ms"), latency_attr_ms)
            elif config.span_type == SpanType.LLM_CALL:
                span.set_attribute(config_attr("llm.latency_ms"), latency_attr_ms)
            elif config.span_type == SpanType.RAG_RETRIEVE:
                span.set_attribute(config_attr("rag.latency_ms"), latency_attr_ms)
            elif config.span_type == SpanType.A2A_CALL:
                span.set_attribute(config_attr("a2a.latency_ms"), latency_attr_ms)
            elif config.span_type == SpanType.CP_REQUEST:
                span.set_attribute(config_attr("cp.request.duration_ms"), latency_attr_ms)
            elif config.span_type == SpanType.TOOLS_RECOMMEND:
                span.set_attribute(config_attr("mcp.selection.latency.ms"), latency_attr_ms)
            elif config.span_type == SpanType.REQUEST_VALIDATION:
                span.set_attribute(config_attr("request.validation.duration_ms"), latency_attr_ms)
            elif config.span_type == SpanType.RESPONSE_VALIDATION:
                span.set_attribute(config_attr("response.validation.duration_ms"), latency_attr_ms)
            elif config.span_type in (
                SpanType.PLANNER,
                SpanType.TASK_EXECUTE,
                SpanType.RESPONSE_COMPOSE,
                SpanType.PAYLOAD_VALIDATION,
                SpanType.POLICY_VALIDATION,
                SpanType.AUGMENTATION_VALIDATION,
            ):
                span.set_attribute(config_attr("span.duration_ms"), latency_attr_ms)

            # Incoming request at control-plane: trace_id is generated here and shared afterwards.
            if config.span_type == SpanType.CP_REQUEST:
                span.set_attribute(config_attr("cp.incoming_trace_id"), current_trace_id)
                if overrides.get(config_attr("cp.status.code")) in [
                    "ALLOWED",
                    "FLAGGED",
                ]:
                    span.set_attribute(config_attr("cp.outgoing_trace_id"), current_trace_id)

            # End-user or upstream agent raw request: emit as a control-plane event on incoming validation root.
            if config.span_type == SpanType.REQUEST_VALIDATION:
                raw_req = getattr(context, "raw_request", None)
                raw_req_red = getattr(context, "raw_request_redacted", None)
                if raw_req or raw_req_red:
                    event_attrs: dict[str, Any] = {}
                    if raw_req:
                        event_attrs[config_attr("enduser.request.raw")] = raw_req
                    if raw_req_red:
                        event_attrs[config_attr("enduser.request.raw.redacted")] = raw_req_red
                    span.add_event("gentoro.enduser.request", event_attrs)

            # Agent / A2A final response: emit as a control-plane event on outgoing response validation root.
            if config.span_type == SpanType.RESPONSE_VALIDATION:
                final_resp = getattr(context, "final_response", None)
                final_resp_red = getattr(context, "final_response_redacted", None)
                if final_resp or final_resp_red:
                    response_event_attrs: dict[str, Any] = {}
                    if final_resp:
                        response_event_attrs[config_attr("agent.response.raw")] = final_resp
                    if final_resp_red:
                        response_event_attrs[config_attr("agent.response.raw.redacted")] = (
                            final_resp_red
                        )
                    span.add_event("gentoro.agent.response", response_event_attrs)

            # Capture higher_latency_condition on a2a.orchestrate only (same span as orchestration.duration_ms).
            if config.span_type == SpanType.A2A_ORCHESTRATE:
                _set_higher_latency_condition_attributes(span, context, config_attr)

            # LLM span: no direct tool execution at this layer; gen_ai.* on this span only reflects the model call itself
            # (system instructions, input/output messages, usage, etc.). Tool recommendation and execution are captured on
            # gentoro.tools.recommend and gentoro.mcp.tool.execute spans respectively.
            is_error = self.span_builder.should_error(config)
            tool_result = overrides.get(config_attr("tool.status.result"))
            # When overrides set step.outcome=success or mcp.attempt.outcome=success, do not mark span as error.
            step_outcome_override = overrides.get(config_attr("step.outcome"))
            attempt_outcome_override = overrides.get(config_attr("mcp.attempt.outcome"))
            override_success = (
                isinstance(step_outcome_override, str)
                and step_outcome_override.strip().lower() == "success"
            ) or (
                isinstance(attempt_outcome_override, str)
                and attempt_outcome_override.strip().lower() == "success"
            )
            if override_success:
                is_error = False
            # Spec 5.2: logical call failure — parent MCP span MUST set status.code=ERROR when step.outcome=fail.
            if (
                config.span_type == SpanType.MCP_TOOL_EXECUTE
                and isinstance(step_outcome_override, str)
                and step_outcome_override.strip().lower() == "fail"
            ):
                is_error = True
            # a2a.orchestrate root: status.code UNSET on success/partial, ERROR on error.
            # Propagate upstream failure to root so the response reflects error (semconv: root_on_compose_failure).
            if config.span_type == SpanType.A2A_ORCHESTRATE:
                outcome = (overrides.get(config_attr("a2a.outcome")) or "success").lower()
                if outcome == "error":
                    _record_span_error(
                        span,
                        config.span_type,
                        overrides,
                        exception_type=getattr(config, "exception_type", None),
                        exception_message=getattr(config, "exception_message", None),
                    )
                elif child_compose_failure is not None:
                    # A child (e.g. response_compose, MCP, tools_recommend) has configured failure; root must show error
                    # (semconv: root_on_compose_failure). Accumulate same exception_type/exception_message to a2a.orchestrate.
                    span.set_attribute(config_attr("a2a.outcome"), "error")
                    error_type = (
                        overrides.get("error.type")
                        or overrides.get(config_attr("error.type"))
                        or child_compose_failure
                    )
                    if isinstance(error_type, str) and error_type not in SEMCONV_ERROR_TYPE_VALUES:
                        error_type = "tool_error"
                    exc_type = (
                        getattr(failing_child_config, "exception_type", None)
                        if failing_child_config
                        else getattr(config, "exception_type", None)
                    )
                    exc_msg = (
                        getattr(failing_child_config, "exception_message", None)
                        if failing_child_config
                        else getattr(config, "exception_message", None)
                    )
                    _record_span_error(
                        span,
                        config.span_type,
                        overrides,
                        error_type_override=error_type,
                        exception_type=exc_type,
                        exception_message=exc_msg,
                    )
                # success / partial: leave status UNSET (do not set OK)
            # Control-plane request / response validation roots: status.code UNSET on
            # allowed/blocked; ERROR on runtime/system failure (outcome=error).
            elif config.span_type == SpanType.REQUEST_VALIDATION:
                outcome = (overrides.get(config_attr("request.outcome")) or "allowed").lower()
                if outcome == "error":
                    _record_span_error(
                        span,
                        config.span_type,
                        overrides,
                        exception_type=getattr(config, "exception_type", None),
                        exception_message=getattr(config, "exception_message", None),
                    )
            elif config.span_type == SpanType.RESPONSE_VALIDATION:
                outcome = (overrides.get(config_attr("response.outcome")) or "allowed").lower()
                if outcome == "error":
                    _record_span_error(
                        span,
                        config.span_type,
                        overrides,
                        exception_type=getattr(config, "exception_type", None),
                        exception_message=getattr(config, "exception_message", None),
                    )
            elif (
                config.span_type == SpanType.RESPONSE_COMPOSE
                and (is_error or tool_result is False or compose_accumulated_failure)
                and (not override_success or compose_accumulated_failure)
            ):
                span.set_attribute(config_attr("step.outcome"), "fail")
                # Use error.type from overrides or accumulated from upstream (e.g. MCP 4xx → invalid_arguments).
                error_type_override = (
                    overrides.get("error.type")
                    or overrides.get(config_attr("error.type"))
                    or compose_accumulated_failure
                )
                exc_type, exc_msg = _response_compose_exception_defaults(
                    config, overrides, config_attr, error_type_override
                )
                _record_span_error(
                    span,
                    config.span_type,
                    overrides,
                    error_type_override=error_type_override,
                    exception_type=exc_type,
                    exception_message=exc_msg,
                )
            elif (
                config.span_type == SpanType.TASK_EXECUTE
                and (is_error or tool_result is False)
                and not override_success
            ):
                span.set_attribute(config_attr("step.outcome"), "fail")
                _record_span_error(
                    span,
                    config.span_type,
                    overrides,
                    exception_type=getattr(config, "exception_type", None),
                    exception_message=getattr(config, "exception_message", None),
                )
            elif (
                config.span_type == SpanType.TOOLS_RECOMMEND
                and (is_error or tool_result is False)
                and not override_success
            ):
                span.set_attribute(config_attr("step.outcome"), "fail")
                _record_span_error(
                    span,
                    config.span_type,
                    overrides,
                    exception_type=getattr(config, "exception_type", None),
                    exception_message=getattr(config, "exception_message", None),
                )
            elif config.span_type == SpanType.PAYLOAD_VALIDATION and getattr(
                config, "validation_errors", None
            ):
                # Payload validation failed with one or more validation.error events (no exception event).
                validation_errors = config.validation_errors or []
                error_type = overrides.get(
                    config_attr("error.type")
                ) or _ERROR_TYPE_BY_SPAN_TYPE.get(SpanType.PAYLOAD_VALIDATION, "invalid_arguments")
                if error_type not in SEMCONV_ERROR_TYPE_VALUES:
                    error_type = "invalid_arguments"
                span.set_attribute("error.type", error_type)
                if _ERROR_CATEGORY_BY_SPAN_TYPE.get(SpanType.PAYLOAD_VALIDATION):
                    span.set_attribute(
                        config_attr("error.category"),
                        _ERROR_CATEGORY_BY_SPAN_TYPE[SpanType.PAYLOAD_VALIDATION],
                    )
                span.set_status(Status(StatusCode.ERROR, "Validation failed"))
                event_name = config_attr("validation.error")
                for ev in validation_errors:
                    attrs = {}
                    if ev.get("path") is not None:
                        attrs[config_attr("validation.error.path")] = str(ev["path"])
                    if ev.get("rule") is not None:
                        attrs[config_attr("validation.error.rule")] = str(ev["rule"])
                    if ev.get("code") is not None:
                        attrs[config_attr("validation.error.code")] = str(ev["code"])
                    if attrs:
                        span.add_event(event_name, attrs)
            elif (is_error or tool_result is False) and not override_success:
                _record_span_error(
                    span,
                    config.span_type,
                    overrides,
                    exception_type=getattr(config, "exception_type", None),
                    exception_message=getattr(config, "exception_message", None),
                )
            else:
                # Control-plane validation spans (root + children) use UNSET for
                # allowed/blocked outcomes; do not force OK.
                cp_span_types = {
                    SpanType.REQUEST_VALIDATION,
                    SpanType.RESPONSE_VALIDATION,
                    SpanType.PAYLOAD_VALIDATION,
                    SpanType.POLICY_VALIDATION,
                    SpanType.AUGMENTATION_VALIDATION,
                }
                if config.span_type in cp_span_types:
                    pass
                # For other spans, mark successful completion explicitly.
                else:
                    span.set_status(Status(StatusCode.OK))

            # Diagnostic event: {prefix}.agent.tool_selection on tools.recommend spans.
            # Captures combined system+user input and a bounded JSON tool_plan structure.
            if config.span_type == SpanType.TOOLS_RECOMMEND:
                try:
                    user_input_text: str | None = None
                    system_text: str | None = None
                    combined_input_text: str | None = None
                    selected_tool_name: str = ""
                    # Use the OTEL GenAI tool name when available for this logical recommendation.
                    try:
                        span_tool_name = span.attributes.get("gen_ai.tool.name")  # type: ignore[attr-defined]
                        if isinstance(span_tool_name, str):
                            selected_tool_name = span_tool_name
                    except Exception:
                        selected_tool_name = ""

                    # Prefer the actual gen_ai.input.messages attribute on the span, which the
                    # AttributeGenerator serializes as JSON, and fall back to context.llm_input_messages.
                    raw_input_attr = span.attributes.get("gen_ai.input.messages")  # type: ignore[attr-defined]
                    messages_source = None
                    if isinstance(raw_input_attr, str):
                        try:
                            decoded = json.loads(raw_input_attr)
                            if isinstance(decoded, list):
                                messages_source = decoded
                        except Exception:
                            messages_source = None
                    if messages_source is None:
                        messages_source = getattr(context, "llm_input_messages", None)

                    msgs = messages_source
                    if isinstance(msgs, list) and msgs:
                        # Prefer explicit system+user messages when present (new scenario template).
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            role = msg.get("role")
                            content = msg.get("content") or []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_val = block.get("text")
                                    if not isinstance(text_val, str) or not text_val.strip():
                                        continue
                                    text_val = text_val.strip()
                                    if role == "system" and system_text is None:
                                        system_text = text_val
                                    elif role == "user" and user_input_text is None:
                                        user_input_text = text_val
                        # Fallbacks for simpler GenAI message shapes (no content[] wrapper).
                        if user_input_text is None:
                            first = msgs[0]
                            if isinstance(first, dict):
                                direct_text = first.get("text")
                                if isinstance(direct_text, str) and direct_text.strip():
                                    user_input_text = direct_text.strip()
                                if user_input_text is None:
                                    msg_field = first.get("message")
                                    if isinstance(msg_field, str) and msg_field.strip():
                                        user_input_text = msg_field.strip()

                    if system_text and user_input_text:
                        combined_input_text = f"{system_text}\n\n{user_input_text}"
                    elif user_input_text:
                        combined_input_text = user_input_text
                    elif system_text:
                        combined_input_text = system_text

                    # Build tool_plan entries for all recommended tools in the chain when available.
                    tool_plan_entries: list[dict[str, Any]] = []
                    context_tool_args = getattr(context, "tool_call_arguments", None)
                    if isinstance(context_tool_args, dict) and context_tool_args:
                        for tool_name in context_tool_args.keys():
                            tool_plan_entries.append(
                                {
                                    "tool_name": str(tool_name)
                                    or selected_tool_name
                                    or "unknown.tool",
                                    "trigger_summary": user_input_text or "",
                                    "trigger_quote": user_input_text or "",
                                    "missing_info": None,
                                    "confidence": round(random.uniform(0.8, 0.95), 3),
                                }
                            )
                    else:
                        tool_plan_entries.append(
                            {
                                "tool_name": selected_tool_name or "unknown.tool",
                                "trigger_summary": user_input_text or "",
                                "trigger_quote": user_input_text or "",
                                "missing_info": None,
                                "confidence": round(random.uniform(0.8, 0.95), 3),
                            }
                        )

                    tool_plan_payload = {"tool_plan": tool_plan_entries}
                    event_attrs = {
                        config_attr("agent.tool_selection.input.raw"): combined_input_text
                        or user_input_text
                        or "",
                        config_attr("agent.tool_selection.tool.plan"): json.dumps(
                            tool_plan_payload
                        ),
                    }
                    span.add_event(config_attr("agent.tool_selection"), event_attrs)
                except Exception:
                    # Events are best-effort; failures here must not break trace generation.
                    pass

            # Generate children with minimal gap between spans; collect end times so parent end encompasses all children.
            last_child_end_ns = start_time_ns
            child_tool_latencies: list[float] = []
            next_child_start_ns = start_time_ns + _MIN_SPAN_GAP_NS
            for child_hierarchy in hierarchy.children:
                _, _, child_end_ns, child_tool_latency_ms = self._generate_span_recursive(
                    child_hierarchy,
                    context,
                    span,
                    None,
                    compose_accumulated_failure=child_compose_failure,
                    use_single_tracer=use_single_tracer,
                    logical_start_ns=next_child_start_ns,
                    planned_latencies_ms=planned_latencies_ms,
                    span_callback=span_callback,
                )
                last_child_end_ns = max(last_child_end_ns, child_end_ns)
                child_tool_latencies.append(child_tool_latency_ms)
                next_child_start_ns = child_end_ns + _MIN_SPAN_GAP_NS

            # MCP_TOOL_EXECUTE parent: set tool.latency_ms to sum of attempt latencies (not the 10ms floor).
            if config.span_type == SpanType.MCP_TOOL_EXECUTE and child_tool_latencies:
                parent_tool_latency_ms = sum(child_tool_latencies)
                span.set_attribute(config_attr("tool.latency_ms"), round(parent_tool_latency_ms))
                tool_latency_ms_return = parent_tool_latency_ms

            # Set span end time so exported duration matches simulated latency (config + modifier).
            # Parent end must be >= last child end so trace timestamps are logical (no child after parent).
            span_end_from_latency_ns = start_time_ns + int(latency_ms * 1_000_000)
            end_time_ns = max(span_end_from_latency_ns, last_child_end_ns)
            # Align duration attribute with actual span duration (end - start) before ending span.
            actual_duration_ms = round((end_time_ns - start_time_ns) / 1_000_000)
            if config.span_type == SpanType.A2A_ORCHESTRATE:
                span.set_attribute(config_attr("orchestration.duration_ms"), actual_duration_ms)
            elif config.span_type == SpanType.REQUEST_VALIDATION:
                span.set_attribute(
                    config_attr("request.validation.duration_ms"), actual_duration_ms
                )
            elif config.span_type == SpanType.RESPONSE_VALIDATION:
                span.set_attribute(
                    config_attr("response.validation.duration_ms"), actual_duration_ms
                )
            elif config.span_type == SpanType.CP_REQUEST:
                span.set_attribute(config_attr("cp.request.duration_ms"), actual_duration_ms)
            elif config.span_type == SpanType.MCP_TOOL_EXECUTE and child_tool_latencies:
                pass  # already set to sum of attempt latencies above
            elif config.span_type == SpanType.TOOLS_RECOMMEND:
                span.set_attribute(config_attr("mcp.selection.latency.ms"), actual_duration_ms)
            elif config.span_type == SpanType.RAG_RETRIEVE:
                span.set_attribute(config_attr("rag.latency_ms"), actual_duration_ms)
            elif config.span_type == SpanType.A2A_CALL:
                span.set_attribute(config_attr("a2a.latency_ms"), actual_duration_ms)
            elif config.span_type in (
                SpanType.LLM_CALL,
                SpanType.PLANNER,
                SpanType.TASK_EXECUTE,
                SpanType.RESPONSE_COMPOSE,
                SpanType.PAYLOAD_VALIDATION,
                SpanType.POLICY_VALIDATION,
                SpanType.AUGMENTATION_VALIDATION,
            ):
                span.set_attribute(config_attr("span.duration_ms"), actual_duration_ms)
            span.end(end_time=end_time_ns)

            time.sleep(latency_ms / 1000.0 * 0.01)

        return current_trace_id, current_span_id, end_time_ns, tool_latency_ms_return

    def force_flush(self, timeout_millis: int = 5000) -> bool:
        """Flush all tracer providers so the current batch is exported immediately.

        Call after generating a control-plane-only trace so all spans of that trace
        are sent in one OTLP export (one Kafka message when using an OTLP collector).
        """
        ok = True
        for prov in self._providers:
            try:
                if not prov.force_flush(timeout_millis):
                    ok = False
            except Exception:
                ok = False
        return ok

    def shutdown(self):
        """Shutdown all tracer providers. Flush all before any shutdown so OTLP
        receives every batch (control-plane and data-plane) when using multiple providers.
        """
        for prov in self._providers:
            try:
                prov.force_flush(5000)
            except Exception:
                pass
        for prov in self._providers:
            prov.shutdown()
