"""
Generate complete trace hierarchies aligned with vendor-prefixed OTEL semantic conventions.

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
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from ..config import SEMCONV_ERROR_TYPE_VALUES, resource_schema_url
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
    VALIDATION_PAYLOAD = "validation.payload"
    VALIDATION_POLICY = "validation.policy"
    AUGMENTATION = "augmentation"
    REQUEST_VALIDATION = "request.validation"
    RESPONSE_VALIDATION = "response.validation"
    CP_REQUEST = "cp.request"


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
    SpanType.VALIDATION_PAYLOAD,
    SpanType.VALIDATION_POLICY,
    SpanType.AUGMENTATION,
    SpanType.REQUEST_VALIDATION,
    SpanType.RESPONSE_VALIDATION,
}


def get_span_name(span_type: SpanType) -> str:
    """Return the emitted span name (vendor-prefixed or plain) for this span type."""
    if span_type in _VENDOR_SPAN_SUFFIXES:
        return config_span_name(span_type.value)
    return span_type.value


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
    msg = exception_message or message
    if exception_type:
        exc_cls = type(exception_type, (RuntimeError,), {})
        exc = exc_cls(msg)
    else:
        exc = RuntimeError(msg)
    if hasattr(span, "record_exception"):
        span.record_exception(exc)


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
    SpanType.VALIDATION_PAYLOAD: SpanKind.INTERNAL,
    SpanType.VALIDATION_POLICY: SpanKind.INTERNAL,
    SpanType.AUGMENTATION: SpanKind.INTERNAL,
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
    SpanType.VALIDATION_PAYLOAD: "invalid_arguments",
    SpanType.VALIDATION_POLICY: "invalid_arguments",
    SpanType.AUGMENTATION: "unavailable",
    SpanType.REQUEST_VALIDATION: "invalid_arguments",
    SpanType.RESPONSE_VALIDATION: "invalid_arguments",
    SpanType.CP_REQUEST: "invalid_arguments",
}

# Error category (validation | policy | runtime) for classification when spans fail.
_ERROR_CATEGORY_BY_SPAN_TYPE: dict[SpanType, str] = {
    # Control-plane validation
    SpanType.REQUEST_VALIDATION: "validation",
    SpanType.VALIDATION_PAYLOAD: "validation",
    SpanType.VALIDATION_POLICY: "policy",
    SpanType.AUGMENTATION: "runtime",
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

# Convention-required attributes per vendor span type.
# Compose span (response.compose): MUST have span.class, response.format, step.outcome per spec.
CONVENTION_ATTRIBUTES = {
    SpanType.A2A_ORCHESTRATE: {config_attr("span.class"): "a2a.orchestrate"},
    SpanType.PLANNER: {config_attr("span.class"): "planner"},
    SpanType.TASK_EXECUTE: {
        config_attr("span.class"): "task.execute",
        config_attr("step.outcome"): "success",
    },
    SpanType.LLM_CALL: {
        config_attr("span.class"): "llm.call",
        config_attr("step.outcome"): "success",
    },
    SpanType.TOOLS_RECOMMEND: {
        config_attr("span.class"): "tools.recommend",
        config_attr("step.outcome"): "success",
    },
    SpanType.MCP_TOOL_EXECUTE: {config_attr("span.class"): "mcp.tool.execute"},
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: {config_attr("span.class"): "mcp.tool.execute.attempt"},
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: {config_attr("span.class"): "llm.tool.response.bridge"},
    SpanType.RESPONSE_COMPOSE: {
        config_attr("span.class"): "response.compose",
        config_attr("response.format"): "a2a_json",
        config_attr("step.outcome"): "success",
    },
    # Control-plane: span.class set at runtime only (_CP_SPAN_CLASS_VALUES) so CLI --vendor is applied.
    SpanType.VALIDATION_PAYLOAD: {},
    SpanType.VALIDATION_POLICY: {},
    SpanType.AUGMENTATION: {},
    SpanType.REQUEST_VALIDATION: {},
    SpanType.RESPONSE_VALIDATION: {},
}
# Control-plane span class values; key resolved at runtime via config_attr() so --vendor is applied.
_CP_SPAN_CLASS_VALUES = {
    SpanType.REQUEST_VALIDATION: "request.validation",
    SpanType.RESPONSE_VALIDATION: "response.validation",
    SpanType.VALIDATION_PAYLOAD: "validation.payload",
    SpanType.VALIDATION_POLICY: "validation.policy",
    SpanType.AUGMENTATION: "augmentation",
}


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
        """Generate realistic latency based on config."""
        latency = config.latency_mean_ms * (1 + random.gauss(0, config.latency_variance))
        if random.random() < 0.05:
            latency *= random.uniform(2.0, 4.0)
        return max(10.0, latency)

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
        # Ensure prefix.span.class and convention attributes are set for vendor spans.
        convention = CONVENTION_ATTRIBUTES.get(span_type)
        if convention:
            for k, v in convention.items():
                attrs.setdefault(k, v)
        # Control-plane: set span.class at runtime so CLI --vendor (prefix) is applied.
        if span_type in _CP_SPAN_CLASS_VALUES:
            attrs[config_attr("span.class")] = _CP_SPAN_CLASS_VALUES[span_type]
        # Control-plane roots: ensure tenant id, request id, enduser pseudo id (configured prefix).
        if span_type in (SpanType.REQUEST_VALIDATION, SpanType.RESPONSE_VALIDATION):
            attrs[config_attr("tenant.id")] = context.tenant_id
        if span_type == SpanType.REQUEST_VALIDATION:
            attrs[config_attr("request.id")] = context.request_id
            attrs[config_attr("enduser.pseudo.id")] = getattr(context, "user_id", None) or ""
        # vendor.a2a.orchestrate root: only root-level attrs (no response.format, no route).
        # vendor.response.compose: only compose attrs (no a2a.outcome, a2a.agent.target.id).
        if span_type == SpanType.A2A_ORCHESTRATE:
            attrs.pop(config_attr("response.format"), None)
            attrs.pop(config_attr("route"), None)
        elif span_type == SpanType.RESPONSE_COMPOSE:
            attrs.pop(config_attr("a2a.outcome"), None)
            attrs.pop(config_attr("a2a.agent.target.id"), None)
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
        attrs = dict(config_resource_attributes(tenant_id))
        attrs["service.name"] = service_name
        attrs["service.version"] = service_version
        resource = Resource.create(attrs, schema_url=resource_schema_url())

        self.provider = TracerProvider(resource=resource)
        if show_full_spans:
            self.provider.add_span_processor(_PrintSpanProcessor())  # type: ignore[arg-type]
        self.provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(self.provider)

        self.tracer = trace.get_tracer(__name__)
        self.span_builder = SpanBuilder(self.schema, self.attr_generator, self.tracer)

    def generate_trace(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext | None = None,
    ) -> str:
        """Generate a complete trace from hierarchy definition."""
        if context is None:
            context = GenerationContext.create()

        trace_id = self._generate_span_recursive(hierarchy, context, None)
        return trace_id

    def _generate_span_recursive(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext,
        parent_span: Any,
    ) -> str:
        """Recursively generate spans in the hierarchy."""
        config = hierarchy.root_config
        span_name = get_span_name(config.span_type)
        span_kind = SPAN_KIND_MAP.get(config.span_type, SpanKind.INTERNAL)

        latency_ms = self.span_builder.generate_latency(config)

        overrides = config.attribute_overrides or {}
        current_trace_id = ""
        with self.tracer.start_as_current_span(
            span_name,
            kind=span_kind,
            attributes=self.span_builder.get_attributes(
                config.span_type,
                context,
                overrides,
            ),
        ) as span:
            current_trace_id = format(span.get_span_context().trace_id, "032x")

            if config.span_type == SpanType.CP_REQUEST:
                span.set_attribute(config_attr("cp.incoming_trace_id"), current_trace_id)
                if overrides.get(config_attr("cp.status.code")) in [
                    "ALLOWED",
                    "FLAGGED",
                ]:
                    span.set_attribute(config_attr("cp.outgoing_trace_id"), current_trace_id)

            is_error = self.span_builder.should_error(config)
            tool_result = overrides.get(config_attr("tool.status.result"))
            # a2a.orchestrate root: status.code UNSET on success/partial, ERROR on error
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
            elif config.span_type == SpanType.RESPONSE_COMPOSE and (
                is_error or tool_result is False
            ):
                span.set_attribute(config_attr("step.outcome"), "fail")
                _record_span_error(
                    span,
                    config.span_type,
                    overrides,
                    error_type_override="protocol_error",
                    exception_type=getattr(config, "exception_type", None),
                    exception_message=getattr(config, "exception_message", None),
                )
            elif config.span_type == SpanType.TASK_EXECUTE and (is_error or tool_result is False):
                span.set_attribute(config_attr("step.outcome"), "fail")
                _record_span_error(
                    span,
                    config.span_type,
                    overrides,
                    exception_type=getattr(config, "exception_type", None),
                    exception_message=getattr(config, "exception_message", None),
                )
            elif config.span_type == SpanType.TOOLS_RECOMMEND and (
                is_error or tool_result is False
            ):
                span.set_attribute(config_attr("step.outcome"), "fail")
                _record_span_error(
                    span,
                    config.span_type,
                    overrides,
                    exception_type=getattr(config, "exception_type", None),
                    exception_message=getattr(config, "exception_message", None),
                )
            elif config.span_type == SpanType.VALIDATION_PAYLOAD and getattr(
                config, "validation_errors", None
            ):
                # Payload validation failed with one or more validation.error events (no exception event).
                validation_errors = config.validation_errors or []
                error_type = overrides.get(
                    config_attr("error.type")
                ) or _ERROR_TYPE_BY_SPAN_TYPE.get(SpanType.VALIDATION_PAYLOAD, "invalid_arguments")
                if error_type not in SEMCONV_ERROR_TYPE_VALUES:
                    error_type = "invalid_arguments"
                span.set_attribute("error.type", error_type)
                if _ERROR_CATEGORY_BY_SPAN_TYPE.get(SpanType.VALIDATION_PAYLOAD):
                    span.set_attribute(
                        config_attr("error.category"),
                        _ERROR_CATEGORY_BY_SPAN_TYPE[SpanType.VALIDATION_PAYLOAD],
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
            elif is_error or tool_result is False:
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
                    SpanType.VALIDATION_PAYLOAD,
                    SpanType.VALIDATION_POLICY,
                    SpanType.AUGMENTATION,
                }
                if config.span_type in cp_span_types:
                    pass
                # For other spans, mark successful completion explicitly.
                else:
                    span.set_status(Status(StatusCode.OK))

            # Diagnostic event: {prefix}.agent.tool_selection on tools.recommend spans.
            # Captures the raw user input and a bounded JSON tool_plan structure.
            if config.span_type == SpanType.TOOLS_RECOMMEND:
                try:
                    user_input_text: str | None = None
                    tool_name: str = ""
                    # Use the OTEL GenAI tool name when available for this logical recommendation.
                    try:
                        span_tool_name = span.attributes.get("gen_ai.tool.name")  # type: ignore[attr-defined]
                        if isinstance(span_tool_name, str):
                            tool_name = span_tool_name
                    except Exception:
                        tool_name = ""

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
                        first = msgs[0]
                        if isinstance(first, dict):
                            for block in first.get("content") or []:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_val = block.get("text")
                                    if isinstance(text_val, str) and text_val.strip():
                                        user_input_text = text_val.strip()
                                        break
                            # Fallbacks for simpler GenAI message shapes (no content[] wrapper).
                            if user_input_text is None:
                                direct_text = first.get("text")
                                if isinstance(direct_text, str) and direct_text.strip():
                                    user_input_text = direct_text.strip()
                            if user_input_text is None:
                                msg_field = first.get("message")
                                if isinstance(msg_field, str) and msg_field.strip():
                                    user_input_text = msg_field.strip()
                    tool_plan_payload = {
                        "tool_plan": [
                            {
                                "tool_name": tool_name or "unknown.tool",
                                "trigger_summary": user_input_text or "",
                                "trigger_quote": user_input_text or "",
                                "missing_info": None,
                                "confidence": round(random.uniform(0.8, 0.95), 3),
                            }
                        ]
                    }
                    event_attrs = {
                        config_attr("agent.tool_selection.input.raw"): user_input_text or "",
                        config_attr("agent.tool_selection.tool.plan"): json.dumps(
                            tool_plan_payload
                        ),
                    }
                    span.add_event(config_attr("agent.tool_selection"), event_attrs)
                except Exception:
                    # Events are best-effort; failures here must not break trace generation.
                    pass

            for child_hierarchy in hierarchy.children:
                self._generate_span_recursive(child_hierarchy, context, span)

            time.sleep(latency_ms / 1000.0 * 0.01)

        return current_trace_id

    def shutdown(self):
        """Shutdown the trace provider."""
        self.provider.shutdown()
