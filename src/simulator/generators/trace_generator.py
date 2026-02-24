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

from ..config import attr as config_attr
from ..config import resource_attributes as config_resource_attributes
from ..config import resource_schema_url, schema_version_attr
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
    MCP_TOOL_EXECUTE = "mcp.tool.execute"
    MCP_TOOL_EXECUTE_ATTEMPT = "mcp.tool.execute.attempt"
    LLM_TOOL_RESPONSE_BRIDGE = "llm.tool.response.bridge"
    RESPONSE_COMPOSE = "response.compose"
    RAG_RETRIEVE = "rag.retrieve"
    A2A_CALL = "a2a.call"
    CP_REQUEST = "cp.request"


# Span types that use vendor prefix for emitted name; others use enum value as-is.
_VENDOR_SPAN_SUFFIXES = {
    SpanType.A2A_ORCHESTRATE,
    SpanType.PLANNER,
    SpanType.TASK_EXECUTE,
    SpanType.LLM_CALL,
    SpanType.MCP_TOOL_EXECUTE,
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT,
    SpanType.LLM_TOOL_RESPONSE_BRIDGE,
    SpanType.RESPONSE_COMPOSE,
}


def get_span_name(span_type: SpanType) -> str:
    """Return the emitted span name (vendor-prefixed or plain) for this span type."""
    if span_type in _VENDOR_SPAN_SUFFIXES:
        return config_span_name(span_type.value)
    return span_type.value


SPAN_KIND_MAP = {
    SpanType.A2A_ORCHESTRATE: SpanKind.SERVER,
    SpanType.PLANNER: SpanKind.INTERNAL,
    SpanType.TASK_EXECUTE: SpanKind.INTERNAL,
    SpanType.LLM_CALL: SpanKind.CLIENT,
    SpanType.MCP_TOOL_EXECUTE: SpanKind.CLIENT,
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: SpanKind.CLIENT,
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: SpanKind.INTERNAL,
    SpanType.RESPONSE_COMPOSE: SpanKind.INTERNAL,
    SpanType.RAG_RETRIEVE: SpanKind.INTERNAL,
    SpanType.A2A_CALL: SpanKind.CLIENT,
    SpanType.CP_REQUEST: SpanKind.SERVER,
}

# Convention-required span.class attribute (prefix.span.class) per vendor span type.
CONVENTION_ATTRIBUTES = {
    SpanType.A2A_ORCHESTRATE: {config_attr("span.class"): "a2a.orchestrate"},
    SpanType.PLANNER: {config_attr("span.class"): "planner"},
    SpanType.TASK_EXECUTE: {config_attr("span.class"): "task.execute"},
    SpanType.LLM_CALL: {config_attr("span.class"): "llm.call"},
    SpanType.MCP_TOOL_EXECUTE: {config_attr("span.class"): "mcp.tool.execute"},
    SpanType.MCP_TOOL_EXECUTE_ATTEMPT: {config_attr("span.class"): "mcp.tool.execute.attempt"},
    SpanType.LLM_TOOL_RESPONSE_BRIDGE: {config_attr("span.class"): "llm.tool.response.bridge"},
    SpanType.RESPONSE_COMPOSE: {config_attr("span.class"): "response.compose"},
}


@dataclass
class SpanConfig:
    """Configuration for generating a single span."""

    span_type: SpanType
    latency_mean_ms: float = 100.0
    latency_variance: float = 0.3
    error_rate: float = 0.02
    attribute_overrides: dict[str, Any] = field(default_factory=dict)


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
        service_name: str = "telemetry-simulator",
        service_version: str = "1.0.0",
        show_full_spans: bool = False,
    ):
        """Initialize trace generator with exporter."""
        parser = SchemaParser(schema_path)
        self.schema = parser.parse()
        self.attr_generator = AttributeGenerator(self.schema)

        tenant_id = get_default_tenant_ids()[0]
        attrs = dict(config_resource_attributes(tenant_id))
        attrs[schema_version_attr()] = self.schema.schema_version
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
            if is_error or tool_result is False:
                span.set_status(Status(StatusCode.ERROR, "Error occurred"))
            else:
                span.set_status(Status(StatusCode.OK))

            for child_hierarchy in hierarchy.children:
                self._generate_span_recursive(child_hierarchy, context, span)

            time.sleep(latency_ms / 1000.0 * 0.01)

        return current_trace_id

    def shutdown(self):
        """Shutdown the trace provider."""
        self.provider.shutdown()
