"""
OTEL trace IR and renderer.

SpanSpec / TraceSpec are the low-level IR. render_trace materializes a TraceSpec
into OTEL spans via the provided tracer. Scenario semantics and trace structure
are built by scenarios.compiler; this module only renders.
Unified request-lifecycle trace is built from semconv lineage.single_trace_request_lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import SpanKind, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from .. import config as sim_config
from ..schemas.schema_parser import SingleTraceRequestLifecycle


def _session_id_for_trace_export(spec: TraceSpec) -> str | None:
    """Resolve session id from any span using the configured attribute prefix (``--vendor``)."""
    session_key = f"{sim_config.ATTR_PREFIX}.session.id"
    for s in spec.spans:
        v = (s.attributes or {}).get(session_key)
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            return sv
    return None


def _mcp_server_uuid_for_trace_export(spec: TraceSpec) -> str | None:
    """Resolve MCP server UUID from any span (e.g. mcp.tool.execute) for trace-wide propagation."""
    key = f"{sim_config.ATTR_PREFIX}.mcp.server.uuid"
    for s in spec.spans:
        v = (s.attributes or {}).get(key)
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            return sv
    return None


def _span_class_from_attrs(attrs: dict[str, Any] | None) -> str | None:
    if not attrs:
        return None
    for k, v in attrs.items():
        if isinstance(k, str) and k.endswith(".span.class") and v is not None:
            return str(v)
    return None


def _mcp_tool_uuid_from_execute_ancestor(spec: TraceSpec, span_index: int) -> str | None:
    """Walk parents to find mcp.tool.execute and read tool UUID for the active prefix."""
    tool_key = f"{sim_config.ATTR_PREFIX}.mcp.tool.uuid"
    p = spec.spans[span_index].parent_index
    guard = 0
    n = len(spec.spans)
    while 0 <= p < n and guard <= n:
        guard += 1
        attrs = spec.spans[p].attributes or {}
        if _span_class_from_attrs(attrs) == "mcp.tool.execute":
            v = attrs.get(tool_key)
            if v is not None and str(v).strip():
                return str(v).strip()
            return None
        p = spec.spans[p].parent_index
    return None


# Spans that reference MCP semantics but may omit server UUID in the IR.
_MCP_SERVER_PROPAGATION_SPAN_CLASSES: frozenset[str] = frozenset(
    {"tools.recommend", "mcp.tool.execute.attempt"}
)

_SPAN_KIND_MAP = {
    "SERVER": SpanKind.SERVER,
    "CLIENT": SpanKind.CLIENT,
    "INTERNAL": SpanKind.INTERNAL,
    "PRODUCER": SpanKind.PRODUCER,
    "CONSUMER": SpanKind.CONSUMER,
}


@dataclass
class SpanEventSpec:
    """Spec for one span event (name + attributes)."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    # Nanoseconds since epoch. When None, renderer uses the parent span start time.
    timestamp_ns: int | None = None


@dataclass
class SpanSpec:
    """Spec for one span: name, parent index, kind, attributes, duration_ms, events."""

    name: str
    parent_index: int  # -1 for root
    kind: str  # SERVER | INTERNAL | CLIENT
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    events: list[SpanEventSpec] = field(default_factory=list)
    # OpenTelemetry status.code: "UNSET" | "ERROR".
    status_code: str = "UNSET"


@dataclass
class TraceSpec:
    """One trace: list of span specs in dependency order (parent before children)."""

    spans: list[SpanSpec] = field(default_factory=list)


def build_unified_trace_from_semconv(
    cp_request: TraceSpec,
    data_plane: TraceSpec,
    cp_response: TraceSpec,
    single_trace_lifecycle: SingleTraceRequestLifecycle,
) -> TraceSpec:
    """
    Build one TraceSpec from the three segments using semconv lineage.single_trace_request_lifecycle.
    Hierarchy: request.validation (root) → a2a.orchestrate → response.validation.
    """
    merged_spans: list[SpanSpec] = []

    # Sanity-check lifecycle to avoid silently emitting an incorrect dependency graph.
    # We can't rely on the attribute prefix here, so we read the span class from
    # any attribute key that ends with ".span.class".
    def _get_span_class_from_attrs(attrs: dict[str, Any]) -> str | None:
        for k, v in attrs.items():
            if k.endswith(".span.class") and v is not None:
                return str(v)
        return None

    if len(single_trace_lifecycle.child_chain) < 2:
        raise ValueError(
            "lineage.single_trace_request_lifecycle.child_chain must have at least two entries "
            "(a2a.orchestrate, response.validation)."
        )

    if cp_request.spans:
        cp_root_class = _get_span_class_from_attrs(cp_request.spans[0].attributes)
        if cp_root_class and cp_root_class != single_trace_lifecycle.root_span_class:
            raise ValueError(
                "Mismatch between cp_request root span class and semconv "
                f"lineage.single_trace_request_lifecycle.root_span_class: "
                f"{cp_root_class} != {single_trace_lifecycle.root_span_class}"
            )

    if data_plane.spans:
        dp_root_class = _get_span_class_from_attrs(data_plane.spans[0].attributes)
        expected_dp = single_trace_lifecycle.child_chain[0]
        if dp_root_class and dp_root_class != expected_dp:
            raise ValueError(
                "Mismatch between data_plane root span class and semconv "
                f"lineage.single_trace_request_lifecycle.child_chain[0]: {dp_root_class} != {expected_dp}"
            )

    if cp_response.spans:
        resp_root_class = _get_span_class_from_attrs(cp_response.spans[0].attributes)
        expected_resp = single_trace_lifecycle.child_chain[1]
        if resp_root_class and resp_root_class != expected_resp:
            raise ValueError(
                "Mismatch between cp_response root span class and semconv "
                f"lineage.single_trace_request_lifecycle.child_chain[1]: {resp_root_class} != {expected_resp}"
            )

    cp_offset = len(merged_spans)
    merged_spans.extend(cp_request.spans)
    cp_root_global = cp_offset if cp_request.spans else -1

    dp_offset = len(merged_spans)
    dp_root_global = -1
    for idx, span in enumerate(data_plane.spans):
        if span.parent_index == -1:
            parent_index = cp_root_global if cp_root_global >= 0 else -1
            if dp_root_global == -1:
                dp_root_global = dp_offset + idx
        else:
            parent_index = dp_offset + span.parent_index
        merged_spans.append(
            SpanSpec(
                name=span.name,
                parent_index=parent_index,
                kind=span.kind,
                attributes=dict(span.attributes),
                duration_ms=span.duration_ms,
                events=[
                    SpanEventSpec(
                        name=e.name,
                        attributes=dict(e.attributes) if e.attributes else {},
                        timestamp_ns=e.timestamp_ns,
                    )
                    for e in (span.events or [])
                ],
                status_code=span.status_code,
            )
        )

    resp_offset = len(merged_spans)
    parent_for_resp_root = dp_root_global if dp_root_global >= 0 else cp_root_global
    for span in cp_response.spans:
        if span.parent_index == -1:
            parent_index = parent_for_resp_root if parent_for_resp_root >= 0 else -1
        else:
            parent_index = resp_offset + span.parent_index
        merged_spans.append(
            SpanSpec(
                name=span.name,
                parent_index=parent_index,
                kind=span.kind,
                attributes=dict(span.attributes),
                duration_ms=span.duration_ms,
                events=[
                    SpanEventSpec(
                        name=e.name,
                        attributes=dict(e.attributes) if e.attributes else {},
                        timestamp_ns=e.timestamp_ns,
                    )
                    for e in (span.events or [])
                ],
                status_code=span.status_code,
            )
        )

    return TraceSpec(spans=merged_spans)


def render_trace_with_span_ids(
    tracer: Any,
    spec: TraceSpec,
    *,
    tracer_cp: Any | None = None,
    tracer_dp: Any | None = None,
    trace_base_time_ns: int | None = None,
) -> tuple[str | None, list[str]]:
    """
    Materialize a TraceSpec into OTEL spans using the provided tracer.

    *trace_base_time_ns*: UNIX epoch nanoseconds for the trace scheduling origin (defaults
    to wall clock). Used to separate successive traces in time when emitting quickly.

    Returns:
    - trace_id (hex) or None when there are no spans
    - span_ids aligned to `spec.spans` order
    """
    if not spec.spans:
        return (None, [])

    # Build parent -> children relationships to compute a deterministic simulated timeline.
    # This makes span duration/latency in the UI match SpanSpec.duration_ms.
    children: list[list[int]] = [[] for _ in range(len(spec.spans))]
    roots: list[int] = []
    for i, s in enumerate(spec.spans):
        if s.parent_index >= 0 and s.parent_index < len(spec.spans):
            children[s.parent_index].append(i)
        else:
            roots.append(i)

    start_times_ns: list[int] = [0] * len(spec.spans)
    end_times_ns: list[int] = [0] * len(spec.spans)

    base_ns = time.time_ns() if trace_base_time_ns is None else trace_base_time_ns

    # We interpret duration_ms as the intended span runtime. Parent spans may end
    # before children finish if a scenario's sampled parent duration is shorter.
    def schedule(node_idx: int, start_ns: int) -> int:
        start_times_ns[node_idx] = start_ns
        self_end_ns = start_ns + int(spec.spans[node_idx].duration_ms) * 1_000_000

        # Children are scheduled sequentially starting at the parent's start.
        cur = start_ns
        for child_idx in children[node_idx]:
            cur = schedule(child_idx, cur)

        # In OTEL UI timelines, parent spans should fully cover children.
        # Treat duration_ms as a minimum self-duration; extend end_time if children run longer.
        end_times_ns[node_idx] = max(self_end_ns, cur)

        # Return when this subtree finishes so siblings don't start too early.
        return end_times_ns[node_idx]

    # If multiple roots exist, schedule them sequentially.
    cursor_ns = base_ns
    for r in roots:
        cursor_ns = schedule(r, cursor_ns)

    # When both CP+DP tracers are provided, emit CP spans using tracer_cp and
    # DP spans using tracer_dp. This controls `span.resource.attributes`.
    def _pick_tracer(span_spec: SpanSpec) -> Any:
        if tracer_cp is None or tracer_dp is None:
            return tracer

        # Control-plane request/response validation and children.
        cp_suffixes = (
            ".request.validation",
            ".response.validation",
            ".validation.payload",
            ".validation.policy",
            ".augmentation.validation",
        )
        if any(span_spec.name.endswith(sfx) for sfx in cp_suffixes):
            return tracer_cp
        return tracer_dp

    attr_prefix = sim_config.ATTR_PREFIX
    session_key = f"{attr_prefix}.session.id"
    mcp_server_key = f"{attr_prefix}.mcp.server.uuid"
    mcp_tool_key = f"{attr_prefix}.mcp.tool.uuid"

    trace_session_id = _session_id_for_trace_export(spec)
    trace_mcp_server_uuid = _mcp_server_uuid_for_trace_export(spec)

    span_objects: list[Any] = []
    for i, s in enumerate(spec.spans):
        parent_ctx = None
        if s.parent_index >= 0 and s.parent_index < len(spec.spans):
            parent_span = (
                span_objects[s.parent_index] if s.parent_index < len(span_objects) else None
            )
            if parent_span is not None:
                parent_ctx = set_span_in_context(parent_span)

        kind = _SPAN_KIND_MAP.get(s.kind, SpanKind.INTERNAL)
        chosen_tracer = _pick_tracer(s)
        span = chosen_tracer.start_span(
            s.name,
            kind=kind,
            context=parent_ctx,
            start_time=start_times_ns[i],
        )
        for k, v in s.attributes.items():
            if v is None:
                continue
            if isinstance(v, (bool, int, float)):
                span.set_attribute(k, v)
            else:
                span.set_attribute(k, str(v))

        # Propagate session id to every span when missing (same value anywhere in the trace).
        if trace_session_id is not None:
            cur = s.attributes.get(session_key)
            if cur is None or not str(cur).strip():
                span.set_attribute(session_key, trace_session_id)

        span_class = _span_class_from_attrs(s.attributes)
        if trace_mcp_server_uuid is not None and span_class in _MCP_SERVER_PROPAGATION_SPAN_CLASSES:
            cur_mcp_srv = s.attributes.get(mcp_server_key)
            if cur_mcp_srv is None or not str(cur_mcp_srv).strip():
                span.set_attribute(mcp_server_key, trace_mcp_server_uuid)
        if span_class == "mcp.tool.execute.attempt":
            tool_uuid = _mcp_tool_uuid_from_execute_ancestor(spec, i)
            if tool_uuid is not None:
                cur_tool = s.attributes.get(mcp_tool_key)
                if cur_tool is None or not str(cur_tool).strip():
                    span.set_attribute(mcp_tool_key, tool_uuid)

        # semconv-aligned duration attributes must reflect actual OTEL duration.
        # Our schedule may extend parent spans to cover children; after scheduling
        # we know the real end-start duration in ns.
        if s.name.endswith(".a2a.orchestrate"):
            actual_ms = max(
                0,
                int((end_times_ns[i] - start_times_ns[i]) / 1_000_000),
            )
            for k in s.attributes.keys():
                if isinstance(k, str) and k.endswith(".orchestration.duration_ms"):
                    span.set_attribute(k, actual_ms)

        for ev in s.events:
            if not ev.name:
                continue
            event_attrs: dict[str, Any] = {}
            for k, v in (ev.attributes or {}).items():
                if v is None:
                    continue
                if isinstance(v, (bool, int, float, str)):
                    event_attrs[k] = v
                else:
                    event_attrs[k] = str(v)
            ts = ev.timestamp_ns if ev.timestamp_ns is not None else start_times_ns[i]
            span.add_event(ev.name, attributes=event_attrs or None, timestamp=ts)
        code = (s.status_code or "UNSET").upper()
        if code == "ERROR":
            span.set_status(Status(StatusCode.ERROR))
        else:
            span.set_status(Status(StatusCode.UNSET))
        span_objects.append(span)

    root = span_objects[0]
    trace_id = format(root.get_span_context().trace_id, "032x")
    span_ids = [format(s.get_span_context().span_id, "016x") for s in span_objects]

    # End spans using the scheduled end_time so reported durations match the IR.
    # Ending order doesn't matter once explicit end_time is supplied.
    for i, span in enumerate(span_objects):
        span.end(end_time=end_times_ns[i])

    return (trace_id, span_ids)


def render_trace(
    tracer: Any,
    spec: TraceSpec,
    *,
    tracer_cp: Any | None = None,
    tracer_dp: Any | None = None,
    trace_base_time_ns: int | None = None,
) -> str | None:
    """
    Materialize a TraceSpec into OTEL spans using the provided tracer.

    Returns the generated trace_id (hex) or None when there are no spans.
    """
    trace_id, _span_ids = render_trace_with_span_ids(
        tracer,
        spec,
        tracer_cp=tracer_cp,
        tracer_dp=tracer_dp,
        trace_base_time_ns=trace_base_time_ns,
    )
    return trace_id
