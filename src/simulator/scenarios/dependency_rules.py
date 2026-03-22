"""
Validate TraceGraphSpec invariants: ID consistency, tool sequence, semconv structure.

Fail fast in debug mode when invariants break.
"""

from __future__ import annotations

from .ir import TraceGraphSpec


def _get_span_class(attrs: dict, prefix_key: str) -> str | None:
    """Get span class from attributes (vendor-prefixed)."""
    for key in (prefix_key, "vendor.span.class"):
        if key in attrs:
            return str(attrs[key])
    return None


def validate_trace_graph(
    trace_graph: TraceGraphSpec,
    *,
    attr_prefix: str = "vendor",
) -> None:
    """
    Validate invariants on a compiled trace graph.
    Raises ValueError with a clear message if any check fails.
    """
    compiled = trace_graph.compiled_turn
    prefix_key = f"{attr_prefix}.span.class"

    # Session/conversation identity: same IDs used across the three traces
    session_id = compiled.session_id
    conversation_id = compiled.conversation_id
    session_key = f"{attr_prefix}.session.id"
    conv_key = "gen_ai.conversation.id"

    for label, spec in [
        ("cp_request", trace_graph.cp_request),
        ("data_plane", trace_graph.data_plane),
        ("cp_response", trace_graph.cp_response),
    ]:
        if not spec.spans:
            # Control-plane-only scenarios omit data_plane and/or cp_response.
            continue
        for span in spec.spans:
            if session_key in span.attributes and span.attributes[session_key] != session_id:
                raise ValueError(
                    f"Session ID mismatch in {label} span {span.name}: "
                    f"expected {session_id}, got {span.attributes[session_key]}"
                )
            if conv_key in span.attributes and span.attributes[conv_key] != conversation_id:
                raise ValueError(
                    f"Conversation ID mismatch in {label} span {span.name}: "
                    f"expected {conversation_id}, got {span.attributes[conv_key]}"
                )

    # Data-plane: root must be a2a.orchestrate
    dp = trace_graph.data_plane
    if dp.spans:
        root_class = _get_span_class(dp.spans[0].attributes, prefix_key)
        if root_class != "a2a.orchestrate":
            raise ValueError(
                f"Data-plane root span class must be a2a.orchestrate, got {root_class}"
            )

    # Tool sequence: each tool in tool_chain must appear as mcp.tool.execute with gen_ai.tool.name
    tool_names_in_spans: list[str] = []
    for span in dp.spans:
        cls = _get_span_class(span.attributes, prefix_key)
        if cls == "mcp.tool.execute":
            name = span.attributes.get("gen_ai.tool.name")
            if name is not None:
                tool_names_in_spans.append(str(name))
    if tool_names_in_spans != compiled.tool_chain:
        raise ValueError(
            f"Tool sequence mismatch: scenario tool_chain={compiled.tool_chain}, "
            f"mcp.tool.execute spans have gen_ai.tool.name={tool_names_in_spans}"
        )

    # CP request root: request.validation
    if trace_graph.cp_request.spans:
        root_class = _get_span_class(trace_graph.cp_request.spans[0].attributes, prefix_key)
        if root_class != "request.validation":
            raise ValueError(
                f"CP request root span class must be request.validation, got {root_class}"
            )

    # CP response root: response.validation
    if trace_graph.cp_response.spans:
        root_class = _get_span_class(trace_graph.cp_response.spans[0].attributes, prefix_key)
        if root_class != "response.validation":
            raise ValueError(
                f"CP response root span class must be response.validation, got {root_class}"
            )
