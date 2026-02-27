"""
Dependency rules for realistic rule-based telemetry generation.

These invariants are enforced by the simulator; trace validation can check them
against emitted spans.

Rules:
  1. Session and conversation identity: For the same logical session, all spans
     MUST carry the same session.id and gen_ai.conversation.id, and these two
     MUST be equal (OTEL GenAI convention).
  2. IDs from config only: tenant.id, a2a.agent.target.id, mcp.server.uuid,
     mcp.tool.uuid MUST appear in config/config.yaml and be resolved from
     scenario keys (tenant, agent, mcp_server); no inline UUIDs in scenarios.
  3. Conversation consistency: gen_ai.input.messages and gen_ai.output.messages
     MUST be consistent per turn (user message -> assistant message); content
     comes from scenario conversation.turns or config conversation_samples only.
  4. Tool sequence and division: Tool order and MCP server (division) MUST
     match the scenario's correct_flow and mcp_server; wrong_division and
     partial_workflow modifiers only swap or drop via config-defined values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import attr as config_attr

# Attribute keys that must match between session and conversation.
SESSION_ATTR = config_attr("session.id")
CONVERSATION_ATTR = "gen_ai.conversation.id"

# Attributes that must be resolved from config (tenant, agent, MCP).
CONFIG_DRIVEN_ATTRS = (
    config_attr("tenant.id"),
    config_attr("a2a.agent.target.id"),
    config_attr("mcp.server.uuid"),
    config_attr("mcp.tool.uuid"),
)


@dataclass
class RuleCheck:
    """Result of checking one dependency rule."""

    rule_id: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def check_session_equals_conversation(span_attrs: dict[str, Any]) -> RuleCheck:
    """
    Rule 1: session.id and gen_ai.conversation.id must be present and equal.
    """
    sid = span_attrs.get(SESSION_ATTR)
    cid = span_attrs.get(CONVERSATION_ATTR)
    if sid is None and cid is None:
        return RuleCheck(
            rule_id="session_conversation_identity",
            passed=True,
            message="Session and conversation not set (span may not require them)",
            details={},
        )
    if sid is None or cid is None:
        return RuleCheck(
            rule_id="session_conversation_identity",
            passed=False,
            message="One of session.id or gen_ai.conversation.id is missing",
            details={SESSION_ATTR: sid, CONVERSATION_ATTR: cid},
        )
    if str(sid) != str(cid):
        return RuleCheck(
            rule_id="session_conversation_identity",
            passed=False,
            message="session.id and gen_ai.conversation.id must be equal",
            details={SESSION_ATTR: sid, CONVERSATION_ATTR: cid},
        )
    return RuleCheck(
        rule_id="session_conversation_identity",
        passed=True,
        message="session.id equals gen_ai.conversation.id",
        details={},
    )


def check_config_driven_attributes(
    span_attrs: dict[str, Any],
    allowed_tenant_ids: set[str],
    allowed_agent_ids: set[str],
    allowed_mcp_server_uuids: set[str],
    allowed_mcp_tool_uuids: set[str],
) -> RuleCheck:
    """
    Rule 2: tenant.id, a2a.agent.target.id, mcp.server.uuid, mcp.tool.uuid
    must be in the allowed sets (from config).
    """
    tenant_attr = config_attr("tenant.id")
    agent_attr = config_attr("a2a.agent.target.id")
    server_attr = config_attr("mcp.server.uuid")
    tool_attr = config_attr("mcp.tool.uuid")

    invalid = []

    tid = span_attrs.get(tenant_attr)
    if tid is not None and str(tid).strip():
        if str(tid) not in allowed_tenant_ids:
            invalid.append((tenant_attr, tid))
    aid = span_attrs.get(agent_attr)
    if aid is not None and str(aid).strip():
        if str(aid) not in allowed_agent_ids:
            invalid.append((agent_attr, aid))
    srv = span_attrs.get(server_attr)
    if srv is not None and str(srv).strip():
        if str(srv) not in allowed_mcp_server_uuids:
            invalid.append((server_attr, srv))
    tool = span_attrs.get(tool_attr)
    if tool is not None and str(tool).strip():
        if str(tool) not in allowed_mcp_tool_uuids:
            invalid.append((tool_attr, tool))

    if invalid:
        return RuleCheck(
            rule_id="config_driven_ids",
            passed=False,
            message="One or more IDs not in config",
            details={"invalid": invalid},
        )
    return RuleCheck(
        rule_id="config_driven_ids",
        passed=True,
        message="All config-driven IDs present and allowed",
        details={},
    )


def validate_span_rules(
    span_attrs: dict[str, Any],
    allowed_tenant_ids: set[str] | None = None,
    allowed_agent_ids: set[str] | None = None,
    allowed_mcp_server_uuids: set[str] | None = None,
    allowed_mcp_tool_uuids: set[str] | None = None,
) -> list[RuleCheck]:
    """
    Run dependency rule checks on a span's attributes.

    If allowed_* sets are None, config_driven_ids check is skipped.
    """
    results = []
    results.append(check_session_equals_conversation(span_attrs))

    if (
        allowed_tenant_ids is not None
        and allowed_agent_ids is not None
        and allowed_mcp_server_uuids is not None
        and allowed_mcp_tool_uuids is not None
    ):
        results.append(
            check_config_driven_attributes(
                span_attrs,
                allowed_tenant_ids,
                allowed_agent_ids,
                allowed_mcp_server_uuids,
                allowed_mcp_tool_uuids,
            )
        )
    return results
