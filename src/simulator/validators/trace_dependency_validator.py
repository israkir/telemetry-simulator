"""
Validate emitted traces against dependency rules (session=conversation, IDs from config).

Uses scenarios/dependency_rules to check each span; optionally loads config
to validate tenant/agent/MCP IDs.
"""

from pathlib import Path
from typing import Any

from ..config import CONFIG_PATH, load_yaml
from ..scenarios.dependency_rules import RuleCheck, validate_span_rules


def _load_allowed_ids(
    config_path: Path | None = None,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Load tenant, agent, mcp server, mcp tool UUIDs from config."""
    path = config_path or CONFIG_PATH
    data = load_yaml(path)
    if not isinstance(data, dict):
        return set(), set(), set(), set()

    tenant_ids: set[str] = set()
    tenants = data.get("tenants") or {}
    if isinstance(tenants, dict):
        for t in tenants.values():
            if isinstance(t, dict):
                tid = t.get("id")
                if isinstance(tid, str) and tid.strip():
                    tenant_ids.add(tid.strip())

    agent_ids: set[str] = set()
    agents = data.get("agents") or []
    if isinstance(agents, list):
        for a in agents:
            if isinstance(a, dict):
                aid = a.get("id")
                if isinstance(aid, str) and aid.strip():
                    agent_ids.add(aid.strip())

    server_uuids: set[str] = set()
    tool_uuids: set[str] = set()
    mcp_servers = data.get("mcp_servers") or {}
    if isinstance(mcp_servers, dict):
        for s in mcp_servers.values():
            if not isinstance(s, dict):
                continue
            uuid_val = s.get("mcp_server_uuid")
            if isinstance(uuid_val, str) and uuid_val.strip():
                server_uuids.add(uuid_val.strip())
            for t in s.get("tools") or []:
                if isinstance(t, dict):
                    tu = t.get("tool_uuid")
                    if isinstance(tu, str) and tu.strip():
                        tool_uuids.add(tu.strip())

    return tenant_ids, agent_ids, server_uuids, tool_uuids


def validate_trace_dependencies(
    spans: list[dict[str, Any]],
    config_path: Path | None = None,
    check_config_ids: bool = True,
) -> list[RuleCheck]:
    """
    Run dependency rule checks on all spans in a trace.

    :param spans: List of span dicts (each with "attributes").
    :param config_path: Path to config.yaml for allowed IDs.
    :param check_config_ids: If True, validate tenant/agent/mcp IDs against config.
    :return: List of RuleCheck results (one per span per rule).
    """
    results: list[RuleCheck] = []
    allowed_tenant_ids: set[str] = set()
    allowed_agent_ids: set[str] = set()
    allowed_server_uuids: set[str] = set()
    allowed_tool_uuids: set[str] = set()
    if check_config_ids:
        allowed_tenant_ids, allowed_agent_ids, allowed_server_uuids, allowed_tool_uuids = (
            _load_allowed_ids(config_path)
        )

    for span in spans:
        attrs = span.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        checks = validate_span_rules(
            attrs,
            allowed_tenant_ids=allowed_tenant_ids if check_config_ids else None,
            allowed_agent_ids=allowed_agent_ids if check_config_ids else None,
            allowed_mcp_server_uuids=allowed_server_uuids if check_config_ids else None,
            allowed_mcp_tool_uuids=allowed_tool_uuids if check_config_ids else None,
        )
        results.extend(checks)

    return results
