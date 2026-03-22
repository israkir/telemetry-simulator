"""
Resolve scenario context from config.yaml.

Scenarios reference tenant by key, agent by id, mcp_server by key.
This module resolves those to concrete tenant id, agent id, mcp_server_uuid,
and tool name -> tool_uuid mapping.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ResolvedMcpTool:
    """Resolved MCP tool: name and UUID."""

    name: str
    tool_uuid: str


@dataclass
class ResolvedContext:
    """Resolved scenario context from config."""

    tenant_id: str
    agent_id: str
    mcp_server_uuid: str
    tools_by_name: dict[str, ResolvedMcpTool]

    def get_tool_uuid(self, tool_name: str) -> str | None:
        """Return tool UUID for tool name, or None if not found."""
        t = self.tools_by_name.get(tool_name)
        return t.tool_uuid if t else None


def resolve_context(
    config: dict[str, Any],
    *,
    tenant_key: str | None = None,
    agent_id: str | None = None,
    mcp_server_key: str | None = None,
    default_tenant_id: str | None = None,
) -> ResolvedContext:
    """
    Resolve tenant, agent, and mcp_server from config using scenario keys.

    tenant_key: key in config.tenants (e.g. "toro")
    agent_id: id from config.agents (e.g. "toro-customer-assistant-001")
    mcp_server_key: key in config.mcp_servers (e.g. "phone")
    default_tenant_id: explicit tenant id override. When provided, it takes
        precedence over `tenant_key` (including scenarios that set `tenant:` in
        YAML). This is primarily intended for the CLI `--tenant-id` flag.
    """
    tenants = config.get("tenants") or {}
    agents_list = config.get("agents") or []
    mcp_servers = config.get("mcp_servers") or {}

    # Tenant
    if default_tenant_id:
        tenant_id = default_tenant_id
    elif tenant_key:
        if tenant_key not in tenants:
            raise ValueError(
                f"Unable to resolve tenant id: scenario.context.tenant_key={tenant_key!r} "
                "was not found in resource/config/config.yaml (expected `tenants` entry)."
            )
        t = tenants[tenant_key]
        tenant_id = t.get("id", str(t)) if isinstance(t, dict) else str(t)
    else:
        raise ValueError(
            "Unable to resolve tenant id: scenario did not define context.tenant "
            "scenario.context.tenant_key is missing and no default_tenant_id was provided "
            "(expected `tenant` in scenario YAML and/or `tenants` in resource/config/config.yaml "
            "or --tenant-id via CLI)."
        )

    # Agent (list of {id: "..."})
    resolved_agent_id = ""
    if agent_id:
        for a in agents_list:
            aid = a.get("id", "") if isinstance(a, dict) else str(a)
            if aid == agent_id:
                resolved_agent_id = aid
                break
        if not resolved_agent_id:
            resolved_agent_id = agent_id
    elif agents_list:
        first = agents_list[0]
        resolved_agent_id = first.get("id", str(first)) if isinstance(first, dict) else str(first)

    # MCP server and tools
    mcp_server_uuid = ""
    tools_by_name: dict[str, ResolvedMcpTool] = {}
    if mcp_server_key and mcp_server_key in mcp_servers:
        server = mcp_servers[mcp_server_key]
        if isinstance(server, dict):
            mcp_server_uuid = server.get("mcp_server_uuid", "")
            for tool in server.get("tools") or []:
                if isinstance(tool, dict) and tool.get("name"):
                    tools_by_name[tool["name"]] = ResolvedMcpTool(
                        name=tool["name"],
                        tool_uuid=tool.get("tool_uuid", ""),
                    )

    return ResolvedContext(
        tenant_id=tenant_id,
        agent_id=resolved_agent_id,
        mcp_server_uuid=mcp_server_uuid,
        tools_by_name=tools_by_name,
    )
