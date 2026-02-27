"""
Resolve key-based scenario context to UUIDs and refs from config.

Scenarios specify tenant (key), agent (id), mcp_server (key). This module
resolves them against config/config.yaml so that tenant.id, a2a.agent.target.id,
mcp.server.uuid, and mcp.tool.uuid always come from config only.

Returns plain dicts/lists so that scenario_loader can build ScenarioContext
without circular imports.
"""

from pathlib import Path
from typing import Any


class ConfigResolutionError(Exception):
    """Raised when a scenario key cannot be resolved from config."""

    pass


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    from ..config import CONFIG_PATH, load_yaml

    path = config_path or CONFIG_PATH
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigResolutionError("Config is missing or invalid")
    return data


def _get_tenant_uuid(tenants: dict[str, Any], tenant_key: str) -> str:
    key = (tenant_key or "").strip().lower()
    if not key:
        raise ConfigResolutionError("context.tenant is required")
    tenant = tenants.get(key)
    if not isinstance(tenant, dict):
        raise ConfigResolutionError(f"Unknown tenant key: {tenant_key}")
    tid = tenant.get("id")
    if not isinstance(tid, str) or not tid.strip():
        raise ConfigResolutionError(f"Tenant '{tenant_key}' has no id in config")
    return tid.strip()


def _get_agent_id(agents: list[Any], agent_id: str) -> str:
    aid = (agent_id or "").strip()
    if not aid:
        raise ConfigResolutionError("context.agent is required")
    for a in agents or []:
        if not isinstance(a, dict):
            continue
        if (a.get("id") or "").strip() == aid:
            return aid
    raise ConfigResolutionError(f"Unknown agent id: {agent_id}")


def _get_mcp_server_ref(
    mcp_servers: dict[str, Any], mcp_server_key: str
) -> tuple[str, list[dict[str, str]]]:
    key = (mcp_server_key or "").strip().lower()
    if not key:
        raise ConfigResolutionError("context.mcp_server is required")
    server = mcp_servers.get(key)
    if not isinstance(server, dict):
        raise ConfigResolutionError(f"Unknown mcp_server key: {mcp_server_key}")
    uuid_val = server.get("mcp_server_uuid")
    if not isinstance(uuid_val, str) or not uuid_val.strip():
        raise ConfigResolutionError(
            f"MCP server '{mcp_server_key}' has no mcp_server_uuid in config"
        )
    tools_raw = server.get("tools") or []
    tools: list[dict[str, str]] = []
    for t in tools_raw:
        if (
            isinstance(t, dict)
            and isinstance(t.get("name"), str)
            and isinstance(t.get("tool_uuid"), str)
        ):
            tools.append({"name": t["name"], "tool_uuid": t["tool_uuid"]})
    return uuid_val.strip(), tools


def resolve_context(
    tenant_key: str,
    agent_id: str,
    mcp_server_key: str | None = None,
    workflow: str | None = None,
    correct_flow: Any = None,
    error_pattern: str = "happy_path",
    error_config: Any = None,
    redaction_applied: str = "none",
    actual_steps: list[str] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Resolve key-based context to a dict with tenant_uuid and agents (plain structs) from config.

    mcp_server_key is optional: when missing or empty, agents get mcp: [] (e.g. control-plane-only scenarios).
    Caller (scenario_loader) builds ScenarioContext from the returned dict.
    """
    data = _load_config(config_path)
    tenants = data.get("tenants") or {}
    if not isinstance(tenants, dict):
        tenants = {}
    agents_list = data.get("agents")
    if not isinstance(agents_list, list):
        agents_list = []
    mcp_servers = data.get("mcp_servers") or {}
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}

    tenant_uuid = _get_tenant_uuid(tenants, tenant_key)
    resolved_agent_id = _get_agent_id(agents_list, agent_id)
    mcp_key = (mcp_server_key or "").strip() if mcp_server_key is not None else ""
    if mcp_key:
        server_uuid, tools_list = _get_mcp_server_ref(mcp_servers, mcp_key)
    else:
        server_uuid = ""
        tools_list = []

    agents_payload = [
        {
            "uuid": resolved_agent_id,
            "mcp": [{"server_uuid": server_uuid, "tools": tools_list}] if server_uuid else [],
        }
    ]
    return {
        "tenant_uuid": tenant_uuid,
        "agents": agents_payload,
        "workflow": workflow,
        "correct_flow": correct_flow,
        "error_pattern": error_pattern,
        "error_config": error_config,
        "redaction_applied": redaction_applied,
        "actual_steps": actual_steps,
    }


def get_default_tenant_id(config_path: Path | None = None) -> str:
    """Return the first tenant id from config (delegates to config.get_default_tenant_id)."""
    from ..config import CONFIG_PATH
    from ..config import get_default_tenant_id as _config_default_tenant_id

    try:
        return _config_default_tenant_id(config_path or CONFIG_PATH)
    except ValueError as e:
        raise ConfigResolutionError(str(e)) from e
