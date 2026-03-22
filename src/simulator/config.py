"""
Load and expose simulator configuration from YAML.

Configuration is read from resource/config/config.yaml. All correlation IDs
are generated from id_formats only (no fallbacks). Tenant/agent/mcp_server
resolution is done via config_resolver using keys from scenario YAML.
"""

from pathlib import Path
from typing import Any

import yaml

# Attribute prefix for semantic-convention attributes (e.g. vendor.session.id).
#
# This value is set at runtime from CLI args (see `simulator.cli`).
ATTR_PREFIX = "vendor"


def resource_attributes_with_tenant(
    base_attributes: dict[str, Any],
    tenant_id: str,
) -> dict[str, Any]:
    """Return a copy of preset resource attributes with ``{ATTR_PREFIX}.tenant.id`` set.

    ``ATTR_PREFIX`` comes from the CLI ``--vendor`` flag (default ``vendor``).
    """
    out = dict(base_attributes)
    out[f"{ATTR_PREFIX}.tenant.id"] = tenant_id
    return out


def _default_config_dir() -> Path:
    """Default config directory: cwd/resource/config, or package-relative."""
    for base in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
        candidate = base / "resource" / "config"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "resource" / "config"


def load_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml from config_dir. Returns raw dict; use config_resolver for resolution."""
    directory = Path(config_dir) if config_dir else _default_config_dir()
    path = directory / "config.yaml"
    if not path.exists():
        return {
            "id_formats": {},
            "tenants": {},
            "agents": [],
            "mcp_servers": {},
            "mcp_tool_genai_payloads": {},
        }
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resource_schema_url(config_dir: str | Path | None = None) -> str:
    """Schema URL from resource.yaml.

    This must be explicitly defined; no hardcoded fallback is used.
    """
    directory = Path(config_dir) if config_dir else _default_config_dir()
    path = directory / "resource.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            url = data.get("schema_url")
            if url:
                return str(url)
    raise RuntimeError(f"schema_url must be set in {path} (no default schema URL)")


def get_default_tenant_id(config: dict[str, Any] | None = None) -> str | None:
    """First tenant id from config."""
    if config is None:
        config = load_config()
    tenants = config.get("tenants") or {}
    if not tenants:
        return None
    first_key = next(iter(tenants), None)
    if first_key is None:
        return None
    tenant = tenants[first_key]
    if isinstance(tenant, dict) and "id" in tenant:
        return str(tenant["id"])
    return str(tenant)
