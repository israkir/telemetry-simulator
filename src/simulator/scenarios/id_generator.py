"""
Generate correlation IDs from shared scenario config format definitions.

Formats are defined in scenarios_config.yaml with placeholders:
  {hex:N}   - N random hex chars
  {uuid}    - full UUID with dashes
  {tenant_id} - from context when provided
"""

import re
import uuid
from pathlib import Path
from typing import Any

import yaml

# Default config next to this module; overridable for tests.
DEFAULT_CONFIG_PATH = Path(__file__).parent / "scenarios_config.yaml"

_HEX_PLACEHOLDER = re.compile(r"\{hex:(\d+)\}")
_UUID_PLACEHOLDER = re.compile(r"\{uuid\}")
_TENANT_PLACEHOLDER = re.compile(r"\{tenant_id\}")


def _expand_template(template: str, tenant_id: str | None = None) -> str:
    """Expand placeholders in template. Uses random values for {hex:N} and {uuid}."""
    if not template:
        return ""

    def replace_hex(match: re.Match[str]) -> str:
        n = min(max(1, int(match.group(1))), 64)
        return uuid.uuid4().hex[:n]

    def replace_uuid(_: re.Match[str]) -> str:
        return str(uuid.uuid4())

    def replace_tenant(_: re.Match[str]) -> str:
        return tenant_id or ""

    out = _HEX_PLACEHOLDER.sub(replace_hex, template)
    out = _UUID_PLACEHOLDER.sub(replace_uuid, out)
    out = _TENANT_PLACEHOLDER.sub(replace_tenant, out)
    return out


def load_id_formats(config_path: Path | None = None) -> dict[str, str]:
    """Load id_formats from scenarios_config.yaml. Returns format key -> template."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return _default_id_formats()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return _default_id_formats()
    raw = data.get("id_formats")
    if not isinstance(raw, dict):
        return _default_id_formats()
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def _default_id_formats() -> dict[str, str]:
    """Fallback when config is missing (convention-aligned: vendor.session.id, vendor.request.id, vendor.mcp.tool.call.id)."""
    return {
        "session_id": "sess_toro_{hex:12}",
        "conversation_id": "sess_toro_{hex:12}",
        "request_id": "req_{hex:6}",
        "mcp_tool_call_id": "mcp_call_{hex:12}",
    }


class ScenarioIdGenerator:
    """
    Generate IDs from shared format definitions.

    Use for session_id, request_id, conversation_id, mcp_tool_call_id
    so all scenarios share the same ID shape from scenarios_config.yaml.
    """

    def __init__(self, config_path: Path | None = None):
        self._formats = load_id_formats(config_path)

    def generate(self, format_key: str, tenant_id: str | None = None, **kwargs: Any) -> str:
        """
        Generate an ID for the given format key.

        :param format_key: Key from id_formats (e.g. session_id, request_id, mcp_tool_call_id).
        :param tenant_id: Optional tenant id for {tenant_id} placeholder.
        :param kwargs: Ignored; for future placeholders.
        :return: Generated id string.
        """
        template = self._formats.get(format_key)
        if not template:
            # Fallback for unknown keys
            template = f"id_{uuid.uuid4().hex[:12]}"
        return _expand_template(template, tenant_id=tenant_id)

    def session_id(self, tenant_id: str | None = None) -> str:
        return self.generate("session_id", tenant_id=tenant_id)

    def request_id(self, tenant_id: str | None = None) -> str:
        return self.generate("request_id", tenant_id=tenant_id)

    def conversation_id(self, tenant_id: str | None = None) -> str:
        return self.generate("conversation_id", tenant_id=tenant_id)

    def mcp_tool_call_id(self, tenant_id: str | None = None) -> str:
        return self.generate("mcp_tool_call_id", tenant_id=tenant_id)
