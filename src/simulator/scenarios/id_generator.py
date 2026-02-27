"""
Generate correlation IDs from shared scenario config format definitions.

Formats are defined in config/config.yaml with placeholders:
  {hex:N}   - N random hex chars
  {uuid}    - full UUID with dashes
  {tenant_id} - from context when provided
"""

import re
import uuid
from pathlib import Path
from typing import Any

from ..config import CONFIG_PATH, load_yaml
from .config_resolver import ConfigResolutionError

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
    """Load id_formats from config/config.yaml. Config must define id_formats."""
    path = config_path or CONFIG_PATH
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigResolutionError("Config is missing or invalid")
    raw = data.get("id_formats")
    if not isinstance(raw, dict):
        raise ConfigResolutionError("Config must define id_formats")
    result = {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}
    if not result:
        raise ConfigResolutionError("id_formats must contain at least one template")
    return result


class ScenarioIdGenerator:
    """
    Generate IDs from shared format definitions.

    Use for session_id, request_id, conversation_id, mcp_tool_call_id
    so all scenarios share the same ID shape from config/config.yaml.
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
            raise ConfigResolutionError(f"Unknown id format key: {format_key}")
        return _expand_template(template, tenant_id=tenant_id)

    def session_id(self, tenant_id: str | None = None) -> str:
        return self.generate("session_id", tenant_id=tenant_id)

    def request_id(self, tenant_id: str | None = None) -> str:
        return self.generate("request_id", tenant_id=tenant_id)

    def conversation_id(self, tenant_id: str | None = None) -> str:
        return self.generate("conversation_id", tenant_id=tenant_id)

    def mcp_tool_call_id(self, tenant_id: str | None = None) -> str:
        return self.generate("mcp_tool_call_id", tenant_id=tenant_id)
