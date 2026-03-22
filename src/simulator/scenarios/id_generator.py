"""
Generate correlation IDs from config id_formats only.

Templates use placeholders like {hex:N} for N random hex characters.
No fallbacks; all IDs come from config.yaml id_formats.
"""

import re
import secrets
from typing import Any


def _expand_template(template: str) -> str:
    """Replace {hex:N} with N random lowercase hex characters."""

    def repl(match: re.Match[str]) -> str:
        n = int(match.group(1), 10)
        n = max(1, min(n, 64))
        return secrets.token_hex(n // 2 + (n % 2))[:n]

    return re.sub(r"\{hex:(\d+)\}", repl, template)


def generate_id(formats: dict[str, str], key: str) -> str:
    """Generate one ID for the given key (e.g. session_id, request_id)."""
    template = formats.get(key, "")
    if not template:
        return ""
    return _expand_template(template)


def generate_ids_for_turn(
    formats: dict[str, Any],
    *,
    session_id: str | None = None,
    conversation_id: str | None = None,
    tool_count: int = 0,
) -> dict[str, Any]:
    """
    Generate session_id, conversation_id, request_id, and per-tool MCP tool call ids.

    IDs come from config id_formats only (no fallbacks). The `mcp_tool_call_id` template
    is expanded `tool_count` times to produce distinct ids for multi-tool interactions.
    """
    id_formats = formats.get("id_formats") or {}
    sess = session_id or generate_id(id_formats, "session_id")
    conv = conversation_id or generate_id(id_formats, "conversation_id")
    return {
        "session_id": sess,
        "conversation_id": conv,
        "request_id": generate_id(id_formats, "request_id"),
        "mcp_tool_call_ids": [
            generate_id(id_formats, "mcp_tool_call_id") for _ in range(max(0, tool_count))
        ],
    }
