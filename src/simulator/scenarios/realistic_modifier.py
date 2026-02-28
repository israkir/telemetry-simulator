"""
Apply realistic scenario modifiers to trace hierarchies.

Drives scenario-based telemetry for:
- 4xx from bad/missing parameters: error.type=invalid_arguments, optional http.response.status_code 4xx
- Wrong division/disambiguation: same tool name, wrong vendor-prefixed mcp.server.uuid (and mcp.tool.uuid)
- Partial/wrong-order workflows: hierarchy built from actual_steps (fewer or reordered steps)
- Ungrounded answers: response_compose or RAG span with error (retrieval_error / composition)

Config is read from config/config.yaml (realistic_scenarios, mcp_servers).
SemConv-aligned: error.type and attribute names follow conventions/semconv.yaml.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ATTR_PREFIX, CONFIG_PATH, load_yaml
from ..config import attr as config_attr
from ..generators.trace_generator import (
    SpanType,
    TraceHierarchy,
)


@dataclass
class ErrorTemplate:
    """Error semantics for a simulation goal (SemConv-aligned)."""

    error_type: str = "tool_error"
    http_status_codes: list[int] = field(default_factory=list)


@dataclass
class RealisticScenarioConfig:
    """
    Loaded from config/config.yaml realistic_scenarios and mcp_servers.

    - divisions: division name -> mcp_servers key (for wrong_division resolution)
    - error_templates: simulation_goal -> ErrorTemplate
    - mcp_servers_by_key: mcp_servers key -> { mcp_server_uuid, tools: [{ name, tool_uuid }] }

    Latency is driven by config latency_profiles and scenario data_plane.latency_profile (no modifier).
    """

    divisions: dict[str, str] = field(default_factory=dict)
    error_templates: dict[str, ErrorTemplate] = field(default_factory=dict)
    mcp_servers_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "RealisticScenarioConfig":
        path = config_path or CONFIG_PATH
        data = load_yaml(path)
        if not isinstance(data, dict):
            return cls()

        rs = data.get("realistic_scenarios")
        if not isinstance(rs, dict):
            rs = {}

        divisions = {}
        for k, v in (rs.get("divisions") or {}).items():
            if isinstance(k, str) and isinstance(v, str):
                divisions[k.lower()] = v

        error_templates = {}
        for goal, et in (rs.get("error_templates") or {}).items():
            if not isinstance(et, dict):
                continue
            error_templates[goal] = ErrorTemplate(
                error_type=et.get("error_type", "tool_error"),
                http_status_codes=et.get("http_status_codes") or [],
            )

        mcp_servers_by_key = {}
        for key, server in (data.get("mcp_servers") or {}).items():
            if not isinstance(server, dict):
                continue
            uuid_val = server.get("mcp_server_uuid")
            tools_raw = server.get("tools") or []
            tools = []
            for t in tools_raw:
                if (
                    isinstance(t, dict)
                    and isinstance(t.get("name"), str)
                    and isinstance(t.get("tool_uuid"), str)
                ):
                    tools.append({"name": t["name"], "tool_uuid": t["tool_uuid"]})
            mcp_servers_by_key[key] = {"mcp_server_uuid": uuid_val, "tools": tools}

        return cls(
            divisions=divisions,
            error_templates=error_templates,
            mcp_servers_by_key=mcp_servers_by_key,
        )


def _collect_mcp_hierarchies(hierarchy: TraceHierarchy) -> list[TraceHierarchy]:
    """Return MCP_TOOL_EXECUTE hierarchies in tree order (depth-first)."""
    result: list[TraceHierarchy] = []

    def walk(h: TraceHierarchy) -> None:
        if h.root_config.span_type == SpanType.MCP_TOOL_EXECUTE:
            result.append(h)
        for child in h.children:
            walk(child)

    for child in hierarchy.children:
        walk(child)
    return result


def _find_response_compose(hierarchy: TraceHierarchy) -> TraceHierarchy | None:
    """Return the first RESPONSE_COMPOSE hierarchy in tree order."""
    if hierarchy.root_config.span_type == SpanType.RESPONSE_COMPOSE:
        return hierarchy
    for child in hierarchy.children:
        found = _find_response_compose(child)
        if found:
            return found
    return None


def _apply_4xx_invalid_arguments(
    hierarchy: TraceHierarchy,
    template: ErrorTemplate,
    overrides: dict[str, Any],
    attr_prefix: str,
) -> None:
    """Set one MCP attempt span to error with invalid_arguments and optional http.response.status_code."""
    mcp_list = _collect_mcp_hierarchies(hierarchy)
    if not mcp_list:
        return
    step_index = overrides.get("step_index_for_4xx")
    if step_index is None:
        step_index = random.randint(0, len(mcp_list) - 1)
    else:
        step_index = max(0, min(int(step_index), len(mcp_list) - 1))
    mcp_h = mcp_list[step_index]
    if not mcp_h.children:
        return
    attempt = mcp_h.children[0]
    attempt.root_config.error_rate = 1.0
    attrs = dict(attempt.root_config.attribute_overrides or {})
    attrs["error.type"] = template.error_type
    if template.http_status_codes:
        attrs["http.response.status_code"] = random.choice(template.http_status_codes)
    # Ensure prefixed attribute for tenant/schema consistency when used
    attrs[config_attr("error.type")] = template.error_type
    attempt.root_config.attribute_overrides = attrs


def _apply_wrong_division(
    hierarchy: TraceHierarchy,
    correct_server_uuid: str,
    correct_tool_name: str,
    mcp_server_key: str,
    config: RealisticScenarioConfig,
    overrides: dict[str, Any],
) -> None:
    """
    Replace mcp.server.uuid (and mcp.tool.uuid) on one MCP span with a wrong division's server.
    Same tool name, wrong vendor-prefixed mcp.server.uuid (ambiguous customer input / missing division context).
    """
    other_keys = [k for k in config.divisions.values() if k != mcp_server_key]
    if not other_keys:
        return
    wrong_key = overrides.get("wrong_division_target")
    if isinstance(wrong_key, str) and wrong_key.lower() in config.divisions:
        wrong_key = config.divisions[wrong_key.lower()]
    elif wrong_key not in config.mcp_servers_by_key:
        wrong_key = random.choice(other_keys)

    mcp_list = _collect_mcp_hierarchies(hierarchy)
    if not mcp_list:
        return
    target_index = overrides.get("step_index_wrong_division")
    if target_index is None:
        target_index = random.randint(0, len(mcp_list) - 1)
    else:
        target_index = max(0, min(int(target_index), len(mcp_list) - 1))
    mcp_h = mcp_list[target_index]
    # Use tool name from the span we're modifying (same tool name, wrong server).
    tool_name = (mcp_h.root_config.attribute_overrides or {}).get(
        "gen_ai.tool.name"
    ) or correct_tool_name
    wrong_server = config.mcp_servers_by_key.get(wrong_key)
    if not wrong_server:
        return
    wrong_server_uuid = wrong_server.get("mcp_server_uuid")
    wrong_tool_uuid = None
    for t in wrong_server.get("tools") or []:
        if t.get("name") == tool_name:
            wrong_tool_uuid = t.get("tool_uuid")
            break
    if not wrong_tool_uuid and (wrong_server.get("tools") or []):
        wrong_tool_uuid = wrong_server["tools"][0].get("tool_uuid")
    if not wrong_server_uuid:
        return

    overrides_root = dict(mcp_h.root_config.attribute_overrides or {})
    overrides_root[config_attr("mcp.server.uuid")] = wrong_server_uuid
    if wrong_tool_uuid:
        overrides_root[config_attr("mcp.tool.uuid")] = wrong_tool_uuid
    mcp_h.root_config.attribute_overrides = overrides_root
    if mcp_h.children:
        attempt_overrides = dict(mcp_h.children[0].root_config.attribute_overrides or {})
        attempt_overrides[config_attr("mcp.server.uuid")] = wrong_server_uuid
        if wrong_tool_uuid:
            attempt_overrides[config_attr("mcp.tool.uuid")] = wrong_tool_uuid
        mcp_h.children[0].root_config.attribute_overrides = attempt_overrides
    # Optionally mark this tool call as failed (wrong division often returns error)
    template = config.error_templates.get("wrong_division") or ErrorTemplate(
        error_type="tool_error"
    )
    if mcp_h.children:
        mcp_h.children[0].root_config.error_rate = 1.0
        att = mcp_h.children[0].root_config.attribute_overrides
        if att is None:
            att = {}
        att["error.type"] = template.error_type
        att[config_attr("error.type")] = template.error_type
        mcp_h.children[0].root_config.attribute_overrides = att


def _apply_ungrounded_response(
    hierarchy: TraceHierarchy,
    template: ErrorTemplate,
    attr_prefix: str,
) -> None:
    """Set response_compose span to fail (step.outcome=fail, error.type)."""
    comp = _find_response_compose(hierarchy)
    if not comp:
        return
    comp.root_config.error_rate = 1.0
    attrs = dict(comp.root_config.attribute_overrides or {})
    attrs[config_attr("step.outcome")] = "fail"
    attrs["error.type"] = template.error_type
    attrs[config_attr("error.type")] = template.error_type
    comp.root_config.attribute_overrides = attrs


def _apply_partial_workflow(
    hierarchy: TraceHierarchy,
    template: ErrorTemplate,
    attr_prefix: str,
) -> None:
    """Optionally mark response_compose as fail to signal incomplete/wrong-order flow."""
    comp = _find_response_compose(hierarchy)
    if not comp:
        return
    # Partial flow: last step may still succeed but we can signal incompleteness
    comp.root_config.error_rate = 1.0
    attrs = dict(comp.root_config.attribute_overrides or {})
    attrs[config_attr("step.outcome")] = "fail"
    attrs["error.type"] = template.error_type
    attrs[config_attr("error.type")] = template.error_type
    comp.root_config.attribute_overrides = attrs


def apply_realistic_scenario(
    hierarchy: TraceHierarchy,
    simulation_goal: str | None,
    realistic_overrides: dict[str, Any] | None,
    context: Any,
    mcp_server_key: str | None = None,
    config: RealisticScenarioConfig | None = None,
    config_path: Path | None = None,
) -> None:
    """
    Apply realistic scenario modifiers in place to hierarchy.

    - simulation_goal: happy_path | 4xx_invalid_arguments | wrong_division | partial_workflow | ungrounded_response
    - realistic_overrides: optional step_index_for_4xx, wrong_division_target, etc.
    - context: ScenarioContext (for correct server UUID and tool name; agents[0].mcp[0])
    - mcp_server_key: scenario's MCP server key (e.g. "phone") for wrong_division correct division.
    - config: loaded RealisticScenarioConfig; if None, loads from config_path.
    """
    if not simulation_goal or (simulation_goal or "").lower() in ("happy_path", "none", ""):
        return
    overrides = dict(realistic_overrides or {})
    if config is None:
        config = RealisticScenarioConfig.load(config_path)
    goal = (simulation_goal or "").lower()
    attr_prefix = ATTR_PREFIX or "vendor"

    if goal == "4xx_invalid_arguments":
        template = config.error_templates.get("4xx_invalid_arguments") or ErrorTemplate(
            error_type="invalid_arguments", http_status_codes=[400, 404, 422]
        )
        _apply_4xx_invalid_arguments(hierarchy, template, overrides, attr_prefix)
        return

    if goal == "wrong_division":
        correct_server_uuid = ""
        correct_tool_name = ""
        if context and getattr(context, "agents", None):
            agents = context.agents
            if agents and getattr(agents[0], "mcp", None) and agents[0].mcp:
                correct_server_uuid = getattr(agents[0].mcp[0], "server_uuid", "") or ""
                tools = getattr(agents[0].mcp[0], "tools", []) or []
                if tools:
                    correct_tool_name = getattr(tools[0], "name", "") or ""
        key = (mcp_server_key or overrides.get("mcp_server_key") or "").strip().lower()
        _apply_wrong_division(
            hierarchy,
            correct_server_uuid,
            correct_tool_name,
            key,
            config,
            overrides,
        )
        return

    if goal == "ungrounded_response":
        template = config.error_templates.get("ungrounded_response") or ErrorTemplate(
            error_type="unavailable"
        )
        _apply_ungrounded_response(hierarchy, template, attr_prefix)
        return

    if goal == "partial_workflow":
        template = config.error_templates.get("partial_workflow") or ErrorTemplate(
            error_type="tool_error"
        )
        _apply_partial_workflow(hierarchy, template, attr_prefix)
        return

    # higher_latency: latency is set by config latency_profiles + scenario latency_profile at hierarchy build time; no modifier.
