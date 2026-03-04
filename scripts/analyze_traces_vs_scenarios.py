#!/usr/bin/env python3
"""
Analyze generated traces (traces.jsonl) against scenario definitions for semantic,
goal, attribute, and contextual errors. Validates ID formats, tenant/agent/MCP
attributes from config, tools_recommend attributes, and optional semconv alignment.

Usage:
  python scripts/analyze_traces_vs_scenarios.py [--traces traces.jsonl] [--scenarios resource/scenarios/definitions] [--config resource/config/config.yaml]
  python scripts/analyze_traces_vs_scenarios.py --config resource/config/config.yaml --resource resource/config/resource.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# UUID → division and tool mapping from config
MCP_SERVER_UUID_TO_DIVISION = {
    "11111111-1111-4eec-8001-000000000001": "phone",
    "22222222-2222-4eec-8002-000000000002": "electronics",
    "33333333-3333-4eec-8003-000000000003": "appliances",
}

# tool_uuid → (division, tool_name) for each division's tools (from config)
TOOL_UUID_MAP = {
    "11111111-1111-4eec-8001-000000000011": ("phone", "new_claim"),
    "11111111-1111-4eec-8001-000000000012": ("phone", "update_appointment"),
    "11111111-1111-4eec-8001-000000000013": ("phone", "claim_status"),
    "11111111-1111-4eec-8001-000000000014": ("phone", "cancel_claim"),
    "11111111-1111-4eec-8001-000000000015": ("phone", "upload_documents"),
    "11111111-1111-4eec-8001-000000000016": ("phone", "choose_insurance"),
    "11111111-1111-4eec-8001-000000000017": ("phone", "buy_insurance"),
    "11111111-1111-4eec-8001-000000000018": ("phone", "cancel_product"),
    "11111111-1111-4eec-8001-000000000019": ("phone", "pay"),
    "22222222-2222-4eec-8002-000000000021": ("electronics", "new_claim"),
    "22222222-2222-4eec-8002-000000000022": ("electronics", "update_appointment"),
    "22222222-2222-4eec-8002-000000000023": ("electronics", "claim_status"),
    "22222222-2222-4eec-8002-000000000024": ("electronics", "cancel_claim"),
    "22222222-2222-4eec-8002-000000000025": ("electronics", "upload_documents"),
    "22222222-2222-4eec-8002-000000000026": ("electronics", "choose_insurance"),
    "22222222-2222-4eec-8002-000000000027": ("electronics", "buy_insurance"),
    "22222222-2222-4eec-8002-000000000028": ("electronics", "cancel_product"),
    "22222222-2222-4eec-8002-000000000029": ("electronics", "pay"),
    "33333333-3333-4eec-8003-000000000031": ("appliances", "new_claim"),
    "33333333-3333-4eec-8003-000000000032": ("appliances", "update_appointment"),
    "33333333-3333-4eec-8003-000000000033": ("appliances", "claim_status"),
    "33333333-3333-4eec-8003-000000000034": ("appliances", "cancel_claim"),
    "33333333-3333-4eec-8003-000000000035": ("appliances", "upload_documents"),
    "33333333-3333-4eec-8003-000000000036": ("appliances", "choose_insurance"),
    "33333333-3333-4eec-8003-000000000037": ("appliances", "buy_insurance"),
    "33333333-3333-4eec-8003-000000000038": ("appliances", "cancel_product"),
    "33333333-3333-4eec-8003-000000000039": ("appliances", "pay"),
}

# Attribute keys (vendor prefix may be gentoro. in traces)
def _attr(short: str) -> str:
    return f"gentoro.{short}" if not short.startswith("gentoro.") else short


def _id_format_to_regex(template: str) -> re.Pattern[str] | None:
    """Convert config id_formats template (e.g. sess_toro_{hex:12}) to a validation regex."""
    if not template or not isinstance(template, str):
        return None
    parts: list[str] = []
    i = 0
    while i < len(template):
        m = re.match(r"\{hex:(\d+)\}", template[i:])
        if m:
            parts.append(f"[0-9a-f]{{{m.group(1)}}}")
            i += len(m.group(0))
        else:
            # Escape literal run
            j = template.find("{", i)
            if j == -1:
                j = len(template)
            parts.append(re.escape(template[i:j]))
            i = j
    try:
        return re.compile("^" + "".join(parts) + "$")
    except re.error:
        return None


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; return empty dict on missing/invalid."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required: pip install pyyaml")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load config.yaml and derive ID validation regexes, tenant/agent/MCP/tools_recommend."""
    raw = load_yaml(config_path)
    if not raw:
        return {}

    # Build regexes from id_formats
    id_formats = raw.get("id_formats") or {}
    regexes: dict[str, re.Pattern[str]] = {}
    for key, template in id_formats.items():
        if isinstance(template, str):
            r = _id_format_to_regex(template)
            if r:
                regexes[key] = r
    raw["_id_format_regexes"] = regexes

    # Expected tenant id (first tenant from config)
    tenants = raw.get("tenants") or {}
    tenant_id = None
    for t in tenants.values():
        if isinstance(t, dict) and t.get("id"):
            tenant_id = t["id"]
            break
    raw["_expected_tenant_id"] = tenant_id

    # Expected agent id (first agent from config)
    agents = raw.get("agents") or []
    agent_id = None
    if isinstance(agents, list) and agents:
        a = agents[0]
        if isinstance(a, dict) and a.get("id"):
            agent_id = a["id"]
    raw["_expected_agent_id"] = agent_id

    # tools_recommend defaults (config keys -> gentoro attribute names)
    tools_rec = raw.get("tools_recommend") or {}
    raw["_tools_recommend_expected"] = {
        "gentoro.step.outcome": tools_rec.get("step_outcome"),
        "gentoro.mcp.selection.strategy": tools_rec.get("selection_strategy"),
        "gentoro.mcp.selection.constraints": tools_rec.get("selection_constraints"),
        "gentoro.mcp.tools.selected.count": tools_rec.get("tools_selected_count"),
        "gentoro.mcp.selection.fallback.used": tools_rec.get("selection_fallback_used"),
    }

    return raw


def load_resource(resource_path: Path) -> dict[str, Any]:
    """Load resource.yaml for control_plane/data_plane attribute expectations (optional)."""
    return load_yaml(resource_path)


def load_scenarios(definitions_dir: Path) -> list[dict[str, Any]]:
    """Load all scenario YAML files from definitions dir and subdirs (skip _EXAMPLE_)."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required: pip install pyyaml")
    scenarios = []
    for path in sorted(definitions_dir.rglob("*.yaml")):
        if path.name.startswith("_EXAMPLE") or path.name.startswith("EXAMPLE"):
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and data.get("name"):
            data["_path"] = str(path)
            scenarios.append(data)
    return scenarios


def load_traces(traces_path: Path) -> dict[str, list[dict]]:
    """Load traces.jsonl and group spans by trace_id."""
    by_trace: dict[str, list[dict]] = defaultdict(list)
    with open(traces_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            span = json.loads(line)
            tid = span.get("trace_id")
            if tid:
                by_trace[tid].append(span)
    for tid in by_trace:
        by_trace[tid].sort(key=lambda s: (s.get("start_time", 0), s.get("name", "")))
    return dict(by_trace)


def get_attr(span: dict, key: str) -> Any:
    """Get attribute by key; try both gentoro.* and exact key."""
    attrs = span.get("attributes") or {}
    if key in attrs:
        return attrs[key]
    if not key.startswith("gentoro.") and _attr(key) in attrs:
        return attrs[_attr(key)]
    return None


def validate_span_attributes(span: dict, config: dict[str, Any]) -> list[str]:
    """
    Validate a single span's attributes against config (and optional resource/semconv).
    Returns list of human-readable error strings (Attribute/ID/Config: ...).
    """
    errors: list[str] = []
    attrs = span.get("attributes") or {}
    name = span.get("name") or ""

    if not config:
        return errors

    regexes = config.get("_id_format_regexes") or {}
    expected_tenant = config.get("_expected_tenant_id")
    expected_agent = config.get("_expected_agent_id")
    tools_rec_expected = config.get("_tools_recommend_expected") or {}

    # ID format validation (config id_formats)
    session_id = attrs.get("gentoro.session.id")
    if session_id and "session_id" in regexes and not regexes["session_id"].search(session_id):
        errors.append(f"ID: gentoro.session.id format invalid (expected from id_formats.session_id): {session_id!r}")

    conv_id = attrs.get("gen_ai.conversation.id")
    if conv_id and "conversation_id" in regexes and not regexes["conversation_id"].search(conv_id):
        errors.append(f"ID: gen_ai.conversation.id format invalid (expected from id_formats.conversation_id): {conv_id!r}")

    request_id = attrs.get("gentoro.request.id")
    if request_id and "request_id" in regexes and not regexes["request_id"].search(request_id):
        errors.append(f"ID: gentoro.request.id format invalid (expected from id_formats.request_id): {request_id!r}")

    mcp_call_id = attrs.get("gentoro.mcp.tool.call.id")
    if mcp_call_id and "mcp_tool_call_id" in regexes and not regexes["mcp_tool_call_id"].search(mcp_call_id):
        errors.append(f"ID: gentoro.mcp.tool.call.id format invalid (expected from id_formats.mcp_tool_call_id): {mcp_call_id!r}")

    enduser_id = attrs.get("gentoro.enduser.pseudo.id")
    if enduser_id and "enduser_pseudo_id" in regexes and not regexes["enduser_pseudo_id"].search(enduser_id):
        errors.append(f"ID: gentoro.enduser.pseudo.id format invalid (expected from id_formats.enduser_pseudo_id): {enduser_id!r}")

    # Session == conversation (config rule)
    if session_id and conv_id and session_id != conv_id:
        errors.append("Context: session.id and gen_ai.conversation.id must be equal (config)")

    # Tenant must match config
    if expected_tenant:
        tid = attrs.get("gentoro.tenant.id")
        if tid and tid != expected_tenant:
            errors.append(f"Config: gentoro.tenant.id must be {expected_tenant!r}; got {tid!r}")

    # a2a.orchestrate: agent target id must match config
    if expected_agent and ("a2a.orchestrate" in name or get_attr(span, "gentoro.span.class") == "a2a.orchestrate"):
        aid = attrs.get("gentoro.a2a.agent.target.id")
        if aid and aid != expected_agent:
            errors.append(f"Config: gentoro.a2a.agent.target.id must be {expected_agent!r}; got {aid!r}")

    # MCP server/tool UUIDs must be from config (we have TOOL_UUID_MAP / MCP_SERVER_UUID_TO_DIVISION)
    if "mcp.tool.execute" in name:
        server_uuid = attrs.get("gentoro.mcp.server.uuid")
        tool_uuid = attrs.get("gentoro.mcp.tool.uuid")
        if server_uuid and server_uuid not in MCP_SERVER_UUID_TO_DIVISION:
            errors.append(f"Config: gentoro.mcp.server.uuid not in config mcp_servers: {server_uuid!r}")
        if tool_uuid and tool_uuid not in TOOL_UUID_MAP:
            errors.append(f"Config: gentoro.mcp.tool.uuid not in config mcp_servers.tools: {tool_uuid!r}")

    # tools.recommend span: attributes must match config tools_recommend
    if "tools.recommend" in name or get_attr(span, "gentoro.span.class") == "tools.recommend":
        for attr_name, expected_val in tools_rec_expected.items():
            if expected_val is None:
                continue
            actual = attrs.get(attr_name)
            if actual is not None and actual != expected_val:
                errors.append(
                    f"Config: tools.recommend {attr_name} expected {expected_val!r} (from config tools_recommend); got {actual!r}"
                )

    # Optional: semconv required span_class for known span names
    span_class = attrs.get("gentoro.span.class")
    if name == "gentoro.request.validation" and span_class != "request.validation":
        errors.append(
            "Semconv: gentoro.request.validation span must have "
            f"gentoro.span.class=request.validation; got {span_class!r}"
        )
    if name == "gentoro.a2a.orchestrate" and span_class != "a2a.orchestrate":
        errors.append(f"Semconv: gentoro.a2a.orchestrate span must have gentoro.span.class=a2a.orchestrate; got {span_class!r}")
    if name == "gentoro.tools.recommend" and span_class != "tools.recommend":
        errors.append(f"Semconv: gentoro.tools.recommend span must have gentoro.span.class=tools.recommend; got {span_class!r}")
    if "gentoro.mcp.tool.execute" == name and "attempt" not in name and span_class != "mcp.tool.execute":
        errors.append(f"Semconv: gentoro.mcp.tool.execute span must have gentoro.span.class=mcp.tool.execute; got {span_class!r}")

    return errors


def analyze_trace(trace_id: str, spans: list[dict]) -> dict[str, Any]:
    """Extract structured summary of one trace for validation."""
    result: dict[str, Any] = {
        "trace_id": trace_id,
        "request_outcome": None,
        "block_reason": None,
        "roots": [],
        "mcp_tools": [],  # list of {division, tool_name, step_outcome, error_type, server_uuid, tool_uuid}
        "response_compose_outcome": None,
        "session_id": None,
        "conversation_id": None,
        "span_classes": set(),
        "augmentation_outcome": None,
        "policy_decision": None,
    }
    for span in spans:
        name = span.get("name") or ""
        attrs = span.get("attributes") or {}
        span_class = get_attr(span, "gentoro.span.class") or ""
        result["span_classes"].add(span_class or name)

        if "request.validation" in name or span_class == "request.validation":
            result["request_outcome"] = get_attr(span, "gentoro.request.outcome") or result["request_outcome"]
            result["block_reason"] = get_attr(span, "gentoro.block.reason") or result["block_reason"]
            if not span.get("parent_span_id"):
                result["roots"].append(name)

        if "augmentation.validation" in name or span_class == "augmentation.validation":
            result["augmentation_outcome"] = get_attr(span, "gentoro.step.outcome") or result["augmentation_outcome"]

        if "policy.validation" in span_class or "policy.validation" in name:
            result["policy_decision"] = get_attr(span, "gentoro.policy.decision") or result["policy_decision"]

        if "mcp.tool.execute" in name and "attempt" not in name:
            server_uuid = get_attr(span, "gentoro.mcp.server.uuid")
            tool_uuid = get_attr(span, "gentoro.mcp.tool.uuid")
            tool_name = get_attr(span, "gen_ai.tool.name")
            step_outcome = get_attr(span, "gentoro.step.outcome")
            error_type = get_attr(span, "error.type") or get_attr(span, "gentoro.error.type")
            division = MCP_SERVER_UUID_TO_DIVISION.get(server_uuid) if server_uuid else None
            if not division and tool_uuid:
                t = TOOL_UUID_MAP.get(tool_uuid)
                if t:
                    division = t[0]
            if not tool_name and tool_uuid:
                t = TOOL_UUID_MAP.get(tool_uuid)
                if t:
                    tool_name = t[1]
            result["mcp_tools"].append({
                "division": division,
                "tool_name": tool_name,
                "step_outcome": step_outcome,
                "error_type": error_type,
                "server_uuid": server_uuid,
                "tool_uuid": tool_uuid,
            })

        if "response.compose" in name or span_class == "response.compose":
            result["response_compose_outcome"] = get_attr(span, "gentoro.step.outcome")

        sid = get_attr(span, "gentoro.session.id")
        cid = get_attr(span, "gen_ai.conversation.id")
        if sid:
            result["session_id"] = sid
        if cid:
            result["conversation_id"] = cid

    result["span_classes"] = sorted(result["span_classes"])
    return result


def build_scenario_expectations(scenarios: list[dict]) -> dict[str, dict]:
    """Build a map of scenario name -> expected pattern for matching/validation."""
    expectations = {}
    for s in scenarios:
        name = s.get("name")
        if not name:
            continue
        data_plane = s.get("data_plane") or {}
        expected = s.get("expected") or {}
        overrides = s.get("scenario_overrides") or {}
        control_plane = s.get("control_plane") or {}

        goal = (data_plane.get("goal") or "").strip().lower() or "happy_path"
        workflow = data_plane.get("workflow") or ""
        exp_server = expected.get("mcp_server")
        exp_tools = expected.get("tools") or []
        wrong_division_target = overrides.get("wrong_division_target")
        exception_type = overrides.get("exception_type")
        actual_steps = overrides.get("actual_steps")

        expectations[name] = {
            "goal": goal,
            "workflow": workflow,
            "expected_mcp_server": exp_server,
            "expected_tools": exp_tools,
            "wrong_division_target": wrong_division_target,
            "exception_type": exception_type,
            "actual_steps": actual_steps,
            "control_plane_template": (control_plane.get("template") or data_plane.get("control_plane_template") or "").strip(),
            "is_control_only": "control_plane" in (s.get("tags") or []) and "data-plane" not in (s.get("tags") or []),
        }
    return expectations


def infer_scenario(analysis: dict, expectations: dict[str, dict]) -> str | None:
    """
    Infer the single best-matching scenario for this trace from division, tools, outcomes.
    Returns scenario name or None if no match.
    """
    tools = analysis["mcp_tools"]
    mcp_divisions = [t["division"] for t in tools if t["division"]]
    mcp_tool_names = [t["tool_name"] for t in tools if t["tool_name"]]
    tool_outcomes = [t["step_outcome"] for t in tools]
    any_fail = any(o == "fail" for o in tool_outcomes)
    compose_outcome = analysis.get("response_compose_outcome")
    request_outcome = analysis.get("request_outcome")
    block_reason = analysis.get("block_reason")
    error_types = [t.get("error_type") for t in tools if t.get("error_type")]

    # Partial workflow (no MCP calls) — check before control-plane so unified traces with orchestrate match
    if not tools and "a2a.orchestrate" in str(analysis.get("span_classes")):
        return "agent_confusion_new_claim_partial_tools"

    # Control-plane-only (no data-plane orchestrate, or explicit block/error)
    if not tools and (request_outcome or block_reason is not None):
        if block_reason == "invalid_context":
            return "request_blocked_invalid_context_augment_exception"
        if request_outcome == "error":
            return "request_error_policy_unavailable"  # or request_error_policy_runtime
        if block_reason == "request_policy":
            return "request_blocked_by_policy"
        if block_reason == "invalid_payload":
            return "request_blocked_invalid_payload"
        if block_reason == "rate_limited":
            return "request_blocked_rate_limited"
        return None

    # Data-plane: score by division + tools + goal
    div = mcp_divisions[0] if mcp_divisions else None
    first_tool = mcp_tool_names[0] if mcp_tool_names else None

    # Wrong-division (emit shows wrong division, MCP fail)
    if any_fail and div == "electronics" and first_tool == "pay":
        return "agent_confusion_pay_phone_wrong_division_electronics"
    if any_fail and div == "electronics" and first_tool == "cancel_claim":
        return "agent_confusion_cancel_claim_phone_wrong_division_electronics"

    # 4xx invalid params (one tool, fail, invalid_arguments or tool_error)
    if any_fail and any(et in ("invalid_arguments", "tool_error") for et in (error_types or [None])):
        if first_tool == "new_claim":
            if div == "electronics":
                return "new_claim_electronics_tool_4xx_invalid_params"
            if div == "appliances":
                return "new_claim_appliances_tool_4xx_invalid_params"
        if first_tool == "cancel_claim" and div == "appliances":
            return "cancel_claim_appliances_tool_4xx_invalid_params"
        if first_tool == "claim_status":
            if div == "phone":
                return "claim_status_phone_tool_4xx_invalid_params"
            if div == "appliances":
                return "claim_status_appliances_tool_4xx_invalid_params"

    # Wrong tool order: update_appointment then new_claim (phone)
    if div == "phone" and mcp_tool_names == ["update_appointment", "new_claim"]:
        return "agent_confusion_new_claim_wrong_tool_order"

    # Ungrounded / wrong tool: claim_status success + response_compose fail
    if div == "phone" and mcp_tool_names == ["claim_status"] and compose_outcome == "fail":
        return "agent_confusion_claim_status_ungrounded_summarization"
    if div == "electronics" and mcp_tool_names == ["cancel_product"] and compose_outcome == "fail":
        return "agent_confusion_cancel_product_instead_of_claim"

    # Happy path: single tool success
    if not any_fail and len(tools) == 1:
        if first_tool == "new_claim" and div == "phone":
            return "new_claim_phone"
        if first_tool == "new_claim" and div == "appliances":
            return "new_claim_appliances_higher_latency_peak_hours"
        if first_tool == "cancel_claim" and div == "appliances":
            return "cancel_claim_appliances"
        if first_tool == "update_appointment" and div == "phone":
            return "update_appointment_phone_higher_latency"
        if first_tool == "update_appointment" and div == "electronics":
            return "update_appointment_electronics_higher_latency"
        if first_tool == "claim_status" and div == "phone":
            return "claim_status_phone_higher_latency"
        if first_tool == "claim_status" and div == "appliances":
            return "claim_status_appliances"
        # Generic: scenario name "{tool}_{division}" (e.g. cancel_claim_phone, upload_documents_appliances)
        candidate = f"{first_tool}_{div}"
        if candidate in expectations:
            return candidate

    # Multi-tool success (e.g. wrong order already handled)
    if not any_fail and len(tools) == 2 and div == "phone":
        if set(mcp_tool_names) == {"update_appointment", "new_claim"}:
            return "agent_confusion_new_claim_wrong_tool_order"

    # Fallback by division + first tool
    if first_tool == "new_claim" and div == "phone":
        return "new_claim_phone"
    if first_tool == "cancel_claim" and div == "appliances" and not any_fail:
        return "cancel_claim_appliances"
    if first_tool == "update_appointment":
        if div == "phone":
            return "update_appointment_phone_higher_latency"
        if div == "electronics":
            return "update_appointment_electronics_higher_latency"
    if first_tool == "claim_status" and div == "phone":
        return "claim_status_phone_higher_latency"
    return None


def _validate_control_plane_trace(analysis: dict, scenario_name: str, exp: dict) -> list[str]:
    """Validate control-plane-only trace."""
    errs = []
    template = exp.get("control_plane_template") or ""
    request_outcome = analysis.get("request_outcome")
    block_reason = analysis.get("block_reason")
    if template == "blocked_invalid_context_augment_exception":
        if request_outcome != "blocked":
            errs.append(f"Goal: expected request.outcome=blocked; got {request_outcome}")
        if block_reason != "invalid_context":
            errs.append(f"Attribute: expected block.reason=invalid_context; got {block_reason}")
    elif template == "error_policy_runtime" or "error_policy" in template:
        if request_outcome != "error":
            errs.append(f"Goal: expected request.outcome=error; got {request_outcome}")
    return errs


def validate_trace_against_scenario(analysis: dict, scenario_name: str, expectations: dict[str, dict]) -> list[str]:
    """Validate trace against a single scenario; return list of error messages."""
    exp = expectations.get(scenario_name)
    if not exp:
        return [f"Unknown scenario: {scenario_name}"]
    if exp.get("is_control_only"):
        return _validate_control_plane_trace(analysis, scenario_name, exp)
    tools = analysis["mcp_tools"]
    mcp_divisions = [t["division"] for t in tools if t["division"]]
    actual_tool_names = [t["tool_name"] for t in tools if t["tool_name"]]
    any_fail = any(t["step_outcome"] == "fail" for t in tools)
    errs = []

    goal = exp.get("goal")
    exp_server = exp.get("expected_mcp_server")
    exp_tools = exp.get("expected_tools") or []
    wrong_target = exp.get("wrong_division_target")

    # Division
    if exp_server and mcp_divisions:
        if wrong_target:
            if mcp_divisions[0] != wrong_target:
                errs.append(f"Semantic: wrong_division should show division={wrong_target}; got {mcp_divisions}")
        else:
            if mcp_divisions[0] != exp_server:
                errs.append(f"Semantic: expected division {exp_server}; got {mcp_divisions}")

    # Tools (order for wrong_order scenario; partial_tools has no MCP calls)
    if exp_tools and tools:
        if goal == "partial_workflow" and exp.get("actual_steps"):
            if ["update_appointment", "new_claim"] in (exp.get("actual_steps"), list(exp.get("actual_steps") or [])):
                if actual_tool_names != ["update_appointment", "new_claim"]:
                    errs.append(f"Contextual: expected tool order [update_appointment, new_claim]; got {actual_tool_names}")
        elif not wrong_target and set(actual_tool_names) != set(exp_tools):
            if not (goal == "partial_workflow" and set(actual_tool_names) == set(exp_tools)):
                # partial_tools scenario: expected [new_claim] but actual [] is correct
                if goal == "partial_workflow" and not actual_tool_names:
                    pass
                else:
                    errs.append(f"Semantic: expected tools {exp_tools}; got {actual_tool_names}")

    # Goal vs outcome
    if goal == "happy_path":
        if any_fail:
            errs.append("Goal: happy_path expects all MCP step.outcome=success")
        if analysis.get("response_compose_outcome") not in (None, "success") and tools:
            errs.append("Goal: happy_path expects response_compose step.outcome=success")
    elif goal == "wrong_division":
        if not any_fail and tools:
            errs.append("Goal: wrong_division expects MCP step.outcome=fail")
        if tools and not any(t.get("error_type") for t in tools):
            errs.append("Attribute: wrong_division should have error.type on failed MCP span")
    elif goal == "4xx_invalid_arguments":
        if not any_fail and tools:
            errs.append("Goal: 4xx_invalid_arguments expects MCP step.outcome=fail")
        ets = [t.get("error_type") for t in tools if t.get("error_type")]
        if tools and not any(et in ("invalid_arguments", "tool_error") for et in (ets or [None])):
            errs.append("Attribute: 4xx should have error.type=invalid_arguments or tool_error")
    elif goal == "ungrounded_response":
        if analysis.get("response_compose_outcome") != "fail":
            errs.append("Goal: ungrounded_response expects response_compose step.outcome=fail")
    elif goal == "partial_workflow":
        if analysis.get("response_compose_outcome") != "fail" and analysis.get("response_compose_outcome") is not None:
            errs.append("Goal: partial_workflow expects response_compose step.outcome=fail")

    # Context: session_id == conversation_id
    sid = analysis.get("session_id")
    cid = analysis.get("conversation_id")
    if sid and cid and sid != cid:
        errs.append(f"Contextual: session_id and gen_ai.conversation.id must match")

    return errs


def match_trace_to_scenario(analysis: dict, expectations: dict[str, dict]) -> tuple[str | None, list[str]]:
    """
    Infer best scenario for trace, validate, return (scenario_name, list of errors).
    """
    scenario = infer_scenario(analysis, expectations)
    if scenario is None:
        return None, ["No matching scenario inferred"]
    errs = validate_trace_against_scenario(analysis, scenario, expectations)
    return scenario, errs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze traces vs scenario definitions")
    parser.add_argument("--traces", default="traces.jsonl", help="Path to traces.jsonl")
    parser.add_argument("--scenarios", default="resource/scenarios/definitions", help="Path to scenario definitions dir")
    parser.add_argument("--config", default="resource/config/config.yaml", help="Path to config (for ID formats, tenant, agent, tools_recommend)")
    parser.add_argument("--resource", default="", help="Path to resource.yaml (optional; for resource attribute checks)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-trace details")
    parser.add_argument("--no-attributes", action="store_true", help="Skip attribute/config/semconv compliance checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    traces_path = root / args.traces
    scenarios_dir = root / args.scenarios
    config_path = root / args.config
    resource_path = root / args.resource if args.resource else root / "resource/config/resource.yaml"

    if not traces_path.exists():
        raise SystemExit(f"Traces file not found: {traces_path}")
    if not scenarios_dir.is_dir():
        raise SystemExit(f"Scenarios dir not found: {scenarios_dir}")

    scenarios = load_scenarios(scenarios_dir)
    expectations = build_scenario_expectations(scenarios)
    traces = load_traces(traces_path)

    config: dict[str, Any] = {}
    if config_path.exists():
        config = load_config(config_path)
    resource_cfg: dict[str, Any] = {}
    if resource_path.exists():
        resource_cfg = load_resource(resource_path)

    print(f"Loaded {len(scenarios)} scenarios, {len(traces)} traces")
    if config:
        print(f"Config: {config_path.name} (ID formats, tenant, agent, tools_recommend)")
    if resource_cfg:
        print(f"Resource: {resource_path.name}")
    print()

    all_errors: list[tuple[str, str, list[str]]] = []  # trace_id, scenario, errors
    trace_to_best: dict[str, tuple[str | None, list[str]]] = {}  # trace_id -> (scenario, errors)

    for trace_id, spans in traces.items():
        analysis = analyze_trace(trace_id, spans)
        scenario, errs = match_trace_to_scenario(analysis, expectations)
        trace_to_best[trace_id] = (scenario, errs)
        if errs:
            for err in errs:
                all_errors.append((trace_id, scenario or "?", err))  # single err string per tuple

    # Attribute/config/semconv compliance (per-span checks using config)
    attribute_compliance_errors: list[tuple[str, str, str]] = []  # (trace_id, span_name, error)
    if config and not args.no_attributes:
        for trace_id, spans in traces.items():
            for span in spans:
                name = span.get("name") or ""
                for err in validate_span_attributes(span, config):
                    attribute_compliance_errors.append((trace_id, name, err))

    # Group errors by type (each item is (trace_id, scenario, single_err))
    semantic = []
    goal = []
    attribute = []
    contextual = []
    for trace_id, scenario, e in all_errors:
        if e.startswith("Semantic:"):
            semantic.append((trace_id, scenario, e))
        elif e.startswith("Goal:"):
            goal.append((trace_id, scenario, e))
        elif e.startswith("Attribute:"):
            attribute.append((trace_id, scenario, e))
        elif e.startswith("Contextual:"):
            contextual.append((trace_id, scenario, e))
        else:
            semantic.append((trace_id, scenario, e))

    # Summary
    print("=== Summary ===")
    print(f"Traces with at least one error: {len(set(t for t, _, _ in all_errors))} / {len(traces)}")
    print(f"Semantic errors: {len(semantic)}")
    print(f"Goal errors: {len(goal)}")
    print(f"Attribute errors: {len(attribute)}")
    print(f"Contextual errors: {len(contextual)}")
    print()

    if semantic:
        print("=== Semantic / division / tool errors ===")
        for trace_id, scenario, msg in semantic[:30]:
            print(f"  [{trace_id[:8]}...] {scenario}: {msg}")
        if len(semantic) > 30:
            print(f"  ... and {len(semantic) - 30} more")
        print()
    if goal:
        print("=== Goal / outcome errors ===")
        for trace_id, scenario, msg in goal[:30]:
            print(f"  [{trace_id[:8]}...] {scenario}: {msg}")
        if len(goal) > 30:
            print(f"  ... and {len(goal) - 30} more")
        print()
    if attribute:
        print("=== Attribute errors ===")
        for trace_id, scenario, msg in attribute[:20]:
            print(f"  [{trace_id[:8]}...] {scenario}: {msg}")
        if len(attribute) > 20:
            print(f"  ... and {len(attribute) - 20} more")
        print()
    if contextual:
        print("=== Contextual errors ===")
        for trace_id, scenario, msg in contextual[:20]:
            print(f"  [{trace_id[:8]}...] {scenario}: {msg}")
        if len(contextual) > 20:
            print(f"  ... and {len(contextual) - 20} more")
        print()

    # Attribute/config/semconv compliance summary
    if attribute_compliance_errors:
        print("=== Attribute / config / semconv compliance ===")
        print(f"Spans with attribute or ID or config violations: {len(attribute_compliance_errors)}")
        by_kind: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for trace_id, span_name, err in attribute_compliance_errors:
            if err.startswith("ID:"):
                by_kind["ID format"].append((trace_id, span_name, err))
            elif err.startswith("Config:"):
                by_kind["Config"].append((trace_id, span_name, err))
            elif err.startswith("Context:"):
                by_kind["Context"].append((trace_id, span_name, err))
            elif err.startswith("Semconv:"):
                by_kind["Semconv"].append((trace_id, span_name, err))
            else:
                by_kind["Other"].append((trace_id, span_name, err))
        for kind in ["ID format", "Config", "Context", "Semconv", "Other"]:
            items = by_kind.get(kind, [])
            if not items:
                continue
            print(f"  {kind}: {len(items)}")
            for trace_id, span_name, msg in items[:15]:
                print(f"    [{trace_id[:8]}...] {span_name}: {msg}")
            if len(items) > 15:
                print(f"    ... and {len(items) - 15} more")
        print()
    elif config and not args.no_attributes:
        print("=== Attribute / config / semconv compliance ===")
        print("No attribute, ID format, or config violations found.")
        print()

    if args.verbose:
        print("=== Per-trace inferred scenario (first 15) ===")
        for trace_id, (scenario, errs) in list(trace_to_best.items())[:15]:
            status = "OK" if not errs else f"ERR({len(errs)})"
            print(f"  {trace_id[:12]}... -> {scenario or '?'} [{status}]")
            for e in errs[:5]:
                print(f"      {e}")


if __name__ == "__main__":
    main()
