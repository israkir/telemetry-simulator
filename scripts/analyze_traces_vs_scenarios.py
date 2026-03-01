#!/usr/bin/env python3
"""
Analyze generated traces (traces.jsonl) against scenario definitions for semantic,
goal, attribute, and contextual errors.

Usage:
  python scripts/analyze_traces_vs_scenarios.py [--traces traces.jsonl] [--scenarios resource/scenarios/definitions] [--config resource/config/config.yaml]
"""

from __future__ import annotations

import argparse
import json
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


def load_scenarios(definitions_dir: Path) -> list[dict[str, Any]]:
    """Load all scenario YAML files (skip _EXAMPLE_)."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required: pip install pyyaml")
    scenarios = []
    for path in sorted(definitions_dir.glob("*.yaml")):
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

        if "augmentation" in name or span_class == "augmentation":
            result["augmentation_outcome"] = get_attr(span, "gentoro.step.outcome") or result["augmentation_outcome"]

        if "validation.policy" in span_class or "validation.policy" in name:
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
            return "claim_status_appliances_tool_4xx_invalid_params"

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
    parser.add_argument("--config", default="resource/config/config.yaml", help="Path to config (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-trace details")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    traces_path = root / args.traces
    scenarios_dir = root / args.scenarios

    if not traces_path.exists():
        raise SystemExit(f"Traces file not found: {traces_path}")
    if not scenarios_dir.is_dir():
        raise SystemExit(f"Scenarios dir not found: {scenarios_dir}")

    scenarios = load_scenarios(scenarios_dir)
    expectations = build_scenario_expectations(scenarios)
    traces = load_traces(traces_path)

    print(f"Loaded {len(scenarios)} scenarios, {len(traces)} traces\n")

    all_errors: list[tuple[str, str, list[str]]] = []  # trace_id, scenario, errors
    trace_to_best: dict[str, tuple[str | None, list[str]]] = {}  # trace_id -> (scenario, errors)

    for trace_id, spans in traces.items():
        analysis = analyze_trace(trace_id, spans)
        scenario, errs = match_trace_to_scenario(analysis, expectations)
        trace_to_best[trace_id] = (scenario, errs)
        if errs:
            for err in errs:
                all_errors.append((trace_id, scenario or "?", err))  # single err string per tuple

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

    if args.verbose:
        print("=== Per-trace inferred scenario (first 15) ===")
        for trace_id, (scenario, errs) in list(trace_to_best.items())[:15]:
            status = "OK" if not errs else f"ERR({len(errs)})"
            print(f"  {trace_id[:12]}... -> {scenario or '?'} [{status}]")
            for e in errs[:5]:
                print(f"      {e}")


if __name__ == "__main__":
    main()
