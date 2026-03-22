"""
Validate scenario gen_ai.tool.call.arguments / .result against config.yaml.

resource/config/config.yaml mcp_tool_genai_payloads defines per-tool structure
(field catalog). Scenario overrides must use only those field names; every
tool in tool_chain must have a payload contract in config.
"""

from __future__ import annotations

from typing import Any

from .scenario_loader import Scenario, ScenarioTurn


def _structure_keys(structure: Any) -> set[str]:
    if not isinstance(structure, dict):
        return set()
    return {str(k) for k in structure.keys()}


def _validate_one_blob(
    *,
    scenario_name: str,
    enduser_id: str,
    turn_index: int,
    tool_name: str,
    blob_kind: str,
    payload: dict[str, Any],
    allowed: set[str],
) -> list[str]:
    errs: list[str] = []
    unknown = set(payload.keys()) - allowed
    if unknown:
        errs.append(
            f"{scenario_name} enduser={enduser_id!r} turn={turn_index} "
            f"tool={tool_name!r} {blob_kind}: unknown field(s) {sorted(unknown)!r}; "
            f"allowed per resource/config/config.yaml structure: {sorted(allowed)!r}"
        )
    return errs


def validate_turn_tool_payloads(
    *,
    scenario_name: str,
    enduser_id: str,
    turn: ScenarioTurn,
    payloads: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    chain = list(turn.tool_chain or [])
    chain_set = set(chain)

    args_blob = turn.gen_ai_tool_call_arguments
    res_blob = turn.gen_ai_tool_call_result
    if not isinstance(args_blob, dict):
        args_blob = {}
    if not isinstance(res_blob, dict):
        res_blob = {}

    for t in chain:
        if t not in payloads:
            errors.append(
                f"{scenario_name} enduser={enduser_id!r} turn={turn.turn_index}: "
                f"tool_chain includes {t!r} but resource/config/config.yaml "
                f"`mcp_tool_genai_payloads` has no entry for that tool."
            )

    for tool_name in args_blob:
        if tool_name not in chain_set:
            errors.append(
                f"{scenario_name} enduser={enduser_id!r} turn={turn.turn_index}: "
                f"gen_ai.tool.call.arguments defines {tool_name!r} "
                f"but it is not in tool_chain {chain!r}."
            )

    for tool_name in res_blob:
        if tool_name not in chain_set:
            errors.append(
                f"{scenario_name} enduser={enduser_id!r} turn={turn.turn_index}: "
                f"gen_ai.tool.call.result defines {tool_name!r} "
                f"but it is not in tool_chain {chain!r}."
            )

    for tool_name, obj in args_blob.items():
        if not isinstance(obj, dict):
            errors.append(
                f"{scenario_name} enduser={enduser_id!r} turn={turn.turn_index}: "
                f"gen_ai.tool.call.arguments.{tool_name} must be a mapping, not {type(obj).__name__}."
            )
            continue
        tool_def = payloads.get(tool_name)
        if not isinstance(tool_def, dict):
            continue
        arg_sem = tool_def.get("gen_ai.tool.call.arguments")
        if not isinstance(arg_sem, dict):
            continue
        allowed = _structure_keys(arg_sem.get("structure"))
        if not allowed:
            continue
        errors.extend(
            _validate_one_blob(
                scenario_name=scenario_name,
                enduser_id=enduser_id,
                turn_index=turn.turn_index,
                tool_name=tool_name,
                blob_kind="gen_ai.tool.call.arguments",
                payload=obj,
                allowed=allowed,
            )
        )

    for tool_name, obj in res_blob.items():
        if not isinstance(obj, dict):
            errors.append(
                f"{scenario_name} enduser={enduser_id!r} turn={turn.turn_index}: "
                f"gen_ai.tool.call.result.{tool_name} must be a mapping, not {type(obj).__name__}."
            )
            continue
        tool_def = payloads.get(tool_name)
        if not isinstance(tool_def, dict):
            continue
        res_sem = tool_def.get("gen_ai.tool.call.result")
        if not isinstance(res_sem, dict):
            continue
        allowed = _structure_keys(res_sem.get("structure"))
        if not allowed:
            continue
        errors.extend(
            _validate_one_blob(
                scenario_name=scenario_name,
                enduser_id=enduser_id,
                turn_index=turn.turn_index,
                tool_name=tool_name,
                blob_kind="gen_ai.tool.call.result",
                payload=obj,
                allowed=allowed,
            )
        )

    return errors


def validate_scenario_tool_payloads(scenario: Scenario, config: dict[str, Any]) -> None:
    """
    Raise ValueError if any turn's tool payloads violate mcp_tool_genai_payloads structures.
    """
    payloads = config.get("mcp_tool_genai_payloads") or {}
    if not isinstance(payloads, dict):
        payloads = {}
    all_errs: list[str] = []
    for eu in scenario.endusers:
        eid = eu.id or eu.enduser_pseudo_id or ""
        for turn in eu.turns:
            all_errs.extend(
                validate_turn_tool_payloads(
                    scenario_name=scenario.name,
                    enduser_id=eid,
                    turn=turn,
                    payloads=payloads,
                )
            )
    if all_errs:
        msg = (
            "Scenario tool payload(s) do not comply with resource/config/config.yaml:\n"
            + "\n".join(f"  - {e}" for e in all_errs)
        )
        raise ValueError(msg)
