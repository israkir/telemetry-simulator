"""
Typed intermediate representation for scenario-driven simulation.

ScenarioSpec / EnduserSpec / TurnSpec are aliases to the loader types.
CompiledTurnSpec and TraceGraphSpec are the compiled output consumed by the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..generators.trace_generator import TraceSpec
from .scenario_loader import Scenario, ScenarioEnduser, ScenarioTurn

# Scenario-level IR: use loader types as source of truth.
ScenarioSpec = Scenario
EnduserSpec = ScenarioEnduser
TurnSpec = ScenarioTurn


@dataclass
class CompiledTurnSpec:
    """Compiled turn: IDs, context, tool chain, and latency params for one logical request."""

    scenario_name: str
    enduser_id: str
    turn_index: int

    tenant_id: str
    agent_id: str
    mcp_server_uuid: str

    tool_chain: list[str]

    session_id: str
    conversation_id: str
    request_id: str
    redaction_applied: str
    mcp_tool_call_ids: list[str]

    latency_profile_name: str
    higher_latency_attributes: dict[str, Any]


@dataclass
class TraceGraphSpec:
    """Three traces per logical request: CP request, data-plane, CP response."""

    cp_request: TraceSpec
    data_plane: TraceSpec
    cp_response: TraceSpec
    compiled_turn: CompiledTurnSpec
