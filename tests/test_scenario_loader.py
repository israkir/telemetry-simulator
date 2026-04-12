"""Tests for scenario loading and sample definitions."""

import textwrap
from pathlib import Path

import pytest

from simulator.config import load_config
from simulator.scenarios import SAMPLE_DEFINITIONS_DIR, ScenarioLoader
from simulator.scenarios.scenario_loader import (
    EXAMPLE_SCENARIO_NAME,
    _parse_tool_chain_for_turn,
    load_scenario_yaml,
)
from simulator.scenarios.tool_payload_validation import validate_scenario_tool_payloads


def test_sample_definitions_dir_exists() -> None:
    """Bundled sample definitions directory exists."""
    assert SAMPLE_DEFINITIONS_DIR.is_dir()


def test_loader_default_uses_sample_definitions() -> None:
    """ScenarioLoader with no path uses sample definitions dir."""
    loader = ScenarioLoader()
    assert loader.scenarios_dir == SAMPLE_DEFINITIONS_DIR


def test_loader_list_scenarios_excludes_example_when_sample_dir() -> None:
    """When using sample definitions, example_scenario is excluded from list."""
    loader = ScenarioLoader()
    names = loader.list_scenarios()
    assert EXAMPLE_SCENARIO_NAME not in names
    assert len(names) >= 1  # e.g. single_tool_call


def test_loader_load_all_excludes_example_when_sample_dir() -> None:
    """When using sample definitions, example_scenario is excluded from load_all."""
    loader = ScenarioLoader()
    scenarios = loader.load_all()
    names = [s.name for s in scenarios]
    assert EXAMPLE_SCENARIO_NAME not in names


def test_loader_can_load_single_tool_call() -> None:
    """Scenario single_tool_call loads and has expected structure (correct_flow, mcp_server)."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    assert scenario.name == "single_tool_call"
    assert scenario.repeat_count >= 1
    assert scenario.context is not None
    assert scenario.context.correct_flow is not None
    assert scenario.context.correct_flow.steps


def test_scenarios_do_not_depend_on_config_conversation_samples() -> None:
    """Scenarios define their own conversation via endusers/turns; conversation_samples is derived."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    # Scenario with endusers/turns has conversation_samples derived from them.
    assert getattr(scenario, "conversation_samples", None) is not None
    assert len(scenario.conversation_samples) >= 1


def test_multi_tool_chain_catalog_per_enduser_mcp_server() -> None:
    """Endusers may set `mcp_server` to switch MCP UUIDs within one scenario file."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    keys = {eu.mcp_server_key for eu in scenario.endusers}
    assert keys == {"phone", "electronics", "appliances"}


def test_single_tool_call_tool_payloads_validate_against_config() -> None:
    """Bundled scenario tool args/result match mcp_tool_genai_payloads structures."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    validate_scenario_tool_payloads(scenario, load_config())


def test_parse_tool_chain_rich_mcp_per_step() -> None:
    """Rich tool_chain maps expand to parallel tool names and mcp_server keys."""
    data = {
        "tool_chain": [
            {"mcp_server": "phone", "tool": "new_claim"},
            {"mcp_server": "electronics", "name": "get_available_slots"},
        ],
    }
    tools, servers = _parse_tool_chain_for_turn(data)
    assert tools == ["new_claim", "get_available_slots"]
    assert servers == ["phone", "electronics"]


def test_parse_tool_chain_legacy_parallel_servers() -> None:
    """Legacy string tool_chain plus tool_chain_mcp_servers is preserved."""
    data = {
        "tool_chain": ["new_claim", "pay"],
        "tool_chain_mcp_servers": ["phone", "phone"],
    }
    tools, servers = _parse_tool_chain_for_turn(data)
    assert tools == ["new_claim", "pay"]
    assert servers == ["phone", "phone"]


def test_rich_tool_chain_scenario_loads(tmp_path: Path) -> None:
    """YAML with embedded mcp_server per tool_chain step loads and validates."""
    yml = textwrap.dedent(
        """\
        name: rich_chain
        tenant: toro
        agent: toro-customer-assistant-001
        mcp_server: phone
        endusers:
          - id: u1
            turns:
              - turn_index: 1
                tool_chain:
                  - mcp_server: phone
                    tool: new_claim
                  - mcp_server: electronics
                    tool: get_available_slots
                gen_ai.tool.call.arguments:
                  new_claim:
                    product_type: phone
                    description: test
                    incident_date: '2026-01-01'
                  get_available_slots:
                    claim_id: CLM-1
                    service_type: repair_collection
                    date_range_start: '2026-01-01'
                    date_range_end: '2026-01-31'
                gen_ai.tool.call.result:
                  new_claim:
                    claim_id: CLM-1
                    product_type: phone
                    status: opened
                    coverage:
                      type: device_damage
                      deductible: 75
                  get_available_slots:
                    claim_id: CLM-1
                    slots: []
                    timezone: America/Los_Angeles
        """
    )
    path = tmp_path / "rich.yaml"
    path.write_text(yml, encoding="utf-8")
    scenario = load_scenario_yaml(path, tmp_path)
    turn = scenario.endusers[0].turns[0]
    assert turn.tool_chain == ["new_claim", "get_available_slots"]
    assert turn.tool_chain_mcp_servers == ["phone", "electronics"]


def test_tool_payload_validation_rejects_unknown_argument_field(tmp_path: Path) -> None:
    """Turn-level gen_ai.tool.call.arguments must not use keys absent from config structure."""
    bad = textwrap.dedent(
        """\
        name: bad_tool_fields
        endusers:
          - id: u1
            turns:
              - turn_index: 1
                tool_chain: [new_claim]
                gen_ai.tool.call.arguments:
                  new_claim:
                    product_type: phone
                    not_in_config: "x"
        """
    )
    path = tmp_path / "bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field"):
        load_scenario_yaml(path, tmp_path)
