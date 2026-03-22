"""Tests for scenario loading and sample definitions."""

import textwrap
from pathlib import Path

import pytest

from simulator.config import load_config
from simulator.scenarios import SAMPLE_DEFINITIONS_DIR, ScenarioLoader
from simulator.scenarios.scenario_loader import EXAMPLE_SCENARIO_NAME, load_scenario_yaml
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
