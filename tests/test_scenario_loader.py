"""Tests for scenario loading and sample definitions."""

import pytest

from simulator.scenarios import SAMPLE_DEFINITIONS_DIR, ScenarioLoader
from simulator.scenarios.scenario_loader import EXAMPLE_SCENARIO_NAME


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
    assert len(names) >= 1  # e.g. new_claim_phone


def test_loader_load_all_excludes_example_when_sample_dir() -> None:
    """When using sample definitions, example_scenario is excluded from load_all."""
    loader = ScenarioLoader()
    scenarios = loader.load_all()
    names = [s.name for s in scenarios]
    assert EXAMPLE_SCENARIO_NAME not in names


def test_loader_can_load_example_scenario_explicitly() -> None:
    """example_scenario can still be loaded by name when using sample dir."""
    loader = ScenarioLoader()
    scenario = loader.load(EXAMPLE_SCENARIO_NAME)
    assert scenario.name == EXAMPLE_SCENARIO_NAME


def test_loader_can_load_new_claim_phone() -> None:
    """Scenario new_claim_phone (file) loads and has expected structure (correct_flow, mcp_server)."""
    loader = ScenarioLoader()
    scenario = loader.load("new_claim_phone")
    assert scenario.name == "new_claim_phone"
    assert scenario.repeat_count >= 1
    assert scenario.context is not None
    assert scenario.context.correct_flow is not None
    assert scenario.context.correct_flow.steps
    assert scenario.mcp_server == "phone"
    assert scenario.goal == "happy_path"
