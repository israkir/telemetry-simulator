from datetime import datetime
from zoneinfo import ZoneInfo

from simulator.scenarios.latency import LatencyModel
from simulator.scenarios.scenario_loader import Scenario


def _scenario_with_peak_window() -> Scenario:
    return Scenario(
        name="latency_peak_window",
        latency_profiles={"normal": {}, "peak_hours": {}, "special": {}},
        default_latency_profile="normal",
        raw={
            "workload_schedule": {
                "peak_hours": {
                    "timezone": "America/Los_Angeles",
                    "weekdays": [1, 2, 3, 4, 5],
                    "start_hour": 9,
                    "end_hour": 12,
                }
            }
        },
    )


def test_select_profile_uses_peak_hours_when_window_active() -> None:
    scenario = _scenario_with_peak_window()
    model = LatencyModel.from_scenario(scenario)
    simulated = datetime(2026, 4, 6, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))  # Monday

    profile, higher = model.select_profile(scenario, simulated)

    assert profile == "peak_hours"
    assert higher is None


def test_select_profile_uses_default_when_outside_peak_window() -> None:
    scenario = _scenario_with_peak_window()
    model = LatencyModel.from_scenario(scenario)
    simulated = datetime(2026, 4, 6, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles"))  # Monday

    profile, higher = model.select_profile(scenario, simulated)

    assert profile == "normal"
    assert higher is None


def test_select_profile_keeps_explicit_condition_precedence() -> None:
    scenario = _scenario_with_peak_window()
    scenario.latency_profile_conditions = [{"profile": "special"}]
    model = LatencyModel.from_scenario(scenario)
    simulated = datetime(2026, 4, 6, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))  # Monday

    profile, higher = model.select_profile(scenario, simulated)

    assert profile == "special"
    assert higher == {}
