"""
Centralized latency model: profile selection and span duration sampling.

Preserves lognormal-ish behavior (mean + deviation, gaussian sampling, non-negative).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .scenario_loader import Scenario


class LatencyModel:
    """Select latency profile and sample span durations from scenario latency_profiles."""

    def __init__(
        self,
        latency_profiles: Mapping[str, Any],
        default_profile: str = "normal",
    ) -> None:
        self.latency_profiles = dict(latency_profiles) if latency_profiles else {}
        self.default_profile = default_profile or "normal"

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> LatencyModel:
        return cls(
            latency_profiles=scenario.latency_profiles or {},
            default_profile=scenario.default_latency_profile or "normal",
        )

    def select_profile(
        self,
        scenario: Scenario,
        _simulated_clock: Any = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Select latency profile from scenario.latency_profile_conditions.
        First condition whose 'profile' exists in latency_profiles wins.
        Returns (profile_name, higher_latency_attributes or None).
        """
        conditions = scenario.latency_profile_conditions or []
        for cond in conditions:
            profile = cond.get("profile")
            if profile and profile in self.latency_profiles:
                higher = cond.get("higher_latency_attributes") or {}
                return (profile, higher)
        if self._peak_hours_window_active(scenario, _simulated_clock):
            return ("peak_hours", None)
        return (self.default_profile, None)

    def _peak_hours_window_active(self, scenario: Scenario, simulated_clock: Any = None) -> bool:
        """
        Return True when scenario workload_schedule peak window is active and
        the scenario has a peak_hours latency profile.
        """
        if "peak_hours" not in self.latency_profiles:
            return False
        raw = scenario.raw if isinstance(scenario.raw, dict) else {}
        schedule = raw.get("workload_schedule")
        if not isinstance(schedule, dict):
            return False
        peak = schedule.get("peak_hours")
        if not isinstance(peak, dict):
            return False

        tz_name = str(peak.get("timezone") or "UTC")
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            return False

        now_utc = self._coerce_clock_to_utc(simulated_clock)
        now_local = now_utc.astimezone(tz)

        weekdays_raw = peak.get("weekdays")
        if isinstance(weekdays_raw, list) and weekdays_raw:
            weekdays: list[int] = []
            for item in weekdays_raw:
                try:
                    weekdays.append(int(item))
                except (TypeError, ValueError):
                    continue
            if weekdays and now_local.isoweekday() not in weekdays:
                return False

        start_hour_raw = peak.get("start_hour")
        end_hour_raw = peak.get("end_hour")
        if start_hour_raw is None or end_hour_raw is None:
            return True
        try:
            start_hour = int(start_hour_raw)
            end_hour = int(end_hour_raw)
        except (TypeError, ValueError):
            return True
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            return False

        hour = now_local.hour
        if start_hour == end_hour:
            return True
        if start_hour < end_hour:
            return start_hour <= hour < end_hour
        return hour >= start_hour or hour < end_hour

    def _coerce_clock_to_utc(self, simulated_clock: Any = None) -> datetime:
        """Normalize optional simulated clock into an aware UTC datetime."""
        if isinstance(simulated_clock, datetime):
            return (
                simulated_clock.replace(tzinfo=UTC)
                if simulated_clock.tzinfo is None
                else simulated_clock.astimezone(UTC)
            )
        if isinstance(simulated_clock, (int, float)):
            ts = float(simulated_clock)
            # Heuristic support for ns and ms epoch inputs.
            if abs(ts) > 1e14:
                ts = ts / 1_000_000_000.0
            elif abs(ts) > 1e11:
                ts = ts / 1_000.0
            return datetime.fromtimestamp(ts, tz=UTC)
        return datetime.now(UTC)

    def _sample_duration(self, mean: float, deviation: float) -> int:
        d = mean + (random.gauss(0, 1) * deviation)
        return max(0, int(d))

    def duration_for_span(self, profile_name: str, span_class: str) -> int:
        """Sample duration in ms for a span class from the given profile."""
        profile = self.latency_profiles.get(profile_name) or {}
        span_durations = profile.get("span_durations_ms") or profile.get("span_durations") or {}
        spec = span_durations.get(span_class)
        if spec is None:
            return 50
        if isinstance(spec, (int, float)):
            return int(spec)
        mean = float(spec.get("mean", 50))
        deviation = float(spec.get("deviation", 10))
        return self._sample_duration(mean, deviation)
