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
        return (self.default_profile, None)

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
