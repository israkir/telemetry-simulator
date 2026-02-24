"""
Load and parse YAML-based scenario definitions.

Scenarios define reproducible telemetry generation patterns for:
- Testing specific failure modes
- Load testing with realistic distributions
- Dashboard visualization testing
- Pipeline validation

Supports both deterministic scenarios (fixed structure) and statistical
scenarios (probabilistic branching, distributions, retries).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import ATTR_PREFIX
from ..config import attr as config_attr
from ..config import span_name as config_span_name
from ..defaults import get_tenant_distribution
from ..generators.trace_generator import (
    SpanConfig,
    SpanType,
    TraceHierarchy,
)
from ..statistics.correlations import ErrorPropagation, RetryConfig, RetrySequence
from ..statistics.distributions import Distribution, DistributionFactory


@dataclass
class LatencyConfig:
    """Latency distribution configuration."""

    mean_ms: float = 100.0
    variance: float = 0.3
    spike_rate: float = 0.05
    spike_multiplier: float = 3.0
    distribution: Distribution | None = None

    def sample(self) -> float:
        """Sample latency using configured distribution or default behavior."""
        if self.distribution:
            return max(1.0, self.distribution.sample())
        import random

        latency = self.mean_ms * (1 + random.gauss(0, self.variance))
        if random.random() < self.spike_rate:
            latency *= random.uniform(1.5, self.spike_multiplier)
        return max(1.0, latency)


@dataclass
class ErrorConfig:
    """Error rate and type configuration."""

    rate: float = 0.02
    types: list[str] = field(default_factory=lambda: ["timeout", "validation", "upstream_5xx"])
    retryable_types: list[str] = field(default_factory=lambda: ["timeout", "upstream_5xx"])
    propagation: ErrorPropagation | None = None

    def should_error(self, parent_errored: bool = False, sibling_errored: bool = False) -> bool:
        """Determine if error should occur, considering propagation."""
        if self.propagation:
            return self.propagation.should_error(parent_errored, sibling_errored)
        import random

        return random.random() < self.rate


@dataclass
class RetryBehavior:
    """Configuration for retry behavior in statistical scenarios."""

    enabled: bool = False
    max_attempts: int = 3
    config: RetryConfig | None = None
    force_initial_failure: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetryBehavior":
        """Create RetryBehavior from YAML data."""
        if not data:
            return cls(enabled=False)

        return cls(
            enabled=data.get("enabled", True),
            max_attempts=data.get("max_attempts", 3),
            force_initial_failure=data.get("force_initial_failure", False),
            config=RetryConfig(
                max_attempts=data.get("max_attempts", 3),
                backoff_base_ms=data.get("backoff_base_ms", 100.0),
                backoff_multiplier=data.get("backoff_multiplier", 2.0),
                backoff_jitter=data.get("backoff_jitter", 0.2),
                success_rate_per_attempt=data.get(
                    "success_rate_per_attempt", [0.0, 0.7, 0.85, 0.95]
                ),
            ),
        )


@dataclass
class ScenarioStep:
    """A single step in a scenario (one span in the trace)."""

    span_type: SpanType
    latency: LatencyConfig
    error: ErrorConfig
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["ScenarioStep"] = field(default_factory=list)
    probability: float = 1.0
    count_min: int = 1
    count_max: int = 1
    count_distribution: Distribution | None = None
    retry: RetryBehavior | None = None
    attribute_distributions: dict[str, Distribution] = field(default_factory=dict)

    def should_include(self) -> bool:
        """Determine if this step should be included based on probability."""
        if self.probability >= 1.0:
            return True
        import random

        return random.random() < self.probability

    def sample_count(self) -> int:
        """Sample how many instances of this span to generate."""
        if self.count_distribution:
            return max(self.count_min, min(self.count_max, self.count_distribution.sample_int()))
        import random

        return random.randint(self.count_min, self.count_max)

    def sample_attributes(self) -> dict[str, Any]:
        """Sample attribute values from configured distributions."""
        attrs = dict(self.attributes)
        for key, dist in self.attribute_distributions.items():
            attrs[key] = dist.sample()
        return attrs

    def to_hierarchy(self, parent_errored: bool = False) -> TraceHierarchy:
        """Convert scenario step to trace hierarchy."""
        sampled_latency = self.latency.sample()
        has_error = self.error.should_error(parent_errored)
        attrs = self.sample_attributes()

        # MCP tool: conventions require parent {prefix}.mcp.tool.execute + child {prefix}.mcp.tool.execute.attempt.
        if self.span_type == SpanType.MCP_TOOL_EXECUTE:
            execute_config = SpanConfig(
                span_type=SpanType.MCP_TOOL_EXECUTE,
                latency_mean_ms=0.0,
                latency_variance=0.0,
                error_rate=0.0,
                attribute_overrides=dict(attrs),
            )
            attempt_config = SpanConfig(
                span_type=SpanType.MCP_TOOL_EXECUTE_ATTEMPT,
                latency_mean_ms=sampled_latency,
                latency_variance=0.1,
                error_rate=1.0 if has_error else 0.0,
                attribute_overrides=dict(attrs),
            )
            return TraceHierarchy(
                root_config=execute_config,
                children=[TraceHierarchy(root_config=attempt_config, children=[])],
            )

        config = SpanConfig(
            span_type=self.span_type,
            latency_mean_ms=sampled_latency,
            latency_variance=0.1,
            error_rate=1.0 if has_error else 0.0,
            attribute_overrides=attrs,
        )

        child_hierarchies = []
        sibling_errored = False

        for child in self.children:
            if not child.should_include():
                continue

            count = child.sample_count()
            for _ in range(count):
                child_hier = child.to_hierarchy(parent_errored=has_error or sibling_errored)
                child_hierarchies.append(child_hier)

                if child_hier.root_config.error_rate > 0:
                    sibling_errored = True

        return TraceHierarchy(
            root_config=config,
            children=child_hierarchies,
        )

    def to_retry_hierarchies(self) -> list[TraceHierarchy]:
        """Generate hierarchies for retry attempts."""
        if not self.retry or not self.retry.enabled:
            return [self.to_hierarchy()]

        retry_seq = RetrySequence(
            config=self.retry.config or RetryConfig(),
            error_propagation=self.error.propagation or ErrorPropagation(base_rate=self.error.rate),
        )

        base_latency = self.latency.sample()
        attempts = retry_seq.generate(
            base_latency,
            force_initial_failure=self.retry.force_initial_failure,
        )

        # MCP tool: one hierarchy with root {prefix}.mcp.tool.execute and one child per attempt.
        if self.span_type == SpanType.MCP_TOOL_EXECUTE:
            base_attrs = self.sample_attributes()
            execute_config = SpanConfig(
                span_type=SpanType.MCP_TOOL_EXECUTE,
                latency_mean_ms=0.0,
                latency_variance=0.0,
                error_rate=0.0,
                attribute_overrides=dict(base_attrs),
            )
            attempt_hierarchies = []
            for idx, attempt in enumerate(attempts):
                attrs = dict(base_attrs)
                attrs["retry.attempt"] = attempt["attempt"]
                attrs["retry.is_retry"] = attempt["attempt"] > 1
                attrs[config_attr("mcp.attempt.index")] = idx
                attrs[config_attr("mcp.attempt.outcome")] = (
                    "success" if attempt["success"] else "failure"
                )
                if not attempt["success"]:
                    attrs["error.type"] = attempt["error_type"]

                attempt_config = SpanConfig(
                    span_type=SpanType.MCP_TOOL_EXECUTE_ATTEMPT,
                    latency_mean_ms=attempt["latency_ms"],
                    latency_variance=0.05,
                    error_rate=0.0 if attempt["success"] else 1.0,
                    attribute_overrides=attrs,
                )
                attempt_hierarchies.append(TraceHierarchy(root_config=attempt_config, children=[]))
            return [TraceHierarchy(root_config=execute_config, children=attempt_hierarchies)]

        hierarchies = []
        for attempt in attempts:
            attrs = self.sample_attributes()
            attrs["retry.attempt"] = attempt["attempt"]
            attrs["retry.is_retry"] = attempt["attempt"] > 1

            if not attempt["success"]:
                attrs["error.type"] = attempt["error_type"]

            config = SpanConfig(
                span_type=self.span_type,
                latency_mean_ms=attempt["latency_ms"],
                latency_variance=0.05,
                error_rate=0.0 if attempt["success"] else 1.0,
                attribute_overrides=attrs,
            )

            child_hierarchies = []
            for child in self.children:
                if attempt["success"] or child.probability >= 0.5:
                    child_hier = child.to_hierarchy(parent_errored=not attempt["success"])
                    child_hierarchies.append(child_hier)

            hierarchies.append(
                TraceHierarchy(
                    root_config=config,
                    children=child_hierarchies,
                )
            )

        return hierarchies


@dataclass
class Scenario:
    """Complete scenario definition."""

    name: str
    description: str
    tags: list[str]
    tenant_distribution: dict[str, float]
    repeat_count: int
    interval_ms: float
    root_step: ScenarioStep
    emit_metrics: bool = True
    emit_logs: bool = True
    is_statistical: bool = False

    def get_trace_hierarchy(self) -> TraceHierarchy:
        """Get the trace hierarchy for this scenario."""
        return self.root_step.to_hierarchy()

    def get_trace_hierarchies(self) -> list[TraceHierarchy]:
        """
        Get trace hierarchies, potentially multiple for retry scenarios.

        For scenarios with retry behavior, this returns a hierarchy per attempt.
        For normal scenarios, returns a single-item list.
        """
        if self.root_step.retry and self.root_step.retry.enabled:
            return self.root_step.to_retry_hierarchies()
        return [self.root_step.to_hierarchy()]


# Default prefix used in bundled scenario YAML; normalize to ATTR_PREFIX when loading.
_DEFAULT_YAML_PREFIX = "vendor"


def _normalize_attr_key(key: str) -> str:
    """Rewrite attribute keys from default YAML prefix to configured ATTR_PREFIX."""
    if key.startswith(_DEFAULT_YAML_PREFIX + "."):
        return f"{ATTR_PREFIX}.{key[len(_DEFAULT_YAML_PREFIX) + 1:]}"
    return key


# Span type resolution: YAML type is prefix.suffix (e.g. vendor.a2a.orchestrate) or bare suffix (e.g. rag.retrieve).
# Longer suffixes first so "mcp.tool.execute.attempt" matches before "mcp.tool.execute".
_SPAN_SUFFIXES = [
    ("a2a.orchestrate", SpanType.A2A_ORCHESTRATE),
    ("planner", SpanType.PLANNER),
    ("task.execute", SpanType.TASK_EXECUTE),
    ("llm.call", SpanType.LLM_CALL),
    ("mcp.tool.execute.attempt", SpanType.MCP_TOOL_EXECUTE_ATTEMPT),
    ("mcp.tool.execute", SpanType.MCP_TOOL_EXECUTE),
    ("llm.tool.response.bridge", SpanType.LLM_TOOL_RESPONSE_BRIDGE),
    ("response.compose", SpanType.RESPONSE_COMPOSE),
    ("rag.retrieve", SpanType.RAG_RETRIEVE),
    ("a2a.call", SpanType.A2A_CALL),
    ("cp.request", SpanType.CP_REQUEST),
]


# Default directory of sample scenario definitions (bundled with the package).
# Users can provide a custom folder via ScenarioLoader(scenarios_dir=...) or CLI --scenarios-dir.
SAMPLE_DEFINITIONS_DIR = Path(__file__).parent / "definitions"

# Reference scenario excluded from list and mixed workload when using sample definitions.
# It can still be run explicitly with: scenario --name example_scenario
EXAMPLE_SCENARIO_NAME = "example_scenario"


class ScenarioLoader:
    """Load scenarios from YAML files."""

    def __init__(self, scenarios_dir: Path | str | None = None):
        """Initialize loader with scenarios directory.

        If scenarios_dir is None, uses the bundled sample definitions
        (SAMPLE_DEFINITIONS_DIR). Pass a path to use custom scenario YAML files.
        """
        if scenarios_dir is None:
            self.scenarios_dir = SAMPLE_DEFINITIONS_DIR
        else:
            self.scenarios_dir = Path(scenarios_dir)

    def _get_span_type(self, type_str: str) -> SpanType:
        """Resolve YAML type string to SpanType (prefix.suffix or bare suffix)."""
        for suffix, span_type in _SPAN_SUFFIXES:
            if type_str == config_span_name(suffix) or type_str == suffix:
                return span_type
        return SpanType.A2A_ORCHESTRATE

    def load(self, scenario_name: str) -> Scenario:
        """Load a scenario by name."""
        scenario_file = self.scenarios_dir / f"{scenario_name}.yaml"
        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario not found: {scenario_name}")

        with open(scenario_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_scenario(data)

    def _is_sample_definitions_dir(self) -> bool:
        """True when using the bundled sample definitions (exclude reference scenarios)."""
        try:
            return self.scenarios_dir.resolve() == SAMPLE_DEFINITIONS_DIR.resolve()
        except (OSError, RuntimeError):
            return False

    def load_all(self) -> list[Scenario]:
        """Load all scenarios from the directory."""
        scenarios: list[Scenario] = []
        if not self.scenarios_dir.exists():
            return scenarios

        exclude_example = self._is_sample_definitions_dir()
        for scenario_file in self.scenarios_dir.glob("*.yaml"):
            if exclude_example and scenario_file.stem == EXAMPLE_SCENARIO_NAME:
                continue
            with open(scenario_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            scenarios.append(self._parse_scenario(data))

        return scenarios

    def list_scenarios(self) -> list[str]:
        """List available scenario names."""
        if not self.scenarios_dir.exists():
            return []
        exclude_example = self._is_sample_definitions_dir()
        return [
            f.stem
            for f in self.scenarios_dir.glob("*.yaml")
            if not (exclude_example and f.stem == EXAMPLE_SCENARIO_NAME)
        ]

    def _parse_scenario(self, data: dict) -> Scenario:
        """Parse scenario from YAML data."""
        data = data if isinstance(data, dict) else {}
        root_raw = data.get("root") or data.get("spans")
        if isinstance(root_raw, list) and root_raw:
            root_raw = root_raw[0]
        root_step = self._parse_step(root_raw if isinstance(root_raw, dict) else {})

        tenant_dist = get_tenant_distribution()

        root_for_detect = data.get("root")
        is_statistical = self._detect_statistical(
            root_for_detect if isinstance(root_for_detect, dict) else {}
        )

        return Scenario(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            tenant_distribution=tenant_dist,
            repeat_count=data.get("repeat_count", 1),
            interval_ms=data.get("interval_ms", 500),
            root_step=root_step,
            emit_metrics=data.get("emit_metrics", True),
            emit_logs=data.get("emit_logs", True),
            is_statistical=is_statistical,
        )

    def _detect_statistical(self, data: dict) -> bool:
        """Detect if a scenario uses statistical features."""
        if not data:
            return False

        statistical_keys = {"probability", "count", "retry", "distribution"}
        if any(key in data for key in statistical_keys):
            return True

        latency = data.get("latency", {})
        if isinstance(latency, dict) and "distribution" in latency:
            return True

        error = data.get("error", {})
        if isinstance(error, dict) and "propagation" in error:
            return True

        for attr_val in data.get("attributes", {}).values():
            if isinstance(attr_val, dict) and "distribution" in attr_val:
                return True

        for child in data.get("children", []):
            if self._detect_statistical(child):
                return True

        return False

    def _parse_step(self, data: dict | list | None) -> ScenarioStep:
        """Parse a scenario step from YAML data (one step = one dict; list → use first element)."""
        if isinstance(data, list) and data:
            data = data[0]
        data = data if isinstance(data, dict) else {}
        span_type_str = data.get("type", data.get("span_type", config_span_name("a2a.orchestrate")))
        span_type = self._get_span_type(span_type_str)

        latency_data = data.get("latency") or {}
        if not isinstance(latency_data, (dict, int, float)):
            latency_data = {}
        latency_dist = None
        if isinstance(latency_data, (int, float)):
            latency = LatencyConfig(mean_ms=float(latency_data))
        else:
            if "distribution" in latency_data:
                latency_dist = DistributionFactory.create_latency(latency_data)
            latency = LatencyConfig(
                mean_ms=latency_data.get("mean", latency_data.get("mean_ms", 100)),
                variance=latency_data.get("variance", 0.3),
                spike_rate=latency_data.get("spike_rate", 0.05),
                spike_multiplier=latency_data.get("spike_multiplier", 3.0),
                distribution=latency_dist,
            )

        error_data = data.get("error") or {}
        if not isinstance(error_data, (dict, int, float)):
            error_data = {}
        error_prop = None
        if isinstance(error_data, (int, float)):
            error = ErrorConfig(rate=float(error_data))
        else:
            if "propagation" in error_data:
                error_prop = ErrorPropagation.from_config(error_data)
            error = ErrorConfig(
                rate=error_data.get("rate", 0.02),
                types=error_data.get("types", ["timeout", "validation"]),
                retryable_types=error_data.get("retryable_types", ["timeout"]),
                propagation=error_prop,
            )

        count_data = data.get("count") or {}
        count_dist = None
        count_min = 1
        count_max = 1
        if isinstance(count_data, int):
            count_min = count_max = count_data
        elif isinstance(count_data, dict) and count_data:
            count_min = count_data.get("min", 1)
            count_max = count_data.get("max", 1)
            if "distribution" in count_data:
                count_dist = DistributionFactory.create(count_data)

        retry_data = data.get("retry")
        retry_data = retry_data if isinstance(retry_data, dict) else {}
        retry = RetryBehavior.from_dict(retry_data) if retry_data else None

        attr_dists = {}
        plain_attrs = {}
        for key, value in (data.get("attributes") or {}).items():
            norm_key = _normalize_attr_key(key)
            if isinstance(value, dict) and "distribution" in value:
                attr_dists[norm_key] = DistributionFactory.create(value)
            else:
                plain_attrs[norm_key] = value

        children = [self._parse_step(child) for child in (data.get("children") or [])]

        return ScenarioStep(
            span_type=span_type,
            latency=latency,
            error=error,
            attributes=plain_attrs,
            children=children,
            probability=data.get("probability", 1.0),
            count_min=count_min,
            count_max=count_max,
            count_distribution=count_dist,
            retry=retry,
            attribute_distributions=attr_dists,
        )

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        """Create a scenario from a dictionary (for inline definitions)."""
        loader = cls()
        return loader._parse_scenario(data)
