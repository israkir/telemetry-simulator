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

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import ATTR_PREFIX, CONFIG_PATH, load_yaml
from ..config import attr as config_attr
from ..config import span_name as config_span_name
from ..generators.trace_generator import (
    SpanConfig,
    SpanType,
    TraceHierarchy,
)
from ..statistics.correlations import ErrorPropagation, RetryConfig, RetrySequence
from ..statistics.distributions import Distribution, DistributionFactory
from .config_resolver import get_default_tenant_id, resolve_context
from .control_plane_loader import get_request_scenario_registry
from .id_generator import ScenarioIdGenerator
from .realistic_modifier import apply_realistic_scenario


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
class MCPToolRef:
    """One tool registered on an MCP server (name + uuid)."""

    name: str
    uuid: str


@dataclass
class FlowConfig:
    """Ordered path/flow: step identifiers (e.g. planner, tool names, response_compose)."""

    steps: list[str] = field(default_factory=list)


@dataclass
class ErrorPatternConfig:
    """Error pattern and optional config (rate, which step)."""

    pattern: str = "happy_path"
    rate: float = 0.1
    step_index: int | str = "random"  # int = that MCP step index, "random" = pick one


@dataclass
class MCPServerRef:
    """MCP server reference with ordered list of tools."""

    server_uuid: str
    tools: list[MCPToolRef]


@dataclass
class AgentRef:
    """Agent reference with optional MCP servers it can use."""

    uuid: str
    mcp: list[MCPServerRef] = field(default_factory=list)


@dataclass
class ScenarioContext:
    """
    Scenario generator context: tenant, agents, MCP servers, tools, correct_flow, error patterns.
    Only the first agent and its first MCP server are used when applying context to
    hierarchies (tenant, mcp.server.uuid, mcp.tool.uuid, gen_ai.tool.name, mcp.tool.call.id).
    When actual_steps is set (e.g. for partial_workflow), hierarchy is built from it
    instead of correct_flow.steps to simulate missing or wrong-order steps.
    """

    tenant_uuid: str
    agents: list[AgentRef] = field(default_factory=list)
    workflow: str | None = None
    correct_flow: FlowConfig | None = None
    error_pattern: str = "happy_path"
    error_config: ErrorPatternConfig | None = None
    redaction_applied: str = "none"
    # When set, hierarchy is built from actual_steps instead of correct_flow.steps (partial_workflow).
    actual_steps: list[str] | None = None


def _parse_and_resolve_context(
    data: dict[str, Any] | None,
    config_path: Path,
) -> ScenarioContext | None:
    """
    Parse key-based context and resolve to ScenarioContext from config.
    Requires tenant and agent. mcp_server is optional (control-plane-only scenarios need no MCP).
    workflow, correct_flow, error_pattern are optional; only needed for data-plane hierarchy.
    """
    if not data or not isinstance(data, dict):
        return None
    tenant_key = data.get("tenant")
    agent_id = data.get("agent")
    mcp_server_key = data.get("mcp_server")
    if not tenant_key or not isinstance(tenant_key, str):
        return None
    if not agent_id or not isinstance(agent_id, str):
        return None
    if mcp_server_key is not None and not isinstance(mcp_server_key, str):
        mcp_server_key = None

    workflow = data.get("workflow")
    if workflow is not None and not isinstance(workflow, str):
        workflow = None

    correct_flow_cfg: FlowConfig | None = None
    correct_flow_raw = data.get("correct_flow")
    if isinstance(correct_flow_raw, dict) and isinstance(correct_flow_raw.get("steps"), list):
        correct_flow_cfg = FlowConfig(steps=[str(s) for s in correct_flow_raw["steps"]])
    elif isinstance(correct_flow_raw, list):
        correct_flow_cfg = FlowConfig(steps=[str(s) for s in correct_flow_raw])

    error_pattern = data.get("error_pattern", "happy_path")
    if not isinstance(error_pattern, str):
        error_pattern = "happy_path"
    redaction_applied = data.get("redaction_applied", "none")
    if not isinstance(redaction_applied, str):
        redaction_applied = "none"

    err_cfg: ErrorPatternConfig | None = None
    err_cfg_raw = data.get("error_config")
    if isinstance(err_cfg_raw, dict):
        step_idx = err_cfg_raw.get("step_index", "random")
        if not isinstance(step_idx, int):
            step_idx = "random"
        err_cfg = ErrorPatternConfig(
            pattern=error_pattern,
            rate=float(err_cfg_raw.get("rate", 0.1)),
            step_index=step_idx,
        )
    elif error_pattern not in ("happy_path", "none"):
        err_cfg = ErrorPatternConfig(pattern=error_pattern)

    resolved = resolve_context(
        tenant_key=tenant_key.strip(),
        agent_id=agent_id.strip(),
        mcp_server_key=mcp_server_key.strip() if isinstance(mcp_server_key, str) else None,
        workflow=workflow,
        correct_flow=correct_flow_cfg,
        error_pattern=error_pattern,
        error_config=err_cfg,
        redaction_applied=redaction_applied,
        config_path=config_path,
    )
    agents: list[AgentRef] = []
    for a in resolved.get("agents") or []:
        if not isinstance(a, dict):
            continue
        agent_uuid = a.get("uuid")
        if not isinstance(agent_uuid, str):
            continue
        mcp_list: list[MCPServerRef] = []
        for m in a.get("mcp") or []:
            if not isinstance(m, dict):
                continue
            server_uuid = m.get("server_uuid")
            if not isinstance(server_uuid, str):
                continue
            tools = [
                MCPToolRef(name=t["name"], uuid=t["tool_uuid"])
                for t in (m.get("tools") or [])
                if isinstance(t, dict)
                and isinstance(t.get("name"), str)
                and isinstance(t.get("tool_uuid"), str)
            ]
            mcp_list.append(MCPServerRef(server_uuid=server_uuid, tools=tools))
        agents.append(AgentRef(uuid=agent_uuid, mcp=mcp_list))

    return ScenarioContext(
        tenant_uuid=resolved["tenant_uuid"],
        agents=agents,
        workflow=resolved.get("workflow"),
        correct_flow=resolved.get("correct_flow"),
        error_pattern=resolved.get("error_pattern", "happy_path"),
        error_config=resolved.get("error_config"),
        redaction_applied=resolved.get("redaction_applied", "none"),
        actual_steps=resolved.get("actual_steps"),
    )


def _apply_data_plane_template(
    context: ScenarioContext,
    template_name: str,
    config_path: Path,
) -> tuple[ScenarioContext, str | None, str | None]:
    """
    Enrich context from config data_plane_templates (workflow + correct_flow) and return simulation_goal
    and optional control_plane_template. When the data-plane template sets control_plane_template,
    scenarios using it will use that request-validation template unless they override control_plane.template.
    Returns (context, simulation_goal, control_plane_template). If template not found, returns (context, None, None).
    """
    from ..config import load_yaml

    data = load_yaml(config_path)
    if not isinstance(data, dict):
        return context, None, None
    templates = data.get("data_plane_templates")
    if not isinstance(templates, dict):
        return context, None, None
    template = templates.get((template_name or "").strip())
    if not isinstance(template, dict):
        return context, None, None
    workflow = template.get("workflow")
    if not isinstance(workflow, str):
        return context, None, None
    rs = data.get("realistic_scenarios")
    workflow_templates = rs.get("workflow_templates") if isinstance(rs, dict) else None
    if not isinstance(workflow_templates, dict):
        return context, None, None
    steps = workflow_templates.get(workflow)
    if not isinstance(steps, list):
        return context, None, None
    correct_flow = FlowConfig(steps=[str(s) for s in steps])
    simulation_goal = template.get("simulation_goal")
    if simulation_goal is not None and not isinstance(simulation_goal, str):
        simulation_goal = None
    cp_template = template.get("control_plane_template")
    if cp_template is not None and not isinstance(cp_template, str):
        cp_template = None
    if cp_template and not (cp_template := cp_template.strip()):
        cp_template = None
    updated = ScenarioContext(
        tenant_uuid=context.tenant_uuid,
        agents=context.agents,
        workflow=workflow,
        correct_flow=correct_flow,
        error_pattern=context.error_pattern,
        error_config=context.error_config,
        redaction_applied=context.redaction_applied,
        actual_steps=context.actual_steps,
    )
    return updated, simulation_goal, cp_template


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
                # Conventions: llm_call, tool_recommendation, tool_execution require task.execute parent.
                child_hier = _ensure_task_execute_parent(child_hier, self.span_type)
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
                    child_hier = _ensure_task_execute_parent(child_hier, self.span_type)
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
    interval_deviation_ms: float
    root_step: ScenarioStep
    emit_metrics: bool = True
    emit_logs: bool = True
    is_statistical: bool = False
    context: ScenarioContext | None = None
    id_generator: ScenarioIdGenerator | None = None
    # Simulation goal (e.g. happy_path, higher_latency, 4xx_invalid_arguments); defined per scenario.
    simulation_goal: str | None = None
    # MCP server context (e.g. phone, electronics, appliances); identifies which mcp_servers entry.
    mcp_server: str | None = None
    # Optional multi-turn conversation: raw role/text or user_input/llm_response (reference).
    conversation_turns: list[dict[str, str]] | None = None
    # Per-turn (input_messages, output_messages) for one span per interaction; same session_id for all.
    # Built from conversation.turns in loader. When set, runner emits one trace per turn with same session_id.
    conversation_turn_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None
    # Redaction level for {prefix}.redaction.applied (none, basic, strict). From scenario YAML or context; default none.
    redaction_applied: str = "none"
    # Overrides for realistic scenario modifier (step_index_for_4xx, wrong_division_target, skip_steps, actual_steps).
    realistic_overrides: dict[str, Any] | None = None
    # Control-plane: request outcome (allowed | blocked | error). When blocked, no a2a.orchestrate or response.validation.
    control_plane_request_outcome: str = "allowed"
    # When control_plane_request_outcome=blocked: invalid_payload | request_policy | invalid_context (semconv gentoro.block.reason).
    control_plane_block_reason: str | None = None
    # Optional: use a specific request_validation_templates key from config (overrides outcome + block_reason resolution).
    control_plane_template: str | None = None
    # Optional: override policy exception (type, message) from template; e.g. {"type": "PolicyEngineUnavailable", "message": "..."}.
    control_plane_policy_exception_override: dict[str, str] | None = None

    def get_trace_hierarchy(self) -> TraceHierarchy:
        """Get the trace hierarchy for this scenario."""
        if self.context and self.context.correct_flow and self.context.correct_flow.steps:
            hierarchy = _hierarchy_from_context(self.context)
            _apply_context_to_hierarchy(hierarchy, self.context, self.id_generator)
            _apply_realistic_scenario_if_needed(self, hierarchy)
            return hierarchy
        hierarchy = self.root_step.to_hierarchy()
        if self.context:
            _apply_context_to_hierarchy(hierarchy, self.context, self.id_generator)
        return hierarchy

    def get_trace_hierarchies(self) -> list[TraceHierarchy]:
        """
        Get trace hierarchies, potentially multiple for retry scenarios.

        When context.correct_flow.steps is set, the happy path is expressed in context:
        hierarchy is built from context.correct_flow.steps and context (tenant, agent,
        tools, error_pattern) is applied. Otherwise hierarchy comes from YAML root.
        When scenario is control-plane-only (trace_flow is [incoming_validation] and no
        data-plane definition), returns [] so only incoming validation is emitted.
        """
        if self.context and self.context.correct_flow and self.context.correct_flow.steps:
            hierarchy = _hierarchy_from_context(self.context)
            _apply_context_to_hierarchy(hierarchy, self.context, self.id_generator)
            _apply_realistic_scenario_if_needed(self, hierarchy)
            return [hierarchy]
        # Control-plane-only: no data-plane hierarchy to emit
        from .control_plane_loader import get_request_validation_template_id, get_trace_flow

        flow = get_trace_flow(
            self.control_plane_request_outcome or "allowed",
            template_id=get_request_validation_template_id(
                self.control_plane_request_outcome or "allowed",
                self.control_plane_block_reason,
                self.control_plane_template,
            ),
        )
        if flow == ["incoming_validation"]:
            return []
        if self.root_step.retry and self.root_step.retry.enabled:
            hierarchies = self.root_step.to_retry_hierarchies()
        else:
            hierarchies = [self.root_step.to_hierarchy()]
        if self.context:
            for h in hierarchies:
                _apply_context_to_hierarchy(h, self.context, self.id_generator)
        return hierarchies


def _load_happy_path_latencies() -> dict[SpanType, float]:
    """
    Load default latencies for happy-path context flows from config/config.yaml.

    Config shape in config/config.yaml:

    happy_path_latencies_ms:
      a2a_orchestrate: 1500.0
      planner: 250.0
      mcp_tool_execute: 150.0
      response_compose: 60.0
    """
    defaults: dict[SpanType, float] = {
        SpanType.A2A_ORCHESTRATE: 1500.0,
        SpanType.PLANNER: 250.0,
        SpanType.TASK_EXECUTE: 80.0,
        SpanType.LLM_CALL: 400.0,
        SpanType.TOOLS_RECOMMEND: 50.0,
        SpanType.MCP_TOOL_EXECUTE: 150.0,
        SpanType.RESPONSE_COMPOSE: 60.0,
        SpanType.REQUEST_VALIDATION: 40.0,
        SpanType.RESPONSE_VALIDATION: 40.0,
        SpanType.VALIDATION_PAYLOAD: 20.0,
        SpanType.VALIDATION_POLICY: 20.0,
        SpanType.AUGMENTATION: 20.0,
    }

    data = load_yaml(CONFIG_PATH)
    raw_latencies = data.get("happy_path_latencies_ms")
    if not isinstance(raw_latencies, dict):
        return defaults

    key_map: dict[str, SpanType] = {
        "a2a_orchestrate": SpanType.A2A_ORCHESTRATE,
        "planner": SpanType.PLANNER,
        "task_execute": SpanType.TASK_EXECUTE,
        "llm_call": SpanType.LLM_CALL,
        "tools_recommend": SpanType.TOOLS_RECOMMEND,
        "mcp_tool_execute": SpanType.MCP_TOOL_EXECUTE,
        "response_compose": SpanType.RESPONSE_COMPOSE,
        "request_validation": SpanType.REQUEST_VALIDATION,
        "response_validation": SpanType.RESPONSE_VALIDATION,
        "validation_payload": SpanType.VALIDATION_PAYLOAD,
        "validation_policy": SpanType.VALIDATION_POLICY,
        "augmentation": SpanType.AUGMENTATION,
    }

    latencies = dict(defaults)
    for key, span_type in key_map.items():
        value = raw_latencies.get(key)
        if isinstance(value, (int, float)):
            latencies[span_type] = float(value)

    return latencies


# Default latencies when building hierarchy from context.correct_flow (happy path).
_DEFAULT_LATENCY_MS = _load_happy_path_latencies()

# Task types per semantic conventions: llm_call, tool_recommendation, tool_execution.
# These span types MUST have a task.execute parent with the matching {prefix}.task.type.
_TASK_TYPE_BY_SPAN_TYPE: dict[SpanType, str] = {
    SpanType.LLM_CALL: "llm_call",
    SpanType.TOOLS_RECOMMEND: "tool_recommendation",
    SpanType.MCP_TOOL_EXECUTE: "tool_execution",
}


def _ensure_task_execute_parent(
    child_hier: TraceHierarchy,
    parent_span_type: SpanType | None,
) -> TraceHierarchy:
    """
    Wrap hierarchy in task.execute when required by conventions.

    llm_call, tool_recommendation, tool_execution MUST have task.execute as parent
    with {prefix}.task.type = llm_call | tool_recommendation | tool_execution.
    MCP tool execution is never a child of llm.call; it is always under task (tool_execution).
    """
    st = child_hier.root_config.span_type
    task_type = _TASK_TYPE_BY_SPAN_TYPE.get(st)
    if task_type is None:
        return child_hier
    task_config = SpanConfig(
        span_type=SpanType.TASK_EXECUTE,
        latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.TASK_EXECUTE],
        latency_variance=0.2,
        error_rate=0.0,
        attribute_overrides={
            config_attr("step.outcome"): "success",
            config_attr("task.type"): task_type,
        },
    )
    return TraceHierarchy(root_config=task_config, children=[child_hier])


def _apply_realistic_scenario_if_needed(scenario: "Scenario", hierarchy: TraceHierarchy) -> None:
    """Apply realistic scenario modifier when simulation_goal is set and not happy_path."""
    goal = getattr(scenario, "simulation_goal", None)
    if not goal or (goal or "").lower() in ("happy_path", "none", ""):
        return
    overrides = getattr(scenario, "realistic_overrides", None) or {}
    apply_realistic_scenario(
        hierarchy,
        simulation_goal=goal,
        realistic_overrides=overrides,
        context=scenario.context,
        mcp_server_key=getattr(scenario, "mcp_server", None),
    )


def _hierarchy_from_context(context: ScenarioContext) -> TraceHierarchy:
    """
    Build trace hierarchy from context.correct_flow.steps (or context.actual_steps for partial_workflow).
    The generator knows the correct path from context; no YAML structure required.
    """
    steps = None
    if getattr(context, "actual_steps", None):
        steps = context.actual_steps
    if not steps and context.correct_flow and context.correct_flow.steps:
        steps = context.correct_flow.steps
    if not steps:
        raise ValueError(
            "context.correct_flow.steps or context.actual_steps required to build hierarchy from context"
        )

    root_config = SpanConfig(
        span_type=SpanType.A2A_ORCHESTRATE,
        latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.A2A_ORCHESTRATE],
        latency_variance=0.25,
        error_rate=0.0,
        attribute_overrides={config_attr("a2a.outcome"): "success"},
    )
    children: list[TraceHierarchy] = []
    steps = list(steps)
    i = 0
    while i < len(steps):
        step = steps[i]
        step_lower = (step or "").strip().lower()

        if step_lower in ("planner", "planning"):
            children.append(
                TraceHierarchy(
                    root_config=SpanConfig(
                        span_type=SpanType.PLANNER,
                        latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.PLANNER],
                        latency_variance=0.2,
                        error_rate=0.0,
                        attribute_overrides={config_attr("step.outcome"): "success"},
                    ),
                    children=[],
                )
            )
            i += 1
        elif step_lower in ("response_compose", "response"):
            children.append(
                TraceHierarchy(
                    root_config=SpanConfig(
                        span_type=SpanType.RESPONSE_COMPOSE,
                        latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.RESPONSE_COMPOSE],
                        latency_variance=0.1,
                        error_rate=0.0,
                        attribute_overrides={
                            config_attr("response.format"): "a2a_json",
                            config_attr("step.outcome"): "success",
                        },
                    ),
                    children=[],
                )
            )
            i += 1
        elif step_lower in ("task", "task_execute", "task.execute"):
            # llm.call parent MUST be task.execute (task.type=llm_call).
            # MCP tool execution has its own task (task.type=tool_execution); never nest under llm.call.
            task_overrides = {
                config_attr("step.outcome"): "success",
                config_attr("task.type"): "llm_call",
            }
            llm_call_hierarchy = TraceHierarchy(
                root_config=SpanConfig(
                    span_type=SpanType.LLM_CALL,
                    latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.LLM_CALL],
                    latency_variance=0.2,
                    error_rate=0.0,
                    attribute_overrides={config_attr("step.outcome"): "success"},
                ),
                children=[],
            )
            children.append(
                TraceHierarchy(
                    root_config=SpanConfig(
                        span_type=SpanType.TASK_EXECUTE,
                        latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.TASK_EXECUTE],
                        latency_variance=0.2,
                        error_rate=0.0,
                        attribute_overrides=task_overrides,
                    ),
                    children=[llm_call_hierarchy],
                )
            )
            i += 1
        elif step_lower in ("tools_recommend", "tools.recommend"):
            # tools.recommend parent MUST be task.execute (task.type=tool_recommendation).
            tools_recommend_hierarchy = TraceHierarchy(
                root_config=SpanConfig(
                    span_type=SpanType.TOOLS_RECOMMEND,
                    latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.TOOLS_RECOMMEND],
                    latency_variance=0.2,
                    error_rate=0.0,
                    attribute_overrides={config_attr("step.outcome"): "success"},
                ),
                children=[],
            )
            task_config = SpanConfig(
                span_type=SpanType.TASK_EXECUTE,
                latency_mean_ms=_DEFAULT_LATENCY_MS[SpanType.TASK_EXECUTE],
                latency_variance=0.2,
                error_rate=0.0,
                attribute_overrides={
                    config_attr("step.outcome"): "success",
                    config_attr("task.type"): "tool_recommendation",
                },
            )
            children.append(
                TraceHierarchy(root_config=task_config, children=[tools_recommend_hierarchy])
            )
            i += 1
        else:
            # Tool step at root level (no preceding task). Conventions: tool_execution requires task.execute parent.
            latency = _DEFAULT_LATENCY_MS[SpanType.MCP_TOOL_EXECUTE]
            execute_config = SpanConfig(
                span_type=SpanType.MCP_TOOL_EXECUTE,
                latency_mean_ms=0.0,
                latency_variance=0.0,
                error_rate=0.0,
                attribute_overrides={config_attr("step.outcome"): "success"},
            )
            attempt_config = SpanConfig(
                span_type=SpanType.MCP_TOOL_EXECUTE_ATTEMPT,
                latency_mean_ms=latency,
                latency_variance=0.2,
                error_rate=0.0,
                attribute_overrides={},
            )
            mcp_hierarchy = TraceHierarchy(
                root_config=execute_config,
                children=[TraceHierarchy(root_config=attempt_config, children=[])],
            )
            wrapped = _ensure_task_execute_parent(mcp_hierarchy, SpanType.A2A_ORCHESTRATE)
            children.append(wrapped)
            i += 1

    return TraceHierarchy(root_config=root_config, children=children)


def _collect_mcp_hierarchies_in_order(hierarchy: TraceHierarchy) -> list[TraceHierarchy]:
    """Return list of MCP_TOOL_EXECUTE hierarchies in tree order (depth-first)."""
    result: list[TraceHierarchy] = []

    def walk(h: TraceHierarchy) -> None:
        if h.root_config.span_type == SpanType.MCP_TOOL_EXECUTE:
            result.append(h)
        for child in h.children:
            walk(child)

    for child in hierarchy.children:
        walk(child)
    return result


def _apply_error_pattern(hierarchy: TraceHierarchy, context: ScenarioContext) -> None:
    """Set error_rate on root and MCP attempt spans according to context.error_pattern."""
    import random

    pattern = (context.error_pattern or "happy_path").lower()
    if pattern in ("happy_path", "none", ""):
        hierarchy.root_config.error_rate = 0.0
        for child in hierarchy.children:
            if child.root_config.span_type == SpanType.MCP_TOOL_EXECUTE and child.children:
                child.children[0].root_config.error_rate = 0.0
            _apply_error_pattern(child, context)
        return

    mcp_list = _collect_mcp_hierarchies_in_order(hierarchy)
    err_cfg = context.error_config or ErrorPatternConfig()
    rate = err_cfg.rate

    def zero_all_mcp() -> None:
        for mcp_h in mcp_list:
            if mcp_h.children:
                mcp_h.children[0].root_config.error_rate = 0.0

    if pattern == "single_tool_failure":
        zero_all_mcp()
        if mcp_list:
            step_idx = err_cfg.step_index
            if step_idx == "random":
                step_idx = random.randint(0, len(mcp_list) - 1)
            else:
                step_idx = max(0, min(int(step_idx), len(mcp_list) - 1))
            if mcp_list[step_idx].children:
                mcp_list[step_idx].children[0].root_config.error_rate = rate
        return

    if pattern == "first_tool_failure":
        zero_all_mcp()
        if mcp_list and mcp_list[0].children:
            mcp_list[0].children[0].root_config.error_rate = rate
        return

    if pattern == "last_tool_failure":
        zero_all_mcp()
        if mcp_list and mcp_list[-1].children:
            mcp_list[-1].children[0].root_config.error_rate = rate
        return

    if pattern == "cascade":
        hierarchy.root_config.error_rate = rate
        zero_all_mcp()
        if mcp_list and mcp_list[0].children:
            mcp_list[0].children[0].root_config.error_rate = rate
        return

    if pattern == "random":
        for mcp_h in mcp_list:
            if mcp_h.children:
                mcp_h.children[0].root_config.error_rate = 1.0 if random.random() < rate else 0.0
        return


def _apply_context_to_hierarchy(
    hierarchy: TraceHierarchy,
    context: ScenarioContext,
    id_generator: ScenarioIdGenerator | None = None,
) -> None:
    """
    Merge scenario context into hierarchy in place: tenant, agent id on root;
    MCP server uuid and tool name/uuid on each MCP span in order; flow validation;
    error pattern applied to root and MCP attempt spans.
    IDs (e.g. mcp.tool.call.id) are generated from shared config if id_generator is set.
    """
    if not context.agents:
        overrides = dict(hierarchy.root_config.attribute_overrides or {})
        overrides[config_attr("tenant.id")] = context.tenant_uuid
        hierarchy.root_config.attribute_overrides = overrides
        _apply_error_pattern(hierarchy, context)
        return

    agent = context.agents[0]
    root_overrides = dict(hierarchy.root_config.attribute_overrides or {})
    root_overrides[config_attr("tenant.id")] = context.tenant_uuid
    root_overrides[config_attr("a2a.agent.target.id")] = agent.uuid
    hierarchy.root_config.attribute_overrides = root_overrides

    tools_by_index: list[MCPToolRef] = []
    if agent.mcp:
        tools_by_index = agent.mcp[0].tools
    server_uuid = agent.mcp[0].server_uuid if agent.mcp else ""
    mcp_index = [0]

    def walk(h: TraceHierarchy) -> None:
        cfg = h.root_config
        if cfg.span_type == SpanType.MCP_TOOL_EXECUTE:
            if id_generator:
                call_id = id_generator.mcp_tool_call_id(tenant_id=context.tenant_uuid)
            else:
                call_id = f"mcp_call_{uuid.uuid4().hex[:12]}"
            overrides = dict(cfg.attribute_overrides or {})
            overrides[config_attr("mcp.server.uuid")] = server_uuid
            overrides[config_attr("mcp.tool.call.id")] = call_id
            idx = mcp_index[0]
            if idx < len(tools_by_index):
                tool = tools_by_index[idx]
                overrides["gen_ai.tool.name"] = tool.name
                overrides[config_attr("mcp.tool.uuid")] = tool.uuid
            mcp_index[0] += 1
            cfg.attribute_overrides = overrides
            if h.children:
                attempt_cfg = h.children[0].root_config
                attempt_overrides = dict(attempt_cfg.attribute_overrides or {})
                attempt_overrides[config_attr("mcp.tool.call.id")] = call_id
                attempt_cfg.attribute_overrides = attempt_overrides
        elif cfg.span_type == SpanType.TOOLS_RECOMMEND:
            # For tool recommendation spans, propagate the primary MCP tool name when available
            # so downstream diagnostics (e.g. {prefix}.agent.tool_selection) can align tool_name
            # with gen_ai.tool.name used on the subsequent MCP tool execution.
            overrides = dict(cfg.attribute_overrides or {})
            primary_tool_name = tools_by_index[0].name if tools_by_index else ""
            if primary_tool_name:
                overrides["gen_ai.tool.name"] = primary_tool_name
            cfg.attribute_overrides = overrides
        for child in h.children:
            walk(child)

    for child in hierarchy.children:
        walk(child)

    # When flow.steps tool count != context.agents[0].mcp[0].tools count, hierarchy is
    # left as-is (no auto-fix); ensure scenario YAML matches.

    _apply_error_pattern(hierarchy, context)


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
    ("tools.recommend", SpanType.TOOLS_RECOMMEND),
    ("mcp.tool.execute.attempt", SpanType.MCP_TOOL_EXECUTE_ATTEMPT),
    ("mcp.tool.execute", SpanType.MCP_TOOL_EXECUTE),
    ("llm.tool.response.bridge", SpanType.LLM_TOOL_RESPONSE_BRIDGE),
    ("response.compose", SpanType.RESPONSE_COMPOSE),
    ("rag.retrieve", SpanType.RAG_RETRIEVE),
    ("a2a.call", SpanType.A2A_CALL),
    ("request.validation", SpanType.REQUEST_VALIDATION),
    ("response.validation", SpanType.RESPONSE_VALIDATION),
    ("validation.payload", SpanType.VALIDATION_PAYLOAD),
    ("validation.policy", SpanType.VALIDATION_POLICY),
    ("augmentation", SpanType.AUGMENTATION),
    ("cp.request", SpanType.CP_REQUEST),
]


# Default directory of sample scenario definitions (bundled with the package).
# Users can provide a custom folder via ScenarioLoader(scenarios_dir=...) or CLI --scenarios-dir.
SAMPLE_DEFINITIONS_DIR = Path(__file__).parent / "definitions"

# Reference scenario excluded from list and mixed workload when using sample definitions.
# It can still be run explicitly with: scenario --name example_scenario
EXAMPLE_SCENARIO_NAME = "example_scenario"


def _control_plane_scenario_data(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Build minimal scenario dict for control_plane.request_scenarios registry (no YAML file)."""
    return {
        "name": name,
        "description": entry.get("description", ""),
        "tags": ["control-plane"],
        "control_plane": {"template": entry["template"]},
        "context": {"tenant": "toro", "agent": "toro-customer-assistant-001"},
        "mcp_server": "phone",
        "repeat_count": 2,
        "interval_ms": 50,
        "emit_metrics": False,
        "emit_logs": False,
        "redaction_applied": "none",
    }


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
        self._id_generator: ScenarioIdGenerator | None = None

    def get_id_generator(self) -> ScenarioIdGenerator:
        """Return shared ID generator (loads config/config.yaml once)."""
        if self._id_generator is None:
            self._id_generator = ScenarioIdGenerator(config_path=CONFIG_PATH)
        return self._id_generator

    def _get_span_type(self, type_str: str) -> SpanType:
        """Resolve YAML type string to SpanType (prefix.suffix or bare suffix)."""
        for suffix, span_type in _SPAN_SUFFIXES:
            if type_str == config_span_name(suffix) or type_str == suffix:
                return span_type
        return SpanType.A2A_ORCHESTRATE

    def load(self, scenario_name: str) -> Scenario:
        """Load a scenario by name (from YAML or from control_plane.request_scenarios registry)."""
        scenario_file = self.scenarios_dir / f"{scenario_name}.yaml"
        if scenario_file.exists():
            with open(scenario_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return self._parse_scenario(data)
        if self._is_sample_definitions_dir():
            registry = get_request_scenario_registry(CONFIG_PATH)
            entry = registry.get(scenario_name) if isinstance(registry, dict) else None
            if isinstance(entry, dict) and entry.get("template"):
                data = _control_plane_scenario_data(scenario_name, entry)
                return self._parse_scenario(data)
        raise FileNotFoundError(f"Scenario not found: {scenario_name}")

    def _is_sample_definitions_dir(self) -> bool:
        """True when using the bundled sample definitions (exclude reference scenarios)."""
        try:
            return self.scenarios_dir.resolve() == SAMPLE_DEFINITIONS_DIR.resolve()
        except (OSError, RuntimeError):
            return False

    def load_all(self) -> list[Scenario]:
        """Load all scenarios from the directory and from control_plane.request_scenarios when using sample dir."""
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

        if exclude_example:
            loaded_names = {s.name for s in scenarios}
            registry = get_request_scenario_registry(CONFIG_PATH)
            for name, entry in registry.items():
                if name not in loaded_names and isinstance(entry, dict) and entry.get("template"):
                    scenarios.append(
                        self._parse_scenario(_control_plane_scenario_data(name, entry))
                    )

        return scenarios

    def list_scenarios(self) -> list[str]:
        """List available scenario names (YAML stems + control_plane.request_scenarios when using sample dir)."""
        if not self.scenarios_dir.exists():
            return []
        exclude_example = self._is_sample_definitions_dir()
        from_files = [
            f.stem
            for f in self.scenarios_dir.glob("*.yaml")
            if not (exclude_example and f.stem == EXAMPLE_SCENARIO_NAME)
        ]
        if not exclude_example:
            return from_files
        registry = get_request_scenario_registry(CONFIG_PATH)
        file_set = set(from_files)
        extra = [n for n in registry if n not in file_set]
        return from_files + sorted(extra)

    def _parse_scenario(self, data: dict) -> Scenario:
        """Parse scenario from YAML data."""
        data = data if isinstance(data, dict) else {}
        root_raw = data.get("root")
        if isinstance(root_raw, list) and root_raw:
            root_raw = root_raw[0]
        root_step = self._parse_step(root_raw if isinstance(root_raw, dict) else {})

        scenario_context = _parse_and_resolve_context(data.get("context"), CONFIG_PATH)
        if scenario_context:
            tenant_dist = {scenario_context.tenant_uuid: 1.0}
        else:
            tenant_dist = {get_default_tenant_id(CONFIG_PATH): 1.0}

        # Optional data_plane.template: workflow + correct_flow + simulation_goal + control_plane_template from config.
        simulation_goal_from_template: str | None = None
        control_plane_template_from_dp: str | None = None
        if scenario_context:
            dp = data.get("data_plane")
            if isinstance(dp, dict):
                template_raw = dp.get("template")
                if isinstance(template_raw, str) and template_raw.strip():
                    template_name = template_raw.strip()
                    (
                        scenario_context,
                        simulation_goal_from_template,
                        control_plane_template_from_dp,
                    ) = _apply_data_plane_template(scenario_context, template_name, CONFIG_PATH)

        root_for_detect = data.get("root")
        is_statistical = self._detect_statistical(
            root_for_detect if isinstance(root_for_detect, dict) else {}
        )

        simulation_goal = data.get("simulation_goal")
        if simulation_goal is not None and not isinstance(simulation_goal, str):
            simulation_goal = None
        if simulation_goal is None and simulation_goal_from_template is not None:
            simulation_goal = simulation_goal_from_template
        mcp_server = data.get("mcp_server")
        if mcp_server is not None and not isinstance(mcp_server, str):
            mcp_server = None

        redaction_applied = data.get("redaction_applied")
        if not isinstance(redaction_applied, str) and scenario_context:
            redaction_applied = scenario_context.redaction_applied
        if not isinstance(redaction_applied, str):
            redaction_applied = "none"

        # Optional multi-turn conversation: one span per interaction (user input → LLM response).
        # Format A: [{ user_input: "...", llm_response: "..." }, ...]
        # Format B: [{ role: user|assistant, text: "..." }, ...] (consecutive user/assistant pairs).
        conversation_turns: list[dict[str, str]] | None = None
        conversation_turn_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = (
            None
        )
        conv_raw = data.get("conversation")
        if isinstance(conv_raw, dict):
            turns_raw = conv_raw.get("turns")
            if isinstance(turns_raw, list):
                turn_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
                for t in turns_raw:
                    if not isinstance(t, dict):
                        continue
                    user_input = t.get("user_input")
                    llm_response = t.get("llm_response")
                    if isinstance(user_input, str) and isinstance(llm_response, str):
                        input_msgs = [
                            {"role": "user", "content": [{"type": "text", "text": user_input}]},
                        ]
                        output_msgs = [
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": llm_response}],
                            },
                        ]
                        turn_pairs.append((input_msgs, output_msgs))
                if turn_pairs:
                    conversation_turns = [
                        {"role": "user", "text": inp[0]["content"][0]["text"]}
                        for inp, _ in turn_pairs
                    ] + [
                        {"role": "assistant", "text": out[0]["content"][0]["text"]}
                        for _, out in turn_pairs
                    ]
                else:
                    cleaned: list[dict[str, str]] = []
                    for t in turns_raw:
                        if not isinstance(t, dict):
                            continue
                        role = t.get("role")
                        text = t.get("text")
                        if isinstance(role, str) and isinstance(text, str):
                            cleaned.append({"role": role, "text": text})
                    if cleaned:
                        conversation_turns = cleaned
                        i = 0
                        while i + 1 < len(cleaned):
                            u, a = cleaned[i], cleaned[i + 1]
                            if u.get("role") == "user" and a.get("role") == "assistant":
                                input_msgs = [
                                    {
                                        "role": "user",
                                        "content": [{"type": "text", "text": u["text"]}],
                                    },
                                ]
                                output_msgs = [
                                    {
                                        "role": "assistant",
                                        "content": [{"type": "text", "text": a["text"]}],
                                    },
                                ]
                                turn_pairs.append((input_msgs, output_msgs))
                                i += 2
                            else:
                                i += 1
                if turn_pairs:
                    conversation_turn_pairs = turn_pairs

        realistic_overrides_raw = data.get("realistic_overrides")
        realistic_overrides: dict[str, Any] = {}
        if isinstance(realistic_overrides_raw, dict):
            realistic_overrides = dict(realistic_overrides_raw)

        control_plane_request_outcome = "allowed"
        control_plane_block_reason = None
        control_plane_template: str | None = None
        control_plane_policy_exception_override: dict[str, str] | None = None
        cp_raw = data.get("control_plane")
        if isinstance(cp_raw, dict):
            outcome = cp_raw.get("request_outcome")
            if isinstance(outcome, str) and outcome.strip().lower() in (
                "allowed",
                "blocked",
                "error",
            ):
                control_plane_request_outcome = outcome.strip().lower()
            if control_plane_request_outcome == "blocked":
                reason = cp_raw.get("block_reason")
                if isinstance(reason, str) and reason.strip():
                    control_plane_block_reason = reason.strip().lower()
            template = cp_raw.get("template")
            if isinstance(template, str) and template.strip():
                control_plane_template = template.strip()
            pe = cp_raw.get("policy_exception")
            if isinstance(pe, dict) and (pe.get("type") or pe.get("message")):
                control_plane_policy_exception_override = {
                    k: str(v) for k, v in pe.items() if k in ("type", "message") and v is not None
                }
        # When scenario uses data_plane.template and did not set control_plane.template, use the
        # control_plane_template from the data-plane template (e.g. allowed = no error/exception).
        if control_plane_template is None and control_plane_template_from_dp is not None:
            control_plane_template = control_plane_template_from_dp

        # For partial_workflow, set context.actual_steps from overrides so hierarchy is built with fewer/wrong-order steps.
        if (
            scenario_context
            and (simulation_goal or "").lower() == "partial_workflow"
            and realistic_overrides
        ):
            actual_steps_list = realistic_overrides.get("actual_steps")
            skip_steps = realistic_overrides.get("skip_steps")
            if isinstance(actual_steps_list, list) and actual_steps_list:
                scenario_context.actual_steps = [str(s) for s in actual_steps_list]
            elif (
                isinstance(skip_steps, list)
                and scenario_context.correct_flow
                and scenario_context.correct_flow.steps
            ):
                steps_copy = list(scenario_context.correct_flow.steps)
                for idx in sorted(
                    (int(i) for i in skip_steps if isinstance(i, (int, float))), reverse=True
                ):
                    if 0 <= idx < len(steps_copy):
                        steps_copy.pop(idx)
                scenario_context.actual_steps = steps_copy

        tags_raw = data.get("tags", [])
        if isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if t is not None]
        elif isinstance(tags_raw, str) and tags_raw.strip():
            tags = [tags_raw.strip()]
        else:
            tags = []

        return Scenario(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            tags=tags,
            tenant_distribution=tenant_dist,
            repeat_count=data.get("repeat_count", 1),
            interval_ms=data.get("interval_ms", 500),
            interval_deviation_ms=float(data.get("interval_deviation_ms", 0)),
            root_step=root_step,
            emit_metrics=data.get("emit_metrics", True),
            emit_logs=data.get("emit_logs", True),
            is_statistical=is_statistical,
            context=scenario_context,
            id_generator=self.get_id_generator(),
            simulation_goal=simulation_goal,
            mcp_server=mcp_server,
            conversation_turns=conversation_turns,
            conversation_turn_pairs=conversation_turn_pairs,
            redaction_applied=redaction_applied or "none",
            realistic_overrides=realistic_overrides or None,
            control_plane_request_outcome=control_plane_request_outcome,
            control_plane_block_reason=control_plane_block_reason,
            control_plane_template=control_plane_template,
            control_plane_policy_exception_override=control_plane_policy_exception_override,
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
        span_type_str = data.get("type", config_span_name("a2a.orchestrate"))
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
