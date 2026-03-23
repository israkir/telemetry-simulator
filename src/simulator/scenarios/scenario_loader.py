"""
Load scenario definitions from YAML.

Scenarios are stored under resource/scenarios/definitions (or --scenarios-dir).
Each file defines name, tenant/agent/mcp_server keys, endusers with turns (optional per-enduser
`mcp_server` override),
latency_profiles, and latency_profile_conditions. All processing logic
is driven by these YAML files and semconv conventions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Example scenario file name (excluded from list when using sample definitions).
EXAMPLE_SCENARIO_NAME = "_EXAMPLE_SCENARIO_"


def _default_definitions_dir() -> Path:
    """Default scenario definitions directory."""
    for base in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
        candidate = base / "resource" / "scenarios" / "definitions"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "resource" / "scenarios" / "definitions"


# Bundled sample definitions path (for tests and default CLI).
SAMPLE_DEFINITIONS_DIR = _default_definitions_dir()


@dataclass
class ScenarioTurn:
    """One user turn: request, response, tool chain, and optional planner/compose."""

    turn_index: int
    request_raw: str = ""
    request_raw_redacted: str = ""
    agent_response: str = ""
    tool_chain: list[str] = field(default_factory=list)
    gen_ai_tool_call_arguments: dict[str, Any] = field(default_factory=dict)
    gen_ai_tool_call_result: dict[str, Any] = field(default_factory=dict)
    planner: dict[str, Any] = field(default_factory=dict)
    response_compose: dict[str, Any] = field(default_factory=dict)
    # Presence-driven trace plan declared in scenario YAML.
    span_plan: dict[str, Any] = field(default_factory=dict)
    # Presence flags (key exists in YAML turn block). Used to decide whether
    # data-plane/response traces should be emitted for partial flows.
    has_agent_response: bool = True
    has_tool_chain: bool = True
    # Optional: OTEL span events to emit, targeted by span-class.
    # Example:
    # span_events:
    #   - target_span_class: request.validation
    #     name: gentoro.enduser.request
    #     attributes:
    #       vendor.enduser.request.raw.redacted: "..."
    #     timestamp_ns: 123456789 (optional)
    span_events: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioEnduser:
    """Enduser: id, pseudo id, channel, turns, and optional stable conversation correlation."""

    id: str
    enduser_pseudo_id: str = ""
    channel: str = ""
    turns: list[ScenarioTurn] = field(default_factory=list)
    # Optional YAML override for single-turn scenarios. When an enduser has multiple turns,
    # ScenarioRunner generates one session + conversation id pair for the whole thread.
    session_id: str = ""
    conversation_id: str = ""
    # When set, overrides scenario-level mcp_server for this enduser (compile + export).
    mcp_server_key: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectFlow:
    """Correct flow steps (ordered tool/step names from scenario)."""

    steps: list[str] = field(default_factory=list)


@dataclass
class ScenarioContext:
    """Scenario context: tenant/agent/mcp_server keys and correct flow."""

    tenant_key: str = ""
    agent_id: str = ""
    mcp_server_key: str = ""
    correct_flow: CorrectFlow = field(default_factory=CorrectFlow)
    workload_weight: float = 1.0
    redaction_applied: str = "none"


@dataclass
class Scenario:
    """Loaded scenario: name, context, endusers, latency config, repeat/interval."""

    name: str
    # Source filename stem (from the YAML path, without directories or extension).
    # Used for stable OpenTelemetry instrumentation scope naming.
    source_filename: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    definition_group: str = ""
    context: ScenarioContext = field(default_factory=ScenarioContext)
    endusers: list[ScenarioEnduser] = field(default_factory=list)
    latency_profiles: dict[str, Any] = field(default_factory=dict)
    latency_profile_conditions: list[dict[str, Any]] = field(default_factory=list)
    default_latency_profile: str = "normal"
    trace_interval_ms: float = 200.0
    trace_interval_deviation_ms: float = 0.0
    repeat_count: int = 1
    emit_metrics: bool = True
    emit_logs: bool = True
    # Derived: conversation_samples synthesized from endusers/turns when loading YAML.
    conversation_samples: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_turn(data: dict[str, Any], turn_index: int) -> ScenarioTurn:
    """Build ScenarioTurn from YAML turn block (vendor-prefixed keys accepted)."""
    # Prefer vendor-prefixed keys; also accept short YAML aliases (e.g. request_raw).
    raw = data.get("vendor.enduser.request.raw") or data.get("request_raw", "")
    redacted = data.get("vendor.enduser.request.raw.redacted") or data.get(
        "request_raw_redacted", raw
    )
    span_events_raw = data.get("span_events") or []
    span_events = span_events_raw if isinstance(span_events_raw, list) else []
    return ScenarioTurn(
        turn_index=turn_index,
        request_raw=raw,
        request_raw_redacted=redacted,
        agent_response=data.get("agent_response", ""),
        tool_chain=data.get("tool_chain") or [],
        gen_ai_tool_call_arguments=data.get("gen_ai.tool.call.arguments") or {},
        gen_ai_tool_call_result=data.get("gen_ai.tool.call.result") or {},
        planner=data.get("planner") or {},
        response_compose=data.get("response_compose") or {},
        span_plan=data.get("span_plan") or {},
        has_agent_response="agent_response" in data,
        has_tool_chain="tool_chain" in data,
        span_events=span_events,
        extra={
            k: v
            for k, v in data.items()
            if k
            not in (
                "turn_index",
                "vendor.enduser.request.raw",
                "request_raw",
                "vendor.enduser.request.raw.redacted",
                "request_raw_redacted",
                "agent_response",
                "tool_chain",
                "gen_ai.tool.call.arguments",
                "gen_ai.tool.call.result",
                "planner",
                "response_compose",
                "span_plan",
                "span_events",
            )
        },
    )


def _parse_enduser(data: dict[str, Any]) -> ScenarioEnduser:
    """Build ScenarioEnduser from YAML enduser block."""
    pseudo = data.get("vendor.enduser.pseudo.id") or data.get("enduser_pseudo_id", "")
    session_id = str(data.get("vendor.session.id") or data.get("session_id") or "").strip()
    conversation_id = str(
        data.get("gen_ai.conversation.id") or data.get("conversation_id") or ""
    ).strip()
    turns = []
    for i, t in enumerate(data.get("turns") or []):
        idx = t.get("turn_index", i + 1)
        turns.append(_parse_turn(t, idx))
    excluded = {
        "id",
        "channel",
        "turns",
        "vendor.enduser.pseudo.id",
        "enduser_pseudo_id",
        "vendor.session.id",
        "session_id",
        "gen_ai.conversation.id",
        "conversation_id",
        "mcp_server",
    }
    return ScenarioEnduser(
        id=data.get("id", ""),
        enduser_pseudo_id=pseudo,
        channel=data.get("channel", ""),
        turns=turns,
        session_id=session_id,
        conversation_id=conversation_id,
        mcp_server_key=str(data.get("mcp_server", "") or "").strip(),
        extra={k: v for k, v in data.items() if k not in excluded},
    )


def _build_correct_flow(endusers: list[ScenarioEnduser]) -> CorrectFlow:
    """Derive correct_flow.steps from all turns' tool_chains."""
    steps: list[str] = []
    for eu in endusers:
        for t in eu.turns:
            steps.extend(t.tool_chain)
    return CorrectFlow(steps=steps)


def _build_conversation_samples(scenario: Scenario) -> list[dict[str, Any]]:
    """Derive conversation_samples from endusers/turns."""
    samples: list[dict[str, Any]] = []
    for eu in scenario.endusers:
        for t in eu.turns:
            samples.append(
                {
                    "enduser_id": eu.id,
                    "enduser_pseudo_id": eu.enduser_pseudo_id,
                    "turn_index": t.turn_index,
                    "request_raw": t.request_raw,
                    "agent_response": t.agent_response,
                    "tool_chain": t.tool_chain,
                    "gen_ai_tool_call_arguments": t.gen_ai_tool_call_arguments,
                    "gen_ai_tool_call_result": t.gen_ai_tool_call_result,
                }
            )
    return samples


def load_scenario_yaml(path: Path, scenarios_dir: Path | None = None) -> Scenario:
    """Load a single scenario from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = scenarios_dir or path.parent

    name = data.get("name", path.stem)
    source_filename = path.stem
    tenant_key = data.get("tenant", "")
    agent_id = data.get("agent", "")
    mcp_server_key = data.get("mcp_server", "")
    endusers_data = data.get("endusers") or []
    endusers = [_parse_enduser(e) for e in endusers_data]
    correct_flow = _build_correct_flow(endusers)
    redaction = data.get("vendor.redaction.applied") or "none"

    definition_group = ""
    try:
        path_relative = path.resolve().relative_to(base.resolve())
        if len(path_relative.parts) > 1:
            definition_group = path_relative.parts[0]
    except (ValueError, TypeError):
        pass

    scenario = Scenario(
        name=name,
        source_filename=source_filename,
        description=data.get("description", ""),
        tags=data.get("tags") or [],
        definition_group=definition_group,
        context=ScenarioContext(
            tenant_key=tenant_key,
            agent_id=agent_id,
            mcp_server_key=mcp_server_key,
            correct_flow=correct_flow,
            workload_weight=float(data.get("workload_weight", 1.0)),
            redaction_applied=redaction,
        ),
        endusers=endusers,
        latency_profiles=data.get("latency_profiles") or {},
        latency_profile_conditions=data.get("latency_profile_conditions") or [],
        default_latency_profile=data.get("default_latency_profile") or "normal",
        trace_interval_ms=float(data.get("trace_interval_ms", 200)),
        trace_interval_deviation_ms=float(data.get("trace_interval_deviation_ms", 0)),
        repeat_count=int(data.get("repeat_count", 1)),
        emit_metrics=data.get("emit_metrics", True),
        emit_logs=data.get("emit_logs", True),
        raw=data,
    )
    scenario.conversation_samples = _build_conversation_samples(scenario)

    from ..config import load_config
    from .tool_payload_validation import validate_scenario_tool_payloads

    validate_scenario_tool_payloads(scenario, load_config())
    return scenario


class ScenarioLoader:
    """Load scenarios from a definitions directory."""

    def __init__(self, scenarios_dir: str | Path | None = None):
        self.scenarios_dir = Path(scenarios_dir) if scenarios_dir else SAMPLE_DEFINITIONS_DIR

    def list_scenarios(self) -> list[str]:
        """List scenario names (file stem) excluding example when using sample dir."""
        names: list[str] = []
        if not self.scenarios_dir.is_dir():
            return names
        for path in sorted(self.scenarios_dir.rglob("*.yaml")):
            name = path.stem
            if name == EXAMPLE_SCENARIO_NAME and self.scenarios_dir == SAMPLE_DEFINITIONS_DIR:
                continue
            names.append(name)
        return names

    def load(self, name: str) -> Scenario:
        """Load scenario by name (file stem). Raises FileNotFoundError if not found."""
        if not self.scenarios_dir.is_dir():
            raise FileNotFoundError(
                f"Scenario not found: {name} (dir missing: {self.scenarios_dir})"
            )
        lookup_names = {name}
        if "_multi_turn" in name:
            lookup_names.add(name.replace("_multi_turn", "_multiturn"))
        if name.startswith("new_"):
            legacy_name = name[len("new_") :]
            lookup_names.add(legacy_name)
            if "_multi_turn" in legacy_name:
                lookup_names.add(legacy_name.replace("_multi_turn", "_multiturn"))
        # Prefer exact path, then any file with stem matching name (e.g. normal/single_tool_call.yaml)
        direct = self.scenarios_dir / f"{name}.yaml"
        if direct.is_file():
            return load_scenario_yaml(direct, self.scenarios_dir)
        for path in sorted(self.scenarios_dir.rglob("*.yaml")):
            if path.stem in lookup_names:
                return load_scenario_yaml(path, self.scenarios_dir)
        raise FileNotFoundError(f"Scenario not found: {name} (looked in {self.scenarios_dir})")

    def load_all(self) -> list[Scenario]:
        """Load all scenarios (excluding example when using sample dir)."""
        names = self.list_scenarios()
        result: list[Scenario] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            try:
                result.append(self.load(name))
            except (FileNotFoundError, yaml.YAMLError):
                continue
        return result
