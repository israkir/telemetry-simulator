"""Tests for scenario compiler and dependency_rules validation."""

import dataclasses
import json
from pathlib import Path

import pytest

from simulator import config as sim_config
from simulator.config import load_config
from simulator.defaults import get_default_tenant_ids
from simulator.exporters.file_exporter import FileSpanExporter
from simulator.resource_loader import load_resource_presets
from simulator.scenarios.compiler import compile_turn
from simulator.scenarios.config_resolver import resolve_context
from simulator.scenarios.dependency_rules import validate_trace_graph
from simulator.scenarios.id_generator import generate_ids_for_turn
from simulator.scenarios.latency import LatencyModel
from simulator.scenarios.scenario_loader import ScenarioLoader
from simulator.scenarios.scenario_runner import ScenarioRunner


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _resolve_ctx():
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    default_tenant_id = get_default_tenant_ids()[0] if get_default_tenant_ids() else None
    return resolve_context(
        config,
        tenant_key=scenario.context.tenant_key,
        agent_id=scenario.context.agent_id,
        mcp_server_key=scenario.context.mcp_server_key,
        default_tenant_id=default_tenant_id,
    )


def test_compile_turn_produces_three_traces() -> None:
    """compile_turn returns TraceGraphSpec with cp_request, data_plane, cp_response."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario, enduser, turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    assert graph.cp_request.spans
    assert graph.data_plane.spans
    assert graph.cp_response.spans
    assert graph.compiled_turn.scenario_name == "single_tool_call"
    assert graph.compiled_turn.tool_chain == ["new_claim"]


def test_multiturn_enduser_shares_session_when_using_runner_style_shared_ids() -> None:
    """Same session + conversation id for every turn; distinct request ids (runner contract)."""
    loader = ScenarioLoader()
    scenario = loader.load("new_claim_phone_multi_turn")
    assert scenario.name == "claim_phone_multiturn"
    assert len(scenario.endusers[0].turns) == 4
    config = load_config()
    default_tenant_id = get_default_tenant_ids()[0] if get_default_tenant_ids() else None
    ctx = resolve_context(
        config,
        tenant_key=scenario.context.tenant_key,
        agent_id=scenario.context.agent_id,
        mcp_server_key=scenario.context.mcp_server_key,
        default_tenant_id=default_tenant_id,
    )
    latency_model = LatencyModel.from_scenario(scenario)
    eu = scenario.endusers[0]
    shared = generate_ids_for_turn(config, tool_count=0)
    sess, conv = shared["session_id"], shared["conversation_id"]
    request_ids: list[str] = []
    for turn in eu.turns:
        graph = compile_turn(
            scenario,
            eu,
            turn,
            resolved_ctx=ctx,
            latency_model=latency_model,
            config=config,
            shared_session_id=sess,
            shared_conversation_id=conv,
        )
        assert graph.compiled_turn.session_id == sess
        assert graph.compiled_turn.conversation_id == conv
        request_ids.append(graph.compiled_turn.request_id)
    assert len(set(request_ids)) == 4


def test_data_plane_root_span_has_redaction_and_orchestration_attrs() -> None:
    """A2A orchestration root should include semconv correlation + redaction attrs."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    assert graph.data_plane.spans, "expected data_plane spans"
    a2a_span = graph.data_plane.spans[0]

    prefix = sim_config.ATTR_PREFIX
    assert f"{prefix}.request.id" in a2a_span.attributes
    assert f"{prefix}.redaction.applied" in a2a_span.attributes
    assert a2a_span.attributes[f"{prefix}.redaction.applied"] == scenario.context.redaction_applied
    assert f"{prefix}.orchestration.duration_ms" in a2a_span.attributes
    assert (
        a2a_span.attributes[f"{prefix}.orchestration.duration_ms"] == a2a_span.duration_ms
    ), "orchestration.duration_ms span attribute should match sampled span duration"


def test_error_type_propagates_to_a2a_orchestrate_on_cp_runtime_error(
    tmp_path: Path,
) -> None:
    """If CP hits runtime error, data-plane orchestration root should get error.type."""
    scenario_yaml = """
name: test_cp_runtime_error_propagates_to_a2a
description: test
tags: [control-plane, error]
tenant: toro
agent: toro-customer-assistant-001
mcp_server: phone
vendor.redaction.applied: none
workload_weight: 1.0
trace_interval_ms: 200
trace_interval_deviation_ms: 0
repeat_count: 1
latency_profiles:
  normal:
    span_durations_ms:
      request.validation: { mean: 20, deviation: 1 }
      payload.validation: { mean: 10, deviation: 1 }
      policy.validation: { mean: 10, deviation: 1 }
      a2a.orchestrate: { mean: 30, deviation: 1 }
      response.validation: { mean: 10, deviation: 1 }
default_latency_profile: normal
endusers:
  - id: user_x
    vendor.enduser.pseudo.id: "user_x12345"
    turns:
      - turn_index: 1
        span_plan:
          cp_request:
            root: request.validation
            children: [payload.validation, policy.validation]
          data_plane:
            root: a2a.orchestrate
            children: []
        span_events:
          - target_span_class: request.validation
            name: gentoro.request.error
            attributes:
              vendor.request.outcome: error
              error.type: planning_timeout
          - target_span_class: payload.validation
            name: gentoro.validation.payload.result
            attributes:
              vendor.step.outcome: pass
              vendor.validation.result: valid
          - target_span_class: policy.validation
            name: gentoro.validation.policy.decision
            attributes:
              vendor.step.outcome: pass
              vendor.policy.engine: dlp
              vendor.policy.decision: allow
        vendor.enduser.request.raw: "test"
"""

    from simulator.scenarios.scenario_loader import load_scenario_yaml

    path = tmp_path / "test_cp_error.yaml"
    path.write_text(scenario_yaml, encoding="utf-8")
    scenario = load_scenario_yaml(path, scenarios_dir=tmp_path)

    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    assert graph.data_plane.spans, "expected data-plane orchestration root"
    a2a_span = graph.data_plane.spans[0]
    assert a2a_span.attributes.get("error.type") == "planning_timeout"


def test_llm_call_span_has_genai_attributes_and_tool_events() -> None:
    """LLM call span should include llm_call_model attributes (events disabled for now)."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    # Second enduser: successful new_claim (first enduser exercises ambiguous scheduling_time_resolution failure).
    enduser = scenario.endusers[1]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    dp_spans = graph.data_plane.spans
    assert dp_spans
    prefix = sim_config.ATTR_PREFIX

    llm_spans = [
        s for s in dp_spans if s.attributes.get(f"{prefix}.span.class") == "llm.call"
    ]
    assert len(llm_spans) == 1, "expected a single llm.call span"
    llm = llm_spans[0]

    assert llm.attributes["gen_ai.request.type"] == "completion"
    assert llm.attributes["gen_ai.response.finish_reason"] == "stop"
    assert llm.attributes["gen_ai.usage.input_tokens"] == 920
    assert llm.attributes["gen_ai.usage.output_tokens"] == 155
    # Inferred from scenario turn_index (1) and one captured user message on the span.
    assert llm.attributes[f"{prefix}.llm.turn.count"] == 1
    assert llm.attributes[f"{prefix}.llm.tool.request.count"] == 1
    assert llm.attributes[f"{prefix}.llm.streaming"] is False

    assert llm.attributes[f"{prefix}.llm.content.capture.enabled"] is True
    assert llm.attributes[f"{prefix}.llm.content.redaction.enabled"] is False
    in_msgs = json.loads(llm.attributes["gen_ai.input.messages"])
    out_msgs = json.loads(llm.attributes["gen_ai.output.messages"])
    assert in_msgs[0]["role"] == "user"
    assert in_msgs[0]["content"][0]["text"].startswith("My mobile was in my bag with my keys")
    assert out_msgs[0]["role"] == "assistant"
    assert "CLAIM-PH-8102" in out_msgs[0]["content"][0]["text"]

    assert llm.events == [], "llm.call span events are disabled for now"


def test_llm_turn_count_matches_longer_gen_ai_input_messages() -> None:
    """When YAML supplies a full message list longer than the structural minimum, use that length."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]
    six_msgs = [{"role": "user", "content": [{"type": "text", "text": f"m{i}"}]} for i in range(6)]
    new_turn = dataclasses.replace(turn, extra={**turn.extra, "gen_ai.input.messages": six_msgs})
    new_eu = dataclasses.replace(enduser, turns=[new_turn])

    graph = compile_turn(
        scenario,
        new_eu,
        new_turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    prefix = sim_config.ATTR_PREFIX
    llm = next(
        s
        for s in graph.data_plane.spans
        if s.attributes.get(f"{prefix}.span.class") == "llm.call"
    )
    assert llm.attributes[f"{prefix}.llm.turn.count"] == 6


def test_llm_turn_count_yaml_explicit_overrides_inference() -> None:
    """gentoro.llm.turn.count in scenario YAML must win over structural/message-derived values."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]
    new_turn = dataclasses.replace(turn, extra={**turn.extra, "gentoro.llm.turn.count": 9})
    new_eu = dataclasses.replace(enduser, turns=[new_turn])

    graph = compile_turn(
        scenario,
        new_eu,
        new_turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    prefix = sim_config.ATTR_PREFIX
    llm = next(
        s
        for s in graph.data_plane.spans
        if s.attributes.get(f"{prefix}.span.class") == "llm.call"
    )
    assert llm.attributes[f"{prefix}.llm.turn.count"] == 9


def test_mcp_tool_execute_span_has_retry_and_genai_call_attrs() -> None:
    """mcp.tool.execute span should include gen_ai.tool.call.id and retry.* attributes."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    dp_spans = graph.data_plane.spans
    prefix = sim_config.ATTR_PREFIX
    mcp_spans = [
        s for s in dp_spans if s.attributes.get(f"{prefix}.span.class") == "mcp.tool.execute"
    ]
    assert len(mcp_spans) == 1
    mcp = mcp_spans[0]

    assert "gen_ai.tool.name" in mcp.attributes
    assert "gen_ai.tool.call.id" in mcp.attributes
    assert isinstance(mcp.attributes["gen_ai.tool.call.id"], str)
    assert f"{prefix}.mcp.tool.call.id" not in mcp.attributes

    assert f"{prefix}.retry.count" in mcp.attributes
    assert mcp.attributes[f"{prefix}.retry.count"] == 0
    assert f"{prefix}.retry.policy" in mcp.attributes
    assert mcp.attributes[f"{prefix}.retry.policy"] == "none"
    assert mcp.attributes.get(f"{prefix}.llm.tool.execution.count") == 1


def test_mcp_tool_retries_scenario_emits_attempt_indices_and_retry_counts() -> None:
    """mcp_tool_retries in turn YAML yields N attempt spans, parent retry.count = N-1, execution indices 1..M."""
    loader = ScenarioLoader()
    scenario = loader.load("mcp_tool_retries_booking")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    prefix = sim_config.ATTR_PREFIX

    graph_slots = compile_turn(
        scenario,
        enduser,
        enduser.turns[0],
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    validate_trace_graph(graph_slots)
    dp_slots = graph_slots.data_plane.spans
    mcp_slots = [s for s in dp_slots if s.attributes.get(f"{prefix}.span.class") == "mcp.tool.execute"]
    assert len(mcp_slots) == 1
    assert mcp_slots[0].attributes["gen_ai.tool.name"] == "get_available_slots"
    assert mcp_slots[0].attributes[f"{prefix}.retry.count"] == 0
    slot_attempts_only = [
        s
        for s in dp_slots
        if s.attributes.get(f"{prefix}.span.class") == "mcp.tool.execute.attempt"
    ]
    assert len(slot_attempts_only) == 1

    graph = compile_turn(
        scenario,
        enduser,
        enduser.turns[1],
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    validate_trace_graph(graph)

    dp = graph.data_plane.spans
    mcp_parents = [s for s in dp if s.attributes.get(f"{prefix}.span.class") == "mcp.tool.execute"]
    assert len(mcp_parents) == 1
    assert mcp_parents[0].attributes["gen_ai.tool.name"] == "update_appointment"
    assert mcp_parents[0].attributes[f"{prefix}.retry.count"] == 2
    assert mcp_parents[0].attributes[f"{prefix}.retry.policy"] == "exponential_jitter"
    assert mcp_parents[0].attributes.get(f"{prefix}.llm.tool.execution.count") == 1

    attempts = [s for s in dp if s.attributes.get(f"{prefix}.span.class") == "mcp.tool.execute.attempt"]
    assert len(attempts) == 3

    book_attempts = [
        s
        for s in attempts
        if s.attributes.get("gen_ai.tool.name") == "update_appointment"
    ]
    assert len(book_attempts) == 3
    assert [a.attributes[f"{prefix}.mcp.attempt.index"] for a in book_attempts] == [1, 2, 3]
    assert book_attempts[0].attributes[f"{prefix}.mcp.attempt.outcome"] == "fail"
    assert book_attempts[0].attributes.get("error.type") == "unavailable"
    assert book_attempts[0].status_code == "ERROR"
    assert book_attempts[1].attributes[f"{prefix}.mcp.attempt.outcome"] == "fail"
    assert book_attempts[1].attributes.get("error.type") == "timeout"
    assert book_attempts[1].attributes.get(f"{prefix}.retry.reason") == "timeout"
    assert book_attempts[1].attributes.get(f"{prefix}.retry.backoff.ms") == 480
    assert book_attempts[1].status_code == "ERROR"
    assert book_attempts[2].attributes[f"{prefix}.mcp.attempt.outcome"] == "success"
    assert book_attempts[2].attributes.get(f"{prefix}.retry.backoff.ms") == 620
    assert book_attempts[2].status_code == "UNSET"
    assert "gen_ai.tool.call.result" in book_attempts[2].attributes


def test_mcp_tool_retries_exhausted_all_attempts_fail() -> None:
    """After retries, logical MCP call fails: parent step.fail, error.type from last attempt."""
    loader = ScenarioLoader()
    scenario = loader.load("mcp_tool_retries_exhausted_failed")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[1]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    validate_trace_graph(graph)

    prefix = sim_config.ATTR_PREFIX
    dp = graph.data_plane.spans
    mcp_parents = [s for s in dp if s.attributes.get(f"{prefix}.span.class") == "mcp.tool.execute"]
    assert len(mcp_parents) == 1
    pay_parent = next(s for s in mcp_parents if s.attributes.get("gen_ai.tool.name") == "pay")
    assert pay_parent.attributes[f"{prefix}.step.outcome"] == "fail"
    assert pay_parent.attributes[f"{prefix}.retry.count"] == 2
    assert pay_parent.attributes.get("error.type") == "tool_error"
    assert pay_parent.status_code == "ERROR"

    pay_attempts = [
        s
        for s in dp
        if s.attributes.get(f"{prefix}.span.class") == "mcp.tool.execute.attempt"
        and s.attributes.get("gen_ai.tool.name") == "pay"
    ]
    assert len(pay_attempts) == 3
    assert all(s.attributes[f"{prefix}.mcp.attempt.outcome"] == "fail" for s in pay_attempts)
    assert pay_attempts[-1].attributes.get("error.type") == "tool_error"
    assert all(s.status_code == "ERROR" for s in pay_attempts)
    assert "gen_ai.tool.call.result" not in pay_attempts[-1].attributes


def test_validate_trace_graph_passes_for_valid_graph() -> None:
    """validate_trace_graph does not raise for a valid compiled graph."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]
    graph = compile_turn(
        scenario, enduser, turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    validate_trace_graph(graph)


def test_request_validation_root_span_includes_required_and_optional_attrs() -> None:
    """Request validation root includes semconv-required and scenario/config-driven optional attrs."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario, enduser, turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    root = graph.cp_request.spans[0]
    attrs = root.attributes
    compiled = graph.compiled_turn

    # Required for request.validation root
    assert attrs["vendor.span.class"] == "request.validation"
    assert attrs["vendor.request.id"] == compiled.request_id
    assert attrs["vendor.request.outcome"] == "allowed"
    assert attrs["vendor.tenant.id"] == compiled.tenant_id

    # Optional attributes filled from scenario/config-derived context when present
    assert attrs["gen_ai.conversation.id"] == compiled.conversation_id
    assert attrs["vendor.enduser.pseudo.id"] == compiled.enduser_id
    assert attrs["vendor.a2a.agent.target.id"] == compiled.agent_id


def test_request_validation_request_id_is_unique_per_compiled_request() -> None:
    """Each compile_turn call emits a distinct request id on request.validation root."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph_1 = compile_turn(
        scenario, enduser, turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    graph_2 = compile_turn(
        scenario, enduser, turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    req_id_1 = graph_1.cp_request.spans[0].attributes["vendor.request.id"]
    req_id_2 = graph_2.cp_request.spans[0].attributes["vendor.request.id"]

    assert req_id_1
    assert req_id_2
    assert req_id_1 != req_id_2


def test_augmentation_span_includes_expected_augment_action_attributes() -> None:
    """Augmentation span includes the expected low-cardinality augment action attributes."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario, enduser, turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    augmentation_span = next(
        s for s in graph.cp_request.spans if s.name == "vendor.augmentation.validation"
    )
    attrs = augmentation_span.attributes
    assert attrs["vendor.augment.conversation_id.action"] == "propagated"
    assert attrs["vendor.augment.request_id.action"] == "created"
    assert attrs["vendor.augment.enduser_id.action"] == "missing"
    assert attrs["vendor.augment.target_agent_id.action"] == "attached"


def test_validate_trace_graph_fails_on_tool_sequence_mismatch() -> None:
    """validate_trace_graph raises when mcp.tool.execute spans don't match tool_chain."""
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]
    graph = compile_turn(
        scenario, enduser, turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )
    # Corrupt: set compiled_turn.tool_chain to something that doesn't match data_plane spans
    graph.compiled_turn.tool_chain = ["other_tool"]
    with pytest.raises(ValueError, match="Tool sequence mismatch"):
        validate_trace_graph(graph)


def test_request_validation_span_has_user_prompt_event(tmp_path: Path) -> None:
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    scenario.repeat_count = 1

    spans_path = tmp_path / "spans.jsonl"
    runner = ScenarioRunner(
        trace_exporter=FileSpanExporter(spans_path, append=False),
        metric_exporter=None,
        log_exporter=None,
    )
    runner.run_scenario(scenario)
    runner.shutdown()

    spans = _read_jsonl(spans_path)
    request_validation = [s for s in spans if s.get("name") == "vendor.request.validation"]
    assert request_validation, "expected a vendor.request.validation span"

    events = request_validation[0].get("events") or []
    prompt_event = [e for e in events if e.get("name") == "gentoro.enduser.request"]
    assert prompt_event, "expected gentoro.enduser.request span event on request.validation"

    attrs = prompt_event[0].get("attributes") or {}
    assert "vendor.enduser.request.raw" in attrs
    assert "open a claim" in str(attrs["vendor.enduser.request.raw"])

    span_resource = request_validation[0].get("resource") or {}
    cp_attrs = (load_resource_presets().get("control-plane") or {}).get("attributes") or {}
    expected_tenant_id = get_default_tenant_ids()[0] if get_default_tenant_ids() else None
    for key, value in cp_attrs.items():
        assert span_resource.get(key) == value
    if expected_tenant_id is not None:
        assert span_resource.get("vendor.tenant.id") == expected_tenant_id
    assert span_resource.get("telemetry.sdk.language") == "python"
    assert span_resource.get("telemetry.sdk.name") == "opentelemetry"
    assert "telemetry.sdk.version" in span_resource


def test_response_validation_span_has_agent_response_event(tmp_path: Path) -> None:
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    scenario.repeat_count = 1

    spans_path = tmp_path / "spans.jsonl"
    runner = ScenarioRunner(
        trace_exporter=FileSpanExporter(spans_path, append=False),
        metric_exporter=None,
        log_exporter=None,
    )
    runner.run_scenario(scenario)
    runner.shutdown()

    spans = _read_jsonl(spans_path)
    response_validation = [s for s in spans if s.get("name") == "vendor.response.validation"]
    assert response_validation, "expected a vendor.response.validation span"

    events = response_validation[0].get("events") or []
    agent_event = [e for e in events if e.get("name") == "gentoro.agent.response"]
    assert agent_event, "expected gentoro.agent.response span event on response.validation"

    attrs = agent_event[0].get("attributes") or {}
    assert "vendor.agent.response.raw" in attrs
    agent_raws: list[str] = []
    for rv in response_validation:
        for ev in rv.get("events") or []:
            if ev.get("name") != "gentoro.agent.response":
                continue
            raw = (ev.get("attributes") or {}).get("vendor.agent.response.raw")
            if raw is not None:
                agent_raws.append(str(raw))
    assert any("CLAIM-PH-8102" in t for t in agent_raws), "expected a successful claim id somewhere in the run"


def test_tool_recommendation_span_has_tool_selection_event(tmp_path: Path) -> None:
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    scenario.repeat_count = 1

    spans_path = tmp_path / "spans.jsonl"
    runner = ScenarioRunner(
        trace_exporter=FileSpanExporter(spans_path, append=False),
        metric_exporter=None,
        log_exporter=None,
    )
    runner.run_scenario(scenario)
    runner.shutdown()

    spans = _read_jsonl(spans_path)
    tools_recommend = [s for s in spans if s.get("name") == "vendor.tools.recommend"]
    assert tools_recommend, "expected a vendor.tools.recommend span"

    tr_attrs = tools_recommend[0].get("attributes") or {}
    assert tr_attrs.get("gen_ai.tool.name") == "new_claim"

    events = tools_recommend[0].get("events") or []
    tool_selection = [e for e in events if e.get("name") == "gentoro.agent.tool_selection"]
    assert tool_selection, "expected gentoro.agent.tool_selection event on tools.recommend span"

    attrs = tool_selection[0].get("attributes") or {}
    assert "new_claim" in str(attrs.get("vendor.agent.tool_selection.tool.plan", ""))
    assert "product_type" in str(attrs.get("vendor.agent.tool_selection.tool.plan", ""))


def test_compile_turn_honors_span_plan_presence_for_invalid_payload_blocked() -> None:
    """span_plan presence controls emitted trace segments/classes."""
    loader = ScenarioLoader()
    scenario = loader.load("invalid_payload_blocked")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    span_plan = getattr(turn, "span_plan", {})
    assert "cp_request" in span_plan
    assert "data_plane" not in span_plan
    assert "cp_response" not in span_plan

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    assert graph.cp_request.spans, "cp_request trace should still be emitted"
    req_classes = [s.attributes.get("vendor.span.class") for s in graph.cp_request.spans]
    assert req_classes == ["request.validation", "payload.validation"]
    assert not any(
        s.attributes.get("vendor.span.class") == "augmentation.validation"
        for s in graph.cp_request.spans
    ), "augmentation.validation should not be emitted when policy does not pass"
    assert graph.data_plane.spans == [], "data_plane should not be emitted when data_plane plan is absent"
    assert graph.cp_response.spans == [], "cp_response should not be emitted when cp_response plan is absent"


def test_policy_validation_span_is_enforced_for_allowed_request(tmp_path: Path) -> None:
    """If request.outcome is allowed, policy.validation must exist even if span_plan omits it."""
    from simulator.scenarios.scenario_loader import load_scenario_yaml

    scenario_yaml = """
name: allowed_span_plan_omits_policy
description: test enforcement
tags: [control-plane]
tenant: toro
agent: toro-customer-assistant-001
mcp_server: phone
vendor.redaction.applied: none
workload_weight: 1.0
trace_interval_ms: 200
trace_interval_deviation_ms: 0
repeat_count: 1
latency_profiles:
  normal:
    span_durations_ms:
      request.validation: { mean: 10, deviation: 1 }
      payload.validation: { mean: 5, deviation: 1 }
      policy.validation: { mean: 7, deviation: 1 }
      augmentation.validation: { mean: 6, deviation: 1 }
default_latency_profile: normal
endusers:
  - id: user_a
    vendor.enduser.pseudo.id: "user_x"
    turns:
      - turn_index: 1
        span_plan:
          cp_request:
            root: request.validation
            children: [payload.validation, policy.validation]
        span_events:
          - target_span_class: request.validation
            name: gentoro.enduser.request
            attributes:
              vendor.enduser.request.raw: "Hello"
        vendor.enduser.request.raw: "Hello"
""".strip()

    path = tmp_path / "allowed_span_plan_omits_policy.yaml"
    path.write_text(scenario_yaml, encoding="utf-8")
    scenario = load_scenario_yaml(path, scenarios_dir=tmp_path)

    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    classes = [s.attributes.get("vendor.span.class") for s in graph.cp_request.spans]
    assert classes == [
        "request.validation",
        "payload.validation",
        "policy.validation",
        "augmentation.validation",
    ]

    policy = next(s for s in graph.cp_request.spans if s.attributes.get("vendor.span.class") == "policy.validation")
    assert policy.attributes.get("vendor.step.outcome") == "pass"
    assert policy.attributes.get("vendor.policy.decision") == "allow"
    assert policy.parent_index == 0, "policy.validation must be a direct child of request.validation"


def test_response_compose_span_is_enforced_for_successful_a2a_when_omitted(tmp_path: Path) -> None:
    """If a2a.orchestrate is success, response.compose must exist even if span_plan omits it."""
    from simulator.scenarios.scenario_loader import load_scenario_yaml

    scenario_yaml = """
name: data_plane_compose_enforcement
description: test enforcement
tags: [data-plane]
tenant: toro
agent: toro-customer-assistant-001
mcp_server: phone
vendor.redaction.applied: none
workload_weight: 1.0
trace_interval_ms: 200
trace_interval_deviation_ms: 0
repeat_count: 1
latency_profiles:
  normal:
    span_durations_ms:
      a2a.orchestrate: { mean: 10, deviation: 1 }
      planner: { mean: 5, deviation: 1 }
      response.compose: { mean: 7, deviation: 1 }
default_latency_profile: normal
endusers:
  - id: user_a
    vendor.enduser.pseudo.id: "user_x"
    turns:
      - turn_index: 1
        span_plan:
          data_plane:
            root: a2a.orchestrate
            children: [planner]
        # tool_chain intentionally omitted for this test
""".strip()

    path = tmp_path / "data_plane_compose_enforcement.yaml"
    path.write_text(scenario_yaml, encoding="utf-8")
    scenario = load_scenario_yaml(path, scenarios_dir=tmp_path)

    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    compose = next(
        s for s in graph.data_plane.spans if s.attributes.get("vendor.span.class") == "response.compose"
    )
    assert compose.parent_index == 0
    assert compose.kind == "INTERNAL"


def test_invalid_payload_validation_sets_error_status_but_request_root_unset() -> None:
    loader = ScenarioLoader()
    scenario = loader.load("invalid_payload_blocked")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    req_root = graph.cp_request.spans[0]
    payload = next(s for s in graph.cp_request.spans if s.attributes.get("vendor.span.class") == "payload.validation")

    assert req_root.status_code == "UNSET"
    assert payload.status_code == "ERROR"


def test_policy_engine_runtime_exception_sets_error_on_request_and_policy() -> None:
    loader = ScenarioLoader()
    scenario = loader.load("policy_engine_runtime_exception")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    req_root = graph.cp_request.spans[0]
    payload = next(s for s in graph.cp_request.spans if s.attributes.get("vendor.span.class") == "payload.validation")
    policy = next(s for s in graph.cp_request.spans if s.attributes.get("vendor.span.class") == "policy.validation")

    assert req_root.status_code == "ERROR"
    assert payload.status_code == "UNSET"
    assert policy.status_code == "ERROR"


def test_render_trace_produces_trace_id() -> None:
    """render_trace returns a 32-char hex trace_id when given a non-empty TraceSpec."""
    from opentelemetry.sdk.trace import TracerProvider

    from simulator.generators.trace_generator import SpanSpec, TraceSpec, render_trace

    spec = TraceSpec(spans=[
        SpanSpec("test.span", parent_index=-1, kind="INTERNAL", attributes={"foo": "bar"}),
    ])
    provider = TracerProvider()
    tracer = provider.get_tracer("test", "1.0.0")
    trace_id = render_trace(tracer, spec)
    assert trace_id is not None
    assert len(trace_id) == 32


def test_render_trace_propagates_session_id_to_all_spans() -> None:
    """Renderer copies configured-prefix session id onto spans that omit it."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ReadableSpan,
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    from simulator.generators.trace_generator import SpanSpec, TraceSpec, render_trace_with_span_ids

    class _Collect(SpanExporter):
        def __init__(self) -> None:
            self.spans: list[ReadableSpan] = []

        def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

    prefix = sim_config.ATTR_PREFIX
    session_key = f"{prefix}.session.id"
    sc = f"{prefix}.span.class"
    exporter = _Collect()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test", "1.0.0")

    spec = TraceSpec(
        spans=[
            SpanSpec(
                f"{prefix}.request.validation",
                -1,
                "SERVER",
                {
                    sc: "request.validation",
                    session_key: "sess-from-root",
                    f"{prefix}.tenant.id": "tenant-1",
                },
            ),
            SpanSpec(
                f"{prefix}.validation.payload",
                0,
                "INTERNAL",
                {sc: "payload.validation", f"{prefix}.step.outcome": "pass"},
            ),
        ]
    )
    render_trace_with_span_ids(tracer, spec)
    assert len(exporter.spans) == 2
    for rs in exporter.spans:
        attrs = dict(rs.attributes)
        assert attrs.get(session_key) == "sess-from-root"


def test_render_trace_propagates_mcp_server_uuid_to_tools_recommend_and_attempt() -> None:
    """tools.recommend / mcp.tool.execute.attempt get mcp.server.uuid from elsewhere in the trace."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ReadableSpan,
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    from simulator.generators.trace_generator import SpanSpec, TraceSpec, render_trace_with_span_ids

    class _Collect(SpanExporter):
        def __init__(self) -> None:
            self.spans: list[ReadableSpan] = []

        def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

    prefix = sim_config.ATTR_PREFIX
    sc = f"{prefix}.span.class"
    exporter = _Collect()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test", "1.0.0")

    # tools.recommend listed before mcp.tool.execute — export must still resolve server from later span.
    mcp_srv = f"{prefix}.mcp.server.uuid"
    mcp_tool = f"{prefix}.mcp.tool.uuid"
    spec = TraceSpec(
        spans=[
            SpanSpec(
                f"{prefix}.tools.recommend",
                -1,
                "INTERNAL",
                {
                    sc: "tools.recommend",
                    f"{prefix}.session.id": "s1",
                    f"{prefix}.mcp.tools.available.count": 3,
                    f"{prefix}.mcp.tools.selected.count": 1,
                },
            ),
            SpanSpec(
                f"{prefix}.mcp.tool.execute",
                -1,
                "CLIENT",
                {
                    sc: "mcp.tool.execute",
                    f"{prefix}.session.id": "s1",
                    mcp_srv: "srv-uuid-aaaaaaaa",
                    mcp_tool: "tool-uuid-bbbbbbbb",
                    "gen_ai.tool.call.id": "c1",
                },
            ),
            SpanSpec(
                f"{prefix}.mcp.tool.execute.attempt",
                1,
                "CLIENT",
                {
                    sc: "mcp.tool.execute.attempt",
                    f"{prefix}.session.id": "s1",
                    "gen_ai.tool.call.id": "c1",
                    f"{prefix}.mcp.attempt.index": 1,
                    f"{prefix}.mcp.attempt.outcome": "success",
                },
            ),
        ]
    )
    render_trace_with_span_ids(tracer, spec)
    by_name = {rs.name: dict(rs.attributes) for rs in exporter.spans}
    assert by_name[f"{prefix}.tools.recommend"].get(mcp_srv) == "srv-uuid-aaaaaaaa"
    assert by_name[f"{prefix}.mcp.tool.execute.attempt"].get(mcp_srv) == "srv-uuid-aaaaaaaa"
    assert by_name[f"{prefix}.mcp.tool.execute.attempt"].get(mcp_tool) == "tool-uuid-bbbbbbbb"


def test_request_validation_root_outcome_is_blocked_when_event_blocks() -> None:
    """CP request.validation root must reflect gentoro.request.blocked outcome."""
    loader = ScenarioLoader()
    scenario = loader.load("request_policy_blocked")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    prefix = sim_config.ATTR_PREFIX
    root = graph.cp_request.spans[0]
    assert root.attributes.get(f"{prefix}.span.class") == "request.validation"
    assert root.attributes.get(f"{prefix}.request.outcome") == "blocked"
    assert root.status_code == "UNSET"


def test_augmentation_step_outcome_is_fail_when_event_indicates_fail() -> None:
    """CP augmentation span step.outcome must match augmentation result event."""
    loader = ScenarioLoader()
    scenario = loader.load("augmentation_failure_hard_blocked")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    prefix = sim_config.ATTR_PREFIX
    augmentation = next(
        s
        for s in graph.cp_request.spans
        if s.attributes.get(f"{prefix}.span.class") == "augmentation.validation"
    )
    assert augmentation.status_code == "ERROR"
    assert augmentation.attributes.get(f"{prefix}.step.outcome") == "fail"


def test_payload_step_outcome_inferred_from_validation_result_invalid() -> None:
    """payload.validation step.outcome must be fail when validation.result is invalid."""
    loader = ScenarioLoader()
    scenario = loader.load("invalid_payload_blocked")
    config = load_config()
    ctx = _resolve_ctx()
    latency_model = LatencyModel.from_scenario(scenario)
    enduser = scenario.endusers[0]
    turn = enduser.turns[0]

    graph = compile_turn(
        scenario,
        enduser,
        turn,
        resolved_ctx=ctx,
        latency_model=latency_model,
        config=config,
    )

    prefix = sim_config.ATTR_PREFIX
    payload = next(
        s
        for s in graph.cp_request.spans
        if s.attributes.get(f"{prefix}.span.class") == "payload.validation"
    )
    assert payload.status_code == "ERROR"
    assert payload.attributes.get(f"{prefix}.validation.result") == "invalid"
    assert payload.attributes.get(f"{prefix}.step.outcome") == "fail"


def test_otel_scope_name_includes_scenario_filename() -> None:
    """ScenarioRunner uses `otelsim.<scenario filename>`; with enduser id, `otelsim.<stem>.<id>`."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ReadableSpan,
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    from simulator.generators.trace_generator import SpanSpec, TraceSpec, render_trace
    from simulator.scenarios.scenario_loader import ScenarioLoader
    from simulator.scenarios.scenario_runner import ScenarioRunner

    class CollectingSpanExporter(SpanExporter):
        def __init__(self) -> None:
            super().__init__()
            self.spans: list[ReadableSpan] = []

        def export(self, spans: list[ReadableSpan]) -> SpanExportResult:  # type: ignore[override]
            self.spans.extend(list(spans))
            return SpanExportResult.SUCCESS

    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    assert scenario.source_filename, "scenario.source_filename should be populated from YAML filename"

    exporter = CollectingSpanExporter()
    runner = ScenarioRunner(trace_exporter=exporter)
    expected_scope_name = runner._get_instrumentation_scope_name(scenario)
    assert expected_scope_name == f"otelsim.{scenario.source_filename}"
    first_eu_id = scenario.endusers[0].id
    assert first_eu_id
    assert runner._get_instrumentation_scope_name(
        scenario, enduser_id=first_eu_id
    ) == f"otelsim.{scenario.source_filename}.{first_eu_id}"

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(expected_scope_name, "1.0.0")

    spec = TraceSpec(
        spans=[
            SpanSpec(
                "test.span",
                parent_index=-1,
                kind="INTERNAL",
                attributes={"foo": "bar"},
                duration_ms=1,
            )
        ]
    )
    _ = render_trace(tracer, spec)
    assert exporter.spans, "expected at least one exported span"
    assert exporter.spans[0].instrumentation_scope.name == expected_scope_name
