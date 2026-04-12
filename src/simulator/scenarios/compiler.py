"""
Compile scenario turn into TraceGraphSpec (cp_request, data_plane, cp_response).

Scenario YAML is the source of truth; this module builds TraceSpecs from
CompiledTurnSpec + semconv structure. Latency and IDs are resolved here.
"""

from __future__ import annotations

import json
from typing import Any

from .. import config as sim_config
from ..config import get_default_tenant_id, load_config
from ..generators.trace_generator import SpanEventSpec, SpanSpec, TraceSpec
from .config_resolver import ResolvedContext, resolve_context
from .id_generator import generate_ids_for_turn
from .ir import CompiledTurnSpec, EnduserSpec, TraceGraphSpec, TurnSpec
from .latency import LatencyModel
from .scenario_loader import Scenario

# gentoro-enterprise analytics (GentoroSpanDerivedStorageMapper.SPANCLASS_AUGMENTATION_VALIDATION)
CP_AUGMENTATION_VALIDATION_CLASS = "augmentation.validation"


def _per_tool_resolved_contexts(
    scenario: Scenario,
    enduser: EnduserSpec,
    turn: TurnSpec,
    config: dict[str, Any],
    default_tenant_id: str | None,
) -> list[ResolvedContext] | None:
    """Resolve MCP context per tool when the turn has per-step server keys (rich ``tool_chain`` or legacy ``tool_chain_mcp_servers``)."""
    keys = getattr(turn, "tool_chain_mcp_servers", None) or []
    chain = turn.tool_chain or []
    if not keys or not chain:
        return None
    dtid = default_tenant_id if default_tenant_id else get_default_tenant_id(config)
    fallback = (enduser.mcp_server_key or scenario.context.mcp_server_key or "phone").strip() or "phone"
    out: list[ResolvedContext] = []
    for i in range(len(chain)):
        k = keys[i] if i < len(keys) else fallback
        k = str(k).strip() if k else fallback
        if not k:
            k = fallback
        out.append(
            resolve_context(
                config,
                tenant_key=scenario.context.tenant_key,
                agent_id=scenario.context.agent_id,
                mcp_server_key=k,
                default_tenant_id=dtid,
            )
        )
    return out


def _cp_request_shared_attrs(prefix: str, compiled: CompiledTurnSpec) -> dict[str, Any]:
    """
    Context copied onto every control-plane request-validation span.

    Mirrors gentoro-enterprise dataplane ScenarioContext/setCore fields used on
    gentoro.request.validation and child spans (tenant, session, conversation, agent, request id).
    """
    out: dict[str, Any] = {
        _prefix_attr("vendor.tenant.id", prefix): compiled.tenant_id,
        _prefix_attr("vendor.session.id", prefix): compiled.session_id,
        _prefix_attr("vendor.request.id", prefix): compiled.request_id,
    }
    if compiled.conversation_id:
        out["gen_ai.conversation.id"] = compiled.conversation_id
    if compiled.agent_id:
        out[_prefix_attr("vendor.a2a.agent.target.id", prefix)] = compiled.agent_id
    return out


def _cp_response_shared_attrs(prefix: str, compiled: CompiledTurnSpec) -> dict[str, Any]:
    """Same correlation context as request path, for response.validation and policy child spans."""
    out: dict[str, Any] = {
        _prefix_attr("vendor.tenant.id", prefix): compiled.tenant_id,
        _prefix_attr("vendor.session.id", prefix): compiled.session_id,
        _prefix_attr("vendor.request.id", prefix): compiled.request_id,
    }
    if compiled.conversation_id:
        out["gen_ai.conversation.id"] = compiled.conversation_id
    if compiled.agent_id:
        out[_prefix_attr("vendor.a2a.agent.target.id", prefix)] = compiled.agent_id
    return out


def _prefix_attr(attr: str, prefix: str) -> str:
    """Replace leading `vendor.` or `gentoro.` with runtime prefix from `--vendor`."""
    if attr.startswith("vendor."):
        return f"{prefix}.{attr[7:]}"
    if attr.startswith("gentoro."):
        return f"{prefix}.{attr[len('gentoro.'):]}"
    return attr


def _prefix_attrs(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {_prefix_attr(k, prefix): v for k, v in d.items()}


def _gen_ai_messages_json(val: Any) -> str | None:
    """Serialize OTEL gen_ai.*.messages for span attributes (JSON string)."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return None


def _truthy_content_capture_flag(val: Any) -> bool:
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in ("true", "1", "yes"):
        return True
    return False


def _gen_ai_input_messages_as_list(turn_extra: dict[str, Any]) -> list[Any] | None:
    """Parse gen_ai.input.messages from scenario extra (YAML list or JSON string)."""
    msg_raw = turn_extra.get("gen_ai.input.messages")
    if isinstance(msg_raw, list):
        return msg_raw
    if isinstance(msg_raw, str) and msg_raw.strip():
        try:
            p = json.loads(msg_raw)
            if isinstance(p, list):
                return p
        except json.JSONDecodeError:
            return None
    return None


def _effective_gen_ai_input_message_count(turn: TurnSpec, turn_extra: dict[str, Any]) -> int:
    """
    How many user/model input messages will appear on the llm.call span: explicit
    gen_ai.input.messages in YAML, else one synthesized user message when content capture
    is enabled and request_raw is set (mirrors llm.call attribute assembly below).
    """
    msgs = _gen_ai_input_messages_as_list(turn_extra)
    if msgs:
        return len(msgs)
    cap_raw = turn_extra.get("gentoro.llm.content.capture.enabled")
    if _truthy_content_capture_flag(cap_raw) and (turn.request_raw or "").strip():
        return 1
    return 0


def _infer_llm_turn_count(turn: TurnSpec, turn_extra: dict[str, Any]) -> int:
    """
    gentoro.llm.turn.count: explicit YAML wins; else max(scenario turn_index,
    effective gen_ai.input.messages count) so multi-turn scenarios advance per turn
    while long explicit message lists still dominate.
    """
    v = turn_extra.get("gentoro.llm.turn.count")
    if v is not None:
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip():
            try:
                return int(float(v.strip()))
            except ValueError:
                pass

    turn_idx = int(getattr(turn, "turn_index", 1) or 1)
    msg_n = _effective_gen_ai_input_message_count(turn, turn_extra)
    if msg_n > 0:
        return max(turn_idx, msg_n)
    return turn_idx


def _config_tool_payload_sample(config: dict[str, Any], tool_name: str, key: str) -> dict[str, Any]:
    """Return sample object from mcp_tool_genai_payloads (semconv: JSON-serialized on attempt)."""
    payloads = config.get("mcp_tool_genai_payloads") or {}
    if not isinstance(payloads, dict):
        return {}
    tool_block = payloads.get(tool_name)
    if not isinstance(tool_block, dict):
        return {}
    sem_block = tool_block.get(key)
    if not isinstance(sem_block, dict):
        return {}
    sample = sem_block.get("sample")
    return dict(sample) if isinstance(sample, dict) else {}


def _merged_genai_tool_call_json(
    *,
    tool_name: str,
    turn: TurnSpec,
    config: dict[str, Any],
) -> tuple[str | None, str | None]:
    """
    Merge config samples with per-turn overrides (scenario YAML gen_ai.tool.call.* keyed by tool).

    OTEL GenAI / Gentoro semconv: gen_ai.tool.call.arguments and gen_ai.tool.call.result
    are JSON strings on gentoro.mcp.tool.execute.attempt only.
    """

    def _scenario_tool_dict(blob: dict[str, Any], name: str) -> dict[str, Any]:
        raw = blob.get(name)
        return dict(raw) if isinstance(raw, dict) else {}

    scenario_args = turn.gen_ai_tool_call_arguments or {}
    scenario_res = turn.gen_ai_tool_call_result or {}
    if not isinstance(scenario_args, dict):
        scenario_args = {}
    if not isinstance(scenario_res, dict):
        scenario_res = {}

    merged_args = {
        **_config_tool_payload_sample(config, tool_name, "gen_ai.tool.call.arguments"),
        **_scenario_tool_dict(scenario_args, tool_name),
    }
    merged_res = {
        **_config_tool_payload_sample(config, tool_name, "gen_ai.tool.call.result"),
        **_scenario_tool_dict(scenario_res, tool_name),
    }
    args_json = json.dumps(merged_args, ensure_ascii=True, sort_keys=True) if merged_args else None
    res_json = json.dumps(merged_res, ensure_ascii=True, sort_keys=True) if merged_res else None
    return args_json, res_json


def _mcp_attempt_rows_for_tool(turn: TurnSpec, tool_name: str) -> tuple[list[dict[str, Any]], str]:
    """
    Per-tool MCP attempt sequence from turn.extra['mcp_tool_retries'].

    Returns (yaml_row_dicts, parent_retry_policy). Default one successful attempt, policy none.

    Supported shapes under mcp_tool_retries.<tool_name>:
    - attempts: [ { outcome, error.type?, gentoro.retry.*? }, ... ]
    - retry_count: N  -> N fails (fail_error_type or unavailable) then one success
    """
    default: tuple[list[dict[str, Any]], str] = ([{"outcome": "success"}], "none")
    extra = getattr(turn, "extra", None) or {}
    mtr = extra.get("mcp_tool_retries")
    if not isinstance(mtr, dict):
        return default
    cfg = mtr.get(tool_name)
    if not isinstance(cfg, dict):
        return default

    attempts_raw = cfg.get("attempts")
    if isinstance(attempts_raw, list) and attempts_raw:
        rows: list[dict[str, Any]] = []
        for r in attempts_raw:
            rows.append(dict(r) if isinstance(r, dict) else {})
        pol = str(cfg.get("policy") or "").strip()
        if not pol:
            pol = "none" if len(rows) <= 1 else "exponential_jitter"
        return (rows, pol)

    rc_raw = cfg.get("retry_count")
    if rc_raw is None:
        return default
    try:
        rc_int = max(0, int(rc_raw))
    except (TypeError, ValueError):
        return default
    fail_et = str(cfg.get("fail_error_type") or "unavailable")
    rows = [{"outcome": "fail", "error.type": fail_et} for _ in range(rc_int)] + [
        {"outcome": "success"}
    ]
    pol = str(cfg.get("policy") or "").strip()
    if not pol:
        pol = "none" if rc_int == 0 else "exponential_jitter"
    return (rows, pol)


def _span_attrs_for_mcp_attempt(
    *,
    prefix: str,
    tool_name: str,
    tool_call_id: str,
    attempt_index: int,
    row: dict[str, Any],
    args_json: str | None,
    res_json: str | None,
) -> dict[str, Any]:
    """Build attributes for one gentoro.mcp.tool.execute.attempt span."""
    raw_outcome = row.get("outcome")
    if raw_outcome is None:
        raw_outcome = row.get("gentoro.mcp.attempt.outcome")
    outcome = str(raw_outcome or "success").strip().lower()
    if outcome not in ("success", "fail"):
        outcome = "success"
    reserved = {"outcome", "gentoro.mcp.attempt.outcome"}
    passthrough = {k: v for k, v in row.items() if k not in reserved}

    attrs: dict[str, Any] = {
        _prefix_attr("vendor.span.class", prefix): "mcp.tool.execute.attempt",
        _prefix_attr("vendor.mcp.attempt.index", prefix): attempt_index,
        _prefix_attr("vendor.mcp.attempt.outcome", prefix): outcome,
        "gen_ai.tool.name": tool_name,
        "gen_ai.tool.call.id": tool_call_id,
    }
    if args_json is not None:
        attrs["gen_ai.tool.call.arguments"] = args_json
    if outcome == "success" and res_json is not None:
        attrs["gen_ai.tool.call.result"] = res_json

    for k, v in passthrough.items():
        if v is None:
            continue
        if k == "error.type":
            attrs["error.type"] = v
        elif isinstance(k, str) and (k.startswith("vendor.") or k.startswith("gentoro.")):
            attrs[_prefix_attr(k, prefix)] = v

    if outcome == "fail" and "error.type" not in attrs:
        attrs["error.type"] = "unavailable"

    if attempt_index > 1:
        backoff_key = _prefix_attr("gentoro.retry.backoff.ms", prefix)
        if backoff_key not in attrs:
            attrs[backoff_key] = int(min(2500, 120 * (2 ** (attempt_index - 2))))
        if outcome == "fail":
            reason_key = _prefix_attr("gentoro.retry.reason", prefix)
            if reason_key not in attrs:
                attrs[reason_key] = "transient_error"

    return attrs


def _mcp_parent_and_attempt_durations_ms(
    *,
    latency_model: LatencyModel,
    profile_name: str,
    attempt_count: int,
) -> tuple[int, list[int]]:
    """Wall time: parent equals sum(attempts); each attempt at least 1 ms."""
    base = latency_model.duration_for_span(profile_name, "mcp.tool.execute")
    n = max(1, attempt_count)
    chunk = max(1, base // n)
    durations = [chunk] * n
    durations[-1] += base - sum(durations)
    return base, durations


def _events_for_span_class(
    span_events: list[dict[str, Any]] | None,
    target_span_class: str,
    prefix: str,
) -> list[SpanEventSpec]:
    """Collect scenario-defined events for a given span class."""
    configured_events = span_events or []
    out: list[SpanEventSpec] = []
    for ev in configured_events:
        if not isinstance(ev, dict):
            continue
        if ev.get("target_span_class") != target_span_class:
            continue
        name = ev.get("name")
        if not isinstance(name, str) or not name:
            continue
        attrs = ev.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        prefixed_attrs = _prefix_attrs(attrs, prefix)
        ts_ns = ev.get("timestamp_ns")
        ts: int | None = None
        if isinstance(ts_ns, (int, float)):
            ts = int(ts_ns)
        elif isinstance(ts_ns, str) and ts_ns.isdigit():
            ts = int(ts_ns)
        out.append(SpanEventSpec(name=name, attributes=prefixed_attrs, timestamp_ns=ts))
    return out


def _value_for_attr(attrs: dict[str, Any], suffix: str) -> str | None:
    for k, v in attrs.items():
        if k.endswith(suffix) and v is not None:
            return str(v)
    return None


def _event_value_for_attr(events: list[SpanEventSpec], suffix: str) -> str | None:
    for ev in events:
        for k, v in (ev.attributes or {}).items():
            if k.endswith(suffix) and v is not None:
                return str(v)
    return None


def _event_key_value_for_attr(events: list[SpanEventSpec], suffix: str) -> tuple[str, Any] | None:
    """Return the first matching event attribute key+value (preserve bool/int/float for OTLP)."""
    for ev in events:
        for k, v in (ev.attributes or {}).items():
            if k.endswith(suffix) and v is not None:
                return k, v
    return None


def _infer_status_code(span: SpanSpec) -> str:
    """
    Semconv-aligned status inference:
    - Blocked outcomes are intentional -> UNSET
    - Validation invalid / explicit fail / runtime error outcomes -> ERROR
    """
    span_class = _value_for_attr(span.attributes, ".span.class") or ""
    req_outcome_event = _event_value_for_attr(span.events, ".request.outcome")
    req_outcome_attr = _value_for_attr(span.attributes, ".request.outcome")
    req_outcome = req_outcome_event or req_outcome_attr
    resp_outcome_event = _event_value_for_attr(span.events, ".response.outcome")
    resp_outcome_attr = _value_for_attr(span.attributes, ".response.outcome")
    resp_outcome = resp_outcome_event or resp_outcome_attr
    a2a_outcome = _value_for_attr(span.attributes, ".a2a.outcome")
    step_outcome_attr = _value_for_attr(span.attributes, ".step.outcome")
    step_outcome_event = _event_value_for_attr(span.events, ".step.outcome")
    step_outcome = step_outcome_event or step_outcome_attr
    validation_result_attr = _value_for_attr(span.attributes, ".validation.result")
    validation_result_event = _event_value_for_attr(span.events, ".validation.result")
    validation_result = validation_result_event or validation_result_attr
    attempt_outcome = _value_for_attr(span.attributes, ".mcp.attempt.outcome")

    exception_type = _event_value_for_attr(span.events, "exception.type")
    exception_message = _event_value_for_attr(span.events, "exception.message")

    if req_outcome == "error" or resp_outcome == "error" or a2a_outcome == "error":
        return "ERROR"
    if step_outcome == "fail":
        return "ERROR"
    if exception_type or exception_message:
        return "ERROR"
    if attempt_outcome == "fail":
        return "ERROR"
    if span_class == "payload.validation" and validation_result == "invalid":
        return "ERROR"
    return "UNSET"


def _apply_status_codes(trace: TraceSpec) -> TraceSpec:
    for s in trace.spans:
        s.status_code = _infer_status_code(s)
        #
        # semconv-aligned attribute correction:
        # scenario YAML provides "truth" for many semconv low-cardinality fields
        # via span events (e.g. request.outcome, step.outcome, validation.result).
        # Our compilers currently set some defaults on span attributes; here we
        # overwrite those defaults with any scenario-provided values.
        #
        # This keeps attributes consistent with the status_code we just inferred.
        sync_suffixes = (
            ".request.outcome",
            ".response.outcome",
            ".step.outcome",
            ".validation.result",
            ".policy.engine",
            ".policy.decision",
            ".policy.rule.id",
            ".policy.reason",
            ".policy.severity",
            ".block.reason",
            ".block.stage",
            ".block.http.status_code",
            ".block.http.error_type",
            ".request.audit.flag",
            ".request.audit.source",
            ".policy.fail_mode",
            ".error.category",
            ".error.stage",
        )
        for suffix in sync_suffixes:
            ev_kv = _event_key_value_for_attr(s.events or [], suffix)
            if ev_kv is None:
                continue
            ev_key, ev_val = ev_kv
            s.attributes[ev_key] = ev_val

        # Augmentation actions are low-cardinality and usually provided via
        # augmentation result events.
        for ev in s.events or []:
            for k, v in (ev.attributes or {}).items():
                if v is None or not isinstance(k, str):
                    continue
                if ".augment." in k and k.endswith(".action"):
                    s.attributes[k] = str(v)

        if s.status_code != "ERROR":
            continue

        # semconv (error recording): when status.code=ERROR, we SHOULD set a
        # low-cardinality error.type attribute. The scenario may provide it
        # via span events (common for this simulator), so copy it over to the
        # span attributes when present.
        if "error.type" in s.attributes and s.attributes["error.type"] is not None:
            continue

        error_type: str | None = None
        # 1) span attributes
        for k, v in (s.attributes or {}).items():
            if k == "error.type" and v is not None:
                error_type = str(v)
                break
            if isinstance(k, str) and k.endswith(".error.type") and v is not None:
                error_type = str(v)
                break

        # 2) span events
        if error_type is None:
            for ev in s.events or []:
                for k, v in (ev.attributes or {}).items():
                    if k == "error.type" and v is not None:
                        error_type = str(v)
                        break
                    if isinstance(k, str) and k.endswith(".error.type") and v is not None:
                        error_type = str(v)
                        break
                if error_type is not None:
                    break

        if error_type is not None:
            s.attributes["error.type"] = error_type

        # Runtime failure outcome alignment:
        # - request.outcome/response.outcome should be "error" when status.code=ERROR
        # - step.outcome should be "fail" when status.code=ERROR for validation steps
        for k in list(s.attributes.keys()):
            if isinstance(k, str) and k.endswith(".request.outcome"):
                s.attributes[k] = "error"
            if isinstance(k, str) and k.endswith(".response.outcome"):
                s.attributes[k] = "error"
            if isinstance(k, str) and k.endswith(".step.outcome"):
                s.attributes[k] = "fail"
    return trace


def _first_error_type_from_trace(trace: TraceSpec) -> str | None:
    for s in trace.spans:
        # Prefer explicit error.type attribute.
        if "error.type" in s.attributes and s.attributes["error.type"] is not None:
            return str(s.attributes["error.type"])
        # Also accept vendor-prefixed error.type (some scenario events may carry it).
        for k, v in (s.attributes or {}).items():
            if isinstance(k, str) and k.endswith(".error.type") and v is not None:
                return str(v)
    return None


def _trace_has_status_error(trace: TraceSpec) -> bool:
    return any(s.status_code == "ERROR" for s in trace.spans)


def _segment_cfg(span_plan: dict[str, Any] | None, segment_name: str) -> dict[str, Any]:
    if not isinstance(span_plan, dict):
        return {}
    seg = span_plan.get(segment_name)
    return seg if isinstance(seg, dict) else {}


def _segment_enabled(span_plan: dict[str, Any] | None, segment_name: str) -> bool:
    if not isinstance(span_plan, dict) or not span_plan:
        return False
    return segment_name in span_plan


def _segment_allowed_children(
    span_plan: dict[str, Any] | None,
    segment_name: str,
) -> set[str] | None:
    seg = _segment_cfg(span_plan, segment_name)
    children = seg.get("children")
    if not isinstance(children, list):
        return None
    allowed = {str(c) for c in children if isinstance(c, str) and c}
    return allowed


def _build_cp_request_trace(
    prefix: str,
    compiled: CompiledTurnSpec,
    latency_model: LatencyModel,
    outcome: str = "allowed",
    span_events: list[dict[str, Any]] | None = None,
    allowed_children: set[str] | None = None,
) -> TraceSpec:
    """Build control-plane request validation trace (semconv)."""
    spans: list[SpanSpec] = []
    p = compiled.latency_profile_name

    shared = _cp_request_shared_attrs(prefix, compiled)
    root_attrs = {
        **shared,
        _prefix_attr("vendor.span.class", prefix): "request.validation",
        _prefix_attr("vendor.request.outcome", prefix): outcome,
    }
    if compiled.enduser_id:
        root_attrs[_prefix_attr("vendor.enduser.pseudo.id", prefix)] = compiled.enduser_id
    r_applied = compiled.redaction_applied
    if r_applied is not None and str(r_applied).strip():
        root_attrs[_prefix_attr("vendor.redaction.applied", prefix)] = r_applied
    # semconv diagnostic events targeted at `request.validation`.
    # Events are scenario-driven from YAML `turn.span_events`.
    events_for_request_validation = _events_for_span_class(
        span_events=span_events,
        target_span_class="request.validation",
        prefix=prefix,
    )

    # Consumer rule: when request outcome implies the validation chain should
    # continue, ensure required descendant span classes exist even if the
    # scenario YAML's `span_plan` omits them.
    inferred_request_outcome = (
        _event_value_for_attr(
            events_for_request_validation,
            ".request.outcome",
        )
        or outcome
    )

    # Keep root attribute aligned with any scenario-provided outcome.
    root_attrs[_prefix_attr("vendor.request.outcome", prefix)] = inferred_request_outcome

    policy_events_for_defaulting = _events_for_span_class(
        span_events=span_events,
        target_span_class="policy.validation",
        prefix=prefix,
    )
    policy_decision_from_events = _event_value_for_attr(
        policy_events_for_defaulting, ".policy.decision"
    )
    policy_pass = (
        policy_decision_from_events in {"allow", "allow_with_audit"}
        or inferred_request_outcome == "allowed"
    )

    effective_allowed_children = set(allowed_children) if allowed_children is not None else None
    # Respect an explicit span_plan child list (e.g. policy-only or payload-only short circuits).
    # When policy passes and the plan already includes policy, ensure augmentation is present
    # if scenarios forget to list it (allowed / allow_with_audit paths).
    if (
        effective_allowed_children is not None
        and inferred_request_outcome in {"allowed", "blocked"}
        and policy_pass
        and "policy.validation" in effective_allowed_children
    ):
        effective_allowed_children.add(CP_AUGMENTATION_VALIDATION_CLASS)

    spans.append(
        SpanSpec(
            name=f"{prefix}.request.validation",
            parent_index=-1,
            kind="SERVER",
            attributes=root_attrs,
            duration_ms=latency_model.duration_for_span(p, "request.validation"),
            events=events_for_request_validation,
        )
    )
    child_specs = [
        (
            f"{prefix}.validation.payload",
            "payload.validation",
            "pass",
            {"vendor.validation.result": "valid"},
        ),
        (
            f"{prefix}.validation.policy",
            "policy.validation",
            "pass" if policy_pass else "block",
            {
                "vendor.policy.decision": "allow" if policy_pass else "block",
                "vendor.policy.engine": "default",
            },
        ),
        (
            f"{prefix}.augmentation.validation",
            CP_AUGMENTATION_VALIDATION_CLASS,
            "pass",
            {
                "vendor.augment.conversation_id.action": "propagated",
                "vendor.augment.request_id.action": "created",
                "vendor.augment.enduser_id.action": "missing",
                "vendor.augment.target_agent_id.action": "attached",
            },
        ),
    ]
    for span_name, span_class, default_step_outcome, extra in child_specs:
        if effective_allowed_children is not None and span_class not in effective_allowed_children:
            continue
        attrs = {
            **shared,
            _prefix_attr("vendor.span.class", prefix): span_class,
            _prefix_attr("vendor.step.outcome", prefix): default_step_outcome,
            **_prefix_attrs(extra, prefix),
        }
        if r_applied is not None and str(r_applied).strip():
            attrs[_prefix_attr("vendor.redaction.applied", prefix)] = r_applied
        spans.append(
            SpanSpec(
                name=span_name,
                parent_index=0,
                kind="INTERNAL",
                attributes=attrs,
                duration_ms=latency_model.duration_for_span(p, span_class),
                events=_events_for_span_class(
                    span_events=span_events,
                    target_span_class=span_class,
                    prefix=prefix,
                ),
            )
        )
    return _apply_status_codes(TraceSpec(spans=spans))


def _build_cp_response_trace(
    prefix: str,
    compiled: CompiledTurnSpec,
    latency_model: LatencyModel,
    outcome: str = "allowed",
    span_events: list[dict[str, Any]] | None = None,
    allowed_children: set[str] | None = None,
) -> TraceSpec:
    """Build control-plane response validation trace (semconv)."""
    spans: list[SpanSpec] = []
    p = compiled.latency_profile_name

    shared = _cp_response_shared_attrs(prefix, compiled)
    root_attrs = {
        **shared,
        _prefix_attr("vendor.span.class", prefix): "response.validation",
        _prefix_attr("vendor.response.outcome", prefix): outcome,
    }

    configured_events = span_events or []
    events_for_response_validation: list[SpanEventSpec] = []
    for ev in configured_events:
        if not isinstance(ev, dict):
            continue
        if ev.get("target_span_class") != "response.validation":
            continue
        name = ev.get("name")
        if not isinstance(name, str) or not name:
            continue
        attrs = ev.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}

        prefixed_attrs = _prefix_attrs(attrs, prefix)
        ts_ns = ev.get("timestamp_ns")
        ts: int | None = None
        if isinstance(ts_ns, (int, float)):
            ts = int(ts_ns)
        elif isinstance(ts_ns, str) and ts_ns.isdigit():
            ts = int(ts_ns)

        events_for_response_validation.append(
            SpanEventSpec(
                name=name,
                attributes=prefixed_attrs,
                timestamp_ns=ts,
            )
        )

    inferred_response_outcome = (
        _event_value_for_attr(
            events_for_response_validation,
            ".response.outcome",
        )
        or outcome
    )
    root_attrs[_prefix_attr("vendor.response.outcome", prefix)] = inferred_response_outcome

    effective_allowed_children = set(allowed_children) if allowed_children is not None else None
    if effective_allowed_children is not None and inferred_response_outcome in {
        "allowed",
        "blocked",
    }:
        effective_allowed_children.add("policy.validation")

    spans.append(
        SpanSpec(
            name=f"{prefix}.response.validation",
            parent_index=-1,
            kind="SERVER",
            attributes=root_attrs,
            duration_ms=latency_model.duration_for_span(p, "response.validation"),
            events=events_for_response_validation,
        )
    )
    if effective_allowed_children is None or "policy.validation" in effective_allowed_children:
        pol_attrs = {
            **shared,
            _prefix_attr("vendor.span.class", prefix): "policy.validation",
            _prefix_attr("vendor.policy.engine", prefix): "default",
            _prefix_attr("vendor.policy.decision", prefix): "allow",
            _prefix_attr("vendor.step.outcome", prefix): "pass",
        }
        r_applied = compiled.redaction_applied
        if r_applied is not None and str(r_applied).strip():
            pol_attrs[_prefix_attr("vendor.redaction.applied", prefix)] = r_applied
        spans.append(
            SpanSpec(
                name=f"{prefix}.validation.policy",
                parent_index=0,
                kind="INTERNAL",
                attributes=pol_attrs,
                duration_ms=latency_model.duration_for_span(p, "policy.validation"),
            )
        )
    return _apply_status_codes(TraceSpec(spans=spans))


def _build_data_plane_trace(
    prefix: str,
    ctx: ResolvedContext,
    turn: TurnSpec,
    compiled: CompiledTurnSpec,
    latency_model: LatencyModel,
    span_events: list[dict[str, Any]] | None = None,
    allowed_children: set[str] | None = None,
    config: dict[str, Any] | None = None,
    tool_contexts: list[ResolvedContext] | None = None,
) -> TraceSpec:
    """Build data-plane A2A orchestration trace from scenario turn (semconv)."""
    spans: list[SpanSpec] = []
    p = compiled.latency_profile_name
    allowed = set(allowed_children) if allowed_children is not None else None
    # Consumer rule: if orchestration outcome is success, a response.compose span
    # must be present (even when YAML omits it from span_plan).
    if allowed is not None:
        allowed.add("response.compose")
    cfg = config or {}

    orchestration_duration_ms = latency_model.duration_for_span(p, "a2a.orchestrate")
    root_attrs = {
        _prefix_attr("vendor.span.class", prefix): "a2a.orchestrate",
        _prefix_attr("vendor.a2a.agent.target.id", prefix): ctx.agent_id,
        _prefix_attr("vendor.a2a.outcome", prefix): "success",
        _prefix_attr("vendor.request.id", prefix): compiled.request_id,
        _prefix_attr("vendor.a2a.request.id", prefix): compiled.request_id,
        _prefix_attr("vendor.redaction.applied", prefix): compiled.redaction_applied,
        _prefix_attr("vendor.enduser.pseudo.id", prefix): compiled.enduser_id,
        _prefix_attr("vendor.tenant.id", prefix): ctx.tenant_id,
        _prefix_attr("vendor.session.id", prefix): compiled.session_id,
        _prefix_attr("vendor.orchestration.duration_ms", prefix): orchestration_duration_ms,
        "gen_ai.conversation.id": compiled.conversation_id,
    }
    spans.append(
        SpanSpec(
            name=f"{prefix}.a2a.orchestrate",
            parent_index=-1,
            kind="SERVER",
            attributes=root_attrs,
            duration_ms=orchestration_duration_ms,
        )
    )
    root_idx = 0
    if compiled.higher_latency_attributes:
        spans[-1].attributes.update(_prefix_attrs(compiled.higher_latency_attributes, prefix))

    # Prepare optional span-events targeted to data-plane spans.
    # Events are scenario-driven; no fallback.
    configured_events = span_events or []

    def _build_events_for_target(target_span_class: str) -> list[SpanEventSpec]:
        out: list[SpanEventSpec] = []
        for ev in configured_events:
            if not isinstance(ev, dict):
                continue
            if ev.get("target_span_class") != target_span_class:
                continue
            name = ev.get("name")
            if not isinstance(name, str) or not name:
                continue
            attrs = ev.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            prefixed_attrs = _prefix_attrs(attrs, prefix)

            ts_ns = ev.get("timestamp_ns")
            ts: int | None = None
            if isinstance(ts_ns, (int, float)):
                ts = int(ts_ns)
            elif isinstance(ts_ns, str) and ts_ns.isdigit():
                ts = int(ts_ns)

            out.append(SpanEventSpec(name=name, attributes=prefixed_attrs, timestamp_ns=ts))
        return out

    events_for_tools_recommend = _build_events_for_target("tools.recommend")
    events_for_mcp_execute = _events_for_span_class(span_events, "mcp.tool.execute", prefix)
    events_for_mcp_attempt = _events_for_span_class(span_events, "mcp.tool.execute.attempt", prefix)

    # planner
    if allowed is None or "planner" in allowed:
        spans.append(
            SpanSpec(
                name=f"{prefix}.planner",
                parent_index=root_idx,
                kind="INTERNAL",
                attributes={
                    _prefix_attr("vendor.span.class", prefix): "planner",
                    _prefix_attr("vendor.step.outcome", prefix): "success",
                },
                duration_ms=latency_model.duration_for_span(p, "planner"),
            )
        )

    # LLM task + llm.call
    if allowed is None or "task.execute" in allowed or "llm.call" in allowed:
        spans.append(
            SpanSpec(
                name=f"{prefix}.task.execute",
                parent_index=root_idx,
                kind="INTERNAL",
                attributes={
                    _prefix_attr("vendor.span.class", prefix): "task.execute",
                    _prefix_attr("vendor.task.id", prefix): "task_llm_call",
                    _prefix_attr("vendor.task.type", prefix): "llm_call",
                    _prefix_attr("vendor.step.outcome", prefix): "success",
                },
                duration_ms=latency_model.duration_for_span(p, "task.execute"),
            )
        )
        task_llm_idx = len(spans) - 1
        if allowed is None or "llm.call" in allowed:
            tool_names = list(turn.tool_chain or [])
            tool_count = len(tool_names)

            turn_extra = getattr(turn, "extra", None) or {}
            if not isinstance(turn_extra, dict):
                turn_extra = {}

            llm_turn_count = _infer_llm_turn_count(turn, turn_extra)

            llm_attrs: dict[str, Any] = {
                _prefix_attr("vendor.span.class", prefix): "llm.call",
                _prefix_attr("vendor.step.outcome", prefix): "success",
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4",
                _prefix_attr("gentoro.llm.turn.count", prefix): llm_turn_count,
                _prefix_attr("gentoro.llm.tool.request.count", prefix): tool_count,
            }
            # No defaults: only set values when scenario YAML provides them.
            for src_key, out_key in (
                ("gen_ai.request.type", "gen_ai.request.type"),
                ("gen_ai.response.finish_reason", "gen_ai.response.finish_reason"),
                ("gentoro.llm.streaming", _prefix_attr("gentoro.llm.streaming", prefix)),
                (
                    "gentoro.llm.content.capture.enabled",
                    _prefix_attr("gentoro.llm.content.capture.enabled", prefix),
                ),
                (
                    "gentoro.llm.content.redaction.enabled",
                    _prefix_attr("gentoro.llm.content.redaction.enabled", prefix),
                ),
                ("gen_ai.usage.input_tokens", "gen_ai.usage.input_tokens"),
                ("gen_ai.usage.output_tokens", "gen_ai.usage.output_tokens"),
            ):
                if src_key not in turn_extra:
                    continue
                val = turn_extra.get(src_key)
                if val is None:
                    continue
                # token usage and counts should be numeric; accept numeric and numeric-like strings.
                if out_key in {
                    "gen_ai.usage.input_tokens",
                    "gen_ai.usage.output_tokens",
                }:
                    if isinstance(val, (int, float)):
                        val = int(val)
                llm_attrs[out_key] = val

            for src_key, out_key in (
                ("gen_ai.input.messages", "gen_ai.input.messages"),
                ("gen_ai.output.messages", "gen_ai.output.messages"),
            ):
                if src_key not in turn_extra:
                    continue
                enc = _gen_ai_messages_json(turn_extra.get(src_key))
                if enc is not None:
                    llm_attrs[out_key] = enc

            cap_raw = turn_extra.get("gentoro.llm.content.capture.enabled")
            cap_pref = _prefix_attr("gentoro.llm.content.capture.enabled", prefix)
            capture_on = _truthy_content_capture_flag(llm_attrs.get(cap_pref, cap_raw))
            if capture_on:
                if "gen_ai.input.messages" not in llm_attrs and (turn.request_raw or "").strip():
                    llm_attrs["gen_ai.input.messages"] = json.dumps(
                        [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": turn.request_raw}],
                            }
                        ],
                        ensure_ascii=False,
                    )
                if (
                    "gen_ai.output.messages" not in llm_attrs
                    and (turn.agent_response or "").strip()
                ):
                    llm_attrs["gen_ai.output.messages"] = json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": turn.agent_response}],
                            }
                        ],
                        ensure_ascii=False,
                    )

            spans.append(
                SpanSpec(
                    name=f"{prefix}.llm.call",
                    parent_index=task_llm_idx,
                    kind="CLIENT",
                    attributes=llm_attrs,
                    duration_ms=latency_model.duration_for_span(p, "llm.call"),
                    events=[],
                )
            )

    # Recommendation task + tools.recommend
    if allowed is None or "task.execute" in allowed or "tools.recommend" in allowed:
        spans.append(
            SpanSpec(
                name=f"{prefix}.task.execute",
                parent_index=root_idx,
                kind="INTERNAL",
                attributes={
                    _prefix_attr("vendor.span.class", prefix): "task.execute",
                    _prefix_attr("vendor.task.id", prefix): "task_recommend",
                    _prefix_attr("vendor.task.type", prefix): "tool_recommendation",
                    _prefix_attr("vendor.step.outcome", prefix): "success",
                },
                duration_ms=latency_model.duration_for_span(p, "task.execute"),
            )
        )
        task_rec_idx = len(spans) - 1
        if allowed is None or "tools.recommend" in allowed:
            chain = list(turn.tool_chain or [])
            rec_ctx = tool_contexts[0] if tool_contexts else ctx
            rec_attrs: dict[str, Any] = {
                _prefix_attr("vendor.span.class", prefix): "tools.recommend",
                _prefix_attr("vendor.step.outcome", prefix): "success",
                _prefix_attr("vendor.mcp.tools.available.count", prefix): len(rec_ctx.tools_by_name),
                _prefix_attr("vendor.mcp.tools.selected.count", prefix): min(
                    len(chain), len(rec_ctx.tools_by_name)
                ),
            }
            if chain:
                # OTEL GenAI: identifies the tool(s) the recommender selected (semconv tools_recommend_model).
                rec_attrs["gen_ai.tool.name"] = chain[0] if len(chain) == 1 else ",".join(chain)
            spans.append(
                SpanSpec(
                    name=f"{prefix}.tools.recommend",
                    parent_index=task_rec_idx,
                    kind="INTERNAL",
                    attributes=rec_attrs,
                    duration_ms=latency_model.duration_for_span(p, "tools.recommend"),
                    events=events_for_tools_recommend,
                )
            )

    # 3) Tool execution tasks: one task per tool in tool_chain
    if (
        allowed is None
        or "task.execute" in allowed
        or "mcp.tool.execute" in allowed
        or "mcp.tool.execute.attempt" in allowed
    ):
        for task_i, tool_name in enumerate(turn.tool_chain):
            spans.append(
                SpanSpec(
                    name=f"{prefix}.task.execute",
                    parent_index=root_idx,
                    kind="INTERNAL",
                    attributes={
                        _prefix_attr("vendor.span.class", prefix): "task.execute",
                        _prefix_attr("vendor.task.id", prefix): f"task_tool_{task_i}",
                        _prefix_attr("vendor.task.type", prefix): "tool_execution",
                        _prefix_attr("vendor.step.outcome", prefix): "success",
                    },
                    duration_ms=latency_model.duration_for_span(p, "task.execute"),
                )
            )
            task_exec_idx = len(spans) - 1

            if allowed is None or "mcp.tool.execute" in allowed:
                tctx = (
                    tool_contexts[task_i]
                    if tool_contexts is not None and task_i < len(tool_contexts)
                    else ctx
                )
                tool_uuid = tctx.get_tool_uuid(tool_name) or ""
                tool_call_id = (
                    compiled.mcp_tool_call_ids[task_i]
                    if task_i < len(compiled.mcp_tool_call_ids)
                    else ""
                )
                attempt_rows, retry_policy = _mcp_attempt_rows_for_tool(turn, tool_name)
                n_attempts = len(attempt_rows)
                retry_count = n_attempts - 1
                policy_eff = retry_policy if retry_count > 0 else "none"
                last_raw = attempt_rows[-1].get("outcome") or attempt_rows[-1].get(
                    "gentoro.mcp.attempt.outcome"
                )
                last_outcome = str(last_raw or "success").strip().lower()
                parent_step = "success" if last_outcome == "success" else "fail"
                parent_ms, attempt_durations = _mcp_parent_and_attempt_durations_ms(
                    latency_model=latency_model,
                    profile_name=p,
                    attempt_count=n_attempts,
                )
                spans.append(
                    SpanSpec(
                        name=f"{prefix}.mcp.tool.execute",
                        parent_index=task_exec_idx,
                        kind="CLIENT",
                        attributes={
                            _prefix_attr("vendor.span.class", prefix): "mcp.tool.execute",
                            _prefix_attr("vendor.step.outcome", prefix): parent_step,
                            _prefix_attr("vendor.mcp.server.uuid", prefix): tctx.mcp_server_uuid,
                            _prefix_attr("vendor.mcp.tool.uuid", prefix): tool_uuid,
                            "gen_ai.tool.name": tool_name,
                            # semconv: logical tool call id is gen_ai.tool.call.id (OTEL GenAI).
                            "gen_ai.tool.call.id": tool_call_id,
                            _prefix_attr("gentoro.retry.count", prefix): retry_count,
                            _prefix_attr("gentoro.retry.policy", prefix): policy_eff,
                            _prefix_attr("gentoro.llm.tool.execution.count", prefix): task_i + 1,
                        },
                        duration_ms=parent_ms,
                        events=events_for_mcp_execute,
                    )
                )
                mcp_exec_idx = len(spans) - 1
                if allowed is None or "mcp.tool.execute.attempt" in allowed:
                    args_json, res_json = _merged_genai_tool_call_json(
                        tool_name=tool_name, turn=turn, config=cfg
                    )
                    attempt_events = events_for_mcp_attempt if n_attempts == 1 else []
                    final_attempt_attrs: dict[str, Any] | None = None
                    for ai, row in enumerate(attempt_rows):
                        attempt_attrs = _span_attrs_for_mcp_attempt(
                            prefix=prefix,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            attempt_index=ai + 1,
                            row=row,
                            args_json=args_json,
                            res_json=res_json,
                        )
                        attempt_attrs[_prefix_attr("vendor.mcp.server.uuid", prefix)] = (
                            tctx.mcp_server_uuid
                        )
                        final_attempt_attrs = attempt_attrs
                        spans.append(
                            SpanSpec(
                                name=f"{prefix}.mcp.tool.execute.attempt",
                                parent_index=mcp_exec_idx,
                                kind="CLIENT",
                                attributes=attempt_attrs,
                                duration_ms=attempt_durations[ai],
                                events=attempt_events,
                            )
                        )
                    if (
                        parent_step == "fail"
                        and final_attempt_attrs
                        and final_attempt_attrs.get("error.type") is not None
                    ):
                        spans[mcp_exec_idx].attributes["error.type"] = final_attempt_attrs[
                            "error.type"
                        ]

    if allowed is None or "response.compose" in allowed:
        spans.append(
            SpanSpec(
                name=f"{prefix}.response.compose",
                parent_index=root_idx,
                kind="INTERNAL",
                attributes={
                    _prefix_attr("vendor.span.class", prefix): "response.compose",
                    _prefix_attr("vendor.response.format", prefix): "a2a_json",
                    _prefix_attr("vendor.step.outcome", prefix): "success",
                },
                duration_ms=latency_model.duration_for_span(p, "response.compose"),
            )
        )
    return _apply_status_codes(TraceSpec(spans=spans))


def compile_turn(
    scenario: Scenario,
    enduser: EnduserSpec,
    turn: TurnSpec,
    *,
    resolved_ctx: ResolvedContext,
    latency_model: LatencyModel,
    config: dict[str, Any] | None = None,
    shared_session_id: str | None = None,
    shared_conversation_id: str | None = None,
    default_tenant_id: str | None = None,
) -> TraceGraphSpec:
    """
    Compile one scenario turn into a TraceGraphSpec (three TraceSpecs + CompiledTurnSpec).
    IDs come from config id_formats; latency from scenario latency_profiles via latency_model.

    When *shared_session_id* / *shared_conversation_id* are set (typically by ScenarioRunner for an
    enduser with multiple turns), those values are reused for every turn of that conversation while
    *request_id* and tool call ids stay unique per compile. Otherwise optional YAML fields on the
    enduser (*session_id*, *conversation_id*) apply for single-turn overrides; if unset, all ids are
    generated per turn.
    """
    config = config or load_config()
    prefix = sim_config.ATTR_PREFIX
    if shared_session_id is not None or shared_conversation_id is not None:
        session_for_ids = shared_session_id
        conversation_for_ids = shared_conversation_id
    else:
        session_for_ids = enduser.session_id or None
        conversation_for_ids = enduser.conversation_id or None
    ids = generate_ids_for_turn(
        config,
        tool_count=len(turn.tool_chain),
        session_id=session_for_ids,
        conversation_id=conversation_for_ids,
    )
    profile_name, higher_attrs = latency_model.select_profile(scenario)

    compiled = CompiledTurnSpec(
        scenario_name=scenario.name,
        enduser_id=enduser.enduser_pseudo_id,
        turn_index=turn.turn_index,
        tenant_id=resolved_ctx.tenant_id,
        agent_id=resolved_ctx.agent_id,
        mcp_server_uuid=resolved_ctx.mcp_server_uuid,
        tool_chain=list(turn.tool_chain),
        session_id=ids["session_id"],
        conversation_id=ids["conversation_id"],
        request_id=ids["request_id"],
        redaction_applied=scenario.context.redaction_applied,
        mcp_tool_call_ids=ids["mcp_tool_call_ids"],
        latency_profile_name=profile_name,
        higher_latency_attributes=higher_attrs or {},
    )

    has_span_plan = isinstance(getattr(turn, "span_plan", None), dict) and bool(turn.span_plan)
    emit_cp_request = _segment_enabled(turn.span_plan, "cp_request") if has_span_plan else True
    cp_req = (
        _build_cp_request_trace(
            prefix,
            compiled,
            latency_model,
            span_events=turn.span_events,
            allowed_children=_segment_allowed_children(turn.span_plan, "cp_request"),
        )
        if emit_cp_request
        else TraceSpec(spans=[])
    )
    emit_data_plane = (
        _segment_enabled(turn.span_plan, "data_plane")
        if has_span_plan
        else bool(getattr(turn, "has_tool_chain", True))
    )
    emit_cp_response = (
        _segment_enabled(turn.span_plan, "cp_response")
        if has_span_plan
        else bool(getattr(turn, "has_agent_response", True))
    )

    tool_contexts = _per_tool_resolved_contexts(
        scenario, enduser, turn, config, default_tenant_id
    )

    dp = (
        _build_data_plane_trace(
            prefix,
            resolved_ctx,
            turn,
            compiled,
            latency_model,
            span_events=turn.span_events,
            allowed_children=_segment_allowed_children(turn.span_plan, "data_plane"),
            config=config,
            tool_contexts=tool_contexts,
        )
        if emit_data_plane
        else TraceSpec(spans=[])
    )
    cp_resp = (
        _build_cp_response_trace(
            prefix,
            compiled,
            latency_model,
            span_events=turn.span_events,
            allowed_children=_segment_allowed_children(turn.span_plan, "cp_response"),
        )
        if emit_cp_response
        else TraceSpec(spans=[])
    )

    # Semconv-aligned error propagation:
    # If control-plane traces hit a runtime error (status.code=ERROR),
    # mark the data-plane orchestration root as `a2a.outcome=error` and
    # propagate the low-cardinality `error.type` (when available) so
    # downstream tools can correlate the failure classification.
    if dp.spans and (_trace_has_status_error(cp_req) or _trace_has_status_error(cp_resp)):
        dp_root = dp.spans[0]
        dp_root.attributes[_prefix_attr("vendor.a2a.outcome", prefix)] = "error"
        err_type = _first_error_type_from_trace(cp_req) or _first_error_type_from_trace(cp_resp)
        if err_type is not None:
            dp_root.attributes["error.type"] = err_type
        _apply_status_codes(dp)

    return TraceGraphSpec(
        cp_request=cp_req,
        data_plane=dp,
        cp_response=cp_resp,
        compiled_turn=compiled,
    )
