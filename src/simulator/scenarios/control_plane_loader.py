"""
Load control-plane behavior from config and build trace hierarchies from templates.

All control-plane semantics (request/response validation outcomes, block reasons,
step outcomes, trace flow) are driven by config.yaml. Add new templates or
trace flows by editing config; no code changes required.
Attribute keys in config use short names (e.g. request.outcome); the prefix
is applied at build time from config.attr().

Templates can be merged with control_plane.request_validation_templates._defaults
so repeated blocks (e.g. error_rate, skip augmentation) are defined once.
Control-plane request scenarios can be registered in control_plane.request_scenarios
(name -> { template, description }); the scenario loader will expose them even
without a separate YAML file.
"""

from pathlib import Path
from typing import Any

from ..config import CONFIG_PATH, load_yaml
from ..config import attr as config_attr
from ..generators.trace_generator import (
    SpanConfig,
    SpanType,
    TraceHierarchy,
)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge overrides into base recursively. Overrides win; base is not mutated."""
    out: dict[str, Any] = dict(base)
    for k, v in overrides.items():
        if k not in out:
            out[k] = v
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_control_plane_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or CONFIG_PATH
    data = load_yaml(path)
    if not isinstance(data, dict):
        return {}
    cp = data.get("control_plane")
    return cp if isinstance(cp, dict) else {}


def _prefix_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Apply vendor prefix to attribute keys (e.g. request.outcome -> prefix.request.outcome)."""
    if not attrs:
        return {}
    return {config_attr(k): v for k, v in attrs.items() if v is not None}


def _template_id_from_outcome_and_reason(request_outcome: str, block_reason: str | None) -> str:
    """Resolve template id from scenario request_outcome and block_reason."""
    outcome = (request_outcome or "allowed").strip().lower()
    if outcome == "allowed":
        return "allowed"
    if outcome == "error":
        return "error"
    if outcome == "blocked" and block_reason:
        reason = (block_reason or "").strip().lower().replace("-", "_")
        return f"blocked_{reason}"
    return "blocked_invalid_context"


def get_request_validation_template_id(
    request_outcome: str,
    block_reason: str | None,
    template_override: str | None,
    config_path: Path | None = None,
) -> str:
    """
    Resolve template id for incoming request validation.
    If scenario sets control_plane.template, use it; else derive from outcome + block_reason.
    """
    if template_override and isinstance(template_override, str) and template_override.strip():
        return template_override.strip().lower()
    return _template_id_from_outcome_and_reason(request_outcome, block_reason)


def outcome_from_template_id(template_id: str) -> str:
    """Derive request outcome from template id for trace_flow (allowed | blocked | error)."""
    tid = (template_id or "").strip().lower()
    if tid == "allowed":
        return "allowed"
    if tid == "error" or tid.startswith("error_"):
        return "error"
    if tid.startswith("blocked_"):
        return "blocked"
    return "allowed"


def get_trace_flow(
    request_outcome: str, template_id: str | None = None, config_path: Path | None = None
) -> list[str]:
    """Return list of trace kinds to emit. Uses template_id to infer outcome when provided (e.g. blocked_request_policy -> blocked)."""
    outcome = (request_outcome or "allowed").strip().lower()
    if template_id:
        outcome = outcome_from_template_id(template_id)
    cp = _load_control_plane_config(config_path)
    flow = cp.get("trace_flow")
    if not isinstance(flow, dict):
        return (
            ["incoming_validation", "data_plane", "response_validation"]
            if outcome == "allowed"
            else ["incoming_validation"]
        )
    traces = flow.get(outcome)
    if isinstance(traces, list):
        return [str(t) for t in traces]
    if outcome == "allowed":
        return ["incoming_validation", "data_plane", "response_validation"]
    return ["incoming_validation"]


def get_default_data_plane_workflow_steps(config_path: Path | None = None) -> list[str]:
    """
    Return default workflow step names for data-plane hierarchy when a control-plane
    scenario allows data_plane (trace_flow includes data_plane) but does not define
    correct_flow (orchestrate-only would be emitted otherwise).
    Uses control_plane.default_data_plane_workflow, else first workflow_templates key.
    """
    from ..config import load_yaml

    path = config_path or CONFIG_PATH
    data = load_yaml(path)
    if not isinstance(data, dict):
        return ["planner", "task", "tools_recommend", "new_claim", "response_compose"]
    cp = data.get("control_plane")
    workflow_key = None
    if isinstance(cp, dict):
        workflow_key = cp.get("default_data_plane_workflow")
        if isinstance(workflow_key, str):
            workflow_key = workflow_key.strip() or None
    rs = data.get("scenarios") or data.get("realistic_scenarios")
    workflow_templates = rs.get("workflow_templates") if isinstance(rs, dict) else {}
    if not isinstance(workflow_templates, dict):
        workflow_templates = {}
    if workflow_key and workflow_key in workflow_templates:
        steps = workflow_templates[workflow_key]
        if isinstance(steps, list):
            return [str(s) for s in steps]
    if workflow_templates:
        first_key = next(iter(workflow_templates))
        steps = workflow_templates[first_key]
        if isinstance(steps, list):
            return [str(s) for s in steps]
    return ["planner", "task", "tools_recommend", "new_claim", "response_compose"]


def get_latencies_ms(config_path: Path | None = None) -> dict[str, float]:
    """Return control-plane span latencies from config (request_validation, validation_payload, etc.)."""
    cp = _load_control_plane_config(config_path)
    lat = cp.get("latencies_ms")
    if not isinstance(lat, dict):
        return {}
    return {k: float(v) for k, v in lat.items() if isinstance(v, (int, float))}


def get_request_scenario_registry(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Return control_plane.request_scenarios: scenario name -> { template, description }.
    Used by the scenario loader to expose control-plane request scenarios without requiring a YAML per scenario.
    """
    cp = _load_control_plane_config(config_path)
    reg = cp.get("request_scenarios")
    if not isinstance(reg, dict):
        return {}
    return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in reg.items()}


def build_request_validation_hierarchy_from_template(
    template_id: str,
    config_path: Path | None = None,
    policy_exception_override: dict[str, str] | None = None,
) -> TraceHierarchy:
    """
    Build incoming request validation hierarchy from a config template.
    Template defines root, payload, policy, augmentation attributes and error_rates.
    If policy_exception_override is provided (e.g. from scenario control_plane.policy_exception),
    it overrides the template's policy exception (type, message) for the policy span.
    """
    cp = _load_control_plane_config(config_path)
    templates = cp.get("request_validation_templates")
    if not isinstance(templates, dict) or template_id not in templates:
        raise ValueError(f"Unknown control_plane request_validation template: {template_id}")
    if (template_id or "").strip().lower() == "_defaults":
        raise ValueError("_defaults is not a valid template id")

    raw = templates[template_id]
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid template '{template_id}': not a dict")
    defaults = templates.get("_defaults")
    if isinstance(defaults, dict):
        t = _deep_merge(defaults, raw)
    else:
        t = raw

    latencies = get_latencies_ms(config_path)
    default_lat = 40.0
    root_lat = latencies.get("request_validation", default_lat)
    payload_lat = latencies.get("validation_payload", 20.0)
    policy_lat = latencies.get("validation_policy", 20.0)
    augment_lat = latencies.get("augmentation", 20.0)

    err = t.get("error_rate") or {}
    if not isinstance(err, dict):
        err = {}

    root_attrs = _prefix_attrs(t.get("root") or {})
    root_cfg = SpanConfig(
        span_type=SpanType.REQUEST_VALIDATION,
        latency_mean_ms=float(root_lat),
        latency_variance=0.2,
        error_rate=float(err.get("root", 0)),
        attribute_overrides=root_attrs,
    )

    payload_raw = t.get("payload") or {}
    if not isinstance(payload_raw, dict):
        payload_raw = {}
    payload_attrs = _prefix_attrs(
        {k: v for k, v in payload_raw.items() if k != "validation_errors" and v is not None}
    )
    validation_errors = payload_raw.get("validation_errors")
    if not isinstance(validation_errors, list):
        validation_errors = None
    payload_cfg = SpanConfig(
        span_type=SpanType.VALIDATION_PAYLOAD,
        latency_mean_ms=float(payload_lat),
        latency_variance=0.2,
        error_rate=float(err.get("payload", 0)),
        attribute_overrides=payload_attrs,
        validation_errors=validation_errors,
    )

    policy_raw = t.get("policy") or {}
    policy_attrs = _prefix_attrs(
        {
            k: v
            for k, v in (policy_raw if isinstance(policy_raw, dict) else {}).items()
            if k != "exception"
        }
    )
    exc = policy_raw.get("exception") if isinstance(policy_raw, dict) else None
    exc_type = str(exc.get("type")) if isinstance(exc, dict) and exc.get("type") else None
    exc_msg = str(exc.get("message")) if isinstance(exc, dict) and exc.get("message") else None
    if policy_exception_override:
        if policy_exception_override.get("type"):
            exc_type = str(policy_exception_override["type"])
        if policy_exception_override.get("message"):
            exc_msg = str(policy_exception_override["message"])
    policy_cfg = SpanConfig(
        span_type=SpanType.VALIDATION_POLICY,
        latency_mean_ms=float(policy_lat),
        latency_variance=0.2,
        error_rate=float(err.get("policy", 0)),
        attribute_overrides=policy_attrs,
        exception_type=exc_type,
        exception_message=exc_msg,
    )

    augment_raw = t.get("augmentation") or {}
    if not isinstance(augment_raw, dict):
        augment_raw = {}
    augment_attrs = _prefix_attrs(
        {k: v for k, v in augment_raw.items() if k != "exception" and v is not None}
    )
    aug_exc = augment_raw.get("exception") if isinstance(augment_raw, dict) else None
    aug_exc_type = (
        str(aug_exc.get("type")) if isinstance(aug_exc, dict) and aug_exc.get("type") else None
    )
    aug_exc_msg = (
        str(aug_exc.get("message"))
        if isinstance(aug_exc, dict) and aug_exc.get("message")
        else None
    )
    augment_cfg = SpanConfig(
        span_type=SpanType.AUGMENTATION,
        latency_mean_ms=float(augment_lat),
        latency_variance=0.2,
        error_rate=float(err.get("augmentation", 0)),
        attribute_overrides=augment_attrs,
        exception_type=aug_exc_type,
        exception_message=aug_exc_msg,
    )

    # Optional: omit policy and/or augmentation spans (e.g. rate_limited before policy runs).
    include_policy = t.get("include_policy", True)
    include_augmentation = t.get("include_augmentation", True)
    children: list[TraceHierarchy] = [TraceHierarchy(root_config=payload_cfg, children=[])]
    if include_policy:
        children.append(TraceHierarchy(root_config=policy_cfg, children=[]))
    if include_augmentation:
        children.append(TraceHierarchy(root_config=augment_cfg, children=[]))

    return TraceHierarchy(root_config=root_cfg, children=children)


def build_response_validation_hierarchy_from_template(
    template_id: str = "allowed",
    config_path: Path | None = None,
) -> TraceHierarchy:
    """Build outgoing response validation hierarchy from config template."""
    cp = _load_control_plane_config(config_path)
    templates = cp.get("response_validation_templates")
    if not isinstance(templates, dict) or template_id not in templates:
        raise ValueError(f"Unknown control_plane response_validation template: {template_id}")

    t = templates[template_id]
    if not isinstance(t, dict):
        raise ValueError(f"Invalid template '{template_id}': not a dict")

    latencies = get_latencies_ms(config_path)
    root_lat = latencies.get("response_validation", 40.0)
    policy_lat = latencies.get("validation_policy", 20.0)
    err = t.get("error_rate") or {}
    if not isinstance(err, dict):
        err = {}

    root_attrs = _prefix_attrs(t.get("root") or {})
    root_cfg = SpanConfig(
        span_type=SpanType.RESPONSE_VALIDATION,
        latency_mean_ms=float(root_lat),
        latency_variance=0.2,
        error_rate=float(err.get("root", 0)),
        attribute_overrides=root_attrs,
    )

    policy_attrs = _prefix_attrs(t.get("policy") or {})
    policy_cfg = SpanConfig(
        span_type=SpanType.VALIDATION_POLICY,
        latency_mean_ms=float(policy_lat),
        latency_variance=0.2,
        error_rate=float(err.get("policy", 0)),
        attribute_overrides=policy_attrs,
    )

    return TraceHierarchy(
        root_config=root_cfg,
        children=[TraceHierarchy(root_config=policy_cfg, children=[])],
    )
