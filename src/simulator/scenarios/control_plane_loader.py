"""
Load control-plane behavior from config and build trace hierarchies from templates.

All control-plane semantics (request/response validation outcomes, block reasons,
step outcomes, trace flow) are driven by config.yaml. Add new templates or
trace flows by editing config; no code changes required.
Attribute keys in config use short names (e.g. request.outcome); the prefix
is applied at build time from config.attr().
"""

from pathlib import Path
from typing import Any

from ..config import CONFIG_PATH, attr as config_attr, load_yaml
from ..generators.trace_generator import (
    SpanConfig,
    SpanType,
    TraceHierarchy,
)


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


def get_trace_flow(request_outcome: str, template_id: str | None = None, config_path: Path | None = None) -> list[str]:
    """Return list of trace kinds to emit. Uses template_id to infer outcome when provided (e.g. blocked_request_policy -> blocked)."""
    outcome = (request_outcome or "allowed").strip().lower()
    if template_id:
        outcome = outcome_from_template_id(template_id)
    cp = _load_control_plane_config(config_path)
    flow = cp.get("trace_flow")
    if not isinstance(flow, dict):
        return ["incoming_validation", "data_plane", "response_validation"] if outcome == "allowed" else ["incoming_validation"]
    traces = flow.get(outcome)
    if isinstance(traces, list):
        return [str(t) for t in traces]
    if outcome == "allowed":
        return ["incoming_validation", "data_plane", "response_validation"]
    return ["incoming_validation"]


def get_latencies_ms(config_path: Path | None = None) -> dict[str, float]:
    """Return control-plane span latencies from config (request_validation, validation_payload, etc.)."""
    cp = _load_control_plane_config(config_path)
    lat = cp.get("latencies_ms")
    if not isinstance(lat, dict):
        return {}
    return {k: float(v) for k, v in lat.items() if isinstance(v, (int, float))}


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

    t = templates[template_id]
    if not isinstance(t, dict):
        raise ValueError(f"Invalid template '{template_id}': not a dict")

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

    payload_attrs = _prefix_attrs(t.get("payload") or {})
    payload_cfg = SpanConfig(
        span_type=SpanType.VALIDATION_PAYLOAD,
        latency_mean_ms=float(payload_lat),
        latency_variance=0.2,
        error_rate=float(err.get("payload", 0)),
        attribute_overrides=payload_attrs,
    )

    policy_raw = t.get("policy") or {}
    policy_attrs = _prefix_attrs({k: v for k, v in (policy_raw if isinstance(policy_raw, dict) else {}).items() if k != "exception"})
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

    augment_attrs = _prefix_attrs(t.get("augmentation") or {})
    augment_cfg = SpanConfig(
        span_type=SpanType.AUGMENTATION,
        latency_mean_ms=float(augment_lat),
        latency_variance=0.2,
        error_rate=float(err.get("augmentation", 0)),
        attribute_overrides=augment_attrs,
    )

    return TraceHierarchy(
        root_config=root_cfg,
        children=[
            TraceHierarchy(root_config=payload_cfg, children=[]),
            TraceHierarchy(root_config=policy_cfg, children=[]),
            TraceHierarchy(root_config=augment_cfg, children=[]),
        ],
    )


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
