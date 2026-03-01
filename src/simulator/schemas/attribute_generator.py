"""
Generate attribute values from the OTEL semantic conventions schema and context.

All attributes are driven by the schema file: for each span type the schema
defines which attributes exist (with type, allowed_values, examples, default).
Values are filled from: overrides → default → allowed_values → context (for
correlation keys) → examples → type-based fallback. No hardcoded span-specific
logic; use scenario YAML overrides for domain-specific data.
"""

import hashlib
import json
import random
import re
import string
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import SEMCONV_STEP_OUTCOME_VALUES, schema_version_attr
from ..config import attr as config_attr
from ..defaults import get_default_tenant_ids
from ..scenarios.id_generator import (
    generate_enduser_pseudo_id,
    generate_request_id,
    generate_session_id,
)
from .schema_parser import AttributeDefinition, TelemetrySchema


@dataclass
class GenerationContext:
    """Context for attribute generation to ensure consistency within a trace."""

    tenant_id: str
    session_id: str
    request_id: str
    turn_index: int = 0
    environment: str = "development"
    user_id: str | None = None
    route: str | None = None
    # Redaction level from scenario (e.g. none, basic, strict). Override per scenario via redaction_applied.
    redaction_applied: str = "none"
    # Optional: precomputed LLM messages for content capture (when scenario
    # defines conversation.turns/samples and we want consistent gen_ai.* attributes).
    llm_input_messages: Any | None = None
    llm_output_messages: Any | None = None
    # Optional: precomputed redacted messages for gen_ai.input.redacted / gen_ai.output.redacted.
    # When set (and redaction_applied != "none"), these are used instead of auto-redacting from llm_*_messages.
    llm_input_messages_redacted: Any | None = None
    llm_output_messages_redacted: Any | None = None
    # Optional: scenario name that generated this trace; used for otel.scope.name (e.g. otelsim.new_claim_phone).
    scenario_name: str | None = None
    # Optional: when scenario uses higher_latency profile, condition from data_plane.higher_latency_condition.
    # Captured as span attributes (higher_latency.condition.*) for filtering/correlation.
    higher_latency_condition: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        tenant_id: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> "GenerationContext":
        """Create a generation context. Correlation IDs (session_id, request_id, user_id) come from config id_formats when not passed; no fallbacks."""
        return cls(
            tenant_id=tenant_id or get_default_tenant_ids()[0],
            session_id=session_id or generate_session_id(),
            request_id=kwargs.get("request_id") or generate_request_id(),
            turn_index=kwargs.get("turn_index", 0),
            environment=kwargs.get("environment", "development"),
            user_id=kwargs.get("user_id") or generate_enduser_pseudo_id(),
            route=kwargs.get("route"),
            redaction_applied=kwargs.get("redaction_applied", "none"),
            llm_input_messages=kwargs.get("llm_input_messages"),
            llm_output_messages=kwargs.get("llm_output_messages"),
            llm_input_messages_redacted=kwargs.get("llm_input_messages_redacted"),
            llm_output_messages_redacted=kwargs.get("llm_output_messages_redacted"),
            scenario_name=kwargs.get("scenario_name"),
            higher_latency_condition=kwargs.get("higher_latency_condition"),
        )


def _attr_matches(name: str, *candidates: str) -> bool:
    """True if name equals any candidate or ends with .candidate (e.g. prefix.session.id)."""
    for c in candidates:
        if name == c or name.endswith("." + c):
            return True
    return False


def _sample_llm_input_messages() -> list[dict[str, Any]]:
    """Sample for gen_ai.input.messages: messages sent TO the model for this single LLM call.

    Per convention we emit one span per interaction (user input → LLM response). So input
    for this call is the user message(s) for this turn only, not full conversation history.
    """
    return [
        {"role": "user", "content": [{"type": "text", "text": "What is the status of my claim?"}]},
    ]


def _sample_llm_output_messages() -> list[dict[str, Any]]:
    """Sample for gen_ai.output.messages: the model's response for this single LLM call.

    One assistant message per call; matches the one-interaction-per-span convention.
    """
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Your claim CLM-2024-0042 is in review. Expected completion within 5 business days.",
                },
            ],
        },
    ]


# Patterns for in-place PII/sensitive redaction (replace only sensitive parts, not whole message).
# Order matters: more specific before generic (e.g. claim ID before generic IDs).
_REDACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\S+@\S+\.\S+"), "<EMAIL>"),
    (re.compile(r"https?://[^\s<>]+(?:pay|payment|secure)[^\s<>]*", re.I), "<SECURE_PAYMENT_LINK>"),
    (re.compile(r"https?://[^\s<>]+"), "<LINK>"),
    (re.compile(r"\b(?:PH|EV|HA|CLM)-[A-Z0-9-]{4,}\b", re.I), "<CLAIM_ID>"),
    (re.compile(r"\bpolicy\s*(?:number|#)?\s*[\w\d-]{6,}\b", re.I), "<POLICY_NUMBER>"),
    (
        re.compile(
            r"\b\d{1,5}\s+[\w\s]{3,40}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Drive|Dr|Way|Boulevard|Blvd)\b",
            re.I,
        ),
        "<ADDRESS>",
    ),
    (re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", re.I), "<POSTCODE>"),
    (
        re.compile(
            r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{0,6}\b"
        ),
        "<PHONE>",
    ),
]


def _redact_sensitive_text(text: str, level: str) -> str:
    """Replace only PII/sensitive substrings with placeholders; leave rest of message intact."""
    if not text or level == "none":
        return text
    out = text
    for pattern, placeholder in _REDACTION_PATTERNS:
        out = pattern.sub(placeholder, out)
    return out


def _redact_messages(messages: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    """Deep-copy message list and redact only sensitive text in content[].text."""
    if level == "none" or not messages:
        return list(messages)
    out: list[dict[str, Any]] = []
    for msg in messages:
        copy: dict[str, Any] = {"role": msg.get("role", "user"), "content": []}
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                copy["content"].append(
                    {
                        "type": "text",
                        "text": _redact_sensitive_text(
                            str(text) if text is not None else "", level
                        ),
                    }
                )
            else:
                copy["content"].append(dict(block) if isinstance(block, dict) else block)
        out.append(copy)
    return out


def _sample_llm_redacted(structure: str) -> list[dict[str, Any]]:
    """Fallback when no context messages: sample with placeholders (sensitive parts only), not full wipe."""
    if structure == "input":
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the status of my claim?"}],
            }
        ]
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Your claim <CLAIM_ID> is in review. Expected completion within 5 business days.",
                }
            ],
        },
    ]


def _ensure_int_token_counts(attrs: dict[str, Any]) -> None:
    """Round gen_ai.usage token attributes to int in place (OTEL expects integer counts)."""
    for key in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
        if key in attrs and attrs[key] is not None:
            try:
                attrs[key] = int(round(attrs[key]))
            except (TypeError, ValueError):
                pass


class AttributeGenerator:
    """Generate attribute values from schema definitions and context only.

    The semantic conventions YAML defines which attributes each span type has.
    Values come from: overrides, schema default, allowed_values, context (for
    correlation), schema examples, then type-based generation.
    """

    def __init__(self, schema: TelemetrySchema):
        """Initialize with parsed schema."""
        self.schema = schema

    def generate_attributes_for_span(
        self,
        span_name: str,
        context: GenerationContext,
        overrides: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate attributes for a span type using only the schema and context.

        The schema (semantic conventions YAML) defines which attributes exist
        for this span; values are filled from overrides, schema, and context.
        """
        overrides = dict(overrides) if overrides else {}
        if trace_id is not None:
            overrides.setdefault(config_attr("cp.incoming_trace_id"), trace_id)
            overrides.setdefault(config_attr("cp.outgoing_trace_id"), trace_id)

        attr_defs = self.schema.get_span_attributes(span_name)
        if not attr_defs:
            base = {
                config_attr("tenant.id"): context.tenant_id,
                config_attr("session.id"): context.session_id,
                "gen_ai.conversation.id": context.session_id,
                config_attr("request.id"): context.request_id,
            }
            base.update(overrides)
            for key in list(base.keys()):
                if _attr_matches(key, "redaction.applied"):
                    base[key] = context.redaction_applied
            _ensure_int_token_counts(base)
            return base

        # deployment.environment.name is resource-level only per OTEL/spec; do not set on spans.
        _RESOURCE_ONLY_ATTR = "deployment.environment.name"
        attrs = {}
        for attr_name, attr_def in attr_defs.items():
            if attr_name == _RESOURCE_ONLY_ATTR:
                continue
            value = self.generate_value(attr_def, context, overrides)
            # OTEL span attributes do not accept None; omit optional attrs with no value.
            if value is None:
                if attr_def.required:
                    value = "" if attr_def.attr_type == "string" else 0
                else:
                    continue
            attrs[attr_name] = value
        span_overrides = {
            k: v for k, v in overrides.items() if k != _RESOURCE_ONLY_ATTR and v is not None
        }
        attrs.update(span_overrides)
        # Session and conversation are first-class: all spans in the same logical interaction must carry
        # the same session id. gen_ai.conversation.id SHOULD equal session.id for the same conversation (OTEL).
        attrs[config_attr("session.id")] = context.session_id
        attrs["gen_ai.conversation.id"] = context.session_id
        # Ensure redaction.applied is always the scenario setting (context); overrides must not override it.
        for key in list(attrs.keys()):
            if _attr_matches(key, "redaction.applied"):
                attrs[key] = context.redaction_applied
        _ensure_int_token_counts(attrs)
        return attrs

    def generate_value(
        self,
        attr: AttributeDefinition,
        context: GenerationContext,
        overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Generate a value for one attribute: overrides → default → allowed_values → context/examples/type."""
        if overrides is not None and attr.name in overrides:
            return overrides[attr.name]

        if attr.default is not None:
            return attr.default

        if attr.allowed_values:
            return random.choice(attr.allowed_values)

        return self._generate_by_name_and_type(attr, context)

    def _generate_by_name_and_type(
        self,
        attr: AttributeDefinition,
        context: GenerationContext,
    ) -> Any:
        """Value from context (correlation), schema version, or type-based fallback."""
        name = attr.name

        if _attr_matches(name, "tenant.id"):
            return context.tenant_id
        if _attr_matches(name, "session.id") or name == "gen_ai.conversation.id":
            return context.session_id
        if _attr_matches(name, "request.id"):
            return context.request_id
        # vendor.enduser.pseudo.id only (on request.validation and a2a.orchestrate). Always from config format.
        if _attr_matches(name, "enduser.pseudo.id") or name.endswith("enduser.pseudo.id"):
            return context.user_id if context.user_id else generate_enduser_pseudo_id()
        # deployment.environment.name is resource-level only; do not set on spans.
        if _attr_matches(name, "turn.index"):
            return context.turn_index
        if _attr_matches(name, "route"):
            return context.route if context.route is not None else "default"
        if _attr_matches(name, "redaction.applied"):
            return context.redaction_applied
        if _attr_matches(name, "schema.version") or name == schema_version_attr():
            return self.schema.schema_version
        if name == "service.name":
            return "otelsim"
        if name == "service.version":
            return "1.0.0"

        # LLM content: use scenario conversation when provided, else sample data.
        # Serialize to JSON string so OTel span attributes accept it and it appears in exports (e.g. traces.json).
        if name == "gen_ai.input.messages":
            val = (
                context.llm_input_messages
                if context.llm_input_messages is not None
                else _sample_llm_input_messages()
            )
            return json.dumps(val) if isinstance(val, list) else val
        if name == "gen_ai.output.messages":
            val = (
                context.llm_output_messages
                if context.llm_output_messages is not None
                else _sample_llm_output_messages()
            )
            return json.dumps(val) if isinstance(val, list) else val
        # Redacted variants: only when content redaction is enabled (redaction_applied != "none").
        # When llm.content.redaction.enabled is false, do not include gen_ai.input.redacted / gen_ai.output.redacted.
        # When scenario defines redacted text (llm_*_messages_redacted), use it; else auto-redact from messages.
        if "input.redacted" in name or name.endswith("gen_ai.input.redacted"):
            if context.redaction_applied == "none":
                return None
            if getattr(context, "llm_input_messages_redacted", None) is not None:
                val = context.llm_input_messages_redacted
                return json.dumps(val) if isinstance(val, list) else val
            if context.llm_input_messages:
                return json.dumps(
                    _redact_messages(context.llm_input_messages, context.redaction_applied)
                )
            return json.dumps(_sample_llm_redacted("input"))
        if "output.redacted" in name or name.endswith("gen_ai.output.redacted"):
            if context.redaction_applied == "none":
                return None
            if getattr(context, "llm_output_messages_redacted", None) is not None:
                val = context.llm_output_messages_redacted
                return json.dumps(val) if isinstance(val, list) else val
            if context.llm_output_messages:
                return json.dumps(
                    _redact_messages(context.llm_output_messages, context.redaction_applied)
                )
            return json.dumps(_sample_llm_redacted("output"))

        if "hash" in name:
            return f"sha256:{hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32]}"
        if "redacted" in name:
            return "[REDACTED]"

        # error.type: only set when span is in error (status=ERROR). Trace generator sets it in
        # _record_span_error; do not set here so success/partial spans don't carry an error type.
        if name == "error.type" or name.endswith(".error.type"):
            return None
        # gen_ai.tool.call.arguments: per OTEL convention, parameters as JSON string. When no override
        # (from config tool_call_arguments), use empty object so we never emit placeholder "value_xxx".
        if name == "gen_ai.tool.call.arguments":
            return "{}"
        # SemConv-aligned: step.outcome use only allowed values from schema/conventions.
        if _attr_matches(name, "step.outcome"):
            return random.choice(SEMCONV_STEP_OUTCOME_VALUES)
        # gen_ai.tool.name: never use schema examples (e.g. claims.getClaimStatus, slack.sendMessage).
        # Tool names must come from config (scenario workflow + mcp_servers.<key>.tools); when no override
        # is set, use a safe placeholder so traces only reflect config-driven tool set (new_claim, claim_status, update_appointment).
        if name == "gen_ai.tool.name":
            return "unknown_tool"

        if attr.examples:
            return random.choice(attr.examples)

        return self._generate_by_type(attr.attr_type, name)

    def _generate_by_type(self, attr_type: str, name: str) -> Any:
        """Fallback value by attribute type."""
        if attr_type == "string":
            return self._generate_string(name)
        if attr_type == "int":
            return self._generate_int(name)
        if attr_type == "double":
            return self._generate_double(name)
        if attr_type == "boolean":
            return random.choice([True, False])
        return None

    def _generate_string(self, name: str) -> str:
        if "id" in name.lower():
            return f"{name.split('.')[-1]}_{uuid.uuid4().hex[:8]}"
        if "version" in name.lower():
            return f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        if "name" in name.lower():
            return f"name_{''.join(random.choices(string.ascii_lowercase, k=6))}"
        return f"value_{''.join(random.choices(string.ascii_lowercase, k=8))}"

    def _generate_int(self, name: str) -> int:
        if "latency" in name.lower() or "duration" in name.lower():
            return max(1, int(random.gauss(200, 80)))
        if "count" in name.lower():
            return random.randint(1, 100)
        if "tokens" in name.lower():
            return random.randint(50, 2000)
        if "size" in name.lower() or "bytes" in name.lower():
            return random.randint(100, 10000)
        if "index" in name.lower():
            return random.randint(0, 10)
        if "code" in name.lower() and "status" not in name.lower():
            return random.choice([200, 201, 400, 401, 403, 404, 500, 503])
        return random.randint(0, 1000)

    def _generate_double(self, name: str) -> float:
        if "confidence" in name.lower() or "score" in name.lower():
            return round(random.uniform(0.5, 1.0), 3)
        if "rate" in name.lower():
            return round(random.uniform(0.0, 1.0), 3)
        return round(random.uniform(0, 100), 2)
