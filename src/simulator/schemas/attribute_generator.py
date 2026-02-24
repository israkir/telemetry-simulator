"""
Generate attribute values from the OTEL semantic conventions schema and context.

All attributes are driven by the schema file: for each span type the schema
defines which attributes exist (with type, allowed_values, examples, default).
Values are filled from: overrides → default → allowed_values → context (for
correlation keys) → examples → type-based fallback. No hardcoded span-specific
logic; use scenario YAML overrides for domain-specific data.
"""

import hashlib
import random
import string
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import attr as config_attr
from ..config import schema_version_attr
from ..defaults import get_default_tenant_ids
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

    @classmethod
    def create(
        cls,
        tenant_id: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> "GenerationContext":
        """Create a generation context with defaults (tenant from TENANT_UUID env)."""
        return cls(
            tenant_id=tenant_id or get_default_tenant_ids()[0],
            session_id=session_id or f"sess_{uuid.uuid4().hex[:12]}",
            request_id=kwargs.get("request_id") or f"req_{uuid.uuid4().hex[:12]}",
            turn_index=kwargs.get("turn_index", 0),
            environment=kwargs.get("environment", "development"),
            user_id=kwargs.get("user_id")
            or f"usr_sha256:{hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:16]}",
            route=kwargs.get("route"),
        )


def _attr_matches(name: str, *candidates: str) -> bool:
    """True if name equals any candidate or ends with .candidate (e.g. prefix.session.id)."""
    for c in candidates:
        if name == c or name.endswith("." + c):
            return True
    return False


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
                "tenant.id": context.tenant_id,
                config_attr("session.id"): context.session_id,
                config_attr("request.id"): context.request_id,
            }
            base.update(overrides)
            _ensure_int_token_counts(base)
            return base

        # deployment.environment.name is resource-level only per OTEL/spec; do not set on spans.
        _RESOURCE_ONLY_ATTR = "deployment.environment.name"
        attrs = {}
        for attr_name, attr_def in attr_defs.items():
            if attr_name == _RESOURCE_ONLY_ATTR:
                continue
            attrs[attr_name] = self.generate_value(attr_def, context, overrides)
        span_overrides = {k: v for k, v in overrides.items() if k != _RESOURCE_ONLY_ATTR}
        attrs.update(span_overrides)
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
        if _attr_matches(name, "enduser.id") or "enduser.pseudo.id" in name or "enduser.id" in name:
            return context.user_id
        # deployment.environment.name is resource-level only; do not set on spans.
        if _attr_matches(name, "turn.index"):
            return context.turn_index
        if _attr_matches(name, "route"):
            return context.route if context.route is not None else "default"
        if _attr_matches(name, "schema.version") or name == schema_version_attr():
            return self.schema.schema_version
        if name == "service.name":
            return "telemetry-simulator"
        if name == "service.version":
            return "1.0.0"

        if "hash" in name:
            return f"sha256:{hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32]}"
        if "redacted" in name:
            return "[REDACTED]"

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
