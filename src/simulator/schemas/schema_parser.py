"""
Parse OTEL semantic conventions YAML schema.

Reads a semantic-conventions YAML file (path must be provided by the client via TELEMETRY_SIMULATOR_SCHEMA_PATH or --schema-path) and provides structured access to:
- Span definitions (names, kinds, parent relationships)
- Attribute schemas (types, requirements, allowed values)
- Metrics definitions
- Status codes and enums
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class AttributeRequirement(Enum):
    """Attribute requirement level per OTEL conventions."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class AttributeCategory(Enum):
    """Attribute category for indexing/query optimization."""

    FIRST_CLASS = "first_class"
    METADATA = "metadata"


class SpanKind(Enum):
    """OTEL span kinds."""

    SERVER = "SERVER"
    CLIENT = "CLIENT"
    INTERNAL = "INTERNAL"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


@dataclass
class AttributeDefinition:
    """Definition of a single telemetry attribute."""

    name: str
    attr_type: str
    required: bool
    requirement_level: AttributeRequirement
    category: AttributeCategory
    description: str
    allowed_values: list[str] | None = None
    examples: list[Any] | None = None
    default: Any = None
    sensitive: bool = False

    @classmethod
    def from_yaml(cls, name: str, data: dict | None) -> "AttributeDefinition":
        """Parse attribute definition from YAML data."""
        data = data if isinstance(data, dict) else {}
        req_value = data.get("required", False)
        if req_value is True:
            requirement = AttributeRequirement.REQUIRED
        elif req_value == "recommended":
            requirement = AttributeRequirement.RECOMMENDED
        else:
            requirement = AttributeRequirement.OPTIONAL

        category_str = data.get("category", "metadata")
        category = (
            AttributeCategory.FIRST_CLASS
            if category_str == "first_class"
            else AttributeCategory.METADATA
        )

        return cls(
            name=name,
            attr_type=data.get("type", "string"),
            required=req_value is True,
            requirement_level=requirement,
            category=category,
            description=data.get("description", ""),
            allowed_values=data.get("allowed_values"),
            examples=data.get("examples"),
            default=data.get("default"),
            sensitive=data.get("sensitive", False),
        )


@dataclass
class SpanDefinition:
    """Definition of a canonical span type."""

    name: str
    description: str
    span_kind: list[SpanKind]
    is_root: bool
    parent_spans: list[str]
    attributes: dict[str, AttributeDefinition] = field(default_factory=dict)
    note: str = ""

    @classmethod
    def from_yaml(cls, name: str, data: dict | None) -> "SpanDefinition":
        """Parse span definition from YAML data."""
        data = data if isinstance(data, dict) else {}
        kind_value = data.get("span_kind", "INTERNAL")
        if isinstance(kind_value, list):
            kinds = [SpanKind(k) for k in kind_value]
        else:
            kinds = [SpanKind(kind_value)]

        parent_value = data.get("parent", "")
        if isinstance(parent_value, str) and parent_value:
            parents = [p.strip() for p in parent_value.split("|")]
        else:
            parents = []

        return cls(
            name=name,
            description=data.get("description", ""),
            span_kind=kinds,
            is_root=data.get("is_root", False),
            parent_spans=parents,
            note=data.get("note", ""),
        )


@dataclass
class StatusDefinition:
    """Definition of status codes for a span type."""

    result_type: str = "boolean"
    code_values: list[str] = field(default_factory=list)
    has_metadata: bool = True


@dataclass
class MetricDefinition:
    """Definition of a canonical metric."""

    name: str
    metric_type: str
    unit: str
    description: str
    dimensions: list[str]
    emitted_by: str
    aligns_with_span: str | None = None

    @classmethod
    def from_yaml(cls, name: str, data: dict | None) -> "MetricDefinition":
        """Parse metric definition from YAML data."""
        data = data if isinstance(data, dict) else {}
        return cls(
            name=name,
            metric_type=data.get("type", "counter"),
            unit=data.get("unit", "1"),
            description=data.get("description", ""),
            dimensions=data.get("dimensions", []),
            emitted_by=data.get("emitted_by", "data-plane"),
            aligns_with_span=data.get("aligns_with_span"),
        )


# Map from span name suffix to attribute section key used in otel-semantic.yaml.
# Order: longer suffixes first so "mcp.tool.execute.attempt" matches before "mcp.tool.execute".
_SPEC_SPAN_SECTION_ALIASES = [
    ("mcp.tool.execute.attempt", "mcp_tool_attributes"),
    ("mcp.tool.execute", "mcp_tool_attributes"),
    ("llm.tool.response.bridge", "llm_tool_response_bridge_attributes"),
    ("context.augment", "context_augment_attributes"),
    ("a2a.orchestrate", "a2a_orchestration_attributes"),
    ("response.compose", "a2a_orchestration_attributes"),
    ("task.execute", "task_execute_attributes"),
    ("llm.call", "llm_call_attributes"),
    ("planner", "planner_attributes"),
    ("tools.recommend", "tools_recommend_attributes"),
]


@dataclass
class TelemetrySchema:
    """Complete telemetry schema parsed from YAML."""

    schema_version: str
    spans: dict[str, SpanDefinition]
    resource_attributes: dict[str, AttributeDefinition]
    common_attributes: dict[str, AttributeDefinition]
    span_attributes: dict[str, dict[str, AttributeDefinition]]
    metrics: dict[str, MetricDefinition]
    status_codes: dict[str, StatusDefinition]

    def get_span_attributes(self, span_name: str) -> dict[str, AttributeDefinition]:
        """Get all attributes for a span type (common + span-specific)."""
        attrs = dict(self.common_attributes)
        span_key = self._span_name_to_attr_key(span_name)
        if span_key in self.span_attributes:
            attrs.update(self.span_attributes[span_key])
        else:
            # Fallback: match by suffix so otel-semantic.yaml section names work (e.g. a2a_orchestration_attributes).
            for suffix, section in _SPEC_SPAN_SECTION_ALIASES:
                if span_name == suffix or span_name.endswith("." + suffix):
                    if section in self.span_attributes:
                        attrs.update(self.span_attributes[section])
                    break
        return attrs

    def get_required_attributes(self, span_name: str) -> list[str]:
        """Get list of required attribute names for a span type."""
        attrs = self.get_span_attributes(span_name)
        return [name for name, attr in attrs.items() if attr.required]

    def _span_name_to_attr_key(self, span_name: str) -> str:
        """Convert span name to attribute section key (e.g. vendor.mcp.tool.execute -> vendor_mcp_tool_execute_attributes)."""
        return span_name.replace(".", "_") + "_attributes"


class SchemaParser:
    """Parse OTEL semantic conventions from YAML."""

    def __init__(self, schema_path: Path | str | None = None):
        """Initialize parser with schema path. Path must be provided by the client (env or argument)."""
        if schema_path is not None:
            self.schema_path = Path(schema_path)
            return
        env_path = os.environ.get("TELEMETRY_SIMULATOR_SCHEMA_PATH") or os.environ.get(
            "SCHEMA_PATH"
        )
        if not env_path or not Path(env_path).exists():
            raise FileNotFoundError(
                "Schema path is required. Set TELEMETRY_SIMULATOR_SCHEMA_PATH or pass --schema-path with the full path to your semantic-conventions YAML file."
            )
        self.schema_path = Path(env_path)

    def parse(self) -> TelemetrySchema:
        """Parse the schema file and return structured schema."""
        with open(self.schema_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}

        return TelemetrySchema(
            schema_version=data.get("schema_version", "1.0.0"),
            spans=self._parse_spans(data.get("span_names")),
            resource_attributes=self._parse_attributes(data.get("resource_attributes")),
            common_attributes=self._parse_attributes(data.get("common_attributes")),
            span_attributes=self._parse_span_attributes(data),
            metrics=self._parse_metrics(data.get("canonical_metrics")),
            status_codes=self._parse_status_codes(data.get("status_schema")),
        )

    def _parse_spans(self, span_data: dict | None) -> dict[str, SpanDefinition]:
        """Parse span definitions."""
        span_data = span_data if isinstance(span_data, dict) else {}
        return {name: SpanDefinition.from_yaml(name, defn) for name, defn in span_data.items()}

    def _parse_attributes(self, attr_data: dict | None) -> dict[str, AttributeDefinition]:
        """Parse attribute definitions."""
        attr_data = attr_data if isinstance(attr_data, dict) else {}
        return {name: AttributeDefinition.from_yaml(name, defn) for name, defn in attr_data.items()}

    def _parse_span_attributes(self, data: dict) -> dict[str, dict[str, AttributeDefinition]]:
        """Parse span-specific attribute sections (any key ending with _attributes)."""
        result = {}
        for key, val in data.items():
            if isinstance(key, str) and key.endswith("_attributes") and isinstance(val, dict):
                result[key] = self._parse_attributes(val)
        return result

    def _parse_metrics(self, metric_data: dict | None) -> dict[str, MetricDefinition]:
        """Parse metric definitions."""
        metric_data = metric_data if isinstance(metric_data, dict) else {}
        return {name: MetricDefinition.from_yaml(name, defn) for name, defn in metric_data.items()}

    def _parse_status_codes(self, status_data: dict | None) -> dict[str, StatusDefinition]:
        """Parse status code definitions."""
        status_data = status_data if isinstance(status_data, dict) else {}
        result = {}
        for applies_to in status_data.get("applies_to") or []:
            parts = applies_to.split("(")
            if len(parts) == 2:
                span_type = parts[1].rstrip(")")
                result[span_type] = StatusDefinition(
                    result_type="boolean",
                    code_values=[],
                    has_metadata=True,
                )
        return result
