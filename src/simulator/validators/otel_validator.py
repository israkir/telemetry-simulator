"""
Validate OTEL telemetry payloads against schema and semantic conventions.

Validates:
- Required attributes are present per span type
- Attribute types match schema definitions
- Allowed values are respected for enums
- OTEL GenAI semantic conventions compliance
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import VENDOR_NAME, schema_version_attr
from ..config import attr as config_attr
from ..config import span_name as config_span_name
from ..schemas.schema_parser import SchemaParser


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """A single validation error or warning."""

    severity: ValidationSeverity
    attribute: str
    message: str
    span_type: str | None = None
    expected: Any = None
    actual: Any = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        span_info = f" ({self.span_type})" if self.span_type else ""
        return f"{prefix}{span_info} {self.attribute}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a telemetry payload."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(self, error: ValidationError):
        """Add an error to the result."""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.valid = False
        else:
            self.warnings.append(error)

    def merge(self, other: "ValidationResult"):
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False

    def __str__(self) -> str:
        lines = []
        if self.valid:
            lines.append("✅ Validation passed")
        else:
            lines.append("❌ Validation failed")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        return "\n".join(lines)


# Vendor span suffixes that require GenAI attributes (validator builds full names at init).
_GENAI_REQUIRED_BY_SUFFIX = {
    "mcp.tool.execute": ["gen_ai.tool.name"],
    "mcp.tool.execute.attempt": [],  # minimal: mcp.tool.call.id, attempt.index, attempt.outcome, error.type, retry.reason only
    "llm.call": [
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    ],
}

# Span suffixes that do not carry correlation/context attributes (tenant, session, enduser, redaction).
_MINIMAL_SPAN_SUFFIXES = frozenset(
    {
        "task.execute",
        "llm.call",
        "tools.recommend",
        "mcp.tool.execute",
        "mcp.tool.execute.attempt",
        "response.compose",
    }
)

# Status suffix by span (vendor names built at init).
_STATUS_SUFFIX_BY_NONVENDOR = {
    "rag.retrieve": "rag.status",
    "a2a.call": "a2a.status",
    "cp.request": "cp.status",
}
_STATUS_SUFFIX_BY_VENDOR_SUFFIX = {
    "a2a.orchestrate": "turn.status",
    "mcp.tool.execute": "tool.status",
    "mcp.tool.execute.attempt": "tool.status",
}


class OtelValidator:
    """Validate OTEL telemetry payloads against schema and vendor conventions."""

    def __init__(self, schema_path: str | None = None):
        """Initialize validator with schema."""
        parser = SchemaParser(schema_path)
        self.schema = parser.parse()
        # Build vendor-prefixed span names for validation (no hardcoded vendor).
        self._vendor_root_name = config_span_name("a2a.orchestrate")
        self._valid_roots = [self._vendor_root_name, "cp.request"]
        self._genai_required = {
            config_span_name(s): attrs for s, attrs in _GENAI_REQUIRED_BY_SUFFIX.items()
        }
        self._llm_call_name = config_span_name("llm.call")
        self._status_suffixes = dict(_STATUS_SUFFIX_BY_NONVENDOR)
        for suffix, status in _STATUS_SUFFIX_BY_VENDOR_SUFFIX.items():
            self._status_suffixes[config_span_name(suffix)] = status

    def validate_span(
        self,
        span_name: str,
        attributes: dict[str, Any],
        resource_attributes: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate a span's attributes against the schema."""
        result = ValidationResult(valid=True)

        all_attrs = dict(attributes)
        if resource_attributes:
            for k, v in resource_attributes.items():
                if k not in all_attrs:
                    all_attrs[k] = v

        self._validate_required_attributes(span_name, all_attrs, result)

        self._validate_attribute_types(span_name, all_attrs, result)

        self._validate_allowed_values(span_name, all_attrs, result)

        self._validate_otel_genai_conventions(span_name, all_attrs, result)

        self._validate_vendor_conventions(span_name, all_attrs, result)

        return result

    def _validate_required_attributes(
        self,
        span_name: str,
        attributes: dict[str, Any],
        result: ValidationResult,
    ):
        """Check that all required attributes are present."""
        required_attrs = self.schema.get_required_attributes(span_name)

        for attr_name in required_attrs:
            if attr_name not in attributes:
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.ERROR,
                        attribute=attr_name,
                        message="Required attribute missing",
                        span_type=span_name,
                    )
                )

    def _validate_attribute_types(
        self,
        span_name: str,
        attributes: dict[str, Any],
        result: ValidationResult,
    ):
        """Check that attribute values match expected types."""
        schema_attrs = self.schema.get_span_attributes(span_name)

        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "int": int,
            "double": (int, float),
            "boolean": bool,
        }

        for attr_name, value in attributes.items():
            if attr_name not in schema_attrs:
                continue

            attr_def = schema_attrs[attr_name]
            expected_type = type_map.get(attr_def.attr_type)

            if expected_type is not None and not isinstance(value, expected_type):
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.WARNING,
                        attribute=attr_name,
                        message=f"Type mismatch: expected {attr_def.attr_type}, got {type(value).__name__}",
                        span_type=span_name,
                        expected=attr_def.attr_type,
                        actual=type(value).__name__,
                    )
                )

    def _validate_allowed_values(
        self,
        span_name: str,
        attributes: dict[str, Any],
        result: ValidationResult,
    ):
        """Check that enum attributes have valid values."""
        schema_attrs = self.schema.get_span_attributes(span_name)

        for attr_name, value in attributes.items():
            if attr_name not in schema_attrs:
                continue

            attr_def = schema_attrs[attr_name]
            if attr_def.allowed_values and value not in attr_def.allowed_values:
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.ERROR,
                        attribute=attr_name,
                        message=f"Invalid value: '{value}' not in allowed values",
                        span_type=span_name,
                        expected=attr_def.allowed_values,
                        actual=value,
                    )
                )

    def _validate_otel_genai_conventions(
        self,
        span_name: str,
        attributes: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate OTEL GenAI semantic convention compliance."""
        if span_name not in self._genai_required:
            return

        required = self._genai_required[span_name]
        for attr_name in required:
            if attr_name not in attributes:
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.WARNING,
                        attribute=attr_name,
                        message="OTEL GenAI recommended attribute missing",
                        span_type=span_name,
                    )
                )

        if span_name == self._llm_call_name:
            valid_ops = ["chat", "text_completion", "generate_content"]
            op = attributes.get("gen_ai.operation.name")
            if op and op not in valid_ops:
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.WARNING,
                        attribute="gen_ai.operation.name",
                        message="OTEL GenAI: invalid operation name for inference",
                        span_type=span_name,
                        expected=valid_ops,
                        actual=op,
                    )
                )

    def _validate_vendor_conventions(
        self,
        span_name: str,
        attributes: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate vendor-specific conventions (attribute prefix from config)."""
        suffix = span_name.split(".", 1)[-1] if "." in span_name else span_name
        is_minimal = suffix in _MINIMAL_SPAN_SUFFIXES

        tenant_attr = config_attr("tenant.id")
        if not is_minimal and tenant_attr not in attributes:
            result.add_error(
                ValidationError(
                    severity=ValidationSeverity.ERROR,
                    attribute=tenant_attr,
                    message=f"{VENDOR_NAME}: tenant.id is required on this span",
                    span_type=span_name,
                )
            )

        session_attr = config_attr("session.id")
        if not is_minimal and session_attr not in attributes:
            result.add_error(
                ValidationError(
                    severity=ValidationSeverity.WARNING,
                    attribute=session_attr,
                    message=f"{VENDOR_NAME}: {session_attr} should be present for correlation",
                    span_type=span_name,
                )
            )

        schema_attr = schema_version_attr()
        if "schema.version" not in attributes and schema_attr not in attributes:
            result.add_error(
                ValidationError(
                    severity=ValidationSeverity.INFO,
                    attribute="schema.version",
                    message=f"{VENDOR_NAME}: schema.version not present (may be added by collector)",
                    span_type=span_name,
                )
            )

        if span_name in self._status_suffixes:
            suffix = self._status_suffixes[span_name]
            result_attr = config_attr(f"{suffix}.result")
            code_attr = config_attr(f"{suffix}.code")

            if result_attr not in attributes:
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.WARNING,
                        attribute=result_attr,
                        message=f"{VENDOR_NAME}: status.result should be present",
                        span_type=span_name,
                    )
                )

            if code_attr not in attributes:
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.WARNING,
                        attribute=code_attr,
                        message=f"{VENDOR_NAME}: status.code should be present",
                        span_type=span_name,
                    )
                )

    def validate_trace(
        self,
        spans: list[dict[str, Any]],
        resource_attributes: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate all spans in a trace."""
        result = ValidationResult(valid=True)

        for span in spans:
            span_name = span.get("name", "unknown")
            span_attrs = span.get("attributes", {})
            span_result = self.validate_span(span_name, span_attrs, resource_attributes)
            result.merge(span_result)

        self._validate_trace_structure(spans, result)

        return result

    def _validate_trace_structure(
        self,
        spans: list[dict[str, Any]],
        result: ValidationResult,
    ):
        """Validate trace hierarchy structure."""
        span_ids = {span.get("span_id") for span in spans}
        root_spans = [s for s in spans if s.get("parent_span_id") not in span_ids]

        if len(root_spans) == 0:
            result.add_error(
                ValidationError(
                    severity=ValidationSeverity.WARNING,
                    attribute="trace_structure",
                    message="No root span found in trace",
                )
            )
        elif len(root_spans) > 1:
            result.add_error(
                ValidationError(
                    severity=ValidationSeverity.INFO,
                    attribute="trace_structure",
                    message=f"Multiple root spans found: {len(root_spans)}",
                )
            )

        for root in root_spans:
            root_name = root.get("name", "")
            if root_name and root_name not in self._valid_roots:
                result.add_error(
                    ValidationError(
                        severity=ValidationSeverity.WARNING,
                        attribute="trace_structure",
                        message=f"Unexpected root span type: {root_name}",
                        expected=self._valid_roots,
                        actual=root_name,
                    )
                )

    def get_schema_summary(self) -> dict[str, Any]:
        """Get a summary of the schema for documentation."""
        return {
            "schema_version": self.schema.schema_version,
            "span_types": list(self.schema.spans.keys()),
            "resource_attributes": list(self.schema.resource_attributes.keys()),
            "common_attributes": list(self.schema.common_attributes.keys()),
            "metrics": list(self.schema.metrics.keys()),
        }
