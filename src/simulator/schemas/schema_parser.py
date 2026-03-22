"""
Parse semantic conventions YAML (semconv) for span names, hierarchy, and metrics.

All processing is driven by the YAML; no hardcoded span names or attributes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SpanDef:
    """Span type definition from semconv span_names."""

    name: str
    description: str = ""
    span_kind: str = "INTERNAL"
    parent_spans: list[str] = field(default_factory=list)
    is_root: bool = False


@dataclass
class SingleTraceRequestLifecycle:
    """From lineage.single_trace_request_lifecycle: parent-child order when emitting one trace per request."""

    root_span_class: str
    child_chain: list[str]


@dataclass
class ParsedSchema:
    """Parsed semconv schema."""

    schema_version: str = ""
    spans: dict[str, SpanDef] = field(default_factory=dict)
    resource_attributes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    common_attributes: list[str] = field(default_factory=list)
    single_trace_request_lifecycle: SingleTraceRequestLifecycle | None = None


class SchemaParser:
    """Load and parse semconv YAML."""

    def __init__(self, schema_path: str | Path | None = None):
        if schema_path is None:
            for base in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
                candidate = base / "resource" / "scenarios" / "conventions" / "semconv.yaml"
                if candidate.exists():
                    schema_path = candidate
                    break
            else:
                schema_path = Path.cwd() / "resource" / "scenarios" / "conventions" / "semconv.yaml"
        self.schema_path = Path(schema_path)

    def parse(self) -> ParsedSchema:
        """Parse semconv YAML into ParsedSchema."""
        if not self.schema_path.exists():
            return ParsedSchema()

        with open(self.schema_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        schema_version = data.get("schema_version", "")
        spans: dict[str, SpanDef] = {}
        span_names = data.get("span_names") or {}
        for name, defn in span_names.items():
            if not isinstance(defn, dict):
                continue
            kind = defn.get("span_kind", "INTERNAL")
            parent = defn.get("parent")
            parents = []
            if isinstance(parent, str):
                parents = [p.strip() for p in parent.split("|")]
            elif isinstance(parent, list):
                parents = list(parent)
            is_root = defn.get("is_root", False)
            spans[name] = SpanDef(
                name=name,
                description=defn.get("description", ""),
                span_kind=kind,
                parent_spans=parents,
                is_root=is_root,
            )

        resource_attributes = data.get("resource_attributes") or {}
        common_attributes = list(data.get("common_attributes") or {})
        metrics = data.get("canonical_metrics") or {}

        single_trace = None
        lineage = data.get("lineage") or {}
        st = lineage.get("single_trace_request_lifecycle")
        if (
            isinstance(st, dict)
            and st.get("root_span_class")
            and isinstance(st.get("child_chain"), list)
        ):
            single_trace = SingleTraceRequestLifecycle(
                root_span_class=str(st["root_span_class"]),
                child_chain=[str(c) for c in st["child_chain"]],
            )

        return ParsedSchema(
            schema_version=schema_version,
            spans=spans,
            resource_attributes=resource_attributes,
            metrics=metrics,
            common_attributes=common_attributes,
            single_trace_request_lifecycle=single_trace,
        )
