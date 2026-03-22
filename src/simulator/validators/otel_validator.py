"""
OTEL schema validator and summary.

Uses parsed semconv schema to provide schema summary for CLI validate.
"""

from pathlib import Path
from typing import Any

from ..schemas.schema_parser import ParsedSchema, SchemaParser


class OtelValidator:
    """Validate and summarize semantic conventions schema."""

    def __init__(self, schema_path: str | Path | None = None):
        self._parser = SchemaParser(schema_path)
        self._schema: ParsedSchema | None = None

    def _get_schema(self) -> ParsedSchema:
        if self._schema is None:
            self._schema = self._parser.parse()
        return self._schema

    def get_schema_summary(self) -> dict[str, Any]:
        """Return summary dict: span_types, resource_attributes count, common_attributes count, metrics count."""
        schema = self._get_schema()
        return {
            "span_types": list(schema.spans.keys()),
            "resource_attributes": len(schema.resource_attributes),
            "common_attributes": len(schema.common_attributes),
            "metrics": len(schema.metrics),
        }
