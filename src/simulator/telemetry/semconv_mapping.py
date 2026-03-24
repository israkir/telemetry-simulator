from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .. import config as sim_config
from ..schemas.schema_parser import SchemaParser


def _resolve_semconv_path(schema_path: str | Path | None) -> Path:
    parser = SchemaParser(schema_path)
    return Path(parser.schema_path)


def _map_namespace_key(key: str, runtime_prefix: str) -> str:
    # semconv uses `gentoro.*`, but the simulator emits `vendor.*` (or `--vendor` override).
    if key.startswith("gentoro."):
        return f"{runtime_prefix}.{key[len('gentoro.'):]}"
    return key


def _find_first_key_recursively(data: Any, target_key: str) -> Any | None:
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for v in data.values():
            found = _find_first_key_recursively(v, target_key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_first_key_recursively(item, target_key)
            if found is not None:
                return found
    return None


@dataclasses.dataclass(frozen=True)
class MetricInstrumentSpec:
    metric_name_semconv: str
    metric_name_runtime: str
    metric_type: str  # "counter" | "histogram"
    unit: str
    description: str
    dimension_keys_runtime: list[str]
    aligns_span_class: str
    value_attribute_key_runtime: str | None
    # semconv canonical_metrics.emitted_by — drives OTLP Resource (CP vs DP) for this instrument.
    emitted_by: str


@dataclasses.dataclass(frozen=True)
class SemconvMapping:
    runtime_prefix: str
    semconv_path: Path
    metric_specs_by_span_class: dict[str, list[MetricInstrumentSpec]]
    log_attribute_keys_runtime: list[str]

    @classmethod
    def from_schema_path(cls, schema_path: str | Path | None = None) -> SemconvMapping:
        runtime_prefix = sim_config.ATTR_PREFIX
        semconv_path = _resolve_semconv_path(schema_path)
        return _build_cached_mapping(str(semconv_path), runtime_prefix)


@lru_cache(maxsize=16)
def _build_cached_mapping(semconv_path_str: str, runtime_prefix: str) -> SemconvMapping:
    """Build and cache semconv mapping by path + runtime prefix."""
    semconv_path = Path(semconv_path_str)

    data: dict[str, Any] = {}
    if semconv_path.exists():
        raw = yaml.safe_load(semconv_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw

    canonical_metrics = data.get("canonical_metrics") or {}
    metric_specs_by_span_class: dict[str, list[MetricInstrumentSpec]] = {}

    for metric_key, metric_def in canonical_metrics.items():
        if not isinstance(metric_def, dict):
            continue
        metric_type = str(metric_def.get("type") or "").lower()
        if metric_type not in {"counter", "histogram"}:
            continue

        aligns_with_span = metric_def.get("aligns_with_span")
        if not isinstance(aligns_with_span, str) or not aligns_with_span:
            continue

        # semconv aligns_with_span is like `gentoro.mcp.tool.execute`; our TraceSpec stores
        # it under `<runtime_prefix>.span.class` without the `gentoro.` namespace.
        aligns_span_class = (
            aligns_with_span[len("gentoro.") :]
            if aligns_with_span.startswith("gentoro.")
            else aligns_with_span
        )

        metric_name_runtime = _map_namespace_key(str(metric_key), runtime_prefix)
        dimension_keys = metric_def.get("dimensions") or []
        if not isinstance(dimension_keys, list):
            dimension_keys = []

        dimension_keys_runtime = [
            _map_namespace_key(str(k), runtime_prefix) for k in dimension_keys
        ]

        value_attribute = metric_def.get("value_attribute")
        value_attribute_key_runtime = (
            _map_namespace_key(str(value_attribute), runtime_prefix)
            if isinstance(value_attribute, str) and value_attribute
            else None
        )

        emitted_by_raw = metric_def.get("emitted_by")
        emitted_by = (
            str(emitted_by_raw).strip().lower()
            if isinstance(emitted_by_raw, str) and emitted_by_raw.strip()
            else "data-plane"
        )
        if emitted_by not in ("data-plane", "control-plane"):
            emitted_by = "data-plane"

        spec = MetricInstrumentSpec(
            metric_name_semconv=str(metric_key),
            metric_name_runtime=metric_name_runtime,
            metric_type=metric_type,
            unit=str(metric_def.get("unit") or ""),
            description=str(metric_def.get("description") or ""),
            dimension_keys_runtime=dimension_keys_runtime,
            aligns_span_class=aligns_span_class,
            value_attribute_key_runtime=value_attribute_key_runtime,
            emitted_by=emitted_by,
        )

        metric_specs_by_span_class.setdefault(aligns_span_class, []).append(spec)

    logs_attributes = _find_first_key_recursively(data, "logs_attributes") or []
    if not isinstance(logs_attributes, list):
        logs_attributes = []

    log_attribute_keys_runtime = [
        _map_namespace_key(str(k), runtime_prefix) for k in logs_attributes if isinstance(k, str)
    ]

    return SemconvMapping(
        runtime_prefix=runtime_prefix,
        semconv_path=semconv_path,
        metric_specs_by_span_class=metric_specs_by_span_class,
        log_attribute_keys_runtime=log_attribute_keys_runtime,
    )
