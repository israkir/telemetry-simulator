from __future__ import annotations

import dataclasses
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from ..generators.trace_generator import TraceSpec
from .export_constants import FLUSH_TIMEOUT_MILLIS
from .semconv_mapping import MetricInstrumentSpec, SemconvMapping


@dataclasses.dataclass(frozen=True)
class _SpanCommonValues:
    tenant_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None


class SemconvMetricsEmitter:
    """
    Emit semconv canonical metrics using span classes + SpanSpec.duration_ms.

    Metrics whose semconv ``emitted_by`` is ``control-plane`` use *resource_control_plane*
    (so OTLP ``gentoro.module`` matches CP spans); all others use *resource_data_plane*.
    Downstream UIs often join trace details to metrics by resource + span id.

    Note: The current simulator doesn't replicate all semconv common attributes onto every
    span. This emitter backfills a small set of commonly-used dimension values by scanning
    the TraceSpec for those attributes.

    IMPORTANT: By default, trace_id/span_id are NOT included in metric attributes to avoid
    cardinality explosion. Enable ``include_span_trace_ids=True`` only for debugging.
    """

    def __init__(
        self,
        exporter_data_plane: MetricExporter,
        *,
        exporter_control_plane: MetricExporter | None = None,
        schema_path: str | None = None,
        resource_data_plane: Resource,
        resource_control_plane: Resource | None = None,
        export_interval_ms: int = 500,
        include_span_trace_ids: bool = False,
    ) -> None:
        self._include_span_trace_ids = include_span_trace_ids
        self._mapping = SemconvMapping.from_schema_path(schema_path)
        self._resource_dp = resource_data_plane
        self._resource_cp = resource_control_plane or resource_data_plane
        # One PeriodicExportingMetricReader per exporter instance (OTLP clients are not
        # designed for concurrent export() from two readers on the same object).
        exporter_cp = exporter_control_plane or exporter_data_plane
        reader_dp = PeriodicExportingMetricReader(
            exporter_data_plane,
            export_interval_millis=export_interval_ms,
        )
        reader_cp = PeriodicExportingMetricReader(
            exporter_cp,
            export_interval_millis=export_interval_ms,
        )
        self._meter_provider_dp = MeterProvider(
            resource=resource_data_plane,
            metric_readers=[reader_dp],
            shutdown_on_exit=False,
        )
        self._meter_provider_cp = MeterProvider(
            resource=self._resource_cp,
            metric_readers=[reader_cp],
            shutdown_on_exit=False,
        )
        # Avoid overriding global meter provider (tests/other components may already set one).
        self._meter_dp = metrics.get_meter(
            "telemetry-simulator.metrics.dp",
            meter_provider=self._meter_provider_dp,
        )
        self._meter_cp = metrics.get_meter(
            "telemetry-simulator.metrics.cp",
            meter_provider=self._meter_provider_cp,
        )

        self._instrument_by_key: dict[tuple[Any, ...], Any] = {}
        self._init_instruments()

        # Keys we expect to exist in span attributes (when they are present at all).
        self._span_class_key = f"{self._mapping.runtime_prefix}.span.class"

    @staticmethod
    def _spec_key(spec: MetricInstrumentSpec) -> tuple[Any, ...]:
        # MetricInstrumentSpec is immutable but contains list fields, so use a fully hashable key.
        return (
            spec.metric_name_runtime,
            spec.metric_type,
            tuple(spec.dimension_keys_runtime),
            spec.aligns_span_class,
            spec.value_attribute_key_runtime,
        )

    def _init_instruments(self) -> None:
        for specs in self._mapping.metric_specs_by_span_class.values():
            for spec in specs:
                key = self._spec_key(spec)
                if key in self._instrument_by_key:
                    continue

                meter = self._meter_cp if spec.emitted_by == "control-plane" else self._meter_dp
                instr: Any
                if spec.metric_type == "counter":
                    instr = meter.create_counter(
                        spec.metric_name_runtime,
                        description=spec.description or spec.metric_name_runtime,
                        unit=spec.unit or "1",
                    )
                else:
                    instr = meter.create_histogram(
                        spec.metric_name_runtime,
                        description=spec.description or spec.metric_name_runtime,
                        unit=spec.unit or "",
                    )

                self._instrument_by_key[key] = instr

    def _extract_common_values(self, trace_spec: TraceSpec) -> _SpanCommonValues:
        tenant_id_key = f"{self._mapping.runtime_prefix}.tenant.id"
        session_id_key = f"{self._mapping.runtime_prefix}.session.id"
        conv_id_key = "gen_ai.conversation.id"

        tenant_id = None
        session_id = None
        conversation_id = None

        for span in trace_spec.spans:
            if tenant_id is None:
                v = span.attributes.get(tenant_id_key)
                if v is not None:
                    tenant_id = str(v)
            if session_id is None:
                v = span.attributes.get(session_id_key)
                if v is not None:
                    session_id = str(v)
            if conversation_id is None:
                v = span.attributes.get(conv_id_key)
                if v is not None:
                    conversation_id = str(v)
            if tenant_id is not None and session_id is not None and conversation_id is not None:
                break
        return _SpanCommonValues(
            tenant_id=tenant_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )

    def record_trace(
        self,
        trace_spec: TraceSpec,
        *,
        trace_id_hex: str | None = None,
        span_id_hex_by_index: list[str] | None = None,
    ) -> None:
        """
        Record metrics for all spans in the trace.

        Uses:
        - `span.duration_ms` for histogram values when `value_attribute` isn't provided.
        - span.attributes for dimension values when present.
        - backfilled common values for a small subset of dimension keys.

        When *trace_id_hex* / *span_id_hex_by_index* are set (after spans are rendered), point
        attributes include ``span_id``, ``span.id``, ``trace_id``, and ``trace.id`` so UIs such
        as Gentoro's trace details view can filter metric rows by selected span.
        """

        common = self._extract_common_values(trace_spec)

        for span_index, span in enumerate(trace_spec.spans):
            span_class = span.attributes.get(self._span_class_key)
            if span_class is None:
                continue
            span_class = str(span_class)

            specs = self._mapping.metric_specs_by_span_class.get(span_class) or []
            if not specs:
                continue

            for spec in specs:
                res = self._resource_cp if spec.emitted_by == "control-plane" else self._resource_dp
                resource_attrs = dict(res.attributes) if hasattr(res, "attributes") else {}

                dim_attrs: dict[str, Any] = {}
                missing = False
                for dim_key in spec.dimension_keys_runtime:
                    if dim_key in span.attributes:
                        val = span.attributes.get(dim_key)
                    elif dim_key in resource_attrs:
                        val = resource_attrs.get(dim_key)
                    else:
                        # Backfill a few common values that we don't currently attach to every span.
                        if dim_key == f"{self._mapping.runtime_prefix}.tenant.id":
                            val = common.tenant_id
                        elif dim_key == f"{self._mapping.runtime_prefix}.session.id":
                            val = common.session_id
                        elif dim_key == "gen_ai.conversation.id":
                            val = common.conversation_id
                        else:
                            val = None

                    if val is None:
                        missing = True
                        break

                    # OTel metric SDK expects primitive attribute values.
                    if isinstance(val, (bool, int, float, str)):
                        dim_attrs[dim_key] = val
                    else:
                        dim_attrs[dim_key] = str(val)

                if missing:
                    continue

                # High-cardinality trace/span IDs: only include when explicitly enabled.
                # Including them by default causes linear memory/CPU growth and can trigger
                # backend policy rejections (400 Bad Request) due to cardinality limits.
                if self._include_span_trace_ids:
                    if trace_id_hex:
                        dim_attrs["trace_id"] = trace_id_hex
                        dim_attrs["trace.id"] = trace_id_hex
                    if span_id_hex_by_index is not None and span_index < len(span_id_hex_by_index):
                        sid = span_id_hex_by_index[span_index]
                        if sid:
                            dim_attrs["span_id"] = sid
                            dim_attrs["span.id"] = sid

                instr = self._instrument_by_key.get(self._spec_key(spec))
                if instr is None:
                    continue

                if spec.metric_type == "counter":
                    instr.add(1, dim_attrs)
                else:
                    if spec.value_attribute_key_runtime:
                        v = span.attributes.get(spec.value_attribute_key_runtime)
                        if v is None or not isinstance(v, (int, float)):
                            continue
                        instr.record(v, dim_attrs)
                    else:
                        # Default duration-based histogram.
                        instr.record(int(span.duration_ms), dim_attrs)

    def force_flush(self, *, timeout_millis: int | None = None) -> None:
        deadline = timeout_millis if timeout_millis is not None else FLUSH_TIMEOUT_MILLIS
        for prov in (self._meter_provider_dp, self._meter_provider_cp):
            if hasattr(prov, "force_flush"):
                prov.force_flush(timeout_millis=deadline)

    def shutdown(self, *, timeout_millis: int | None = None) -> None:
        deadline = timeout_millis if timeout_millis is not None else FLUSH_TIMEOUT_MILLIS
        self._meter_provider_dp.shutdown(timeout_millis=deadline)
        self._meter_provider_cp.shutdown(timeout_millis=deadline)
