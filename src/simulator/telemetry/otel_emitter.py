from __future__ import annotations

from dataclasses import dataclass

from opentelemetry.sdk._logs.export import LogRecordExporter
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Tracer

from ..generators.trace_generator import TraceSpec, render_trace_with_span_ids
from .export_constants import FLUSH_TIMEOUT_MILLIS
from .logs_emitter import SemconvLogsEmitter
from .metrics_emitter import SemconvMetricsEmitter


@dataclass(frozen=True)
class EmitTraceResult:
    trace_id: str | None
    span_ids: list[str]


class OtelEmitter:
    """Facade for emitting traces + semconv-derived metrics and logs."""

    def __init__(
        self,
        *,
        metric_exporter: MetricExporter | None,
        log_exporter: LogRecordExporter | None,
        schema_path: str | None,
        dp_resource: Resource,
        cp_resource: Resource | None = None,
        metric_exporter_control_plane: MetricExporter | None = None,
        export_metrics_interval_ms: int = 500,
        include_metric_span_trace_ids: bool = False,
    ) -> None:
        self._metrics_emitter = (
            SemconvMetricsEmitter(
                metric_exporter,
                exporter_control_plane=metric_exporter_control_plane,
                schema_path=schema_path,
                resource_data_plane=dp_resource,
                resource_control_plane=cp_resource,
                export_interval_ms=export_metrics_interval_ms,
                include_span_trace_ids=include_metric_span_trace_ids,
            )
            if metric_exporter is not None
            else None
        )
        self._logs_emitter = (
            SemconvLogsEmitter(
                log_exporter,
                schema_path=schema_path,
                resource=dp_resource,
            )
            if log_exporter is not None
            else None
        )

    def emit_trace(
        self,
        tracer_cp: Tracer,
        tracer_dp: Tracer,
        trace_spec: TraceSpec,
        *,
        trace_base_time_ns: int | None = None,
    ) -> EmitTraceResult:
        trace_id, span_ids = render_trace_with_span_ids(
            tracer_dp,
            trace_spec,
            tracer_cp=tracer_cp,
            tracer_dp=tracer_dp,
            trace_base_time_ns=trace_base_time_ns,
        )

        if self._metrics_emitter is not None:
            self._metrics_emitter.record_trace(
                trace_spec,
                trace_id_hex=trace_id,
                span_id_hex_by_index=span_ids,
            )

        if self._logs_emitter is not None and trace_id is not None:
            self._logs_emitter.record_trace(
                trace_spec,
                trace_id_hex=trace_id,
                span_id_hex_by_index=span_ids,
            )

        # Rely on BatchSpanProcessor (traces), PeriodicExportingMetricReader (metrics),
        # and BatchLogRecordProcessor (logs) for batching. Per-trace force_flush would
        # serialize OTLP exports and overload collectors under high turn volume.

        return EmitTraceResult(trace_id=trace_id, span_ids=span_ids)

    def force_flush(self, *, timeout_millis: int | None = None) -> None:
        deadline = timeout_millis if timeout_millis is not None else FLUSH_TIMEOUT_MILLIS
        if self._metrics_emitter is not None:
            self._metrics_emitter.force_flush(timeout_millis=deadline)
        if self._logs_emitter is not None:
            self._logs_emitter.force_flush(timeout_millis=deadline)

    def shutdown(self, *, timeout_millis: int | None = None) -> None:
        deadline = timeout_millis if timeout_millis is not None else FLUSH_TIMEOUT_MILLIS
        if self._metrics_emitter is not None:
            self._metrics_emitter.shutdown(timeout_millis=deadline)
        if self._logs_emitter is not None:
            self._logs_emitter.shutdown()
