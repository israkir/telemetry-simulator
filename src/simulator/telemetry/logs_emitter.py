from __future__ import annotations

import dataclasses
import time
from typing import Any

from opentelemetry._logs import SeverityNumber, get_logger
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    TraceState,
    set_span_in_context,
)

from ..generators.trace_generator import TraceSpec
from .export_constants import FLUSH_TIMEOUT_MILLIS
from .semconv_mapping import SemconvMapping


@dataclasses.dataclass(frozen=True)
class _SpanCorrelation:
    trace_id_hex: str
    span_id_hex_by_index: list[str]


class SemconvLogsEmitter:
    """
    Emit logs derived from the compiled TraceSpec.

    This simulator cannot fully mirror application-emitted logs, but it does:
    - attach trace_id/span_id from the generated OTEL spans
    - attach semconv-required correlation attributes from span attributes
    - emit logs at key span classes for debugging/correlation
    """

    def __init__(
        self,
        exporter: LogRecordExporter,
        *,
        schema_path: str | None = None,
        resource: Resource,
        export_timeout_millis: float | None = None,
    ) -> None:
        self._mapping = SemconvMapping.from_schema_path(schema_path)

        # Logger provider is per-emitter so tests can use isolated FileLogExporter outputs.
        self._provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
        if export_timeout_millis is not None:
            processor = BatchLogRecordProcessor(
                exporter,
                export_timeout_millis=export_timeout_millis,
            )
        else:
            processor = BatchLogRecordProcessor(exporter)
        self._provider.add_log_record_processor(processor)
        self._logger = get_logger("telemetry-simulator.logs", logger_provider=self._provider)

        self._span_class_attr_key = f"{self._mapping.runtime_prefix}.span.class"

        # Initial set of “interesting” spans for correlated logs.
        # semconv canonical span classes for this simulator’s generated structure.
        self._interesting_span_classes = {
            "request.validation",
            "response.validation",
            "a2a.orchestrate",
            "mcp.tool.execute",
            "llm.call",
        }

    @staticmethod
    def _hex_to_int(hex_str: str, *, expected_len: int) -> int:
        # Defensive conversion: tests depend on consistent formatting.
        if len(hex_str) != expected_len:
            raise ValueError(f"Expected hex length {expected_len}, got {len(hex_str)}: {hex_str}")
        return int(hex_str, 16)

    def _build_required_attrs(
        self, trace_spec: TraceSpec, *, keys_runtime: list[str]
    ) -> dict[str, Any]:
        # Precompute key->value by scanning spans in dependency order.
        # Keep first encountered value for stable output.
        values: dict[str, Any] = {}
        for span in trace_spec.spans:
            for k in keys_runtime:
                if k in values:
                    continue
                if k in span.attributes:
                    values[k] = span.attributes[k]
            if len(values) == len(keys_runtime):
                break
        return values

    def _emit_for_span(
        self,
        *,
        index: int,
        trace_spec: TraceSpec,
        span_id_hex_by_index: list[str],
        trace_id_hex: str,
        required_attrs: dict[str, Any],
    ) -> None:
        span = trace_spec.spans[index]
        span_class = span.attributes.get(self._span_class_attr_key)
        if span_class is None:
            return
        span_class = str(span_class)
        if span_class not in self._interesting_span_classes:
            return

        span_id_hex = span_id_hex_by_index[index]

        trace_id_int = self._hex_to_int(trace_id_hex, expected_len=32)
        span_id_int = self._hex_to_int(span_id_hex, expected_len=16)

        # Create a non-recording span context so the log record can carry trace/span IDs.
        span_ctx = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),  # keep it simple; used only for propagation
            trace_state=TraceState(),
        )
        parent_span = NonRecordingSpan(span_ctx)
        ctx = set_span_in_context(parent_span)

        attrs: dict[str, Any] = dict(required_attrs)
        attrs[self._span_class_attr_key] = span_class

        # Give dashboards a human-friendly event name.
        attrs["event.name"] = f"span.{span_class}"

        # FileLogExporter records body + severity fields.
        now_ns = time.time_ns()
        self._logger.emit(
            body=f"telemetry-simulator {span_class}",
            attributes=attrs,
            context=ctx,
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            event_name="telemetry.span",
            timestamp=now_ns,
            observed_timestamp=now_ns,
        )

    def record_trace(
        self,
        trace_spec: TraceSpec,
        *,
        trace_id_hex: str,
        span_id_hex_by_index: list[str],
    ) -> None:
        required_attrs = self._build_required_attrs(
            trace_spec,
            keys_runtime=self._mapping.log_attribute_keys_runtime,
        )
        corr = _SpanCorrelation(
            trace_id_hex=trace_id_hex,
            span_id_hex_by_index=span_id_hex_by_index,
        )

        for i in range(len(trace_spec.spans)):
            self._emit_for_span(
                index=i,
                trace_spec=trace_spec,
                span_id_hex_by_index=corr.span_id_hex_by_index,
                trace_id_hex=corr.trace_id_hex,
                required_attrs=required_attrs,
            )

    def force_flush(self, *, timeout_millis: int | None = None) -> None:
        deadline = timeout_millis if timeout_millis is not None else FLUSH_TIMEOUT_MILLIS
        self._provider.force_flush(timeout_millis=deadline)

    def shutdown(self) -> None:
        self._provider.shutdown()
