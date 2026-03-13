"""
File-based exporters for offline analysis and debugging.

Writes telemetry to JSON files for:
- Offline validation
- Test fixtures
- Pipeline debugging
"""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry.sdk._logs.export import LogExporter, LogExportResult
from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult, MetricsData
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class FileSpanExporter(SpanExporter):
    """Export spans to a JSON file."""

    def __init__(self, output_path: str | Path, append: bool = True):
        """Initialize file exporter."""
        self.output_path = Path(output_path)
        self.append = append
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not append and self.output_path.exists():
            self.output_path.unlink()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to file."""
        try:
            span_dicts: list[dict[str, Any]] = []
            for span in spans:
                events_serialized: list[dict[str, Any]] = []
                if getattr(span, "events", None):
                    for ev in span.events:
                        events_serialized.append(
                            {
                                "name": ev.name,
                                "timestamp": ev.timestamp,
                                "attributes": dict(ev.attributes) if ev.attributes else {},
                            }
                        )
                span_dict: dict[str, Any] = {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "status": {
                        "status_code": span.status.status_code.name,
                        "description": span.status.description,
                    },
                    "attributes": dict(span.attributes) if span.attributes else {},
                    "kind": span.kind.name if span.kind else "INTERNAL",
                    "resource": dict(span.resource.attributes) if span.resource else {},
                }
                if events_serialized:
                    span_dict["events"] = events_serialized
                span_dicts.append(span_dict)

            mode = "a" if self.append else "w"
            with open(self.output_path, mode, encoding="utf-8") as f:
                for span_dict in span_dicts:
                    f.write(json.dumps(span_dict, default=str) + "\n")

            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush."""
        return True


class FileMetricExporter(MetricExporter):
    """Export metrics to a JSON file."""

    def __init__(self, output_path: str | Path, append: bool = True):
        """Initialize file exporter."""
        super().__init__()
        self.output_path = Path(output_path)
        self.append = append
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not append and self.output_path.exists():
            self.output_path.unlink()

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10000,
        **kwargs,
    ) -> MetricExportResult:
        """Export metrics to file."""
        try:
            metric_dicts = []
            for resource_metrics in metrics_data.resource_metrics:
                resource_attrs = (
                    dict(resource_metrics.resource.attributes) if resource_metrics.resource else {}
                )

                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        metric_dict: dict[str, Any] = {
                            "name": metric.name,
                            "description": metric.description,
                            "unit": metric.unit,
                            "resource": resource_attrs,
                            "timestamp": datetime.now().isoformat(),
                        }

                        if hasattr(metric, "data") and metric.data:
                            data_points = []
                            for dp in metric.data.data_points:
                                point: dict[str, Any] = {
                                    "attributes": (
                                        dict(dp.attributes) if hasattr(dp, "attributes") else {}  # type: ignore[arg-type]
                                    ),
                                    "start_time": (
                                        dp.start_time_unix_nano
                                        if hasattr(dp, "start_time_unix_nano")
                                        else None
                                    ),
                                    "time": (
                                        dp.time_unix_nano if hasattr(dp, "time_unix_nano") else None
                                    ),
                                }
                                if hasattr(dp, "value"):
                                    point["value"] = dp.value
                                if hasattr(dp, "count"):
                                    point["count"] = dp.count
                                if hasattr(dp, "sum"):
                                    point["sum"] = dp.sum
                                data_points.append(point)
                            metric_dict["data_points"] = data_points

                        metric_dicts.append(metric_dict)

            mode = "a" if self.append else "w"
            with open(self.output_path, mode, encoding="utf-8") as f:
                for metric_dict in metric_dicts:
                    f.write(json.dumps(metric_dict, default=str) + "\n")

            return MetricExportResult.SUCCESS
        except Exception:
            return MetricExportResult.FAILURE

    def shutdown(self, timeout_millis: float = 30000, **kwargs) -> None:
        """Shutdown exporter."""
        pass

    def force_flush(self, timeout_millis: float = 10000) -> bool:
        """Force flush."""
        return True


class FileLogExporter(LogExporter):
    """Export logs to a JSON file."""

    def __init__(self, output_path: str | Path, append: bool = True):
        """Initialize file exporter."""
        self.output_path = Path(output_path)
        self.append = append
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not append and self.output_path.exists():
            self.output_path.unlink()

    def export(self, batch: Sequence) -> LogExportResult:  # type: ignore[override]
        """Export logs to file."""
        try:
            log_dicts = []
            for log_record in batch:
                # OpenTelemetry 1.40+ wraps LogRecord in LogData; unwrap when needed so we
                # can read timestamp, body, attributes, trace_id, and span_id consistently.
                record = getattr(log_record, "log_record", log_record)
                log_dict = {
                    "timestamp": getattr(record, "timestamp", None),
                    "observed_timestamp": (
                        getattr(record, "observed_timestamp", None)
                    ),
                    "severity_number": (
                        getattr(record.severity_number, "value", None)
                        if getattr(record, "severity_number", None) is not None
                        else None
                    ),
                    "severity_text": (
                        getattr(record, "severity_text", None)
                    ),
                    "body": (
                        str(getattr(record, "body", None))
                        if getattr(record, "body", None) is not None
                        else None
                    ),
                    "attributes": (
                        dict(getattr(record, "attributes", {}) or {})
                    ),
                    "trace_id": (
                        format(getattr(record, "trace_id"), "032x")
                        if getattr(record, "trace_id", 0)
                        else None
                    ),
                    "span_id": (
                        format(getattr(record, "span_id"), "016x")
                        if getattr(record, "span_id", 0)
                        else None
                    ),
                    "resource": (
                        dict(log_record.resource.attributes)
                        if hasattr(log_record, "resource") and log_record.resource
                        else {}
                    ),
                }
                log_dicts.append(log_dict)

            mode = "a" if self.append else "w"
            with open(self.output_path, mode, encoding="utf-8") as f:
                for log_dict in log_dicts:
                    f.write(json.dumps(log_dict, default=str) + "\n")

            return LogExportResult.SUCCESS
        except Exception:
            return LogExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass
