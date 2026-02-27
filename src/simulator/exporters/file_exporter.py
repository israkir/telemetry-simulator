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
            span_dicts = []
            for span in spans:
                span_dict = {
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
                log_dict = {
                    "timestamp": log_record.timestamp if hasattr(log_record, "timestamp") else None,
                    "observed_timestamp": (
                        log_record.observed_timestamp
                        if hasattr(log_record, "observed_timestamp")
                        else None
                    ),
                    "severity_number": (
                        log_record.severity_number.value
                        if hasattr(log_record, "severity_number") and log_record.severity_number
                        else None
                    ),
                    "severity_text": (
                        log_record.severity_text if hasattr(log_record, "severity_text") else None
                    ),
                    "body": (
                        str(log_record.body)
                        if hasattr(log_record, "body") and log_record.body
                        else None
                    ),
                    "attributes": (
                        dict(log_record.attributes)
                        if hasattr(log_record, "attributes") and log_record.attributes
                        else {}
                    ),
                    "trace_id": (
                        format(log_record.trace_id, "032x")
                        if hasattr(log_record, "trace_id") and log_record.trace_id
                        else None
                    ),
                    "span_id": (
                        format(log_record.span_id, "016x")
                        if hasattr(log_record, "span_id") and log_record.span_id
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
