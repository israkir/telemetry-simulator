"""
Console exporters for debugging and development.

Prints telemetry to stdout for quick verification.
"""

from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter


def create_console_exporters():
    """
    Create console exporters for all signal types.

    Returns:
        Tuple of (trace_exporter, metric_exporter, log_exporter)
    """
    return (
        ConsoleSpanExporter(),
        ConsoleMetricExporter(),
        ConsoleLogRecordExporter(),
    )
