"""
OTLP exporters for traces, metrics, and logs.

Provides factory functions for creating OTLP exporters with proper configuration.
Supports both HTTP and gRPC protocols.
"""

from typing import Any


def create_otlp_trace_exporter(
    endpoint: str = "http://localhost:4318",
    protocol: str = "http",
    headers: dict[str, str] | None = None,
    **kwargs: Any,
):
    """
    Create an OTLP trace exporter.

    Args:
        endpoint: OTLP endpoint URL
        protocol: "http" or "grpc"
        headers: Optional headers to include
        **kwargs: Additional exporter configuration

    Returns:
        Configured SpanExporter
    """
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(
            endpoint=endpoint.replace("http://", "").replace("https://", ""),
            headers=headers,
            **kwargs,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[assignment]
            OTLPSpanExporter,
        )

        traces_endpoint = endpoint
        if not traces_endpoint.endswith("/v1/traces"):
            traces_endpoint = f"{endpoint}/v1/traces"
        return OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers=headers,
            **kwargs,
        )


def create_otlp_metric_exporter(
    endpoint: str = "http://localhost:4318",
    protocol: str = "http",
    headers: dict[str, str] | None = None,
    **kwargs: Any,
):
    """
    Create an OTLP metric exporter.

    Args:
        endpoint: OTLP endpoint URL
        protocol: "http" or "grpc"
        headers: Optional headers to include
        **kwargs: Additional exporter configuration

    Returns:
        Configured MetricExporter
    """
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter(
            endpoint=endpoint.replace("http://", "").replace("https://", ""),
            headers=headers,
            **kwargs,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore[assignment]
            OTLPMetricExporter,
        )

        metrics_endpoint = endpoint
        if not metrics_endpoint.endswith("/v1/metrics"):
            metrics_endpoint = f"{endpoint}/v1/metrics"
        return OTLPMetricExporter(
            endpoint=metrics_endpoint,
            headers=headers,
            **kwargs,
        )


def create_otlp_log_exporter(
    endpoint: str = "http://localhost:4318",
    protocol: str = "http",
    headers: dict[str, str] | None = None,
    **kwargs: Any,
):
    """
    Create an OTLP log exporter.

    Args:
        endpoint: OTLP endpoint URL
        protocol: "http" or "grpc"
        headers: Optional headers to include
        **kwargs: Additional exporter configuration

    Returns:
        Configured LogExporter
    """
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        return OTLPLogExporter(
            endpoint=endpoint.replace("http://", "").replace("https://", ""),
            headers=headers,
            **kwargs,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (  # type: ignore[assignment]
            OTLPLogExporter,
        )

        logs_endpoint = endpoint
        if not logs_endpoint.endswith("/v1/logs"):
            logs_endpoint = f"{endpoint}/v1/logs"
        return OTLPLogExporter(
            endpoint=logs_endpoint,
            headers=headers,
            **kwargs,
        )
