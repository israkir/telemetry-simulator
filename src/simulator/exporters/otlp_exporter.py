"""
OTLP exporters for traces, metrics, and logs.

Provides factory functions for creating OTLP exporters with proper configuration.
Supports both HTTP and gRPC protocols.
"""

from typing import Any


def _normalize_http_base_endpoint(endpoint: str) -> str:
    """
    Normalize an OTLP HTTP base endpoint URL.

    The simulator CLI passes a single `--endpoint` value to all exporters. Users may
    include an OTLP signal suffix (e.g. `/v1/traces`) in that value; we strip any known
    signal suffix so each exporter can append its correct `/v1/<signal>` path.
    """

    e = endpoint.strip().rstrip("/")
    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if e.endswith(suffix):
            e = e[: -len(suffix)].rstrip("/")
            break
    return e


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

        traces_endpoint = f"{_normalize_http_base_endpoint(endpoint)}/v1/traces"
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

        metrics_endpoint = f"{_normalize_http_base_endpoint(endpoint)}/v1/metrics"
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

        logs_endpoint = f"{_normalize_http_base_endpoint(endpoint)}/v1/logs"
        return OTLPLogExporter(
            endpoint=logs_endpoint,
            headers=headers,
            **kwargs,
        )
