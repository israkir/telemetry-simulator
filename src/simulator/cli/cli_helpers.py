"""CLI helpers: exporter setup, OTLP probes, and progress formatting."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from ..exporters.file_exporter import (
    FileLogExporter,
    FileMetricCoordinator,
    FileMetricExporter,
    FileSpanCoordinator,
    FileSpanExporter,
)
from ..exporters.otlp_exporter import create_otlp_log_exporter, create_otlp_metric_exporter
from ..scenarios.scenario_loader import ScenarioLoader

# Max span names and trace_id display length to avoid container log line truncation.
_MAX_SPAN_NAMES_IN_LOG = 12
_TRACE_ID_DISPLAY_LEN = 16


def format_workload_pick_tag(pick: int, pick_total: int | None) -> str:
    """Label for a mixed-workload pick; omit ``/total`` when there is no ``--count`` limit."""
    if pick_total is None:
        return f"[pick {pick}]"
    return f"[pick {pick}/{pick_total}]"


def normalize_otlp_http_base_endpoint(endpoint: str) -> str:
    """Treat `endpoint` as OTLP base and strip any known signal suffix."""
    normalized = endpoint.strip().rstrip("/")
    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")
            break
    return normalized


def otlp_http_path_exists(base_endpoint: str, path: str, *, timeout_s: float = 1.0) -> bool:
    """
    Probe whether an OTLP HTTP endpoint path exists.

    Uses `HEAD` to avoid request body. Treat 405 as existing path and only
    disable a signal on 404.
    """
    url = f"{base_endpoint}{path}"
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return int(response.status) != 404
    except urllib.error.HTTPError as error:
        return error.code != 404
    except urllib.error.URLError:
        return False


def create_metric_exporter_with_fallback(
    *,
    endpoint: str,
    disabled: bool,
    fallback_file: str,
    warn: Callable[[str], None] = print,
) -> Any:
    """Create OTLP metric exporter, falling back to file when `/v1/metrics` is missing."""
    if disabled:
        return None
    base = normalize_otlp_http_base_endpoint(endpoint)
    if otlp_http_path_exists(base, "/v1/metrics"):
        return create_otlp_metric_exporter(endpoint)
    missing_url = f"{base}/v1/metrics"
    warn(f"Warning: OTLP metrics endpoint missing at {missing_url}; writing to {fallback_file}")
    return FileMetricExporter(fallback_file)


def create_metric_exporter_pair_with_fallback(
    *,
    endpoint: str,
    disabled: bool,
    fallback_file: str,
    warn: Callable[[str], None] = print,
) -> tuple[Any, Any]:
    """Two metric exporter instances for paired CP/DP ``PeriodicExportingMetricReader``s.

    OpenTelemetry: each reader should use its own exporter instance so concurrent
    ``export()`` calls do not contend on a single OTLP client lock.
    """
    if disabled:
        return (None, None)
    base = normalize_otlp_http_base_endpoint(endpoint)
    if otlp_http_path_exists(base, "/v1/metrics"):
        return (
            create_otlp_metric_exporter(endpoint),
            create_otlp_metric_exporter(endpoint),
        )
    missing_url = f"{base}/v1/metrics"
    warn(f"Warning: OTLP metrics endpoint missing at {missing_url}; writing to {fallback_file}")
    m_coord = FileMetricCoordinator()
    ex_dp = FileMetricExporter(fallback_file, append=False, coordinator=m_coord)
    ex_cp = FileMetricExporter(fallback_file, append=False, coordinator=m_coord)
    return (ex_dp, ex_cp)


def create_log_exporter_with_fallback(
    *,
    endpoint: str,
    disabled: bool,
    fallback_file: str,
    warn: Callable[[str], None] = print,
) -> Any:
    """Create OTLP log exporter, falling back to file when `/v1/logs` is missing."""
    if disabled:
        return None
    base = normalize_otlp_http_base_endpoint(endpoint)
    if otlp_http_path_exists(base, "/v1/logs"):
        return create_otlp_log_exporter(endpoint)
    missing_url = f"{base}/v1/logs"
    warn(f"Warning: OTLP logs endpoint missing at {missing_url}; writing to {fallback_file}")
    return FileLogExporter(fallback_file)


def format_progress_line(
    current: int,
    total: int,
    trace_id: str,
    tenant_id: str,
    span_names: list[str] | None,
) -> str:
    """Format a compact progress line to avoid truncation in container logs."""
    trace = (
        trace_id[:_TRACE_ID_DISPLAY_LEN] + ".."
        if len(trace_id) > _TRACE_ID_DISPLAY_LEN
        else trace_id
    )
    names = span_names or []
    if len(names) <= _MAX_SPAN_NAMES_IN_LOG:
        spans_str = ",".join(names)
    else:
        hidden_count = len(names) - _MAX_SPAN_NAMES_IN_LOG
        spans_str = ",".join(names[:_MAX_SPAN_NAMES_IN_LOG]) + f",..+{hidden_count}"
    total_str = str(total) if total else "∞"
    return f"   [{current}/{total_str}] trace_id={trace} tenant_id={tenant_id} spans={spans_str}"


def print_traces_by_scenario_grouped(
    scenario_loader: ScenarioLoader,
    traces_by_scenario: dict[str, int],
) -> None:
    """Print traces-per-scenario grouped by scenario definition folder (sorted by group name)."""
    scenarios = scenario_loader.load_all()
    name_to_group = {
        scenario.name: (getattr(scenario, "definition_group", None) or "") for scenario in scenarios
    }
    traces_by_group: dict[str, list[tuple[str, int]]] = {}
    for scenario_name, count in traces_by_scenario.items():
        group = name_to_group.get(scenario_name, "")
        traces_by_group.setdefault(group, []).append((scenario_name, count))
    for group in sorted(traces_by_group.keys()):
        label = group if group else "root"
        entries = sorted(traces_by_group[group], key=lambda item: item[0])
        print(f"   [{label}]")
        for scenario_name, count in entries:
            print(f"      {scenario_name}: {count}")


def build_file_exporters(
    *,
    output_file: str,
    disable_metrics: bool,
    disable_logs: bool,
) -> tuple[
    FileSpanExporter,
    FileSpanExporter,
    FileMetricExporter | None,
    FileMetricExporter | None,
    FileLogExporter | None,
]:
    """Build trace/metric/log exporters that write to local files.

    Returns two span exporters (CP + DP) that append to the same file under one lock,
    and two metric exporters (CP + DP) that append to the same metrics file under one lock,
    so each ``BatchSpanProcessor`` / ``PeriodicExportingMetricReader`` pair uses distinct
    exporter instances (OpenTelemetry best practice; avoids OTLP export lock contention).
    """
    coord = FileSpanCoordinator()
    trace_exporter_cp = FileSpanExporter(output_file, coordinator=coord)
    trace_exporter_dp = FileSpanExporter(output_file, coordinator=coord)
    if disable_metrics:
        metric_exporter_cp = None
        metric_exporter_dp = None
    else:
        m_path = output_file.replace(".jsonl", "_metrics.jsonl")
        m_coord = FileMetricCoordinator()
        metric_exporter_cp = FileMetricExporter(m_path, append=False, coordinator=m_coord)
        metric_exporter_dp = FileMetricExporter(m_path, append=False, coordinator=m_coord)
    log_exporter = (
        FileLogExporter(output_file.replace(".jsonl", "_logs.jsonl")) if not disable_logs else None
    )
    return (
        trace_exporter_cp,
        trace_exporter_dp,
        metric_exporter_dp,
        metric_exporter_cp,
        log_exporter,
    )
