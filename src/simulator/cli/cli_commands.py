"""
Command handlers used by the CLI entrypoint.
"""

from __future__ import annotations

import argparse
import sys

from ..defaults import get_default_tenant_ids
from ..exporters.otlp_exporter import create_otlp_trace_exporter
from ..scenarios.scenario_loader import ScenarioLoader
from ..scenarios.scenario_runner import ScenarioRunner
from ..resource_loader import load_resource_presets
from ..schemas.schema_parser import SchemaParser
from ..validators.otel_validator import OtelValidator
from .cli_helpers import (
    build_file_exporters,
    create_log_exporter_with_fallback,
    create_metric_exporter_pair_with_fallback,
    format_progress_line,
    format_workload_pick_tag,
    print_traces_by_scenario_grouped,
)


def _get_service_version_for_banner() -> str | None:
    """
    Read `service.version` from the configured OTEL Resource presets.

    This is intentionally derived from the same `resource/config/resource.yaml`
    that drives exported `resource` attributes, so the banner matches what
    collectors (e.g., Jaeger) will observe.
    """
    presets = load_resource_presets()
    for key in ("control-plane", "data-plane"):
        attrs = (presets.get(key) or {}).get("attributes") or {}
        value = attrs.get("service.version")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def cmd_run(args: argparse.Namespace) -> None:
    """Run mixed workload generation."""
    print("Starting mixed workload telemetry generation...")
    print(f"   Endpoint: {args.endpoint}")
    service_version = _get_service_version_for_banner()
    if service_version:
        print(f"   service.version: {service_version}")
    else:
        print("   service.version: (unknown)")
    each_once = args.each_once
    if each_once:
        print("   Mode: each (tagged) scenario once")
    else:
        count_text = args.count if args.count is not None else "until interrupted"
        print(f"   Count: {count_text}")
    print(f"   Interval: {args.interval}ms")

    tags_list: list[str] | None = None
    if args.tags:
        tags_list = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        if tags_list:
            print(f"   Tags filter: {', '.join(tags_list)}")

    tenant_ids = get_default_tenant_ids()
    tenant_override = args.tenant_id
    if isinstance(tenant_override, str):
        tenant = tenant_override.strip()
        if tenant:
            tenant_ids = [item for item in tenant_ids if item != tenant]
            tenant_ids.insert(0, tenant)
    print(f"   Tenants: {', '.join(tenant_ids)}")
    print()

    if args.output_file:
        (
            trace_exporter,
            trace_exporter_dp,
            metric_exporter,
            metric_exporter_cp,
            log_exporter,
        ) = build_file_exporters(
            output_file=args.output_file,
            disable_metrics=bool(args.no_metrics),
            disable_logs=bool(args.no_logs),
        )
        print(f"   Output: {args.output_file}")
    else:
        # Separate OTLP exporter instances: each TracerProvider's BatchSpanProcessor
        # calls shutdown() on its exporter exactly once.
        trace_exporter = create_otlp_trace_exporter(args.endpoint)
        trace_exporter_dp = create_otlp_trace_exporter(args.endpoint)
        metric_exporter, metric_exporter_cp = create_metric_exporter_pair_with_fallback(
            endpoint=args.endpoint,
            disabled=args.no_metrics,
            fallback_file="traces_metrics.jsonl",
        )
        log_exporter = create_log_exporter_with_fallback(
            endpoint=args.endpoint,
            disabled=args.no_logs,
            fallback_file="traces_logs.jsonl",
        )
        print("   Output: OTLP")

    print()

    verbose_traces = args.show_spans or args.show_all_attributes or args.show_full_spans

    def progress_callback(
        current: int,
        total: int,
        *,
        trace_id: str = "",
        tenant_id: str = "",
        span_names: list[str] | None = None,
        scenario_name: str | None = None,
        workload_pick: int | None = None,
        workload_total: int | None = None,
    ) -> None:
        """Progress for mixed workload (``workload_pick`` always set for ``run``)."""
        label = scenario_name or "scenario"
        pick_tag = (
            format_workload_pick_tag(workload_pick, workload_total)
            if workload_pick is not None
            else ""
        )

        if verbose_traces:
            detail = format_progress_line(current, total, trace_id, tenant_id, span_names).strip()
            if pick_tag:
                print(f"   {pick_tag} {label} · {detail}")
            else:
                print(f"   {detail}")
            return

        if total == 1:
            return

        if total > 1 and (current % 10 == 0 or current == total):
            print(f"   {pick_tag} {label} — turns completed: {current}/{total}")

    def pick_complete(pick: int, pick_total: int | None, scen_name: str, ids: list[str]) -> None:
        if verbose_traces:
            return
        n_traces = len(ids)
        print(
            f"   {format_workload_pick_tag(pick, pick_total)} {scen_name} — "
            f"{n_traces} trace(s) emitted"
        )

    runner = None
    try:
        runner = ScenarioRunner(
            trace_exporter=trace_exporter,
            trace_exporter_dp=trace_exporter_dp,
            metric_exporter=metric_exporter,
            metric_exporter_control_plane=metric_exporter_cp,
            log_exporter=log_exporter,
            schema_path=args.semconv,
            service_name=args.service_name,
            show_full_spans=args.show_full_spans,
            scenarios_dir=args.scenarios_dir,
            default_tenant_id=args.tenant_id,
        )

        trace_ids, traces_by_scenario = runner.run_mixed_workload(
            count=args.count,
            interval_ms=args.interval,
            progress_callback=progress_callback,
            pick_complete_callback=pick_complete,
            tags=tags_list,
            each_once=each_once,
        )
        runner.shutdown()

        print()
        print(f"Generated {len(trace_ids)} traces")
        if traces_by_scenario:
            total_from_scenarios = sum(traces_by_scenario.values())
            print(f"   Traces per scenario (total: {total_from_scenarios}):")
            assert runner.scenario_loader is not None  # set by run_mixed_workload
            print_traces_by_scenario_grouped(runner.scenario_loader, traces_by_scenario)

        if not (args.show_spans or args.show_all_attributes) and trace_ids:
            print()
            max_show = 10
            to_show = trace_ids if len(trace_ids) <= max_show else trace_ids[:max_show]
            label = (
                "Trace IDs:"
                if len(trace_ids) <= max_show
                else f"Trace IDs (first {max_show} of {len(trace_ids)}):"
            )
            print(label)
            for trace_id in to_show:
                print(f"   {trace_id}")
            logical_requests = 0 if each_once else (args.count if args.count is not None else 0)
            if logical_requests and len(trace_ids) != logical_requests:
                per_pick = len(trace_ids) / float(logical_requests)
                print(
                    f"   \nNote: Trace count ({len(trace_ids)}) differs from pick count "
                    f"({logical_requests}) because each pick runs a scenario that may emit "
                    f"multiple traces (≈{per_pick:.1f} traces/pick on average; one trace_id per turn)."
                )
    except KeyboardInterrupt:
        # Short flush deadlines so a stuck collector does not require a second Ctrl+C.
        if runner is not None:
            try:
                runner.shutdown(fast=True)
            except KeyboardInterrupt:
                pass
            except Exception:
                pass
        print("\nGeneration interrupted")
        sys.exit(0)
    except Exception as error:
        print(f"\nError: {error}")
        sys.exit(1)


def cmd_scenario(args: argparse.Namespace) -> None:
    """Run a single YAML-defined scenario."""
    print(f"Running scenario: {args.name}")
    loader = ScenarioLoader(args.scenarios_dir)
    service_version = _get_service_version_for_banner()

    try:
        scenario = loader.load(args.name)
    except FileNotFoundError:
        available = loader.list_scenarios()
        print(f"Scenario not found: {args.name}")
        print(f"   Available scenarios: {', '.join(available)}")
        sys.exit(1)

    if args.count:
        scenario.repeat_count = args.count

    print(f"   Description: {scenario.description}")
    print(f"   Repeat count: {scenario.repeat_count}")
    if service_version:
        print(f"   service.version: {service_version}")
    else:
        print("   service.version: (unknown)")
    if not args.output_file:
        print(f"   Endpoint: {args.endpoint}")
        print(f"   Service: {args.service_name} (select this in Jaeger UI)")
    if args.interval is not None:
        interval_str = f"{args.interval}ms (CLI override)"
        if args.interval_deviation is not None and args.interval_deviation > 0:
            interval_str += f" ±{args.interval_deviation}ms"
    else:
        interval_str = f"{scenario.trace_interval_ms}ms"
        if scenario.trace_interval_deviation_ms > 0:
            interval_str += f" ±{scenario.trace_interval_deviation_ms}ms"
    print(f"   Interval: {interval_str}")
    print(f"   Tags: {', '.join(scenario.tags)}")
    print()

    if args.output_file:
        (
            trace_exporter,
            trace_exporter_dp,
            metric_exporter,
            metric_exporter_cp,
            log_exporter,
        ) = build_file_exporters(
            output_file=args.output_file,
            disable_metrics=not scenario.emit_metrics or args.no_metrics,
            disable_logs=not scenario.emit_logs or args.no_logs,
        )
        print(f"   Output: {args.output_file}")
    else:
        trace_exporter = create_otlp_trace_exporter(args.endpoint)
        trace_exporter_dp = create_otlp_trace_exporter(args.endpoint)
        metric_exporter, metric_exporter_cp = create_metric_exporter_pair_with_fallback(
            endpoint=args.endpoint,
            disabled=not scenario.emit_metrics or args.no_metrics,
            fallback_file="traces_metrics.jsonl",
        )
        log_exporter = create_log_exporter_with_fallback(
            endpoint=args.endpoint,
            disabled=not scenario.emit_logs or args.no_logs,
            fallback_file="traces_logs.jsonl",
        )

    scenario_verbose = args.show_spans or args.show_all_attributes or args.show_full_spans

    def progress_callback(
        current: int,
        total: int,
        *,
        trace_id: str = "",
        tenant_id: str = "",
        span_names: list[str] | None = None,
        scenario_name: str | None = None,
    ) -> None:
        name = scenario_name or args.name
        if scenario_verbose:
            print(format_progress_line(current, total, trace_id, tenant_id, span_names))
            return
        if total > 1 and (current % 10 == 0 or current == total):
            print(f"   [{name}] turns completed: {current}/{total}")
        elif total == 1 and current == total:
            tid_disp = ""
            if trace_id:
                tid_disp = trace_id[:16] + "…" if len(trace_id) > 16 else trace_id
                tid_disp = f" (trace_id={tid_disp})"
            print(f"   [{name}] completed 1 turn{tid_disp}")

    runner = None
    try:
        runner = ScenarioRunner(
            trace_exporter=trace_exporter,
            trace_exporter_dp=trace_exporter_dp,
            metric_exporter=metric_exporter,
            metric_exporter_control_plane=metric_exporter_cp,
            log_exporter=log_exporter,
            schema_path=args.semconv,
            service_name=args.service_name,
            show_full_spans=args.show_full_spans,
            scenarios_dir=args.scenarios_dir,
            default_tenant_id=args.tenant_id,
        )

        run_kw: dict = {}
        if args.interval is not None:
            run_kw["trace_interval_ms"] = args.interval
            if args.interval_deviation is not None:
                run_kw["trace_interval_deviation_ms"] = args.interval_deviation
        trace_ids = runner.run_scenario(scenario, progress_callback=progress_callback, **run_kw)
        runner.shutdown()

        print()
        print(f"Generated {len(trace_ids)} traces")
        if not (args.show_spans or args.show_all_attributes) and trace_ids:
            max_show = 10
            to_show = trace_ids if len(trace_ids) <= max_show else trace_ids[:max_show]
            label = (
                "Trace IDs:"
                if len(trace_ids) <= max_show
                else f"Trace IDs (first {max_show} of {len(trace_ids)}):"
            )
            print(label)
            for trace_id in to_show:
                print(f"   {trace_id}")
            if scenario.repeat_count and len(trace_ids) != scenario.repeat_count:
                per_repeat = len(trace_ids) / float(scenario.repeat_count)
                print(
                    f"   Note: Trace count ({len(trace_ids)}) differs from repeat_count "
                    f"({scenario.repeat_count}) when the scenario has multiple turns or end users "
                    f"(≈{per_repeat:.1f} traces per repeat on average)."
                )
    except KeyboardInterrupt:
        if runner is not None:
            try:
                runner.shutdown(fast=True)
            except KeyboardInterrupt:
                pass
            except Exception:
                pass
        print("\nGeneration interrupted")
        sys.exit(0)
    except Exception as error:
        print(f"\nError: {error}")
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    """List available scenarios."""
    loader = ScenarioLoader(args.scenarios_dir)
    scenarios = loader.load_all()
    if not scenarios:
        print("No scenarios found.")
        print(f"Looking in: {loader.scenarios_dir}")
        return

    print("Available scenarios:")
    print()
    for scenario in scenarios:
        tags = ", ".join(scenario.tags) if scenario.tags else "none"
        print(f"  - {scenario.name}")
        print(
            f"     {scenario.description[:60]}..."
            if len(scenario.description) > 60
            else f"     {scenario.description}"
        )
        print(f"     Tags: {tags}")
        interval_str = f"{scenario.trace_interval_ms}ms"
        if scenario.trace_interval_deviation_ms > 0:
            interval_str += f" ±{scenario.trace_interval_deviation_ms}ms"
        print(f"     Repeat: {scenario.repeat_count}, Interval: {interval_str}")
        print()


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate schema and show information."""
    try:
        parser = SchemaParser(args.semconv)
        schema = parser.parse()
        validator = OtelValidator(args.semconv)

        print("Schema loaded successfully")
        print(f"   Version: {schema.schema_version}")
        print(f"   Path: {parser.schema_path}")
        print()

        if args.show_schema:
            summary = validator.get_schema_summary()
            print("Schema Summary:")
            print(f"  Span types: {', '.join(summary['span_types'])}")
            print(f"  Resource attributes: {summary['resource_attributes']}")
            print(f"  Common attributes: {summary['common_attributes']}")
            print(f"  Metrics: {summary['metrics']}")
            print()

        if args.show_spans:
            print("Span Type Definitions:")
            for name, span_def in schema.spans.items():
                kind = getattr(span_def, "span_kind", "INTERNAL")
                kinds = (
                    ", ".join(str(item) for item in kind) if isinstance(kind, list) else str(kind)
                )
                root = " (root)" if getattr(span_def, "is_root", False) else ""
                parents = f" <- {', '.join(span_def.parent_spans)}" if span_def.parent_spans else ""
                print(f"  - {name} [{kinds}]{root}{parents}")
                print(f"    {span_def.description}")
            print()

        if args.show_metrics:
            print("Canonical Metrics:")
            for name, metric_def in schema.metrics.items():
                metric = metric_def if isinstance(metric_def, dict) else {}
                dimensions = metric.get("dimensions", [])
                if not isinstance(dimensions, list):
                    dimensions = []
                print(f"  - {name}")
                print(f"    Type: {metric.get('type', '')}, Unit: {metric.get('unit', '')}")
                print(f"    Dimensions: {', '.join(str(item) for item in dimensions)}")
                print(f"    Emitted by: {metric.get('emitted_by', '')}")
            print()

        if not (args.show_schema or args.show_spans or args.show_metrics):
            print("Use --show-schema, --show-spans, or --show-metrics for details")
    except Exception as error:
        print(f"Validation failed: {error}")
        sys.exit(1)
