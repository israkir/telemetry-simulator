"""
Command-line interface for the Telemetry Simulator.

Provides commands for:
- Running YAML-defined scenarios
- Mixed workload simulation
- Validation and schema inspection
"""

import argparse
import os
import sys

from . import config as sim_config
from .defaults import get_default_tenant_ids
from .exporters.file_exporter import FileLogExporter, FileMetricExporter, FileSpanExporter
from .exporters.otlp_exporter import (
    create_otlp_log_exporter,
    create_otlp_metric_exporter,
    create_otlp_trace_exporter,
)
from .scenarios.scenario_loader import ScenarioLoader
from .scenarios.scenario_runner import ScenarioRunner
from .schemas.schema_parser import SchemaParser
from .validators.otel_validator import OtelValidator

# Max span names and trace_id display length to avoid container log line truncation (e.g. 16k limit)
_MAX_SPAN_NAMES_IN_LOG = 12
_TRACE_ID_DISPLAY_LEN = 16


def _format_progress_line(
    current: int,
    total: int,
    trace_id: str,
    tenant_id: str,
    span_names: list[str] | None,
) -> str:
    """Format a single progress line; keep short so container logs don't truncate."""
    tid = (
        (trace_id[:_TRACE_ID_DISPLAY_LEN] + "..")
        if len(trace_id) > _TRACE_ID_DISPLAY_LEN
        else trace_id
    )
    names = span_names or []
    if len(names) <= _MAX_SPAN_NAMES_IN_LOG:
        spans_str = ",".join(names)
    else:
        spans_str = (
            ",".join(names[:_MAX_SPAN_NAMES_IN_LOG]) + f",..+{len(names) - _MAX_SPAN_NAMES_IN_LOG}"
        )
    return f"   [{current}/{total}] trace_id={tid} tenant_id={tenant_id} spans={spans_str}"


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="otelsim",
        description="Schema-driven OTEL telemetry simulator for LLM observability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run mixed workload to OTLP collector
  otelsim run --count 100 --interval 200

  # Run specific scenario
  otelsim scenario --name new_claim_phone

  # Validate schema and show summary
  otelsim validate --show-schema

  # Export to file instead of OTLP
  otelsim run --count 10 --output-file traces.jsonl
        """,
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:4318",
        help="OTLP HTTP endpoint (default: http://localhost:4318)",
    )

    parser.add_argument(
        "--semconv",
        dest="semconv",
        type=str,
        default=None,
        help="Path to semantic-conventions YAML (required unless SEMCONV is set)",
    )

    parser.add_argument(
        "--vendor",
        type=str,
        default=None,
        help=(
            "Attribute prefix for vendor-specific attributes "
            "(e.g. vendor → vendor.session.id, vendor.tenant.id). "
            "Overrides TELEMETRY_SIMULATOR_ATTR_PREFIX."
        ),
    )

    parser.add_argument(
        "--service-name",
        type=str,
        default="otelsim",
        help="Service name for telemetry (default: otelsim)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run mixed workload generation")
    run_parser.add_argument("--endpoint", type=str, help=argparse.SUPPRESS)
    run_parser.add_argument("--semconv", dest="semconv", type=str, help=argparse.SUPPRESS)
    run_parser.add_argument("--service-name", type=str, help=argparse.SUPPRESS)
    run_parser.add_argument("--vendor", type=str, help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of traces to generate (default: 100)",
    )
    run_parser.add_argument(
        "--interval",
        type=float,
        default=500,
        help="Interval between traces in ms (default: 500)",
    )
    run_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (if set, exports to file instead of OTLP)",
    )
    run_parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metric generation",
    )
    run_parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable log generation",
    )
    run_parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Print each generated trace (trace_id, tenant_id, span names) to the terminal",
    )
    run_parser.add_argument(
        "--show-all-attributes",
        action="store_true",
        help="Print each trace (trace_id, tenant_id, span names); default in container",
    )
    run_parser.add_argument(
        "--show-full-spans",
        action="store_true",
        help="Print complete span content (name, trace_id, span_id, kind, status, all attributes) for every span",
    )
    run_parser.add_argument(
        "--scenarios-dir",
        type=str,
        default=None,
        help="Folder with scenario YAML files for mixed workload (default: built-in sample definitions)",
    )
    run_parser.add_argument(
        "--tags",
        type=str,
        default=None,
        metavar="TAG[,TAG...]",
        help="Only run scenarios that have at least one of these tags (e.g. --tags=control-plane or --tags=control-plane,data-plane)",
    )
    run_parser.add_argument(
        "--each-once",
        action="store_true",
        help="Run each (tagged) scenario exactly once instead of --count random picks",
    )

    scenario_parser = subparsers.add_parser("scenario", help="Run YAML-defined scenario")
    scenario_parser.add_argument("--endpoint", type=str, help=argparse.SUPPRESS)
    scenario_parser.add_argument("--semconv", dest="semconv", type=str, help=argparse.SUPPRESS)
    scenario_parser.add_argument("--service-name", type=str, help=argparse.SUPPRESS)
    scenario_parser.add_argument("--vendor", type=str, help=argparse.SUPPRESS)
    scenario_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Scenario name (without .yaml extension)",
    )
    scenario_parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override repeat count from scenario",
    )
    scenario_parser.add_argument(
        "--scenarios-dir",
        type=str,
        default=None,
        help="Folder with scenario YAML files (default: built-in sample definitions)",
    )
    scenario_parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Print each generated trace (trace_id, tenant_id, span names) to the terminal",
    )
    scenario_parser.add_argument(
        "--show-all-attributes",
        action="store_true",
        help="Print each trace (trace_id, tenant_id, span names)",
    )
    scenario_parser.add_argument(
        "--show-full-spans",
        action="store_true",
        help="Print complete span content (name, attributes, etc.) for every span",
    )
    scenario_parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metric export (use when backend accepts traces only, e.g. Jaeger)",
    )
    scenario_parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable log export (use when backend accepts traces only, e.g. Jaeger)",
    )
    scenario_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (if set, exports traces to file instead of OTLP)",
    )

    list_parser = subparsers.add_parser("list", help="List available scenarios")
    list_parser.add_argument(
        "--scenarios-dir",
        type=str,
        default=None,
        help="Folder with scenario YAML files (default: built-in sample definitions)",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate schema and configuration")
    validate_parser.add_argument("--semconv", dest="semconv", type=str, help=argparse.SUPPRESS)
    validate_parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Show schema summary",
    )
    validate_parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Show span type definitions",
    )
    validate_parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show metric definitions",
    )

    return parser


def cmd_run(args: argparse.Namespace):
    """Run mixed workload generation."""
    print("Starting mixed workload telemetry generation...")
    print(f"   Endpoint: {args.endpoint}")
    each_once = getattr(args, "each_once", False)
    if each_once:
        print("   Mode: each (tagged) scenario once")
    else:
        print(f"   Count: {args.count}")
    print(f"   Interval: {args.interval}ms")
    tags_list: list[str] | None = None
    if getattr(args, "tags", None) and isinstance(args.tags, str):
        tags_list = [t.strip() for t in args.tags.split(",") if t.strip()]
        if tags_list:
            print(f"   Tags filter: {', '.join(tags_list)}")
    print(f"   Tenants: {', '.join(get_default_tenant_ids())}")
    print()

    if args.output_file:
        trace_exporter = FileSpanExporter(args.output_file)
        metric_exporter = (
            FileMetricExporter(args.output_file.replace(".jsonl", "_metrics.jsonl"))
            if not args.no_metrics
            else None
        )
        log_exporter = (
            FileLogExporter(args.output_file.replace(".jsonl", "_logs.jsonl"))
            if not args.no_logs
            else None
        )
        print(f"   Output: {args.output_file}")
    else:
        trace_exporter = create_otlp_trace_exporter(args.endpoint)
        metric_exporter = (
            create_otlp_metric_exporter(args.endpoint) if not args.no_metrics else None
        )
        log_exporter = create_otlp_log_exporter(args.endpoint) if not args.no_logs else None
        print("   Output: OTLP")

    print()

    def progress_callback(
        current: int,
        total: int,
        trace_id: str = "",
        tenant_id: str = "",
        span_names: list[str] | None = None,
    ):
        if getattr(args, "show_spans", False) or getattr(args, "show_all_attributes", False):
            print(_format_progress_line(current, total, trace_id, tenant_id, span_names))
        elif current % 10 == 0 or current == total:
            print(f"   Spans generated: {current}/{total}")

    try:
        runner = ScenarioRunner(
            trace_exporter=trace_exporter,
            metric_exporter=metric_exporter,
            log_exporter=log_exporter,
            schema_path=args.semconv,
            service_name=args.service_name,
            show_full_spans=getattr(args, "show_full_spans", False),
            scenarios_dir=getattr(args, "scenarios_dir", None),
        )

        trace_ids, traces_by_scenario = runner.run_mixed_workload(
            count=args.count,
            interval_ms=args.interval,
            progress_callback=progress_callback,
            tags=tags_list,
            each_once=each_once,
        )

        runner.shutdown()

        print()
        print(f"Generated {len(trace_ids)} traces")
        if traces_by_scenario:
            total_from_scenarios = sum(traces_by_scenario.values())
            print(f"   Traces per scenario (total: {total_from_scenarios}):")
            for name in sorted(traces_by_scenario.keys()):
                print(f"      {name}: {traces_by_scenario[name]}")
        if (
            not (getattr(args, "show_spans", False) or getattr(args, "show_all_attributes", False))
            and trace_ids
        ):
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
            logical_requests = 0 if each_once else args.count
            if logical_requests and len(trace_ids) != logical_requests:
                per_request = len(trace_ids) / float(logical_requests)
                print(
                    f"   \nNote: Each logical request currently emits separate traces for "
                    f"control-plane request validation, data-plane orchestration, and "
                    f"response validation (≈{per_request:.1f} traces/request)."
                )

    except KeyboardInterrupt:
        print("\nGeneration interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_scenario(args: argparse.Namespace):
    """Run YAML-defined scenario."""
    print(f"Running scenario: {args.name}")

    loader = ScenarioLoader(args.scenarios_dir)

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
    if not getattr(args, "output_file", None):
        print(f"   Endpoint: {args.endpoint}")
        print(f"   Service: {args.service_name} (select this in Jaeger UI)")
    interval_str = f"{scenario.interval_ms}ms"
    if scenario.interval_deviation_ms > 0:
        interval_str += f" ±{scenario.interval_deviation_ms}ms"
    print(f"   Interval: {interval_str}")
    print(f"   Tags: {', '.join(scenario.tags)}")
    print()

    if getattr(args, "output_file", None):
        trace_exporter = FileSpanExporter(args.output_file)
        metric_exporter = (
            FileMetricExporter(args.output_file.replace(".jsonl", "_metrics.jsonl"))
            if scenario.emit_metrics and not getattr(args, "no_metrics", False)
            else None
        )
        log_exporter = (
            FileLogExporter(args.output_file.replace(".jsonl", "_logs.jsonl"))
            if scenario.emit_logs and not getattr(args, "no_logs", False)
            else None
        )
        print(f"   Output: {args.output_file}")
    else:
        trace_exporter = create_otlp_trace_exporter(args.endpoint)
        metric_exporter = (
            create_otlp_metric_exporter(args.endpoint)
            if scenario.emit_metrics and not getattr(args, "no_metrics", False)
            else None
        )
        log_exporter = (
            create_otlp_log_exporter(args.endpoint)
            if scenario.emit_logs and not getattr(args, "no_logs", False)
            else None
        )

    def progress_callback(
        current: int,
        total: int,
        trace_id: str = "",
        tenant_id: str = "",
        span_names: list[str] | None = None,
    ):
        if getattr(args, "show_spans", False) or getattr(args, "show_all_attributes", False):
            print(_format_progress_line(current, total, trace_id, tenant_id, span_names))
        elif current % 10 == 0 or current == total:
            print(f"   Spans generated: {current}/{total}")

    try:
        runner = ScenarioRunner(
            trace_exporter=trace_exporter,
            metric_exporter=metric_exporter,
            log_exporter=log_exporter,
            schema_path=args.semconv,
            service_name=args.service_name,
            show_full_spans=getattr(args, "show_full_spans", False),
        )

        trace_ids = runner.run_scenario(scenario, progress_callback=progress_callback)
        runner.shutdown()

        print()
        print(f"Generated {len(trace_ids)} traces")
        if (
            not (getattr(args, "show_spans", False) or getattr(args, "show_all_attributes", False))
            and trace_ids
        ):
            max_show = 10
            to_show = trace_ids if len(trace_ids) <= max_show else trace_ids[:max_show]
            label = (
                "Trace IDs:"
                if len(trace_ids) <= max_show
                else f"Trace IDs (first {max_show} of {len(trace_ids)}):"
            )
            print(label)
            for tid in to_show:
                print(f"   {tid}")
            if scenario.repeat_count and len(trace_ids) != scenario.repeat_count:
                per_request = len(trace_ids) / float(scenario.repeat_count)
                print(
                    f"   Note: Each logical request currently emits separate traces for "
                    f"control-plane request validation, data-plane orchestration, and "
                    f"response validation (≈{per_request:.1f} traces/request)."
                )

    except KeyboardInterrupt:
        print("\nGeneration interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_list(args: argparse.Namespace):
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
        interval_str = f"{scenario.interval_ms}ms"
        if scenario.interval_deviation_ms > 0:
            interval_str += f" ±{scenario.interval_deviation_ms}ms"
        print(f"     Repeat: {scenario.repeat_count}, Interval: {interval_str}")
        print()


def cmd_validate(args: argparse.Namespace):
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
            print(f"  Resource attributes: {len(summary['resource_attributes'])}")
            print(f"  Common attributes: {len(summary['common_attributes'])}")
            print(f"  Metrics: {len(summary['metrics'])}")
            print()

        if args.show_spans:
            print("Span Type Definitions:")
            for name, span_def in schema.spans.items():
                kinds = ", ".join(k.value for k in span_def.span_kind)
                root = " (root)" if span_def.is_root else ""
                parents = f" <- {', '.join(span_def.parent_spans)}" if span_def.parent_spans else ""
                print(f"  - {name} [{kinds}]{root}{parents}")
                print(f"    {span_def.description}")
            print()

        if args.show_metrics:
            print("Canonical Metrics:")
            for name, metric_def in schema.metrics.items():
                print(f"  - {name}")
                print(f"    Type: {metric_def.metric_type}, Unit: {metric_def.unit}")
                print(f"    Dimensions: {', '.join(metric_def.dimensions)}")
                print(f"    Emitted by: {metric_def.emitted_by}")
            print()

        if not (args.show_schema or args.show_spans or args.show_metrics):
            print("Use --show-schema, --show-spans, or --show-metrics for details")

    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


_DEFAULT_ENDPOINT = "http://localhost:4318"
_DEFAULT_SERVICE_NAME = "otelsim"


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Optional vendor override: CLI flag wins over env. This controls the
    # attribute prefix used by config.attr/span_name (e.g. vendor.*).
    vendor = getattr(args, "vendor", None)
    if vendor:
        v = vendor.strip().lower()
        if v:
            os.environ["TELEMETRY_SIMULATOR_ATTR_PREFIX"] = v
            # Update module-level config so existing imports see the new prefix.
            sim_config.ATTR_PREFIX = v
            # VENDOR_NAME: use explicit env override when set; otherwise capitalize prefix.
            custom_name = os.environ.get("TELEMETRY_SIMULATOR_VENDOR_NAME", "").strip()
            sim_config.VENDOR_NAME = custom_name or v.capitalize()

    # Subparsers that accept global options can leave endpoint/service_name unset when
    # options are given only after the subcommand; normalize so commands always get defaults.
    if args.command in ("run", "scenario"):
        if getattr(args, "endpoint", None) is None:
            args.endpoint = _DEFAULT_ENDPOINT
        if getattr(args, "service_name", None) is None:
            args.service_name = _DEFAULT_SERVICE_NAME

    if args.command == "run":
        cmd_run(args)
    elif args.command == "scenario":
        cmd_scenario(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
