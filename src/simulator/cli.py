"""
Command-line interface for the Telemetry Simulator.

Provides commands for:
- Running YAML-defined scenarios
- Mixed workload simulation
- Validation and schema inspection
"""

import argparse
import sys

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
        prog="telemetry-simulator",
        description="Schema-driven OTEL telemetry simulator for LLM observability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run mixed workload to OTLP collector
  telemetry-simulator run --count 100 --interval 200

  # Run specific scenario
  telemetry-simulator scenario --name successful_agent_turn --count 50

  # Validate schema and show summary
  telemetry-simulator validate --show-schema

  # Export to file instead of OTLP
  telemetry-simulator run --count 10 --output-file traces.jsonl
        """,
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:4318",
        help="OTLP HTTP endpoint (default: http://localhost:4318)",
    )

    parser.add_argument(
        "--schema-path",
        type=str,
        default=None,
        help="Path to semantic-conventions YAML (required unless TELEMETRY_SIMULATOR_SCHEMA_PATH is set)",
    )

    parser.add_argument(
        "--service-name",
        type=str,
        default="telemetry-simulator",
        help="Service name for telemetry (default: telemetry-simulator)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run mixed workload generation")
    run_parser.add_argument("--endpoint", type=str, help=argparse.SUPPRESS)
    run_parser.add_argument("--schema-path", type=str, help=argparse.SUPPRESS)
    run_parser.add_argument("--service-name", type=str, help=argparse.SUPPRESS)
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
        "--tenants",
        type=str,
        nargs="+",
        default=get_default_tenant_ids(),
        help="Tenant IDs from TENANT_UUID env (required)",
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

    scenario_parser = subparsers.add_parser("scenario", help="Run YAML-defined scenario")
    scenario_parser.add_argument("--endpoint", type=str, help=argparse.SUPPRESS)
    scenario_parser.add_argument("--schema-path", type=str, help=argparse.SUPPRESS)
    scenario_parser.add_argument("--service-name", type=str, help=argparse.SUPPRESS)
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

    list_parser = subparsers.add_parser("list", help="List available scenarios")
    list_parser.add_argument(
        "--scenarios-dir",
        type=str,
        default=None,
        help="Folder with scenario YAML files (default: built-in sample definitions)",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate schema and configuration")
    validate_parser.add_argument("--schema-path", type=str, help=argparse.SUPPRESS)
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
    print(f"   Count: {args.count}")
    print(f"   Interval: {args.interval}ms")
    print(f"   Tenants: {', '.join(args.tenants)}")
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
            schema_path=args.schema_path,
            service_name=args.service_name,
            show_full_spans=getattr(args, "show_full_spans", False),
            scenarios_dir=getattr(args, "scenarios_dir", None),
        )

        trace_ids = runner.run_mixed_workload(
            count=args.count,
            interval_ms=args.interval,
            progress_callback=progress_callback,
        )

        runner.shutdown()

        print()
        print(f"Generated {len(trace_ids)} traces")
        if (
            not (getattr(args, "show_spans", False) or getattr(args, "show_all_attributes", False))
            and trace_ids
        ):
            print()
            print("Sample trace IDs:")
            for trace_id in trace_ids[:5]:
                print(f"   {trace_id}")

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
    print(f"   Interval: {scenario.interval_ms}ms")
    print(f"   Tags: {', '.join(scenario.tags)}")
    print()

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
            schema_path=args.schema_path,
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
            print("Sample trace IDs:")
            for tid in trace_ids[:5]:
                print(f"   {tid}")

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
        print(f"     Repeat: {scenario.repeat_count}, Interval: {scenario.interval_ms}ms")
        print()


def cmd_validate(args: argparse.Namespace):
    """Validate schema and show information."""
    try:
        parser = SchemaParser(args.schema_path)
        schema = parser.parse()
        validator = OtelValidator(args.schema_path)

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
_DEFAULT_SERVICE_NAME = "telemetry-simulator"


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

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
