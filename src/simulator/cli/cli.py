"""Command-line parser and command dispatch for the telemetry simulator."""

import argparse
import sys

from .. import config as sim_config
from .cli_commands import cmd_list, cmd_run, cmd_scenario, cmd_validate


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
  otelsim scenario --name single_tool_call

  # Scenario with fixed 5s between trace starts (overrides YAML trace_interval_ms)
  otelsim scenario --name new_claim_phone_multi_turn --interval 5000

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
        help=(
            "Semantic-conventions YAML path (default: auto-resolve "
            "resource/scenarios/conventions/semconv.yaml under cwd or repo root)"
        ),
    )

    parser.add_argument(
        "--vendor",
        type=str,
        default=None,
        help=(
            "Attribute prefix for vendor-specific attributes on spans, metrics, and logs "
            "(e.g. --vendor gentoro → gentoro.tenant.id on resource). Default: vendor."
        ),
    )

    parser.add_argument(
        "--tenant-id",
        type=str,
        default=None,
        help=(
            "Override the default tenant id used for telemetry. "
            "When omitted, the first tenant id from resource/config/config.yaml is used."
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
    run_parser.add_argument("--tenant-id", dest="tenant_id", type=str, help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of mixed-workload picks (each pick runs one scenario workload); "
            "omit to run until interrupted (Ctrl+C)"
        ),
    )
    run_parser.add_argument(
        "--interval",
        type=float,
        default=500,
        help="Delay between mixed-workload picks in ms (default: 500)",
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
        help="Print each trace (trace_id, tenant_id, span names)",
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
    scenario_parser.add_argument("--tenant-id", dest="tenant_id", type=str, help=argparse.SUPPRESS)
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
        "--interval",
        type=float,
        default=None,
        metavar="MS",
        help=(
            "Delay between successive trace start times in ms (overrides scenario "
            "trace_interval_ms; jitter disabled unless --interval-deviation is set)"
        ),
    )
    scenario_parser.add_argument(
        "--interval-deviation",
        type=float,
        default=None,
        metavar="MS",
        help=(
            "When used with --interval, random jitter ± this many ms around the interval "
            "(omit for fixed spacing)"
        ),
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


_DEFAULT_ENDPOINT = "http://localhost:4318"
_DEFAULT_SERVICE_NAME = "otelsim"


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.vendor:
        v = args.vendor.strip().lower()
        if v:
            sim_config.ATTR_PREFIX = v

    if args.tenant_id:
        t = args.tenant_id.strip()
        args.tenant_id = t or None

    if args.command in ("run", "scenario"):
        if args.endpoint is None:
            args.endpoint = _DEFAULT_ENDPOINT
        if args.service_name is None:
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
