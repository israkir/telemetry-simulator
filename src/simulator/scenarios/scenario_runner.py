"""
Execute scenarios and coordinate trace/metric/log generation.

The runner orchestrates:
- Loading scenarios from YAML
- Generating traces with proper hierarchies
- Emitting correlated metrics and logs
- Managing tenant distribution and timing
"""

import random
import time
from collections.abc import Callable
from typing import Any

from opentelemetry.sdk._logs.export import LogExporter
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.trace.export import SpanExporter

from ..config import attr as config_attr
from ..generators.log_generator import LogGenerator
from ..generators.metric_generator import MetricGenerator
from ..generators.trace_generator import TraceGenerator, TraceHierarchy
from ..schemas.attribute_generator import GenerationContext
from .scenario_loader import Scenario, ScenarioLoader


def _int_token_count(val: Any, default: int) -> int:
    """Coerce token count to int for OTEL (round if float)."""
    if val is None:
        return default
    try:
        return int(round(val))
    except (TypeError, ValueError):
        return default


class ScenarioRunner:
    """Execute telemetry generation scenarios."""

    def __init__(
        self,
        trace_exporter: SpanExporter,
        metric_exporter: MetricExporter | None = None,
        log_exporter: LogExporter | None = None,
        schema_path: str | None = None,
        service_name: str = "telemetry-simulator",
        show_full_spans: bool = False,
        scenarios_dir: str | None = None,
    ):
        """Initialize runner with exporters and optional custom scenario definitions folder."""
        self.trace_generator = TraceGenerator(
            exporter=trace_exporter,
            schema_path=schema_path,
            service_name=service_name,
            show_full_spans=show_full_spans,
        )

        self.metric_generator = None
        if metric_exporter:
            self.metric_generator = MetricGenerator(
                exporter=metric_exporter,
                schema_path=schema_path,
                service_name=service_name,
            )

        self.log_generator = None
        if log_exporter:
            self.log_generator = LogGenerator(
                exporter=log_exporter,
                schema_path=schema_path,
                service_name=service_name,
            )

        self.scenario_loader = ScenarioLoader(scenarios_dir=scenarios_dir)

    def run_scenario(
        self,
        scenario: Scenario | str,
        progress_callback: Callable[[int, int, str, str, list[str]], None] | None = None,
    ) -> list[str]:
        """Run a scenario and return generated trace IDs."""
        if isinstance(scenario, str):
            scenario = self.scenario_loader.load(scenario)

        trace_ids = []
        tenants = list(scenario.tenant_distribution.keys())
        weights = list(scenario.tenant_distribution.values())

        for i in range(scenario.repeat_count):
            tenant_id = random.choices(tenants, weights=weights)[0]

            context = GenerationContext.create(
                tenant_id=tenant_id,
                turn_index=i % 10,
            )

            hierarchies = scenario.get_trace_hierarchies()
            iteration_trace_ids = []

            for hierarchy in hierarchies:
                trace_id = self.trace_generator.generate_trace(hierarchy, context)
                iteration_trace_ids.append(trace_id)
                trace_ids.append(trace_id)

                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(hierarchy, context)

                if scenario.emit_logs and self.log_generator:
                    self._emit_logs_for_hierarchy(hierarchy, context)

            if progress_callback:
                primary_trace_id = iteration_trace_ids[-1] if iteration_trace_ids else ""
                all_span_names = []
                for h in hierarchies:
                    all_span_names.extend(h.span_names())
                progress_callback(
                    i + 1,
                    scenario.repeat_count,
                    primary_trace_id,
                    context.tenant_id,
                    all_span_names,
                )

            if scenario.interval_ms > 0 and i < scenario.repeat_count - 1:
                time.sleep(scenario.interval_ms / 1000.0)

        return trace_ids

    def run_mixed_workload(
        self,
        count: int = 100,
        interval_ms: float = 200,
        progress_callback: Callable[[int, int, str, str, list[str]], None] | None = None,
    ) -> list[str]:
        """Run a mixed workload by picking at random from YAML-defined scenarios."""
        scenarios = self.scenario_loader.load_all()
        if not scenarios:
            dir_path = self.scenario_loader.scenarios_dir
            raise ValueError(
                f"No YAML scenarios found in {dir_path}. "
                "Add at least one .yaml file there, or pass --scenarios-dir to use a custom folder. "
                "Sample definitions are bundled in src/simulator/scenarios/definitions/."
            )

        trace_ids = []

        for i in range(count):
            scenario = random.choice(scenarios)
            tenants = list(scenario.tenant_distribution.keys())
            weights = list(scenario.tenant_distribution.values())
            tenant_id = random.choices(tenants, weights=weights)[0]

            context = GenerationContext.create(
                tenant_id=tenant_id,
                turn_index=i % 10,
            )
            hierarchy = scenario.get_trace_hierarchy()
            trace_id = self.trace_generator.generate_trace(hierarchy, context)
            trace_ids.append(trace_id)

            if scenario.emit_metrics and self.metric_generator:
                self._emit_metrics_for_hierarchy(hierarchy, context)

            if scenario.emit_logs and self.log_generator:
                self._emit_logs_for_hierarchy(hierarchy, context)

            if progress_callback:
                progress_callback(
                    i + 1,
                    count,
                    trace_id,
                    context.tenant_id,
                    hierarchy.span_names(),
                )

            if interval_ms > 0 and i < count - 1:
                time.sleep(interval_ms / 1000.0)

        return trace_ids

    def _emit_metrics_for_hierarchy(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext,
    ):
        """Emit metrics corresponding to a trace hierarchy."""
        if not self.metric_generator:
            return

        self._emit_metrics_recursive(hierarchy.root_config, hierarchy.children, context)

    def _emit_metrics_recursive(self, config, children, context):
        """Recursively emit metrics for span configs."""
        from ..generators.trace_generator import SpanType

        span_type = config.span_type
        latency = config.latency_mean_ms
        attrs = config.attribute_overrides or {}

        if span_type == SpanType.A2A_ORCHESTRATE:
            self.metric_generator.record_turn(
                context,
                duration_ms=latency,
                status_code=attrs.get(config_attr("turn.status.code"), "SUCCESS"),
            )
        elif span_type in (SpanType.MCP_TOOL_EXECUTE, SpanType.MCP_TOOL_EXECUTE_ATTEMPT):
            self.metric_generator.record_tool_call(
                context,
                tool_name=attrs.get("gen_ai.tool.name", "unknown.tool"),
                server_name=attrs.get(config_attr("tool.server.name"), "unknown-server"),
                latency_ms=latency,
                status_code=attrs.get(config_attr("tool.status.code"), "OK"),
            )
        elif span_type == SpanType.LLM_CALL:
            self.metric_generator.record_llm_inference(
                context,
                provider=attrs.get("gen_ai.provider.name", "openai"),
                model=attrs.get("gen_ai.request.model", "gpt-4.1-mini"),
                latency_ms=latency,
                input_tokens=_int_token_count(attrs.get("gen_ai.usage.input_tokens"), 500),
                output_tokens=_int_token_count(attrs.get("gen_ai.usage.output_tokens"), 200),
            )
        elif span_type == SpanType.RAG_RETRIEVE:
            self.metric_generator.record_rag_retrieval(
                context,
                index_name=attrs.get("rag.index.name", "default_index"),
                latency_ms=latency,
                docs_returned=attrs.get("rag.documents.returned", 5),
            )
        elif span_type == SpanType.A2A_CALL:
            self.metric_generator.record_a2a_call(
                context,
                target_agent=attrs.get("a2a.target.agent", "unknown_agent"),
                latency_ms=latency,
                status_code=attrs.get(config_attr("a2a.status.code"), "OK"),
            )
        elif span_type == SpanType.CP_REQUEST:
            self.metric_generator.record_cp_request(
                context,
                duration_ms=latency,
                status_code=attrs.get(config_attr("cp.status.code"), "ALLOWED"),
            )

        for child in children:
            self._emit_metrics_recursive(child.root_config, child.children, context)

    def _emit_logs_for_hierarchy(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext,
    ):
        """Emit logs corresponding to a trace hierarchy."""
        if not self.log_generator:
            return

        self._emit_logs_recursive(hierarchy.root_config, hierarchy.children, context, is_root=True)

    def _emit_logs_recursive(self, config, children, context, is_root=False):
        """Recursively emit logs for span configs."""
        from ..generators.trace_generator import SpanType

        span_type = config.span_type
        attrs = config.attribute_overrides or {}

        if span_type == SpanType.A2A_ORCHESTRATE:
            self.log_generator.log_turn_start(context)

        if span_type in (SpanType.MCP_TOOL_EXECUTE, SpanType.MCP_TOOL_EXECUTE_ATTEMPT):
            self.log_generator.log_tool_call(
                context,
                tool_name=attrs.get("gen_ai.tool.name", "unknown.tool"),
                server_name=attrs.get(config_attr("tool.server.name"), "unknown-server"),
                status_code=attrs.get(config_attr("tool.status.code"), "OK"),
                latency_ms=config.latency_mean_ms,
            )
        elif span_type == SpanType.LLM_CALL:
            self.log_generator.log_llm_inference(
                context,
                provider=attrs.get("gen_ai.provider.name", "openai"),
                model=attrs.get("gen_ai.request.model", "gpt-4.1-mini"),
                operation=attrs.get(config_attr("llm.operation"), "chat"),
                input_tokens=_int_token_count(attrs.get("gen_ai.usage.input_tokens"), 500),
                output_tokens=_int_token_count(attrs.get("gen_ai.usage.output_tokens"), 200),
                latency_ms=config.latency_mean_ms,
            )
        elif span_type == SpanType.RAG_RETRIEVE:
            self.log_generator.log_rag_retrieval(
                context,
                index_name=attrs.get("rag.index.name", "default_index"),
                docs_returned=attrs.get("rag.documents.returned", 5),
                latency_ms=config.latency_mean_ms,
            )
        elif span_type == SpanType.A2A_CALL:
            self.log_generator.log_a2a_call(
                context,
                target_agent=attrs.get("a2a.target.agent", "unknown_agent"),
                operation=attrs.get("a2a.operation", "delegate"),
                status_code=attrs.get(config_attr("a2a.status.code"), "OK"),
                latency_ms=config.latency_mean_ms,
            )
        elif span_type == SpanType.CP_REQUEST:
            self.log_generator.log_cp_request(
                context,
                status_code=attrs.get(config_attr("cp.status.code"), "ALLOWED"),
            )

        for child in children:
            self._emit_logs_recursive(child.root_config, child.children, context, is_root=False)

        if span_type == SpanType.A2A_ORCHESTRATE:
            self.log_generator.log_turn_end(
                context,
                status_code=attrs.get(config_attr("turn.status.code"), "SUCCESS"),
                duration_ms=config.latency_mean_ms,
            )

    def shutdown(self):
        """Shutdown all generators."""
        self.trace_generator.shutdown()
        if self.metric_generator:
            self.metric_generator.shutdown()
        if self.log_generator:
            self.log_generator.shutdown()
