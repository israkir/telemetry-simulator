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

from ..config import CONFIG_PATH, attr as config_attr, load_yaml
from ..generators.log_generator import LogGenerator
from ..generators.metric_generator import MetricGenerator
from ..generators.trace_generator import TraceGenerator, TraceHierarchy
from ..schemas.attribute_generator import GenerationContext
from .id_generator import ScenarioIdGenerator
from .scenario_loader import Scenario, ScenarioLoader


def _conversation_samples_from_config() -> dict[str, Any]:
    """Load conversation_samples from config/config.yaml (workflow -> list of {user_input, llm_response})."""
    data = load_yaml(CONFIG_PATH)
    return data.get("conversation_samples") or {}


def _sample_to_otel_messages(user_input: str, llm_response: str) -> tuple[list[dict], list[dict]]:
    """Convert one conversation sample (user_input, llm_response) to OTEL GenAI message shape."""
    input_msgs = [
        {"role": "user", "content": [{"type": "text", "text": user_input}]},
    ]
    output_msgs = [
        {"role": "assistant", "content": [{"type": "text", "text": llm_response}]},
    ]
    return input_msgs, output_msgs


def _get_conversation_from_config(workflow: str | None) -> tuple[list[dict] | None, list[dict] | None]:
    """
    If workflow matches conversation_samples in config, return (input_messages, output_messages).
    Falls back to "default" when workflow has no samples. Otherwise return (None, None).
    """
    if not workflow or not isinstance(workflow, str):
        return None, None
    samples_by_workflow = _conversation_samples_from_config()
    for key in (workflow, "default"):
        workflow_samples = samples_by_workflow.get(key)
        if not isinstance(workflow_samples, dict):
            continue
        samples_list = workflow_samples.get("samples")
        if not isinstance(samples_list, list) or not samples_list:
            continue
        sample = random.choice(samples_list)
        if not isinstance(sample, dict):
            continue
        user_input = sample.get("user_input")
        llm_response = sample.get("llm_response")
        if isinstance(user_input, str) and isinstance(llm_response, str):
            return _sample_to_otel_messages(user_input, llm_response)
    return None, None


def _context_kwargs_for_scenario(scenario: Scenario, tenant_id: str) -> dict:
    """Build kwargs for GenerationContext.create using scenario id_generator when set."""
    kwargs: dict = {
        "tenant_id": tenant_id,
        "turn_index": 0,
        "redaction_applied": getattr(scenario, "redaction_applied", "none"),
    }
    id_gen = getattr(scenario, "id_generator", None)
    if isinstance(id_gen, ScenarioIdGenerator):
        kwargs["session_id"] = id_gen.session_id(tenant_id=tenant_id)
        kwargs["request_id"] = id_gen.request_id(tenant_id=tenant_id)
    # Per convention: one span per LLM call = one interaction (user input → LLM response).
    # When scenario has conversation_turn_pairs, the runner sets llm_* per turn (one trace per turn, same session_id).
    # Otherwise use a single-turn sample from conversation_samples (workflow) so gen_ai.input/output are realistic.
    if getattr(scenario, "conversation_turn_pairs", None):
        pass  # runner sets llm_input_messages / llm_output_messages per turn
    else:
        context = getattr(scenario, "context", None)
        workflow = context.workflow if context else None
        input_msgs, output_msgs = _get_conversation_from_config(workflow)
        if input_msgs is not None and output_msgs is not None:
            kwargs["llm_input_messages"] = input_msgs
            kwargs["llm_output_messages"] = output_msgs
    return kwargs


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

    def _build_incoming_request_validation_hierarchy(self) -> TraceHierarchy:
        """
        Build a control-plane incoming request validation hierarchy for the
        "allowed" happy path (spec §9.1).

        Structure:
          {prefix}.request.validation (root, SERVER)
            ├── {prefix}.validation.payload  (INTERNAL)
            ├── {prefix}.validation.policy   (INTERNAL)
            └── {prefix}.augmentation        (INTERNAL)
        """
        from ..generators.trace_generator import SpanType, SpanConfig, TraceHierarchy
        from ..config import attr as config_attr  # type: ignore[import-not-found]
        from .scenario_loader import _DEFAULT_LATENCY_MS  # type: ignore[import-not-found]

        root_cfg = SpanConfig(
            span_type=SpanType.REQUEST_VALIDATION,
            latency_mean_ms=_DEFAULT_LATENCY_MS.get(SpanType.REQUEST_VALIDATION, 40.0),
            latency_variance=0.2,
            error_rate=0.0,
            attribute_overrides={
                config_attr("request.outcome"): "allowed",
            },
        )
        payload_cfg = SpanConfig(
            span_type=SpanType.VALIDATION_PAYLOAD,
            latency_mean_ms=_DEFAULT_LATENCY_MS.get(SpanType.VALIDATION_PAYLOAD, 20.0),
            latency_variance=0.2,
            error_rate=0.0,
            attribute_overrides={
                config_attr("step.outcome"): "pass",
                config_attr("validation.result"): "valid",
            },
        )
        policy_cfg = SpanConfig(
            span_type=SpanType.VALIDATION_POLICY,
            latency_mean_ms=_DEFAULT_LATENCY_MS.get(SpanType.VALIDATION_POLICY, 20.0),
            latency_variance=0.2,
            error_rate=0.0,
            attribute_overrides={
                config_attr("step.outcome"): "pass",
                config_attr("policy.engine"): "dlp",
                config_attr("policy.decision"): "allow",
            },
        )
        augment_cfg = SpanConfig(
            span_type=SpanType.AUGMENTATION,
            latency_mean_ms=_DEFAULT_LATENCY_MS.get(SpanType.AUGMENTATION, 20.0),
            latency_variance=0.2,
            error_rate=0.0,
            attribute_overrides={
                config_attr("step.outcome"): "pass",
                config_attr("augment.conversation_id.action"): "propagated",
                config_attr("augment.request_id.action"): "created",
                config_attr("augment.enduser_id.action"): "missing",
                config_attr("augment.target_agent_id.action"): "attached",
            },
        )

        payload_h = TraceHierarchy(root_config=payload_cfg, children=[])
        policy_h = TraceHierarchy(root_config=policy_cfg, children=[])
        augment_h = TraceHierarchy(root_config=augment_cfg, children=[])

        return TraceHierarchy(root_config=root_cfg, children=[payload_h, policy_h, augment_h])

    def _build_outgoing_response_validation_hierarchy(self) -> TraceHierarchy:
        """
        Build a control-plane outgoing response validation hierarchy for the
        "allowed" happy path (spec §6.1).

        Structure:
          {prefix}.response.validation (root, SERVER)
            └── {prefix}.validation.policy (INTERNAL)
        """
        from ..generators.trace_generator import SpanType, SpanConfig, TraceHierarchy
        from ..config import attr as config_attr  # type: ignore[import-not-found]
        from .scenario_loader import _DEFAULT_LATENCY_MS  # type: ignore[import-not-found]

        root_cfg = SpanConfig(
            span_type=SpanType.RESPONSE_VALIDATION,
            latency_mean_ms=_DEFAULT_LATENCY_MS.get(SpanType.RESPONSE_VALIDATION, 40.0),
            latency_variance=0.2,
            error_rate=0.0,
            attribute_overrides={
                config_attr("response.outcome"): "allowed",
            },
        )
        policy_cfg = SpanConfig(
            span_type=SpanType.VALIDATION_POLICY,
            latency_mean_ms=_DEFAULT_LATENCY_MS.get(SpanType.VALIDATION_POLICY, 20.0),
            latency_variance=0.2,
            error_rate=0.0,
            attribute_overrides={
                config_attr("step.outcome"): "pass",
                config_attr("policy.engine"): "dlp",
                config_attr("policy.decision"): "allow",
            },
        )

        policy_h = TraceHierarchy(root_config=policy_cfg, children=[])
        return TraceHierarchy(root_config=root_cfg, children=[policy_h])

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

        turn_pairs = getattr(scenario, "conversation_turn_pairs", None) or []
        use_multi_turn = len(turn_pairs) > 0

        for i in range(scenario.repeat_count):
            tenant_id = random.choices(tenants, weights=weights)[0]
            ctx_kwargs = _context_kwargs_for_scenario(scenario, tenant_id)
            # One session_id per logical session (iteration); all turns in this iteration share it.
            session_id = ctx_kwargs.get("session_id") or GenerationContext.create(**ctx_kwargs).session_id
            id_gen = getattr(scenario, "id_generator", None)

            if use_multi_turn:
                # One trace per interaction (turn); same session_id for all turns in this session.
                hierarchies = scenario.get_trace_hierarchies()
                hierarchy = hierarchies[0] if hierarchies else scenario.get_trace_hierarchy()
                iteration_trace_ids = []
                for turn_index, (input_msgs, output_msgs) in enumerate(turn_pairs):
                    ctx_kwargs["session_id"] = session_id
                    ctx_kwargs["turn_index"] = turn_index
                    ctx_kwargs["llm_input_messages"] = input_msgs
                    ctx_kwargs["llm_output_messages"] = output_msgs
                    if isinstance(id_gen, ScenarioIdGenerator):
                        ctx_kwargs["request_id"] = id_gen.request_id(tenant_id=tenant_id)
                    context = GenerationContext.create(**ctx_kwargs)
                    trace_id = self.trace_generator.generate_trace(hierarchy, context)
                    iteration_trace_ids.append(trace_id)
                    trace_ids.append(trace_id)
                    if scenario.emit_metrics and self.metric_generator:
                        self._emit_metrics_for_hierarchy(hierarchy, context)
                    if scenario.emit_logs and self.log_generator:
                        self._emit_logs_for_hierarchy(hierarchy, context)
            else:
                ctx_kwargs["turn_index"] = i % 10
                context = GenerationContext.create(**ctx_kwargs)
                hierarchies = scenario.get_trace_hierarchies()
                iteration_trace_ids = []

                # Emit incoming and outgoing validation traces (control-plane)
                # when the scenario includes an A2A orchestration trace.
                from ..generators.trace_generator import SpanType  # local import to avoid cycles

                has_a2a = any(h.root_config.span_type is SpanType.A2A_ORCHESTRATE for h in hierarchies)

                if has_a2a:
                    incoming_h = self._build_incoming_request_validation_hierarchy()
                    incoming_trace_id = self.trace_generator.generate_trace(incoming_h, context)
                    iteration_trace_ids.append(incoming_trace_id)
                    trace_ids.append(incoming_trace_id)
                    if scenario.emit_metrics and self.metric_generator:
                        self._emit_metrics_for_hierarchy(incoming_h, context)
                    if scenario.emit_logs and self.log_generator:
                        self._emit_logs_for_hierarchy(incoming_h, context)

                for hierarchy in hierarchies:
                    dp_trace_id = self.trace_generator.generate_trace(hierarchy, context)
                    iteration_trace_ids.append(dp_trace_id)
                    trace_ids.append(dp_trace_id)
                    if scenario.emit_metrics and self.metric_generator:
                        self._emit_metrics_for_hierarchy(hierarchy, context)
                    if scenario.emit_logs and self.log_generator:
                        self._emit_logs_for_hierarchy(hierarchy, context)

                if has_a2a:
                    outgoing_h = self._build_outgoing_response_validation_hierarchy()
                    outgoing_trace_id = self.trace_generator.generate_trace(outgoing_h, context)
                    iteration_trace_ids.append(outgoing_trace_id)
                    trace_ids.append(outgoing_trace_id)
                    if scenario.emit_metrics and self.metric_generator:
                        self._emit_metrics_for_hierarchy(outgoing_h, context)
                    if scenario.emit_logs and self.log_generator:
                        self._emit_logs_for_hierarchy(outgoing_h, context)

            if progress_callback:
                primary_trace_id = iteration_trace_ids[-1] if iteration_trace_ids else ""
                if use_multi_turn:
                    h = scenario.get_trace_hierarchy()
                    all_span_names = h.span_names()
                else:
                    all_span_names = []
                    for h in scenario.get_trace_hierarchies():
                        all_span_names.extend(h.span_names())
                progress_callback(
                    i + 1,
                    scenario.repeat_count,
                    primary_trace_id,
                    context.tenant_id,
                    all_span_names[:50],
                )

            if scenario.interval_ms > 0 and i < scenario.repeat_count - 1:
                delay_ms = scenario.interval_ms
                if scenario.interval_deviation_ms > 0:
                    delay_ms = max(
                        0,
                        delay_ms
                        + random.uniform(
                            -scenario.interval_deviation_ms,
                            scenario.interval_deviation_ms,
                        ),
                    )
                time.sleep(delay_ms / 1000.0)

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

        trace_ids: list[str] = []

        for i in range(count):
            scenario = random.choice(scenarios)
            tenants = list(scenario.tenant_distribution.keys())
            weights = list(scenario.tenant_distribution.values())
            tenant_id = random.choices(tenants, weights=weights)[0]
            ctx_kwargs = _context_kwargs_for_scenario(scenario, tenant_id)
            ctx_kwargs["turn_index"] = i % 10
            context = GenerationContext.create(**ctx_kwargs)

            hierarchies = scenario.get_trace_hierarchies()
            iteration_trace_ids: list[str] = []
            all_span_names: list[str] = []

            from ..generators.trace_generator import SpanType  # local import

            has_a2a = any(h.root_config.span_type is SpanType.A2A_ORCHESTRATE for h in hierarchies)

            # Emit incoming / outgoing validation traces when an A2A orchestration
            # hierarchy is present for this iteration.
            if has_a2a:
                incoming_h = self._build_incoming_request_validation_hierarchy()
                incoming_trace_id = self.trace_generator.generate_trace(incoming_h, context)
                iteration_trace_ids.append(incoming_trace_id)
                trace_ids.append(incoming_trace_id)
                all_span_names.extend(incoming_h.span_names())
                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(incoming_h, context)
                if self.log_generator and scenario.emit_logs:
                    self._emit_logs_for_hierarchy(incoming_h, context)

            for hierarchy in hierarchies:
                dp_trace_id = self.trace_generator.generate_trace(hierarchy, context)
                iteration_trace_ids.append(dp_trace_id)
                trace_ids.append(dp_trace_id)

                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(hierarchy, context)

                if self.log_generator and scenario.emit_logs:
                    self._emit_logs_for_hierarchy(hierarchy, context)

                all_span_names.extend(hierarchy.span_names())

            if has_a2a:
                outgoing_h = self._build_outgoing_response_validation_hierarchy()
                outgoing_trace_id = self.trace_generator.generate_trace(outgoing_h, context)
                iteration_trace_ids.append(outgoing_trace_id)
                trace_ids.append(outgoing_trace_id)
                all_span_names.extend(outgoing_h.span_names())
                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(outgoing_h, context)
                if self.log_generator and scenario.emit_logs:
                    self._emit_logs_for_hierarchy(outgoing_h, context)

            primary_trace_id = iteration_trace_ids[-1] if iteration_trace_ids else ""

            if progress_callback:
                progress_callback(
                    i + 1,
                    count,
                    primary_trace_id,
                    context.tenant_id,
                    all_span_names,
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
                provider=attrs.get("gen_ai.system", "openai"),
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
                provider=attrs.get("gen_ai.system", "openai"),
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
