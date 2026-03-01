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

from ..config import CONFIG_PATH, load_yaml
from ..config import attr as config_attr
from ..generators.log_generator import LogGenerator
from ..generators.metric_generator import MetricGenerator
from ..generators.trace_generator import TraceGenerator, TraceHierarchy
from ..schemas.attribute_generator import GenerationContext
from .id_generator import (
    ScenarioIdGenerator,
    generate_enduser_pseudo_id,
    generate_request_id,
    generate_session_id,
)
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


def _get_conversation_from_config(
    workflow: str | None,
) -> tuple[
    list[dict] | None,
    list[dict] | None,
    list[dict] | None,
    list[dict] | None,
]:
    """
    If workflow matches conversation_samples in config, return (input_messages, output_messages, input_redacted, output_redacted).
    Redacted pair is (None, None) when sample has no user_input_redacted/llm_response_redacted. Otherwise (None, None, None, None).
    """
    if not workflow or not isinstance(workflow, str):
        return None, None, None, None
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
            inp_msgs, out_msgs = _sample_to_otel_messages(user_input, llm_response)
            ui_red = sample.get("user_input_redacted")
            lr_red = sample.get("llm_response_redacted")
            if isinstance(ui_red, str) and isinstance(lr_red, str):
                inp_red, out_red = _sample_to_otel_messages(ui_red, lr_red)
                return inp_msgs, out_msgs, inp_red, out_red
            return inp_msgs, out_msgs, None, None
    return None, None, None, None


def _context_kwargs_for_scenario(
    scenario: Scenario, tenant_id: str, iteration_index: int | None = None
) -> dict:
    """Build kwargs for GenerationContext.create. All correlation IDs from config id_formats (id_generator or standalone generators); no fallbacks.
    When iteration_index is set and scenario.cycle_conversation_samples is True, use conversation_samples[iteration_index % len(samples)] instead of random.
    """
    kwargs: dict = {
        "tenant_id": tenant_id,
        "turn_index": 0,
        "redaction_applied": getattr(scenario, "redaction_applied", "none"),
        "scenario_name": getattr(scenario, "name", None),
        "higher_latency_condition": getattr(scenario, "higher_latency_condition", None),
    }
    id_gen = getattr(scenario, "id_generator", None)
    if isinstance(id_gen, ScenarioIdGenerator):
        kwargs["session_id"] = id_gen.session_id(tenant_id=tenant_id)
        kwargs["request_id"] = id_gen.request_id(tenant_id=tenant_id)
        kwargs["user_id"] = id_gen.enduser_pseudo_id(tenant_id=tenant_id)
    else:
        kwargs["session_id"] = generate_session_id(tenant_id=tenant_id)
        kwargs["request_id"] = generate_request_id(tenant_id=tenant_id)
        kwargs["user_id"] = generate_enduser_pseudo_id(tenant_id=tenant_id)
    # Per convention: one span per LLM call = one interaction (user input → LLM response).
    # When scenario has conversation_turn_pairs, the runner sets llm_* per turn (one trace per turn, same session_id).
    # When scenario has conversation_samples, pick one per iteration: by index if cycle_conversation_samples else random.
    # Otherwise use a single-turn sample from config conversation_samples (workflow).
    if getattr(scenario, "conversation_turn_pairs", None):
        pass  # runner sets llm_input_messages / llm_output_messages per turn
    else:
        scenario_samples = getattr(scenario, "conversation_samples", None) or []
        if scenario_samples:
            if iteration_index is not None and getattr(
                scenario, "cycle_conversation_samples", False
            ):
                sample = scenario_samples[iteration_index % len(scenario_samples)]
            else:
                sample = random.choice(scenario_samples)
            user_input = sample.get("user_input")
            llm_response = sample.get("llm_response")
            if isinstance(user_input, str) and isinstance(llm_response, str):
                input_msgs, output_msgs = _sample_to_otel_messages(user_input, llm_response)
                kwargs["llm_input_messages"] = input_msgs
                kwargs["llm_output_messages"] = output_msgs
                ui_red = sample.get("user_input_redacted")
                lr_red = sample.get("llm_response_redacted")
                if isinstance(ui_red, str) and isinstance(lr_red, str):
                    inp_red, out_red = _sample_to_otel_messages(ui_red, lr_red)
                    kwargs["llm_input_messages_redacted"] = inp_red
                    kwargs["llm_output_messages_redacted"] = out_red
        else:
            context = getattr(scenario, "context", None)
            workflow = context.workflow if context else None
            cfg_inp, cfg_out, cfg_inp_red, cfg_out_red = _get_conversation_from_config(workflow)
            if cfg_inp is not None and cfg_out is not None:
                kwargs["llm_input_messages"] = cfg_inp
                kwargs["llm_output_messages"] = cfg_out
                if cfg_inp_red is not None and cfg_out_red is not None:
                    kwargs["llm_input_messages_redacted"] = cfg_inp_red
                    kwargs["llm_output_messages_redacted"] = cfg_out_red
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
        service_name: str = "otelsim",
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

    def _incoming_validation_hierarchy(self, scenario: Scenario) -> TraceHierarchy:
        """Build incoming request validation hierarchy from config template (no hardcoded outcomes)."""
        from .control_plane_loader import (
            build_request_validation_hierarchy_from_template,
            get_request_validation_template_id,
        )

        outcome = getattr(scenario, "control_plane_request_outcome", "allowed") or "allowed"
        block_reason = getattr(scenario, "control_plane_block_reason", None)
        template_override = getattr(scenario, "control_plane_template", None)
        template_id = get_request_validation_template_id(outcome, block_reason, template_override)
        policy_exception_override = getattr(
            scenario, "control_plane_policy_exception_override", None
        )
        return build_request_validation_hierarchy_from_template(
            template_id, policy_exception_override=policy_exception_override
        )

    def _response_validation_hierarchy(self, scenario: Scenario) -> TraceHierarchy:
        """Build outgoing response validation hierarchy from config template."""
        from .control_plane_loader import build_response_validation_hierarchy_from_template

        return build_response_validation_hierarchy_from_template("allowed")

    def _get_trace_flow_and_hierarchies(
        self, scenario: Scenario
    ) -> tuple[list[str], list[TraceHierarchy], bool]:
        """Resolve trace flow and data-plane hierarchies from scenario + config (no code branches)."""
        from ..generators.trace_generator import SpanType  # local import to avoid cycles
        from .control_plane_loader import get_request_validation_template_id, get_trace_flow

        hierarchies = scenario.get_trace_hierarchies()
        trace_flow = get_trace_flow(
            getattr(scenario, "control_plane_request_outcome", "allowed") or "allowed",
            template_id=get_request_validation_template_id(
                getattr(scenario, "control_plane_request_outcome", "allowed") or "allowed",
                getattr(scenario, "control_plane_block_reason", None),
                getattr(scenario, "control_plane_template", None),
            ),
        )
        has_data_plane = any(
            h.root_config.span_type is SpanType.A2A_ORCHESTRATE for h in hierarchies
        )
        return trace_flow, hierarchies, has_data_plane

    def _emit_traces_for_request(
        self,
        scenario: Scenario,
        context: GenerationContext,
        trace_flow: list[str],
        hierarchies: list[TraceHierarchy],
        has_data_plane: bool,
    ) -> tuple[list[str], list[str]]:
        """Emit traces (and optional metrics/logs) for one logical request from config-driven trace_flow."""
        iteration_trace_ids: list[str] = []
        all_span_names: list[str] = []

        use_unified = (
            "incoming_validation" in trace_flow
            and has_data_plane
            and "data_plane" in trace_flow
            and "response_validation" in trace_flow
            and len(hierarchies) > 0
        )
        if use_unified:
            incoming_h = self._incoming_validation_hierarchy(scenario)
            outgoing_h = self._response_validation_hierarchy(scenario)
            tid = self.trace_generator.generate_unified_request_trace(
                incoming_h, hierarchies[0], outgoing_h, context
            )
            iteration_trace_ids.append(tid)
            all_span_names.extend(incoming_h.span_names())
            all_span_names.extend(hierarchies[0].span_names())
            all_span_names.extend(outgoing_h.span_names())
            if scenario.emit_metrics and self.metric_generator:
                self._emit_metrics_for_hierarchy(incoming_h, context)
                self._emit_metrics_for_hierarchy(hierarchies[0], context)
                self._emit_metrics_for_hierarchy(outgoing_h, context)
            if scenario.emit_logs and self.log_generator:
                self._emit_logs_for_hierarchy(incoming_h, context)
                self._emit_logs_for_hierarchy(hierarchies[0], context)
                self._emit_logs_for_hierarchy(outgoing_h, context)
            return iteration_trace_ids, all_span_names

        if "incoming_validation" in trace_flow:
            incoming_h = self._incoming_validation_hierarchy(scenario)
            tid = self.trace_generator.generate_trace(incoming_h, context)
            iteration_trace_ids.append(tid)
            all_span_names.extend(incoming_h.span_names())
            if scenario.emit_metrics and self.metric_generator:
                self._emit_metrics_for_hierarchy(incoming_h, context)
            if scenario.emit_logs and self.log_generator:
                self._emit_logs_for_hierarchy(incoming_h, context)

        if has_data_plane and "data_plane" in trace_flow:
            for h in hierarchies:
                tid = self.trace_generator.generate_trace(h, context)
                iteration_trace_ids.append(tid)
                all_span_names.extend(h.span_names())
                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(h, context)
                if scenario.emit_logs and self.log_generator:
                    self._emit_logs_for_hierarchy(h, context)

        if has_data_plane and "response_validation" in trace_flow:
            outgoing_h = self._response_validation_hierarchy(scenario)
            tid = self.trace_generator.generate_trace(outgoing_h, context)
            iteration_trace_ids.append(tid)
            all_span_names.extend(outgoing_h.span_names())
            if scenario.emit_metrics and self.metric_generator:
                self._emit_metrics_for_hierarchy(outgoing_h, context)
            if scenario.emit_logs and self.log_generator:
                self._emit_logs_for_hierarchy(outgoing_h, context)
        elif not has_data_plane:
            for h in hierarchies:
                tid = self.trace_generator.generate_trace(h, context)
                iteration_trace_ids.append(tid)
                all_span_names.extend(h.span_names())
                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(h, context)
                if scenario.emit_logs and self.log_generator:
                    self._emit_logs_for_hierarchy(h, context)

        return iteration_trace_ids, all_span_names

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
            ctx_kwargs = _context_kwargs_for_scenario(scenario, tenant_id, iteration_index=i)
            # One session_id per logical session (iteration); all turns in this iteration share it.
            session_id = ctx_kwargs["session_id"]
            id_gen = getattr(scenario, "id_generator", None)

            # Control-plane and data-plane flow from config/template/scenario only (trace_flow).
            trace_flow, hierarchies, has_data_plane = self._get_trace_flow_and_hierarchies(scenario)
            iteration_trace_ids: list[str] = []
            all_span_names: list[str] = []

            if use_multi_turn:
                # Multi-turn: each turn = one new request reaching control-plane (incoming).
                # Different trace_id per turn; same session_id for the whole session.
                turn_pairs_redacted = (
                    getattr(scenario, "conversation_turn_pairs_redacted", None) or []
                )
                for turn_index, (input_msgs, output_msgs) in enumerate(turn_pairs):
                    ctx_kwargs["session_id"] = session_id
                    ctx_kwargs["turn_index"] = turn_index
                    ctx_kwargs["llm_input_messages"] = input_msgs
                    ctx_kwargs["llm_output_messages"] = output_msgs
                    if (
                        turn_index < len(turn_pairs_redacted)
                        and turn_pairs_redacted[turn_index] is not None
                    ):
                        inp_red, out_red = turn_pairs_redacted[turn_index]
                        ctx_kwargs["llm_input_messages_redacted"] = inp_red
                        ctx_kwargs["llm_output_messages_redacted"] = out_red
                    else:
                        ctx_kwargs.pop("llm_input_messages_redacted", None)
                        ctx_kwargs.pop("llm_output_messages_redacted", None)
                    if isinstance(id_gen, ScenarioIdGenerator):
                        ctx_kwargs["request_id"] = id_gen.request_id(tenant_id=tenant_id)
                    else:
                        ctx_kwargs["request_id"] = generate_request_id(tenant_id=tenant_id)
                    context = GenerationContext.create(**ctx_kwargs)
                    tids, names = self._emit_traces_for_request(
                        scenario, context, trace_flow, hierarchies, has_data_plane
                    )
                    iteration_trace_ids.extend(tids)
                    trace_ids.extend(tids)
                    all_span_names = names  # last turn's span names for progress
            else:
                ctx_kwargs["turn_index"] = i % 10
                context = GenerationContext.create(**ctx_kwargs)
                iteration_trace_ids, all_span_names = self._emit_traces_for_request(
                    scenario, context, trace_flow, hierarchies, has_data_plane
                )
                trace_ids.extend(iteration_trace_ids)

            if progress_callback:
                primary_trace_id = iteration_trace_ids[-1] if iteration_trace_ids else ""
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
        tags: list[str] | None = None,
        each_once: bool = False,
    ) -> tuple[list[str], dict[str, int]]:
        """Run a mixed workload by picking at random from YAML-defined scenarios.
        If tags is non-empty, only scenarios that have at least one of these tags are included.
        If each_once is True, run each (filtered) scenario exactly once instead of count random picks.
        Returns (trace_ids, traces_by_scenario) where traces_by_scenario maps scenario name to trace count.
        """
        scenarios = self.scenario_loader.load_all()
        if tags:
            tag_set = {t.strip().lower() for t in tags if t and isinstance(t, str)}
            if tag_set:
                scenarios = [
                    s
                    for s in scenarios
                    if getattr(s, "tags", None)
                    and any((x or "").strip().lower() in tag_set for x in s.tags)
                ]
        if not scenarios:
            dir_path = self.scenario_loader.scenarios_dir
            if tags:
                raise ValueError(
                    f"No scenarios match tags: {', '.join(tags)}. "
                    f"Add tags to scenario YAML (e.g. tags: [control-plane]) or use different --tags."
                )
            raise ValueError(
                f"No YAML scenarios found in {dir_path}. "
                "Add at least one .yaml file there, or pass --scenarios-dir to use a custom folder. "
                "Sample definitions live in scenarios/definitions/ (project root or TELEMETRY_SIMULATOR_ROOT)."
            )

        trace_ids: list[str] = []
        traces_by_scenario: dict[str, int] = {}
        total = len(scenarios) if each_once else count

        # When workload_weight is set on scenarios, use it as a relative sampling weight
        # for mixed workloads so that, for example, successful control-plane outcomes
        # can be more frequent than error/blocked outcomes.
        weights = [max(getattr(s, "workload_weight", 1.0), 0.0) for s in scenarios]
        use_weighted_sampling = not each_once and any(w > 0 for w in weights)

        def run_one_request(scenario: Scenario, index: int) -> None:
            tenants = list(scenario.tenant_distribution.keys())
            weights = list(scenario.tenant_distribution.values())
            tenant_id = random.choices(tenants, weights=weights)[0]
            ctx_kwargs = _context_kwargs_for_scenario(scenario, tenant_id)
            ctx_kwargs["turn_index"] = index % 10
            context = GenerationContext.create(**ctx_kwargs)
            trace_flow, hierarchies, has_data_plane = self._get_trace_flow_and_hierarchies(scenario)
            iteration_trace_ids, all_span_names = self._emit_traces_for_request(
                scenario, context, trace_flow, hierarchies, has_data_plane
            )
            trace_ids.extend(iteration_trace_ids)
            scenario_name = getattr(scenario, "name", "unknown")
            traces_by_scenario[scenario_name] = (
                traces_by_scenario.get(scenario_name, 0) + len(iteration_trace_ids)
            )
            primary_trace_id = iteration_trace_ids[-1] if iteration_trace_ids else ""
            if progress_callback:
                progress_callback(
                    index + 1,
                    total,
                    primary_trace_id,
                    context.tenant_id,
                    all_span_names,
                )

        if each_once:
            for i, scenario in enumerate(scenarios):
                run_one_request(scenario, i)
                if interval_ms > 0 and i < total - 1:
                    time.sleep(interval_ms / 1000.0)
        else:
            for i in range(count):
                if use_weighted_sampling:
                    scenario = random.choices(scenarios, weights=weights, k=1)[0]
                else:
                    scenario = random.choice(scenarios)
                run_one_request(scenario, i)
                if interval_ms > 0 and i < count - 1:
                    time.sleep(interval_ms / 1000.0)

        return trace_ids, traces_by_scenario

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
