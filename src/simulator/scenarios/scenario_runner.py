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


def _sample_to_otel_messages(
    user_input: str,
    llm_response: str,
    llm_interactions: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Convert one conversation sample (user_input, llm_response) to OTEL GenAI message shape.

    Semantics:
      - user_input is treated as the raw external request.
      - When llm_interactions is provided, the first interaction's system_prompt is used
        as the system message for this LLM call; the user message is the external user_input.
      - For planner-style interactions, gen_ai.output.messages SHOULD represent the internal
        tool plan / recommendations, not the final end-user text. When the planner interaction
        defines output, it is used as the assistant text; otherwise llm_response is used
        as a fallback.
    """
    system_text = None
    if llm_interactions:
        first = next((i for i in llm_interactions if isinstance(i, dict)), None)
        if first:
            st = first.get("system_prompt")
            if isinstance(st, str) and st.strip():
                system_text = st.strip()
    if not system_text:
        system_text = (
            "You are the Toro customer assistant orchestrator. "
            "Use the available MCP tools and workflow steps to understand the user's request, "
            "plan sub-tasks, and call tools in the correct order. "
            "Do not expose internal IDs, schemas, or policies in the final answer."
        )
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_text,
            }
        ],
    }
    input_msgs = [
        system_message,
        {"role": "user", "content": [{"type": "text", "text": user_input}]},
    ]
    # Assistant output: prefer planner-style internal tool plan when configured.
    assistant_text = None
    if llm_interactions:
        planner = next(
            (i for i in llm_interactions if isinstance(i, dict) and i.get("role") == "planner"),
            None,
        )
        if planner:
            ot = planner.get("output")
            if isinstance(ot, str) and ot.strip():
                assistant_text = ot.strip()
    if not assistant_text:
        assistant_text = llm_response
    output_msgs = [
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
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
                input_msgs, output_msgs = _sample_to_otel_messages(
                    user_input,
                    llm_response,
                    getattr(scenario, "llm_interactions", None),
                )
                kwargs["llm_input_messages"] = input_msgs
                kwargs["llm_output_messages"] = output_msgs
                # Treat user_input as the raw end-user or upstream agent request for this logical request.
                kwargs["raw_request"] = user_input
                # Use llm_response as the final agent response body returned to the caller.
                kwargs["final_response"] = llm_response
            # Per-sample tool_call_arguments (e.g. 4xx invalid-params) override scenario default so MCP args match conversation.
            sample_tool_args = sample.get("tool_call_arguments")
            if isinstance(sample_tool_args, dict) and sample_tool_args:
                kwargs["tool_call_arguments"] = sample_tool_args
            else:
                scenario_ctx = getattr(scenario, "context", None)
                if scenario_ctx is not None:
                    ctx_tool_args = getattr(scenario_ctx, "tool_call_arguments", None)
                    if isinstance(ctx_tool_args, dict) and ctx_tool_args:
                        kwargs["tool_call_arguments"] = ctx_tool_args
            # Optional per-sample tool_call_results; otherwise fall back to scenario context.
            sample_tool_results = sample.get("tool_call_results")
            if isinstance(sample_tool_results, dict) and sample_tool_results:
                kwargs["tool_call_results"] = sample_tool_results
            else:
                scenario_ctx = getattr(scenario, "context", None)
                if scenario_ctx is not None:
                    ctx_tool_results = getattr(scenario_ctx, "tool_call_results", None)
                    if isinstance(ctx_tool_results, dict) and ctx_tool_results:
                        kwargs["tool_call_results"] = ctx_tool_results
            ui_red = sample.get("user_input_redacted")
            lr_red = sample.get("llm_response_redacted")
            if isinstance(ui_red, str) and isinstance(lr_red, str):
                inp_red, out_red = _sample_to_otel_messages(
                    ui_red,
                    lr_red,
                    getattr(scenario, "llm_interactions", None),
                )
                kwargs["llm_input_messages_redacted"] = inp_red
                kwargs["llm_output_messages_redacted"] = out_red
                # Redacted raw request variant for control-plane event.
                kwargs["raw_request_redacted"] = ui_red
                # Redacted final response variant for response validation event.
                kwargs["final_response_redacted"] = lr_red
        else:
            context = getattr(scenario, "context", None)
            workflow = context.workflow if context else None
            if context and getattr(context, "tool_call_arguments", None):
                kwargs["tool_call_arguments"] = context.tool_call_arguments
            if context and getattr(context, "tool_call_results", None):
                kwargs["tool_call_results"] = context.tool_call_results
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

        # Optional: workload schedule for time-of-day scenario mix. When enabled via
        # use_workload_schedule, run_mixed_workload() can pick a scenario name from
        # config.workload_schedules.default instead of static workload_weight.
        self._workload_schedule: dict[str, Any] | None = None

    def _load_default_workload_schedule(self) -> dict[str, Any] | None:
        """Load workload_schedules.default from config for time-of-day scenario mix."""
        try:
            from .scenario_loader import CONFIG_PATH, load_yaml  # type: ignore[attr-defined]
        except Exception:
            return None
        try:
            data = load_yaml(CONFIG_PATH)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        schedules = data.get("workload_schedules")
        if not isinstance(schedules, dict):
            return None
        default = schedules.get("default")
        if not isinstance(default, dict):
            return None
        buckets = default.get("buckets")
        if not isinstance(buckets, dict) or not buckets:
            return None
        return default

    def _pick_scenario_by_schedule(
        self,
        scenarios: list[Scenario],
        schedule: dict[str, Any],
    ) -> Scenario | None:
        """Pick a Scenario based on workload_schedules.default and current local hour."""
        import random
        from datetime import datetime

        # Resolve timezone; fall back to naive local time if pytz/zoneinfo not available.
        tz_name = schedule.get("timezone") or "UTC"
        now = None
        try:
            from zoneinfo import ZoneInfo  # type: ignore[import]

            now = datetime.now(ZoneInfo(str(tz_name)))
        except Exception:
            now = datetime.now()
        hour = now.hour

        buckets = schedule.get("buckets") or {}
        selected_weights: dict[str, float] | None = None
        for bucket_cfg in buckets.values():
            if not isinstance(bucket_cfg, dict):
                continue
            hours_spec = bucket_cfg.get("hours")
            if not isinstance(hours_spec, list) or not hours_spec:
                continue
            # Each hours entry can be a single "start-end" string or a two-element list [start, end].
            for entry in hours_spec:
                start = end = None
                if isinstance(entry, str) and "-" in entry:
                    try:
                        start_s, end_s = entry.split("-", 1)
                        start = int(start_s)
                        end = int(end_s)
                    except Exception:
                        continue
                elif (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and isinstance(entry[0], (int, float))
                    and isinstance(entry[1], (int, float))
                ):
                    start = int(entry[0])
                    end = int(entry[1])
                if start is None or end is None:
                    continue
                if start <= hour < end:
                    sw = bucket_cfg.get("scenario_weights")
                    if isinstance(sw, dict) and sw:
                        selected_weights = {
                            str(k): float(v)
                            for k, v in sw.items()
                            if isinstance(v, (int, float)) and v > 0
                        }
                    break
            if selected_weights:
                break
        if not selected_weights:
            return None

        # Restrict to scenarios that are present in schedule.
        eligible = [s for s in scenarios if getattr(s, "name", None) in selected_weights]
        if not eligible:
            return None
        weights = [selected_weights.get(s.name, 0.0) for s in eligible]
        if not any(w > 0 for w in weights):
            return None
        return random.choices(eligible, weights=weights, k=1)[0]

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
        augmentation_exception_override = getattr(
            scenario, "control_plane_augmentation_exception_override", None
        )
        return build_request_validation_hierarchy_from_template(
            template_id,
            policy_exception_override=policy_exception_override,
            augmentation_exception_override=augmentation_exception_override,
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

        # Attach trace/span ids directly to SpanConfig instances so metrics/logs can read them later.
        def _capture_span(config: Any, trace_id: str, span_id: str) -> None:
            try:
                config._trace_ids = trace_id, span_id
            except Exception:
                pass

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
                incoming_h,
                hierarchies[0],
                outgoing_h,
                context,
                span_callback=_capture_span,
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
            self.trace_generator.force_flush()
            return iteration_trace_ids, all_span_names

        if "incoming_validation" in trace_flow:
            incoming_h = self._incoming_validation_hierarchy(scenario)
            tid = self.trace_generator.generate_trace(
                incoming_h,
                context,
                span_callback=_capture_span,
            )
            iteration_trace_ids.append(tid)
            all_span_names.extend(incoming_h.span_names())
            if scenario.emit_metrics and self.metric_generator:
                self._emit_metrics_for_hierarchy(incoming_h, context)
            if scenario.emit_logs and self.log_generator:
                self._emit_logs_for_hierarchy(incoming_h, context)
            self.trace_generator.force_flush()

        if has_data_plane and "data_plane" in trace_flow:
            for h in hierarchies:
                tid = self.trace_generator.generate_trace(
                    h,
                    context,
                    span_callback=_capture_span,
                )
                iteration_trace_ids.append(tid)
                all_span_names.extend(h.span_names())
                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(h, context)
                if scenario.emit_logs and self.log_generator:
                    self._emit_logs_for_hierarchy(h, context)
                self.trace_generator.force_flush()

        if has_data_plane and "response_validation" in trace_flow:
            outgoing_h = self._response_validation_hierarchy(scenario)
            tid = self.trace_generator.generate_trace(
                outgoing_h,
                context,
                span_callback=_capture_span,
            )
            iteration_trace_ids.append(tid)
            all_span_names.extend(outgoing_h.span_names())
            if scenario.emit_metrics and self.metric_generator:
                self._emit_metrics_for_hierarchy(outgoing_h, context)
            if scenario.emit_logs and self.log_generator:
                self._emit_logs_for_hierarchy(outgoing_h, context)
            self.trace_generator.force_flush()
        elif not has_data_plane:
            for h in hierarchies:
                tid = self.trace_generator.generate_trace(
                    h,
                    context,
                    span_callback=_capture_span,
                )
                iteration_trace_ids.append(tid)
                all_span_names.extend(h.span_names())
                if scenario.emit_metrics and self.metric_generator:
                    self._emit_metrics_for_hierarchy(h, context)
                if scenario.emit_logs and self.log_generator:
                    self._emit_logs_for_hierarchy(h, context)
                self.trace_generator.force_flush()

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
            # One session_id and one enduser.pseudo.id per logical session (iteration); all turns share them.
            session_id = ctx_kwargs["session_id"]
            user_id = ctx_kwargs.get("user_id") or generate_enduser_pseudo_id(tenant_id=tenant_id)
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
                scenario_ctx = getattr(scenario, "context", None)
                original_actual_steps = (
                    getattr(scenario_ctx, "actual_steps", None)
                    if scenario_ctx is not None
                    else None
                )
                per_turn_steps = getattr(scenario, "per_turn_actual_steps", None) or []
                # Ensure multi-turn traces inherit scenario-level tool_call_arguments/results so
                # MCP attempt spans can emit gen_ai.tool.call.arguments/result just like single-turn.
                if scenario_ctx is not None:
                    ctx_tool_args = getattr(scenario_ctx, "tool_call_arguments", None)
                    if isinstance(ctx_tool_args, dict) and ctx_tool_args:
                        ctx_kwargs["tool_call_arguments"] = ctx_tool_args
                    ctx_tool_results = getattr(scenario_ctx, "tool_call_results", None)
                    if isinstance(ctx_tool_results, dict) and ctx_tool_results:
                        ctx_kwargs["tool_call_results"] = ctx_tool_results
                for turn_index, (input_msgs, output_msgs) in enumerate(turn_pairs):
                    # Introduce a small, realistic delay between turns in the same session
                    # so multi-turn traces do not all share nearly identical timestamps.
                    # Ensure at least ~1s separation between turns, with modest jitter.
                    if turn_index > 0:
                        base_ms = scenario.interval_ms if scenario.interval_ms > 0 else 1000.0
                        delay_ms = max(1000.0, base_ms)
                        jitter_ms = min(delay_ms * 0.3, 300.0)
                        delay_ms = max(
                            0.0,
                            delay_ms
                            + random.uniform(
                                -jitter_ms,
                                jitter_ms,
                            ),
                        )
                        time.sleep(delay_ms / 1000.0)
                    ctx_kwargs["session_id"] = session_id
                    ctx_kwargs["user_id"] = user_id
                    ctx_kwargs["turn_index"] = turn_index
                    ctx_kwargs["llm_input_messages"] = input_msgs
                    ctx_kwargs["llm_output_messages"] = output_msgs
                    # Derive raw_request and final_response for this turn from OTEL GenAI messages
                    # so gentoro.enduser.input and related fields are per-turn instead of shared.
                    user_text = None
                    if isinstance(input_msgs, list):
                        for msg in reversed(input_msgs):
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                contents = msg.get("content") or []
                                if contents and isinstance(contents[0], dict):
                                    user_text = contents[0].get("text")
                                break
                    if isinstance(user_text, str) and user_text:
                        ctx_kwargs["raw_request"] = user_text
                    else:
                        ctx_kwargs.pop("raw_request", None)

                    assistant_text = None
                    if isinstance(output_msgs, list):
                        for msg in reversed(output_msgs):
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                contents = msg.get("content") or []
                                if contents and isinstance(contents[0], dict):
                                    assistant_text = contents[0].get("text")
                                break
                    if isinstance(assistant_text, str) and assistant_text:
                        ctx_kwargs["final_response"] = assistant_text
                    else:
                        ctx_kwargs.pop("final_response", None)
                    if (
                        turn_index < len(turn_pairs_redacted)
                        and turn_pairs_redacted[turn_index] is not None
                    ):
                        inp_red, out_red = turn_pairs_redacted[turn_index]
                        ctx_kwargs["llm_input_messages_redacted"] = inp_red
                        ctx_kwargs["llm_output_messages_redacted"] = out_red
                        red_user_text = None
                        if isinstance(inp_red, list):
                            for msg in reversed(inp_red):
                                if isinstance(msg, dict) and msg.get("role") == "user":
                                    contents = msg.get("content") or []
                                    if contents and isinstance(contents[0], dict):
                                        red_user_text = contents[0].get("text")
                                    break
                        if isinstance(red_user_text, str) and red_user_text:
                            ctx_kwargs["raw_request_redacted"] = red_user_text
                        else:
                            ctx_kwargs.pop("raw_request_redacted", None)

                        red_assistant_text = None
                        if isinstance(out_red, list):
                            for msg in reversed(out_red):
                                if isinstance(msg, dict) and msg.get("role") == "assistant":
                                    contents = msg.get("content") or []
                                    if contents and isinstance(contents[0], dict):
                                        red_assistant_text = contents[0].get("text")
                                    break
                        if isinstance(red_assistant_text, str) and red_assistant_text:
                            ctx_kwargs["final_response_redacted"] = red_assistant_text
                        else:
                            ctx_kwargs.pop("final_response_redacted", None)
                    else:
                        ctx_kwargs.pop("llm_input_messages_redacted", None)
                        ctx_kwargs.pop("llm_output_messages_redacted", None)
                        ctx_kwargs.pop("raw_request_redacted", None)
                        ctx_kwargs.pop("final_response_redacted", None)
                    # Per-turn workflow override: when per_turn_actual_steps is set, use it to
                    # adjust context.actual_steps so data-plane tools differ by turn.
                    if scenario_ctx is not None:
                        if turn_index < len(per_turn_steps) and per_turn_steps[turn_index]:
                            scenario_ctx.actual_steps = per_turn_steps[turn_index]
                        else:
                            scenario_ctx.actual_steps = original_actual_steps
                        # Recompute hierarchies for this turn with the updated context.
                        _, hierarchies, has_data_plane = self._get_trace_flow_and_hierarchies(
                            scenario
                        )

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

                # Restore original actual_steps after all turns in this session.
                if scenario_ctx is not None:
                    scenario_ctx.actual_steps = original_actual_steps
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
        count: int | None = None,
        interval_ms: float = 200,
        progress_callback: Callable[[int, int, str, str, list[str]], None] | None = None,
        tags: list[str] | None = None,
        each_once: bool = False,
        use_workload_schedule: bool = False,
    ) -> tuple[list[str], dict[str, int]]:
        """Run a mixed workload by picking at random from YAML-defined scenarios.
        If tags is non-empty, only scenarios that have at least one of these tags are included.
        If each_once is True, run each (filtered) scenario exactly once instead of count random picks.
        If count is None, run until interrupted (progress total is 0 for display).
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
                "Sample definitions live in resource/scenarios/definitions/ (under project root or TELEMETRY_SIMULATOR_ROOT)."
            )

        trace_ids: list[str] = []
        traces_by_scenario: dict[str, int] = {}
        # 0 for total means unbounded (run until interrupted)
        total = len(scenarios) if each_once else (count if count is not None else 0)

        # When workload_weight is set on scenarios, use it as a relative sampling weight
        # for mixed workloads so that, for example, successful control-plane outcomes
        # can be more frequent than error/blocked outcomes.
        weights = [max(getattr(s, "workload_weight", 1.0), 0.0) for s in scenarios]
        use_weighted_sampling = not each_once and any(w > 0 for w in weights)

        schedule = None
        if not each_once and use_workload_schedule:
            if self._workload_schedule is None:
                self._workload_schedule = self._load_default_workload_schedule()
            schedule = self._workload_schedule

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
            traces_by_scenario[scenario_name] = traces_by_scenario.get(scenario_name, 0) + len(
                iteration_trace_ids
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
        elif count is not None:
            for i in range(count):
                if schedule:
                    picked = self._pick_scenario_by_schedule(scenarios, schedule)
                    scenario = picked or (
                        random.choices(scenarios, weights=weights, k=1)[0]
                        if use_weighted_sampling
                        else random.choice(scenarios)
                    )
                else:
                    if use_weighted_sampling:
                        scenario = random.choices(scenarios, weights=weights, k=1)[0]
                    else:
                        scenario = random.choice(scenarios)
                run_one_request(scenario, i)
                if interval_ms > 0 and i < count - 1:
                    time.sleep(interval_ms / 1000.0)
        else:
            i = 0
            while True:
                if schedule:
                    picked = self._pick_scenario_by_schedule(scenarios, schedule)
                    scenario = picked or (
                        random.choices(scenarios, weights=weights, k=1)[0]
                        if use_weighted_sampling
                        else random.choice(scenarios)
                    )
                else:
                    if use_weighted_sampling:
                        scenario = random.choices(scenarios, weights=weights, k=1)[0]
                    else:
                        scenario = random.choice(scenarios)
                run_one_request(scenario, i)
                if interval_ms > 0:
                    time.sleep(interval_ms / 1000.0)
                i += 1

        return trace_ids, traces_by_scenario

    def _emit_metrics_for_hierarchy(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext,
    ):
        """Emit metrics corresponding to a trace hierarchy."""
        if not self.metric_generator:
            return

        self._emit_metrics_recursive(
            hierarchy.root_config,
            hierarchy.children,
            context,
        )

    def _emit_metrics_recursive(
        self,
        config,
        children,
        context,
    ):
        """Recursively emit metrics for span configs."""
        from ..generators.trace_generator import SpanType

        span_type = config.span_type
        latency = config.latency_mean_ms
        attrs = config.attribute_overrides or {}
        trace_id: str | None = None
        span_id: str | None = None
        ids = getattr(config, "_trace_ids", None)
        if isinstance(ids, tuple) and len(ids) == 2:
            trace_id, span_id = ids

        if span_type == SpanType.A2A_ORCHESTRATE:
            self.metric_generator.record_turn(
                context,
                duration_ms=latency,
                status_code=attrs.get(config_attr("turn.status.code"), "SUCCESS"),
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type in (SpanType.MCP_TOOL_EXECUTE, SpanType.MCP_TOOL_EXECUTE_ATTEMPT):
            self.metric_generator.record_tool_call(
                context,
                tool_name=attrs.get("gen_ai.tool.name", "unknown.tool"),
                server_name=attrs.get(config_attr("tool.server.name"), "unknown-server"),
                latency_ms=latency,
                status_code=attrs.get(config_attr("tool.status.code"), "OK"),
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.LLM_CALL:
            self.metric_generator.record_llm_inference(
                context,
                provider=attrs.get("gen_ai.system", "openai"),
                model=attrs.get("gen_ai.request.model", "gpt-4.1-mini"),
                latency_ms=latency,
                input_tokens=_int_token_count(attrs.get("gen_ai.usage.input_tokens"), 500),
                output_tokens=_int_token_count(attrs.get("gen_ai.usage.output_tokens"), 200),
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.RAG_RETRIEVE:
            self.metric_generator.record_rag_retrieval(
                context,
                index_name=attrs.get("rag.index.name", "default_index"),
                latency_ms=latency,
                docs_returned=attrs.get("rag.documents.returned", 5),
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.A2A_CALL:
            self.metric_generator.record_a2a_call(
                context,
                target_agent=attrs.get("a2a.target.agent", "unknown_agent"),
                latency_ms=latency,
                status_code=attrs.get(config_attr("a2a.status.code"), "OK"),
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.CP_REQUEST:
            self.metric_generator.record_cp_request(
                context,
                duration_ms=latency,
                status_code=attrs.get(config_attr("cp.status.code"), "ALLOWED"),
                trace_id=trace_id,
                span_id=span_id,
            )

        for child in children:
            self._emit_metrics_recursive(child.root_config, child.children, context)

    def _emit_logs_for_hierarchy(
        self,
        hierarchy: TraceHierarchy,
        context: GenerationContext,
    ):
        """Emit logs corresponding to a trace hierarchy."""
        if self.log_generator is None:
            return

        self._emit_logs_recursive(
            hierarchy.root_config,
            hierarchy.children,
            context,
            is_root=True,
        )

    def _emit_logs_recursive(
        self,
        config,
        children,
        context,
        is_root: bool = False,
    ):
        """Recursively emit logs for span configs."""
        from ..generators.trace_generator import SpanType

        if self.log_generator is None:
            return
        log_generator = self.log_generator

        span_type = config.span_type
        attrs = config.attribute_overrides or {}
        trace_id: str | None = None
        span_id: str | None = None
        ids = getattr(config, "_trace_ids", None)
        if isinstance(ids, tuple) and len(ids) == 2:
            trace_id, span_id = ids

        if span_type == SpanType.A2A_ORCHESTRATE:
            log_generator.log_turn_start(
                context,
                trace_id=trace_id,
                span_id=span_id,
            )

        if span_type in (SpanType.MCP_TOOL_EXECUTE, SpanType.MCP_TOOL_EXECUTE_ATTEMPT):
            log_generator.log_tool_call(
                context,
                tool_name=attrs.get("gen_ai.tool.name", "unknown.tool"),
                server_name=attrs.get(config_attr("tool.server.name"), "unknown-server"),
                status_code=attrs.get(config_attr("tool.status.code"), "OK"),
                latency_ms=config.latency_mean_ms,
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.LLM_CALL:
            log_generator.log_llm_inference(
                context,
                provider=attrs.get("gen_ai.system", "openai"),
                model=attrs.get("gen_ai.request.model", "gpt-4.1-mini"),
                operation=attrs.get(config_attr("llm.operation"), "chat"),
                input_tokens=_int_token_count(attrs.get("gen_ai.usage.input_tokens"), 500),
                output_tokens=_int_token_count(attrs.get("gen_ai.usage.output_tokens"), 200),
                latency_ms=config.latency_mean_ms,
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.RAG_RETRIEVE:
            log_generator.log_rag_retrieval(
                context,
                index_name=attrs.get("rag.index.name", "default_index"),
                docs_returned=attrs.get("rag.documents.returned", 5),
                latency_ms=config.latency_mean_ms,
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.A2A_CALL:
            log_generator.log_a2a_call(
                context,
                target_agent=attrs.get("a2a.target.agent", "unknown_agent"),
                operation=attrs.get("a2a.operation", "delegate"),
                status_code=attrs.get(config_attr("a2a.status.code"), "OK"),
                latency_ms=config.latency_mean_ms,
                trace_id=trace_id,
                span_id=span_id,
            )
        elif span_type == SpanType.CP_REQUEST:
            log_generator.log_cp_request(
                context,
                status_code=attrs.get(config_attr("cp.status.code"), "ALLOWED"),
                trace_id=trace_id,
                span_id=span_id,
            )

        for child in children:
            self._emit_logs_recursive(
                child.root_config,
                child.children,
                context,
                is_root=False,
            )

        if span_type == SpanType.A2A_ORCHESTRATE:
            log_generator.log_turn_end(
                context,
                status_code=attrs.get(config_attr("turn.status.code"), "SUCCESS"),
                duration_ms=config.latency_mean_ms,
                trace_id=trace_id,
                span_id=span_id,
            )

    def shutdown(self):
        """Shutdown all generators."""
        self.trace_generator.shutdown()
        if self.metric_generator:
            self.metric_generator.shutdown()
        if self.log_generator:
            self.log_generator.shutdown()
