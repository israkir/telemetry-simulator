"""
Run scenarios: resolve config, compile turns to TraceGraphSpec, render and export spans.

Uses scenarios.compiler for scenario → TraceGraphSpec and trace_generator.render_trace
for OTEL span creation. Resource attributes from resource.yaml (control-plane / data-plane).
"""

import itertools
import random
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from opentelemetry.sdk._logs.export import LogRecordExporter
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from .. import config as sim_config
from ..config import load_config
from ..generators.trace_generator import build_unified_trace_from_semconv
from ..resource_loader import load_resource_presets
from ..scenarios.compiler import compile_turn
from ..scenarios.config_resolver import resolve_context
from ..scenarios.dependency_rules import validate_trace_graph
from ..scenarios.id_generator import generate_ids_for_turn
from ..scenarios.latency import LatencyModel
from ..scenarios.scenario_loader import Scenario, ScenarioLoader
from ..schemas.schema_parser import ParsedSchema, SchemaParser
from ..telemetry.export_constants import (
    BSP_EXPORT_TIMEOUT_MILLIS,
    BSP_MAX_EXPORT_BATCH_SIZE,
    BSP_SCHEDULE_DELAY_MILLIS,
    FLUSH_SHUTDOWN_FAST_MILLIS,
    FLUSH_TIMEOUT_MILLIS,
)
from ..telemetry.otel_emitter import OtelEmitter


class ScenarioRunner:
    """Run scenarios and export traces via the provided exporter.

    Metrics and logs share one ``OtelEmitter`` (and thus CP/DP ``MeterProvider``s plus
    one ``LoggerProvider``) for the runner lifetime, created on the first successful
    ``run_scenario`` and shut down in ``shutdown()``. Batched signals flush on shutdown
    (not after each scenario) to avoid OTLP export storms. The OTLP resource includes
    the tenant resource attribute chosen by ``--vendor`` (e.g. ``vendor.tenant.id`` or
    ``gentoro.tenant.id``) so downstream systems
    such as Gentoro can index points by tenant. Traces use separate CP/DP
    ``BatchSpanProcessor`` instances and exporter objects from the CLI.
    """

    def __init__(
        self,
        trace_exporter: SpanExporter,
        trace_exporter_dp: SpanExporter | None = None,
        metric_exporter: MetricExporter | None = None,
        metric_exporter_control_plane: MetricExporter | None = None,
        log_exporter: LogRecordExporter | None = None,
        schema_path: str | None = None,
        service_name: str = "otelsim",
        show_full_spans: bool = False,
        scenarios_dir: str | Path | None = None,
        config_dir: str | Path | None = None,
        default_tenant_id: str | None = None,
        debug_validate: bool = False,
        include_metric_span_trace_ids: bool = True,
        max_cached_tenants: int = 8,
    ):
        # CP and DP each attach a BatchSpanProcessor, which calls exporter.shutdown().
        # Callers must pass distinct exporter instances (or pair file exporters with a
        # shared coordinator); sharing one OTLP exporter across both processors triggers
        # duplicate shutdown on the same client.
        self.trace_exporter_cp = trace_exporter
        self.trace_exporter_dp = (
            trace_exporter_dp if trace_exporter_dp is not None else trace_exporter
        )
        self.metric_exporter = metric_exporter
        self.metric_exporter_control_plane = metric_exporter_control_plane
        self.log_exporter = log_exporter
        self._otel_emitter: OtelEmitter | None = None
        self.schema_path = schema_path
        self.service_name = service_name
        self.show_full_spans = show_full_spans
        self.scenarios_dir = scenarios_dir
        self.config_dir = config_dir
        self._default_tenant_id_override = (
            default_tenant_id.strip()
            if isinstance(default_tenant_id, str) and default_tenant_id.strip()
            else None
        )
        self.debug_validate = debug_validate
        self._include_metric_span_trace_ids = include_metric_span_trace_ids
        self._max_cached_tenants = max(1, int(max_cached_tenants))
        self._config = load_config(config_dir)
        self._resource_presets = load_resource_presets(config_dir)
        # Semconv parsing can be relatively expensive (yaml.safe_load + graph setup).
        # Cache it for the runner lifetime since it doesn't change per scenario.
        self._schema_parser: SchemaParser = SchemaParser(self.schema_path)
        self._parsed_schema: ParsedSchema | None = None
        # TracerProviders include tenant-scoped resource attributes (``--vendor``),
        # so we cache them per tenant to keep traces consistent without leaking threads.
        self._providers_by_tenant: OrderedDict[str, tuple[TracerProvider, TracerProvider]] = (
            OrderedDict()
        )
        self.scenario_loader: ScenarioLoader | None = None  # lazy init for run_mixed_workload
        # Next trace start time (nanoseconds since UNIX epoch) for spacing in OTLP/Jaeger.
        self._synthetic_next_trace_start_ns: int | None = None
        self._runner_shutdown = False
        # Tenant on metrics/logs Resource uses ``{ATTR_PREFIX}.tenant.id`` (see ``--vendor``).
        self._otel_emitter_resource_tenant: str | None = None

    def _ensure_shared_otel_emitter(self, tenant_id: str) -> OtelEmitter:
        """Lazily create the metrics/logs pipeline (one MeterProvider + LoggerProvider per runner)."""
        if self._otel_emitter is None:
            cp_attrs = self._resource_presets.get("control-plane", {}).get("attributes") or {}
            dp_attrs = self._resource_presets.get("data-plane", {}).get("attributes") or {}
            cp_resource_attrs = sim_config.resource_attributes_with_tenant(
                dict(cp_attrs), tenant_id
            )
            dp_resource_attrs = sim_config.resource_attributes_with_tenant(
                dict(dp_attrs), tenant_id
            )
            self._otel_emitter = OtelEmitter(
                metric_exporter=self.metric_exporter,
                metric_exporter_control_plane=self.metric_exporter_control_plane,
                log_exporter=self.log_exporter,
                schema_path=self.schema_path,
                dp_resource=Resource.create(dp_resource_attrs),
                cp_resource=Resource.create(cp_resource_attrs),
                include_metric_span_trace_ids=self._include_metric_span_trace_ids,
            )
            self._otel_emitter_resource_tenant = tenant_id
            return self._otel_emitter
        if self._otel_emitter_resource_tenant != tenant_id:
            warnings.warn(
                "Metrics and logs OTLP resource tenant was fixed at the first scenario in this "
                "runner; a later scenario uses a different tenant id, so Gentoro may attribute "
                "those signals to the wrong tenant. Use one tenant per runner or separate "
                "CLI invocations for mixed-tenant workloads.",
                UserWarning,
                stacklevel=2,
            )
        return self._otel_emitter

    def _advance_synthetic_trace_schedule(
        self,
        scenario: Scenario,
        trace_base_ns: int,
        *,
        interval_ms: float | None = None,
        interval_deviation_ms: float | None = None,
    ) -> None:
        """Move timeline forward by interval (ms) ± optional deviation."""
        if interval_ms is not None:
            mean = float(interval_ms)
            dev = float(interval_deviation_ms) if interval_deviation_ms is not None else 0.0
        else:
            mean = float(scenario.trace_interval_ms)
            dev = float(scenario.trace_interval_deviation_ms)
        if dev > 0:
            delta_ms = mean + random.uniform(-dev, dev)
        else:
            delta_ms = mean
        delta_ms = max(0.0, delta_ms)
        scheduled_next_ns = trace_base_ns + int(delta_ms * 1_000_000)
        now_ns = time.time_ns()
        # Keep simulated timestamps near real time. Under sustained load, work per
        # turn can exceed the configured interval and otherwise cause event-time to
        # drift behind wall clock by minutes or hours.
        self._synthetic_next_trace_start_ns = max(scheduled_next_ns, now_ns)

    def _get_providers_for_tenant(self, *, tenant_id: str) -> tuple[TracerProvider, TracerProvider]:
        """Return (cp_provider, dp_provider) for a tenant."""
        cached = self._providers_by_tenant.get(tenant_id)
        if cached:
            self._providers_by_tenant.move_to_end(tenant_id)
            return cached

        cp_attrs = self._resource_presets.get("control-plane", {}).get("attributes") or {}
        dp_attrs = self._resource_presets.get("data-plane", {}).get("attributes") or {}
        # Important: use Resource.create(...) so OpenTelemetry default resource
        # attributes are merged in (telemetry.sdk.*).
        cp_resource = Resource.create(
            sim_config.resource_attributes_with_tenant(dict(cp_attrs), tenant_id)
        )
        dp_resource = Resource.create(
            sim_config.resource_attributes_with_tenant(dict(dp_attrs), tenant_id)
        )

        provider_cp = TracerProvider(resource=cp_resource, shutdown_on_exit=False)
        provider_dp = TracerProvider(resource=dp_resource, shutdown_on_exit=False)

        bsp_kw = {
            "schedule_delay_millis": BSP_SCHEDULE_DELAY_MILLIS,
            "max_export_batch_size": BSP_MAX_EXPORT_BATCH_SIZE,
            "export_timeout_millis": BSP_EXPORT_TIMEOUT_MILLIS,
        }
        provider_cp.add_span_processor(BatchSpanProcessor(self.trace_exporter_cp, **bsp_kw))
        provider_dp.add_span_processor(BatchSpanProcessor(self.trace_exporter_dp, **bsp_kw))

        self._providers_by_tenant[tenant_id] = (provider_cp, provider_dp)
        self._providers_by_tenant.move_to_end(tenant_id)
        self._evict_oldest_provider_if_needed()
        return provider_cp, provider_dp

    def _shutdown_provider_pair(
        self, provider_cp: TracerProvider, provider_dp: TracerProvider
    ) -> None:
        """Best-effort flush+shutdown for a provider pair."""
        for provider in (provider_cp, provider_dp):
            try:
                provider.force_flush(timeout_millis=FLUSH_TIMEOUT_MILLIS)
            except Exception:
                pass
        for provider in (provider_cp, provider_dp):
            try:
                provider.shutdown()
            except Exception:
                pass

    def _evict_oldest_provider_if_needed(self) -> None:
        """Bound provider/thread growth when many tenant ids are seen in one runner."""
        while len(self._providers_by_tenant) > self._max_cached_tenants:
            _tenant_id, pair = self._providers_by_tenant.popitem(last=False)
            self._shutdown_provider_pair(*pair)

    def _get_instrumentation_scope_name(
        self, scenario: Scenario, *, enduser_id: str | None = None
    ) -> str:
        """
        OpenTelemetry instrumentation scope name (emits as `otel.scope.name`).

        Uses ``otelsim.<scenario filename stem>``; when *enduser_id* is non-empty,
        appends ``.<enduser_id>`` (YAML ``endusers[].id``).
        """
        stem = getattr(scenario, "source_filename", "") or getattr(scenario, "name", "")
        base = "otelsim" if not stem else f"otelsim.{stem}"
        eid = (enduser_id or "").strip()
        if not eid:
            return base
        return f"{base}.{eid}"

    def _scenario_is_active_now(self, scenario: Scenario) -> bool:
        """
        Return whether a scenario is currently eligible for mixed workload picking.

        Scenarios may define optional schedule metadata in YAML:
          workload_schedule:
            peak_hours:
              timezone: "Europe/Oslo"
              weekdays: [1,2,3,4,5]   # ISO weekday (Mon=1 ... Sun=7)
              start_hour: 17          # inclusive local hour [0..23]
              end_hour: 21            # exclusive local hour [0..23], wrap supported
        """
        raw = scenario.raw if isinstance(scenario.raw, dict) else {}
        schedule = raw.get("workload_schedule")
        if not isinstance(schedule, dict):
            return True
        peak = schedule.get("peak_hours")
        if not isinstance(peak, dict):
            return True

        tz_name = str(peak.get("timezone") or "UTC")
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            warnings.warn(
                f"Scenario '{scenario.name}' has invalid workload_schedule timezone '{tz_name}'; "
                "treating it as always active.",
                UserWarning,
                stacklevel=2,
            )
            return True

        now_local = datetime.now(UTC).astimezone(tz)

        weekdays_raw = peak.get("weekdays")
        if isinstance(weekdays_raw, list) and weekdays_raw:
            weekdays: list[int] = []
            for item in weekdays_raw:
                try:
                    weekdays.append(int(item))
                except (TypeError, ValueError):
                    continue
            if weekdays and now_local.isoweekday() not in weekdays:
                return False

        start_hour_raw = peak.get("start_hour")
        end_hour_raw = peak.get("end_hour")
        if start_hour_raw is None or end_hour_raw is None:
            # Missing hour bounds means no active-hour restriction.
            return True
        try:
            start_hour = int(start_hour_raw)
            end_hour = int(end_hour_raw)
        except (TypeError, ValueError):
            # If hours are missing/invalid, treat schedule as non-restrictive.
            return True

        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            warnings.warn(
                f"Scenario '{scenario.name}' has out-of-range peak_hours window "
                f"({start_hour}, {end_hour}); treating it as always active.",
                UserWarning,
                stacklevel=2,
            )
            return True

        hour = now_local.hour
        if start_hour == end_hour:
            return True
        if start_hour < end_hour:
            return start_hour <= hour < end_hour
        return hour >= start_hour or hour < end_hour

    def run_scenario(
        self,
        scenario: Scenario,
        progress_callback: Callable[..., None] | None = None,
        *,
        _continue_timeline: bool = False,
        trace_interval_ms: float | None = None,
        trace_interval_deviation_ms: float | None = None,
        include_span_names_in_progress: bool = False,
    ) -> list[str]:
        """
        Run scenario: for each repeat and each turn, compile to TraceGraphSpec, validate (if debug), render and export.
        Returns list of trace_ids (hex).

        If *progress_callback* is set, it is invoked after each turn is emitted with
        the first positional arg a 1-based completed count (``1..total``), and keyword
        args including ``trace_id``, ``tenant_id``, ``span_names``, ``scenario_name``.

        Successive traces use ``scenario.trace_interval_ms`` (± jitter) as the gap between
        **trace start** times unless *trace_interval_ms* is set (then optional
        *trace_interval_deviation_ms*; default deviation 0 when only the mean is overridden).
        With ``_continue_timeline=True`` (mixed workload), the timeline continues across
        scenario picks; otherwise it restarts at wall clock for each call.
        """
        if not _continue_timeline:
            self._synthetic_next_trace_start_ns = None

        if self.scenario_loader is None:
            self.scenario_loader = ScenarioLoader(self.scenarios_dir)
        computed_default_tenant_id = next(
            (
                v.get("id")
                for v in (self._config.get("tenants") or {}).values()
                if isinstance(v, dict) and v.get("id")
            ),
            None,
        )
        default_tenant_id = self._default_tenant_id_override or computed_default_tenant_id
        base_mcp_key = (scenario.context.mcp_server_key or "").strip() or "phone"
        resolved_base = resolve_context(
            self._config,
            tenant_key=scenario.context.tenant_key,
            agent_id=scenario.context.agent_id,
            mcp_server_key=base_mcp_key,
            default_tenant_id=default_tenant_id,
        )

        provider_cp, provider_dp = self._get_providers_for_tenant(tenant_id=resolved_base.tenant_id)

        latency_model = LatencyModel.from_scenario(scenario)
        if self._parsed_schema is None:
            self._parsed_schema = self._schema_parser.parse()
        schema = self._parsed_schema
        assert schema is not None
        if schema.single_trace_request_lifecycle is None:
            raise RuntimeError(
                "Missing lineage.single_trace_request_lifecycle in semconv; "
                "required to emit a single parent-child request lifecycle trace."
            )
        otel_emitter = self._ensure_shared_otel_emitter(resolved_base.tenant_id)
        trace_ids: list[str] = []
        total = scenario.repeat_count * sum(len(eu.turns) for eu in scenario.endusers)
        if total == 0:
            total = 1
        # 1-based count of completed compile+emit steps (avoids "0/N" and modulo-10 spam on 0).
        completed = 0
        for _ in range(scenario.repeat_count):
            for eu in scenario.endusers:
                mcp_key = (eu.mcp_server_key or "").strip() or scenario.context.mcp_server_key
                if not mcp_key:
                    mcp_key = base_mcp_key
                resolved = resolve_context(
                    self._config,
                    tenant_key=scenario.context.tenant_key,
                    agent_id=scenario.context.agent_id,
                    mcp_server_key=mcp_key,
                    default_tenant_id=default_tenant_id,
                )
                scope_name = self._get_instrumentation_scope_name(scenario, enduser_id=eu.id)
                tracer_cp = provider_cp.get_tracer(scope_name, "1.0.0")
                tracer_dp = provider_dp.get_tracer(scope_name, "1.0.0")
                shared_session_id: str | None = None
                shared_conversation_id: str | None = None
                if len(eu.turns) > 1:
                    shared = generate_ids_for_turn(self._config, tool_count=0)
                    shared_session_id = shared["session_id"]
                    shared_conversation_id = shared["conversation_id"]
                for turn in eu.turns:
                    graph = compile_turn(
                        scenario,
                        eu,
                        turn,
                        resolved_ctx=resolved,
                        latency_model=latency_model,
                        config=self._config,
                        shared_session_id=shared_session_id,
                        shared_conversation_id=shared_conversation_id,
                    )
                    if self.debug_validate:
                        validate_trace_graph(graph, attr_prefix=sim_config.ATTR_PREFIX)
                    spec = build_unified_trace_from_semconv(
                        graph.cp_request,
                        graph.data_plane,
                        graph.cp_response,
                        schema.single_trace_request_lifecycle,
                    )
                    if self._synthetic_next_trace_start_ns is None:
                        self._synthetic_next_trace_start_ns = time.time_ns()
                    trace_base_ns = self._synthetic_next_trace_start_ns
                    result = otel_emitter.emit_trace(
                        tracer_cp,
                        tracer_dp,
                        spec,
                        trace_base_time_ns=trace_base_ns,
                    )
                    self._advance_synthetic_trace_schedule(
                        scenario,
                        trace_base_ns,
                        interval_ms=trace_interval_ms,
                        interval_deviation_ms=trace_interval_deviation_ms,
                    )
                    tid = result.trace_id
                    if tid:
                        trace_ids.append(tid)
                    completed += 1
                    if progress_callback:
                        span_names = (
                            [s.name for s in spec.spans] if include_span_names_in_progress else None
                        )
                        progress_callback(
                            completed,
                            total,
                            trace_id=tid or "",
                            tenant_id=resolved.tenant_id,
                            span_names=span_names,
                            scenario_name=scenario.name,
                        )
        # Batched export: BatchSpanProcessor, PeriodicExportingMetricReader, and
        # BatchLogRecordProcessor flush at shutdown (or on their timers). Per-scenario
        # force_flush synchronizes all signals and overloads OTLP collectors under load.
        return trace_ids

    def run_mixed_workload(
        self,
        count: int | None = None,
        interval_ms: float = 500,
        progress_callback: Callable[..., None] | None = None,
        pick_complete_callback: Callable[[int, int | None, str, list[str]], None] | None = None,
        tags: list[str] | None = None,
        each_once: bool = False,
        include_span_names_in_progress: bool = False,
    ) -> tuple[list[str], dict[str, int]]:
        """
        Run mixed workload: load scenarios (optionally filtered by tags), then
        pick scenarios by weight and generate traces. Returns (trace_ids, traces_by_scenario).

        *count* ``None`` and *each_once* false: run until the caller stops the loop
        (e.g. CLI Ctrl+C). *each_once* true and *count* ``None``: one pass, ``len(scenarios)``
        picks. Otherwise *count* is the number of random (or weighted) picks.

        *pick_complete_callback*, if set, is invoked after each pick with
        ``(pick_index_1based, pick_total, scenario_name, trace_ids)``.
        *pick_total* is ``None`` when there is no fixed pick limit.

        *interval_ms* adds a pause before each pick after the first (``0`` disables).
        """
        if self.scenario_loader is None:
            self.scenario_loader = ScenarioLoader(self.scenarios_dir)
        scenarios = self.scenario_loader.load_all()
        if tags:
            scenarios = [s for s in scenarios if s.tags and any(t in s.tags for t in tags)]
        if not scenarios:
            return ([], {})
        self._synthetic_next_trace_start_ns = None
        weights = [s.context.workload_weight for s in scenarios]
        total_w = sum(weights)
        if total_w <= 0:
            total_w = 1.0
        trace_ids: list[str] = []
        traces_by_scenario: dict[str, int] = {}
        pick_indices: Iterable[int]
        if each_once:
            n_picks = count if count is not None else len(scenarios)
            pick_indices = range(n_picks)
        elif count is not None:
            n_picks = count
            pick_indices = range(n_picks)
        else:
            n_picks = None  # run until caller stops (e.g. Ctrl+C in CLI)
            pick_indices = itertools.count(0)
        keep_trace_ids = n_picks is not None

        effective_interval_ms = float(interval_ms)
        if n_picks is None and effective_interval_ms <= 0:
            # Safety valve for "run forever" mode to avoid an accidental hot loop.
            effective_interval_ms = 50.0

        pending_interval_sleep = False
        for i in pick_indices:
            if pending_interval_sleep and effective_interval_ms > 0:
                time.sleep(effective_interval_ms / 1000.0)
            pending_interval_sleep = True
            if each_once and i < len(scenarios):
                scenario = scenarios[i]
            else:
                r = random.uniform(0, total_w)
                scenario = scenarios[-1]
                for candidate, w in zip(scenarios, weights, strict=True):
                    r -= w
                    if r <= 0:
                        scenario = candidate
                        break
            pick_index = i + 1

            def _mixed_progress(
                current: int,
                total_turns: int,
                *,
                trace_id: str = "",
                tenant_id: str = "",
                span_names: list[str] | None = None,
                scenario_name: str = "",
                _scenario_name: str = scenario.name,
                _pick_index: int = pick_index,
            ) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    current,
                    total_turns,
                    trace_id=trace_id,
                    tenant_id=tenant_id,
                    span_names=span_names,
                    scenario_name=scenario_name or _scenario_name,
                    workload_pick=_pick_index,
                    workload_total=n_picks,
                )

            wrapped = _mixed_progress if progress_callback else None
            ids = self.run_scenario(
                scenario,
                progress_callback=wrapped,
                _continue_timeline=True,
                include_span_names_in_progress=include_span_names_in_progress,
            )
            if keep_trace_ids:
                trace_ids.extend(ids)
            traces_by_scenario[scenario.name] = traces_by_scenario.get(scenario.name, 0) + 1
            if pick_complete_callback is not None:
                pick_complete_callback(pick_index, n_picks, scenario.name, ids)
        return (trace_ids, traces_by_scenario)

    def shutdown(self, *, fast: bool = False) -> None:
        """Shutdown providers and exporters. *fast* uses shorter flush deadlines (e.g. Ctrl+C)."""
        if self._runner_shutdown:
            return
        self._runner_shutdown = True
        self._synthetic_next_trace_start_ns = None
        deadline = FLUSH_SHUTDOWN_FAST_MILLIS if fast else FLUSH_TIMEOUT_MILLIS
        providers = list(self._providers_by_tenant.values())
        for provider_cp, provider_dp in providers:
            try:
                provider_cp.force_flush(timeout_millis=deadline)
            except Exception:
                pass
            try:
                provider_dp.force_flush(timeout_millis=deadline)
            except Exception:
                pass
        for provider_cp, provider_dp in providers:
            try:
                provider_cp.shutdown()
            except Exception:
                pass
            try:
                provider_dp.shutdown()
            except Exception:
                pass
        self._providers_by_tenant.clear()
        if self._otel_emitter is not None:
            try:
                self._otel_emitter.force_flush(timeout_millis=deadline)
                self._otel_emitter.shutdown(timeout_millis=deadline)
            except Exception:
                pass
            self._otel_emitter = None
            self._otel_emitter_resource_tenant = None
