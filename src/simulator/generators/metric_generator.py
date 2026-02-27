"""
Generate canonical metrics aligned with OTEL semantic conventions schema.

Emits metrics directly via OTLP Meter following the emission_policy:
- Metrics are application-emitted, not derived from spans
- Uses same attribute names as spans for correlation
"""

import random
import time

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from ..config import attr as config_attr
from ..config import resource_attributes as config_resource_attributes
from ..config import resource_schema_url
from ..defaults import get_default_tenant_ids
from ..schemas.attribute_generator import AttributeGenerator, GenerationContext
from ..schemas.schema_parser import SchemaParser


class MetricGenerator:
    """Generate canonical metrics following OTEL semantic conventions schema."""

    def __init__(
        self,
        exporter: MetricExporter,
        schema_path: str | None = None,
        service_name: str = "otelsim",
        export_interval_ms: int = 5000,
    ):
        """Initialize metric generator with exporter."""
        parser = SchemaParser(schema_path)
        self.schema = parser.parse()
        self.attr_generator = AttributeGenerator(self.schema)

        tenant_id = get_default_tenant_ids()[0]
        attrs = dict(config_resource_attributes(tenant_id))
        attrs["service.name"] = service_name
        attrs["service.version"] = "1.0.0"
        resource = Resource.create(attrs, schema_url=resource_schema_url())

        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=export_interval_ms,
        )

        self.provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(self.provider)

        self.meter = metrics.get_meter(__name__)
        self._setup_instruments()

    def _setup_instruments(self):
        """Create metric instruments from schema."""
        self.turn_count = self.meter.create_counter(
            config_attr("turn.count"),
            description="Count of agent turns",
            unit="1",
        )

        self.turn_duration = self.meter.create_histogram(
            config_attr("turn.duration_ms"),
            description="Turn duration distribution",
            unit="ms",
        )

        self.tool_count = self.meter.create_counter(
            config_attr("tool.count"),
            description="Count of MCP tool calls",
            unit="1",
        )

        self.tool_latency = self.meter.create_histogram(
            config_attr("tool.latency_ms"),
            description="Tool call latency distribution",
            unit="ms",
        )

        self.rag_count = self.meter.create_counter(
            config_attr("rag.count"),
            description="Count of RAG retrievals",
            unit="1",
        )

        self.rag_latency = self.meter.create_histogram(
            config_attr("rag.latency_ms"),
            description="RAG retrieval latency distribution",
            unit="ms",
        )

        self.rag_docs_returned = self.meter.create_histogram(
            config_attr("rag.documents_returned"),
            description="Distribution of documents returned",
            unit="1",
        )

        self.llm_count = self.meter.create_counter(
            config_attr("llm.count"),
            description="Count of LLM inference calls",
            unit="1",
        )

        self.llm_latency = self.meter.create_histogram(
            config_attr("llm.latency_ms"),
            description="LLM inference latency distribution",
            unit="ms",
        )

        self.llm_input_tokens = self.meter.create_counter(
            config_attr("llm.tokens.input"),
            description="Total input tokens consumed",
            unit="1",
        )

        self.llm_output_tokens = self.meter.create_counter(
            config_attr("llm.tokens.output"),
            description="Total output tokens generated",
            unit="1",
        )

        self.a2a_count = self.meter.create_counter(
            config_attr("a2a.count"),
            description="Count of A2A calls",
            unit="1",
        )

        self.a2a_latency = self.meter.create_histogram(
            config_attr("a2a.latency_ms"),
            description="A2A call latency distribution",
            unit="ms",
        )

        self.cp_request_count = self.meter.create_counter(
            config_attr("cp.request.count"),
            description="Incoming requests at control-plane by status code",
            unit="1",
        )

        self.cp_request_duration = self.meter.create_histogram(
            config_attr("cp.request.duration_ms"),
            description="Control-plane request duration",
            unit="ms",
        )

    def record_turn(
        self,
        context: GenerationContext,
        duration_ms: float,
        status_code: str = "SUCCESS",
        route: str | None = None,
    ):
        """Record metrics for an agent turn (deployment.environment.name is resource-level only)."""
        attrs = {
            "tenant.id": context.tenant_id,
            config_attr("route"): route or context.route or "default",
            config_attr("turn.status.code"): status_code,
        }

        self.turn_count.add(1, attrs)
        self.turn_duration.record(duration_ms, attrs)

    def record_tool_call(
        self,
        context: GenerationContext,
        tool_name: str,
        server_name: str,
        latency_ms: float,
        status_code: str = "OK",
    ):
        """Record metrics for an MCP tool call."""
        attrs = {
            "tenant.id": context.tenant_id,
            "gen_ai.tool.name": tool_name,
            config_attr("tool.server.name"): server_name,
            config_attr("tool.status.code"): status_code,
        }

        self.tool_count.add(1, attrs)
        self.tool_latency.record(latency_ms, attrs)

    def record_rag_retrieval(
        self,
        context: GenerationContext,
        index_name: str,
        latency_ms: float,
        docs_returned: int,
        status_code: str = "OK",
    ):
        """Record metrics for a RAG retrieval."""
        attrs = {
            "tenant.id": context.tenant_id,
            "rag.index.name": index_name,
            config_attr("rag.status.code"): status_code,
        }

        self.rag_count.add(1, attrs)
        self.rag_latency.record(latency_ms, attrs)
        self.rag_docs_returned.record(
            docs_returned, {"tenant.id": context.tenant_id, "rag.index.name": index_name}
        )

    def record_llm_inference(
        self,
        context: GenerationContext,
        provider: str,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record metrics for an LLM inference call."""
        attrs = {
            "tenant.id": context.tenant_id,
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
        }

        self.llm_count.add(1, attrs)
        self.llm_latency.record(latency_ms, attrs)
        self.llm_input_tokens.add(input_tokens, attrs)
        self.llm_output_tokens.add(output_tokens, attrs)

    def record_a2a_call(
        self,
        context: GenerationContext,
        target_agent: str,
        latency_ms: float,
        status_code: str = "OK",
    ):
        """Record metrics for an A2A call."""
        attrs = {
            "tenant.id": context.tenant_id,
            "a2a.target.agent": target_agent,
            config_attr("a2a.status.code"): status_code,
        }

        self.a2a_count.add(1, attrs)
        self.a2a_latency.record(latency_ms, attrs)

    def record_cp_request(
        self,
        context: GenerationContext,
        duration_ms: float,
        status_code: str = "ALLOWED",
    ):
        """Record metrics for a control-plane request (deployment.environment.name is resource-level only)."""
        attrs = {
            "tenant.id": context.tenant_id,
            config_attr("cp.status.code"): status_code,
        }

        self.cp_request_count.add(1, attrs)
        self.cp_request_duration.record(duration_ms, attrs)

    def generate_sample_metrics(
        self,
        count: int = 100,
        tenant_ids: list[str] | None = None,
        interval_ms: float = 100,
    ):
        """Generate a batch of sample metrics."""
        tenants = tenant_ids or get_default_tenant_ids()

        llm_configs = [
            ("openai", "gpt-4.1-mini"),
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-opus"),
            ("anthropic", "claude-3-sonnet"),
        ]

        mcp_servers = [
            ("atlassian.jiraGetIssue", "atlassian-mcp.prod"),
            ("slack.slack_send_message", "slack-mcp.prod"),
            ("github.github_get_repository", "github-mcp.prod"),
        ]

        rag_indices = ["policy_kb_v7", "claims_docs_v2", "legal_docs_v3"]
        a2a_agents = ["risk_agent", "claims_agent", "document_agent"]

        for i in range(count):
            context = GenerationContext.create(
                tenant_id=random.choice(tenants),
                turn_index=i % 10,
            )

            self.record_turn(
                context,
                duration_ms=max(100, random.gauss(1500, 400)),
                status_code=random.choices(
                    ["SUCCESS", "USER_ERROR", "SYSTEM_ERROR"],
                    weights=[0.95, 0.03, 0.02],
                )[0],
            )

            tool_name, server_name = random.choice(mcp_servers)
            self.record_tool_call(
                context,
                tool_name=tool_name,
                server_name=server_name,
                latency_ms=max(10, random.gauss(200, 80)),
                status_code=random.choices(["OK", "ERROR", "TIMEOUT"], weights=[0.95, 0.03, 0.02])[
                    0
                ],
            )

            if random.random() < 0.4:
                self.record_rag_retrieval(
                    context,
                    index_name=random.choice(rag_indices),
                    latency_ms=max(10, random.gauss(100, 40)),
                    docs_returned=random.randint(1, 10),
                )

            provider, model = random.choice(llm_configs)
            self.record_llm_inference(
                context,
                provider=provider,
                model=model,
                latency_ms=max(50, random.gauss(500, 150)),
                input_tokens=random.randint(100, 2000),
                output_tokens=random.randint(50, 1000),
            )

            if random.random() < 0.2:
                self.record_a2a_call(
                    context,
                    target_agent=random.choice(a2a_agents),
                    latency_ms=max(50, random.gauss(300, 100)),
                )

            self.record_cp_request(
                context,
                duration_ms=max(5, random.gauss(30, 10)),
                status_code=random.choices(
                    ["ALLOWED", "BLOCKED", "FLAGGED"],
                    weights=[0.95, 0.03, 0.02],
                )[0],
            )

            if interval_ms > 0 and i < count - 1:
                time.sleep(interval_ms / 1000.0)

    def shutdown(self):
        """Shutdown the meter provider."""
        self.provider.shutdown()
