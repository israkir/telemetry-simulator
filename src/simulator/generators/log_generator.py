"""
Generate correlated log records aligned with OTEL semantic conventions schema.

Emits logs directly via OTLP Logger following the emission_policy:
- Logs are application-emitted, not derived from spans
- Includes trace_id/span_id for correlation
- Uses same attribute names as spans for consistency
"""

import logging
import random
import time
from typing import Any

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.resources import Resource

from ..config import ATTR_PREFIX, resource_schema_url
from ..config import attr as config_attr
from ..config import resource_attributes as config_resource_attributes
from ..config import span_name as config_span_name
from ..defaults import get_default_tenant_ids
from ..schemas.attribute_generator import AttributeGenerator, GenerationContext
from ..schemas.schema_parser import SchemaParser


class LogGenerator:
    """Generate correlated log records following OTEL semantic conventions schema."""

    def __init__(
        self,
        exporter: LogRecordExporter,
        schema_path: str | None = None,
        service_name: str = "otelsim",
    ):
        """Initialize log generator with exporter."""
        parser = SchemaParser(schema_path)
        self.schema = parser.parse()
        self.attr_generator = AttributeGenerator(self.schema)

        tenant_id = get_default_tenant_ids()[0]
        attrs = dict(config_resource_attributes(tenant_id))
        attrs["service.name"] = service_name
        attrs["service.version"] = "1.0.0"
        resource = Resource.create(attrs, schema_url=resource_schema_url())

        self.provider = LoggerProvider(resource=resource)
        self.provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        set_logger_provider(self.provider)

        handler = LoggingHandler(
            level=logging.DEBUG,
            logger_provider=self.provider,
        )

        self.logger = logging.getLogger(f"{ATTR_PREFIX}.telemetry")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    def _get_base_attrs(self, context: GenerationContext) -> dict[str, Any]:
        """Get base attributes for all log records (deployment.environment.name is resource-level only)."""
        return {
            "tenant.id": context.tenant_id,
            config_attr("session.id"): context.session_id,
            config_attr("request.id"): context.request_id,
            config_attr("turn.index"): context.turn_index,
        }

    def log_turn_start(self, context: GenerationContext, route: str | None = None):
        """Log start of an agent turn."""
        attrs = self._get_base_attrs(context)
        attrs[config_attr("route")] = route or context.route or "default"
        attrs["event.name"] = "turn.start"

        self.logger.info(
            "Agent turn started",
            extra=attrs,
        )

    def log_turn_end(
        self,
        context: GenerationContext,
        status_code: str = "SUCCESS",
        duration_ms: float = 0,
    ):
        """Log end of an agent turn."""
        attrs = self._get_base_attrs(context)
        attrs[config_attr("turn.status.code")] = status_code
        attrs[config_attr("turn.status.result")] = status_code == "SUCCESS"
        attrs["duration_ms"] = duration_ms
        attrs["event.name"] = "turn.end"

        level = logging.INFO if status_code == "SUCCESS" else logging.WARNING
        self.logger.log(
            level,
            f"Agent turn completed with status {status_code}",
            extra=attrs,
        )

    def log_tool_call(
        self,
        context: GenerationContext,
        tool_name: str,
        server_name: str,
        status_code: str = "OK",
        latency_ms: float = 0,
        error_message: str | None = None,
    ):
        """Log an MCP tool call."""
        attrs = self._get_base_attrs(context)
        attrs["gen_ai.tool.name"] = tool_name
        attrs[config_attr("tool.server.name")] = server_name
        attrs[config_attr("tool.status.code")] = status_code
        attrs[config_attr("tool.status.result")] = status_code == "OK"
        attrs["tool.latency_ms"] = latency_ms
        attrs["event.name"] = "tool.call"

        if status_code == "OK":
            self.logger.info(
                f"Tool call to {tool_name} completed",
                extra=attrs,
            )
        else:
            attrs["error.message"] = error_message or f"Tool call failed: {status_code}"
            self.logger.error(
                f"Tool call to {tool_name} failed: {status_code}",
                extra=attrs,
            )

    def log_llm_inference(
        self,
        context: GenerationContext,
        provider: str,
        model: str,
        operation: str = "chat",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0,
    ):
        """Log an LLM inference call."""
        attrs = self._get_base_attrs(context)
        attrs["gen_ai.system"] = provider
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.operation.name"] = operation
        attrs["gen_ai.usage.input_tokens"] = input_tokens
        attrs["gen_ai.usage.output_tokens"] = output_tokens
        attrs["duration_ms"] = latency_ms
        attrs["event.name"] = config_span_name("llm.call")

        self.logger.info(
            f"LLM inference completed: {provider}/{model}",
            extra=attrs,
        )

    def log_rag_retrieval(
        self,
        context: GenerationContext,
        index_name: str,
        docs_returned: int,
        status_code: str = "OK",
        latency_ms: float = 0,
    ):
        """Log a RAG retrieval operation."""
        attrs = self._get_base_attrs(context)
        attrs["rag.index.name"] = index_name
        attrs["rag.documents.returned"] = docs_returned
        attrs[config_attr("rag.status.code")] = status_code
        attrs["rag.latency_ms"] = latency_ms
        attrs["event.name"] = "rag.retrieve"

        self.logger.info(
            f"RAG retrieval from {index_name}: {docs_returned} documents",
            extra=attrs,
        )

    def log_a2a_call(
        self,
        context: GenerationContext,
        target_agent: str,
        operation: str,
        status_code: str = "OK",
        latency_ms: float = 0,
    ):
        """Log an A2A call."""
        attrs = self._get_base_attrs(context)
        attrs["a2a.target.agent"] = target_agent
        attrs["a2a.operation"] = operation
        attrs[config_attr("a2a.status.code")] = status_code
        attrs["a2a.latency_ms"] = latency_ms
        attrs["event.name"] = "a2a.call"

        level = logging.INFO if status_code == "OK" else logging.WARNING
        self.logger.log(
            level,
            f"A2A call to {target_agent} ({operation}): {status_code}",
            extra=attrs,
        )

    def log_cp_request(
        self,
        context: GenerationContext,
        status_code: str = "ALLOWED",
        reason: str | None = None,
    ):
        """Log a control-plane request decision."""
        attrs = self._get_base_attrs(context)
        attrs[config_attr("cp.status.code")] = status_code
        attrs[config_attr("cp.status.result")] = status_code != "BLOCKED"
        attrs["event.name"] = "cp.request"

        if reason:
            attrs[config_attr("cp.status.metadata")] = f'{{"reason":"{reason}"}}'

        if status_code == "ALLOWED":
            self.logger.info("Control-plane allowed request", extra=attrs)
        elif status_code == "FLAGGED":
            self.logger.warning(
                f"Control-plane flagged request: {reason or 'anomaly detected'}",
                extra=attrs,
            )
        else:
            self.logger.error(
                f"Control-plane blocked request: {reason or 'policy violation'}",
                extra=attrs,
            )

    def log_safety_event(
        self,
        context: GenerationContext,
        event_type: str,
        severity: str = "medium",
        details: dict[str, Any] | None = None,
    ):
        """Log a safety-related event."""
        attrs = self._get_base_attrs(context)
        attrs["safety.event.type"] = event_type
        attrs["safety.severity"] = severity
        attrs["event.name"] = "safety.event"

        if details:
            for k, v in details.items():
                attrs[f"safety.{k}"] = v

        level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(severity, logging.WARNING)

        self.logger.log(
            level,
            f"Safety event: {event_type} (severity: {severity})",
            extra=attrs,
        )

    def generate_sample_logs(
        self,
        count: int = 100,
        tenant_ids: list[str] | None = None,
        interval_ms: float = 100,
    ):
        """Generate a batch of sample log records."""
        tenants = tenant_ids or get_default_tenant_ids()

        llm_configs = [
            ("openai", "gpt-4.1-mini"),
            ("anthropic", "claude-3-opus"),
        ]

        mcp_servers = [
            ("atlassian.jiraGetIssue", "atlassian-mcp.prod"),
            ("slack.slack_send_message", "slack-mcp.prod"),
        ]

        for i in range(count):
            context = GenerationContext.create(
                tenant_id=random.choice(tenants),
                turn_index=i % 10,
            )

            self.log_turn_start(context)

            if random.random() < 0.1:
                self.log_cp_request(
                    context,
                    status_code=random.choice(["ALLOWED", "FLAGGED", "BLOCKED"]),
                    reason="policy_check" if random.random() < 0.5 else None,
                )

            tool_name, server_name = random.choice(mcp_servers)
            tool_status = random.choices(["OK", "ERROR", "TIMEOUT"], weights=[0.9, 0.05, 0.05])[0]
            self.log_tool_call(
                context,
                tool_name=tool_name,
                server_name=server_name,
                status_code=tool_status,
                latency_ms=random.gauss(200, 80),
                error_message="Connection timeout" if tool_status == "TIMEOUT" else None,
            )

            if random.random() < 0.4:
                self.log_rag_retrieval(
                    context,
                    index_name=random.choice(["policy_kb_v7", "claims_docs_v2"]),
                    docs_returned=random.randint(1, 10),
                    latency_ms=random.gauss(100, 40),
                )

            provider, model = random.choice(llm_configs)
            self.log_llm_inference(
                context,
                provider=provider,
                model=model,
                input_tokens=random.randint(100, 2000),
                output_tokens=random.randint(50, 1000),
                latency_ms=random.gauss(500, 150),
            )

            if random.random() < 0.05:
                self.log_safety_event(
                    context,
                    event_type=random.choice(
                        ["pii_detected", "prompt_injection", "content_policy"]
                    ),
                    severity=random.choice(["low", "medium", "high"]),
                )

            turn_status = random.choices(
                ["SUCCESS", "USER_ERROR", "SYSTEM_ERROR"],
                weights=[0.9, 0.05, 0.05],
            )[0]
            self.log_turn_end(
                context,
                status_code=turn_status,
                duration_ms=random.gauss(1500, 400),
            )

            if interval_ms > 0 and i < count - 1:
                time.sleep(interval_ms / 1000.0)

    def shutdown(self):
        """Shutdown the logger provider."""
        self.provider.shutdown()
