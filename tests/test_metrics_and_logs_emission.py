import json
from pathlib import Path

from simulator.exporters.file_exporter import FileLogExporter, FileMetricExporter, FileSpanExporter
from simulator.scenarios.scenario_loader import ScenarioLoader
from simulator.scenarios.scenario_runner import ScenarioRunner


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_metrics_emission_writes_canonical_metric(tmp_path: Path) -> None:
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    scenario.repeat_count = 1

    metrics_path = tmp_path / "metrics.jsonl"
    logs_path = tmp_path / "logs.jsonl"
    spans_path = tmp_path / "spans.jsonl"

    runner = ScenarioRunner(
        trace_exporter=FileSpanExporter(spans_path, append=False),
        metric_exporter=FileMetricExporter(metrics_path, append=False),
        log_exporter=FileLogExporter(logs_path, append=False),
    )

    trace_ids = runner.run_scenario(scenario)
    assert trace_ids, "expected at least one generated trace_id"
    runner.shutdown()

    metrics = _read_jsonl(metrics_path)
    assert metrics, "expected at least one metrics export line"

    # At least one canonical metric should have been emitted.
    metric_names = {m.get("name") for m in metrics}
    assert "vendor.orchestration.count" in metric_names

    rv = next(m for m in metrics if m.get("name") == "vendor.request.validation.count")
    assert rv.get("resource", {}).get("vendor.module") == "control-plane"

    orch = next(m for m in metrics if m.get("name") == "vendor.orchestration.count")
    dps = orch.get("data_points") or []
    assert dps, "expected orchestration.count data points"
    attrs = dps[0].get("attributes") or {}
    for key in ("trace_id", "trace.id", "span_id", "span.id"):
        assert attrs.get(key), f"expected metric point attribute {key!r} for span correlation"


def test_logs_emission_writes_trace_and_span_ids(tmp_path: Path) -> None:
    loader = ScenarioLoader()
    scenario = loader.load("single_tool_call")
    scenario.repeat_count = 1

    metrics_path = tmp_path / "metrics.jsonl"
    logs_path = tmp_path / "logs.jsonl"
    spans_path = tmp_path / "spans.jsonl"

    runner = ScenarioRunner(
        trace_exporter=FileSpanExporter(spans_path, append=False),
        metric_exporter=FileMetricExporter(metrics_path, append=False),
        log_exporter=FileLogExporter(logs_path, append=False),
    )

    runner.run_scenario(scenario)
    runner.shutdown()

    logs = _read_jsonl(logs_path)
    assert logs, "expected at least one log export line"

    # FileLogExporter stores trace/span ids as hex strings.
    has_non_null_corr = any(log.get("trace_id") and log.get("span_id") for log in logs)
    assert has_non_null_corr, "expected at least one log record with trace_id and span_id"

