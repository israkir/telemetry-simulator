# Live Trace Visualization

View generated traces in a browser using Jaeger while the simulator runs on your host.

## Prerequisites

- Docker or Podman (Makefile prefers Podman if available)
- Simulator running locally (e.g. `otelsim run --semconv /path/to/conventions.yaml`)

## Steps

1. **Start Jaeger** (OTLP receiver + UI):

   ```bash
   make jaeger-up
   ```

   This starts Jaeger with OTLP on ports 4317 (gRPC) and 4318 (HTTP), and the UI on port 16686.

2. **Run the simulator**:

   ```bash
   otelsim run --semconv /path/to/semconv.yaml
   ```

3. **Open the Jaeger UI**: [http://localhost:16686](http://localhost:16686)

   - Select service **otelsim**
   - Click **Find Traces** to see generated traces

4. **Stop Jaeger when done**:

   ```bash
   make jaeger-down
   ```

## Environment

- **OTLP endpoint**: The simulator uses `http://localhost:4318` by default (use `--endpoint` to change; the Makefile uses `OTLP_ENDPOINT` when set).
- **Schema and tenant**: Set `SEMCONV` as in the [README](../README.md). Tenant ID comes from `config/config.yaml`.

## See Also

- [README](../README.md) – Quick start and CLI reference
- [Generating Telemetry](./generating-telemetry.md) – Scenario usage
