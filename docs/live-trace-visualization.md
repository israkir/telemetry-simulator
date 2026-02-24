# Live Trace Visualization

View generated traces in a browser using Jaeger while the simulator runs on your host.

## Prerequisites

- Docker or Podman (Makefile prefers Podman if available)
- Simulator running locally (e.g. `make run` or `make run-scenario`)

## Steps

1. **Start Jaeger** (OTLP receiver + UI):

   ```bash
   make jaeger-up
   ```

   This starts Jaeger with OTLP on ports 4317 (gRPC) and 4318 (HTTP), and the UI on port 16686.

2. **Run the simulator**:

   ```bash
   make run
   # or a specific scenario:
   SCENARIO=successful_agent_turn make run-scenario
   ```

3. **Open the Jaeger UI**: [http://localhost:16686](http://localhost:16686)

   - Select service **telemetry-simulator**
   - Click **Find Traces** to see generated traces

4. **Stop Jaeger when done**:

   ```bash
   make jaeger-down
   ```

## Environment

- **OTLP endpoint**: The simulator uses `http://localhost:4318` by default (set `OTLP_ENDPOINT` or `--endpoint` to change).
- **Schema and tenant**: Ensure `TELEMETRY_SIMULATOR_SCHEMA_PATH` (or `SCHEMA_PATH`) and `TENANT_UUID` are set as described in the [README](../README.md).

## See Also

- [README](../README.md) – Quick start and CLI reference
- [Generating Telemetry](./generating-telemetry.md) – Scenario usage
