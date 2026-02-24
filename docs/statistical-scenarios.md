# Scenario Configuration Reference

This guide describes how to configure YAML scenarios for generating realistic telemetry data. Scenarios support probabilistic behavior, realistic distributions, correlated error patterns, and retry sequences.

## Overview

Scenarios are defined in YAML files. The simulator includes **sample definitions** in `src/simulator/scenarios/definitions/`; you can use those or point to a custom folder with `--scenarios-dir`. Each scenario describes:

- **Trace structure** - Hierarchy of spans (parent-child relationships). Supported span types include `{prefix}.a2a.orchestrate`, `{prefix}.planner`, `{prefix}.task.execute`, `{prefix}.llm.call`, `{prefix}.mcp.tool.execute`, `{prefix}.response.compose`, and others (see [Generating Telemetry](./generating-telemetry.md)).
- **Latency distributions** - How long each span takes
- **Error behavior** - Error rates and propagation patterns
- **Probabilistic features** - Variable span inclusion, counts, and retries
- **Attributes** - Static values or sampled from distributions (use `vendor.*` in YAML; the loader normalizes to `TELEMETRY_SIMULATOR_ATTR_PREFIX`)

```mermaid
flowchart TB
    subgraph Framework["Scenario Generation"]
        direction TB
        YAML[("Scenario YAML")] --> Loader["ScenarioLoader"]
        
        subgraph Stats["Generation Features"]
            Dist["Distributions (log-normal, poisson, etc.)"]
            Err["Error Propagation (cascade, correlate)"]
            Retry["Retry Sequences (backoff, attempts)"]
        end
        
        Loader --> Stats
        Stats --> Hierarchy["TraceHierarchy"]
        Hierarchy --> Runner["ScenarioRunner"]
        Runner --> Traces["Generated Traces"]
    end
    
    style Framework fill:#f5f5f5,stroke:#333
    style Stats fill:#e3f2fd,stroke:#1976d2
```

---

## Latency Configuration

### Basic Latency

Simple latency with mean and variance:

```yaml
latency:
  mean_ms: 200
  variance: 0.3    # ±30% variation
```

### Log-Normal Distribution (Recommended)

Real-world latencies are right-skewed: most requests are fast, but some take much longer. Log-normal captures this naturally.

```mermaid
---
config:
  xyChart:
    width: 600
    height: 300
---
xychart-beta
    title "Normal vs Log-Normal Latency Distribution"
    x-axis "Latency (ms)" [0, 100, 200, 300, 400, 500, 600, 700, 800]
    y-axis "Frequency" 0 --> 100
    line "Normal (mean=200)" [5, 25, 60, 90, 60, 25, 5, 1, 0]
    line "Log-Normal (median=200)" [10, 80, 55, 30, 15, 8, 5, 3, 2]
```

**Legend:** Blue = **Normal** (mean=200 ms). Green = **Log-Normal** (median=200 ms).

```yaml
latency:
  distribution: log_normal
  median_ms: 200      # 50th percentile
  sigma: 0.8          # Shape parameter
                      # - 0.5: tight, ~90% within 2x of median
                      # - 0.8: moderate spread
                      # - 1.2: heavy tail
```

#### Sigma Parameter Effect

```mermaid
---
config:
  xyChart:
    width: 600
    height: 250
---
xychart-beta
    title "Log-Normal with Different Sigma Values (median=200ms)"
    x-axis "Latency (ms)" [0, 100, 200, 300, 400, 500, 600, 800, 1000]
    y-axis "Frequency" 0 --> 100
    line "sigma=0.5 (tight)" [2, 70, 95, 40, 10, 3, 1, 0, 0]
    line "sigma=0.8 (moderate)" [5, 55, 70, 40, 22, 12, 7, 3, 1]
    line "sigma=1.2 (heavy tail)" [15, 45, 50, 35, 25, 18, 13, 8, 5]
```

**Legend:** Blue = **sigma 0.5** (tight). Orange = **sigma 0.8** (moderate). Green = **sigma 1.2** (heavy tail).

### Mixture Distribution (Bimodal)

Model cache hit/miss or fast path/slow path scenarios:

```mermaid
---
config:
  xyChart:
    width: 600
    height: 280
---
xychart-beta
    title "Mixture Distribution: Cache Hit (70%) vs Cache Miss (30%)"
    x-axis "Latency (ms)" [0, 50, 100, 150, 200, 300, 400, 500, 600, 700]
    y-axis "Frequency" 0 --> 80
    bar "Combined" [5, 70, 20, 5, 8, 15, 25, 30, 18, 8]
```

```yaml
latency:
  distribution: mixture
  components:
    - weight: 0.7
      distribution: normal
      mean: 50          # Cache hit (fast)
      stddev: 10
    - weight: 0.3
      distribution: log_normal
      median: 500       # Cache miss (slow)
      sigma: 0.6
```

---

## Available Distributions

| Distribution | Use Case | Parameters |
|--------------|----------|------------|
| `normal` | Symmetric variations | `mean`, `stddev` |
| `log_normal` | Latencies (right-skewed) | `median`, `sigma` |
| `exponential` | Inter-arrival times | `mean` |
| `uniform` | Random in range | `low`, `high` |
| `poisson` | Event counts | `lambda` |
| `geometric` | Trials until success | `p` |
| `categorical` | Discrete choices | `values` (dict of value→weight) |
| `mixture` | Multi-modal behavior | `components` (list of distributions) |

---

## Probabilistic Spans

Control whether spans appear in each trace using `probability`:

```mermaid
flowchart LR
    subgraph "Trace Generation (100 traces)"
        AT["{prefix}.a2a.orchestrate 100%"] --> AP["{prefix}.planner 100%"]
        AT --> RAG["rag.retrieve 30%"]
        AT --> TOOL["{prefix}.mcp.tool.execute 80%"]
        AT --> LLM["{prefix}.llm.call 95%"]
        
        RAG -->|"~30 traces"| RAG_YES[Include RAG]
        RAG -->|"~70 traces"| RAG_NO[Skip RAG]
        
        TOOL -->|"~80 traces"| TOOL_YES[Include Tool]
        TOOL -->|"~20 traces"| TOOL_NO[Skip Tool]
    end
    
    style RAG fill:#fff3e0,stroke:#ff9800
    style TOOL fill:#fff3e0,stroke:#ff9800
    style RAG_YES fill:#c8e6c9,stroke:#4caf50
    style TOOL_YES fill:#c8e6c9,stroke:#4caf50
    style RAG_NO fill:#ffcdd2,stroke:#f44336
    style TOOL_NO fill:#ffcdd2,stroke:#f44336
```

```yaml
children:
  - type: rag.retrieve
    probability: 0.3    # Only 30% of traces include RAG
    
  - type: vendor.mcp.tool.execute
    probability: 0.8    # 80% include tool call
```

---

## Count Distributions

Generate variable numbers of child spans:

```mermaid
---
config:
  xyChart:
    width: 500
    height: 250
---
xychart-beta
    title "Poisson Distribution: Tool Calls per Turn (lambda=2.5)"
    x-axis "Number of Tool Calls" [0, 1, 2, 3, 4, 5, 6, 7]
    y-axis "Probability %" 0 --> 30
    bar "Probability" [8, 20, 26, 21, 13, 7, 3, 1]
```

```yaml
children:
  - type: vendor.mcp.tool.execute
    count:
      distribution: poisson
      lambda: 2.5       # Average 2.5 tool calls per turn
      min: 1            # At least 1
      max: 5            # At most 5
```

Or simple range:

```yaml
children:
  - type: vendor.mcp.tool.execute
    count:
      min: 1
      max: 3            # Random 1-3 tool calls
```

---

## Error Configuration

### Basic Error Rate

```yaml
error:
  rate: 0.05           # 5% error rate
```

### Error Propagation

Model realistic error cascades through the trace hierarchy:

```mermaid
flowchart TB
    subgraph "Error Propagation Model"
        direction TB
        
        Parent["Parent Span, error rate: 5%"]
        
        Parent -->|"Parent OK (95%)"| ChildOK["Child Span, base rate: 3%"]
        Parent -->|"Parent ERROR (5%)"| ChildCascade["Child Span, 80% cascade + 3% base"]
        
        ChildOK -->|"3% fail"| C1Err["Child Error"]
        ChildOK -->|"97% pass"| C1OK["Child OK"]
        
        ChildCascade -->|"~83% fail"| C2Err["Child Error (cascaded)"]
        ChildCascade -->|"~17% pass"| C2OK["Child OK"]
    end
    
    style Parent fill:#fff3e0,stroke:#ff9800
    style ChildCascade fill:#ffcdd2,stroke:#f44336
    style C1Err fill:#ffcdd2,stroke:#f44336
    style C2Err fill:#ffcdd2,stroke:#f44336
    style C1OK fill:#c8e6c9,stroke:#4caf50
    style C2OK fill:#c8e6c9,stroke:#4caf50
```

```mermaid
flowchart LR
    subgraph "Sibling Cascade (30%)"
        S1["Sibling 1 ERROR"] -->|"30% cascade"| S2["Sibling 2 higher error risk"]
        S2 -->|"30% cascade"| S3["Sibling 3 even higher risk"]
    end
    
    style S1 fill:#ffcdd2,stroke:#f44336
    style S2 fill:#fff3e0,stroke:#ff9800
    style S3 fill:#fff3e0,stroke:#ff9800
```

```yaml
error:
  rate: 0.05           # Base error rate (5%)
  types:
    - timeout
    - upstream_5xx
    - rate_limit
  propagation:
    from_parent: 0.8   # 80% chance child fails if parent failed
    cascade_to_siblings: 0.3  # 30% chance sibling fails if one fails
```

### Error Types

| Error Type | Description |
|------------|-------------|
| `timeout` | Request timeout |
| `validation` | Input validation failure |
| `upstream_5xx` | Upstream service error |
| `rate_limit` | Rate limiting |
| `auth_failure` | Authentication/authorization failure |
| `not_found` | Resource not found |
| `internal_error` | Internal server error |

---

## Retry Behavior

Model operations that retry on failure:

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Service
    
    Note over C,S: Attempt 1 (forced failure)
    C->>S: Request
    S-->>C: ERROR (timeout)
    
    Note over C: Backoff: 100ms + jitter
    
    Note over C,S: Attempt 2 (75% success rate)
    C->>S: Retry
    S-->>C: ERROR (upstream_5xx)
    
    Note over C: Backoff: 200ms + jitter
    
    Note over C,S: Attempt 3 (92% success rate)
    C->>S: Retry
    S-->>C: SUCCESS
```

```mermaid
---
config:
  xyChart:
    width: 450
    height: 220
---
xychart-beta
    title "Cumulative Success Rate by Attempt"
    x-axis "Attempt" [1, 2, 3]
    y-axis "Success Rate %" 0 --> 100
    bar "Success Rate" [0, 75, 92]
```

```yaml
retry:
  enabled: true
  max_attempts: 3
  force_initial_failure: true   # First attempt always fails
  backoff_base_ms: 100
  backoff_multiplier: 2.0       # Exponential backoff
  backoff_jitter: 0.2           # ±20% jitter on backoff
  success_rate_per_attempt:
    - 0.0    # Attempt 1: 0% success (forced failure)
    - 0.75   # Attempt 2: 75% success
    - 0.92   # Attempt 3: 92% success
```

### Exponential Backoff with Jitter

```mermaid
---
config:
  xyChart:
    width: 500
    height: 220
---
xychart-beta
    title "Exponential Backoff Timing (base=100ms, multiplier=2x)"
    x-axis "Retry Attempt" [1, 2, 3, 4]
    y-axis "Backoff (ms)" 0 --> 1000
    bar "Min (with -20% jitter)" [80, 160, 320, 640]
    bar "Base" [100, 200, 400, 800]
    bar "Max (with +20% jitter)" [120, 240, 480, 960]
```

**Legend:** Blue = **Min** (−20% jitter). Orange = **Base**. Green = **Max** (+20% jitter).

### Retry Attributes

Retry scenarios automatically add these span attributes:

| Attribute | Description |
|-----------|-------------|
| `retry.attempt` | Attempt number (1, 2, 3...) |
| `retry.is_retry` | `true` if attempt > 1 |
| `error.type` | Error type if attempt failed |

---

## Attribute Configuration

### Static Attributes

```yaml
attributes:
  gen_ai.operation.name: chat
  vendor.tool.system: mcp
```

### Distributed Attributes

Generate attribute values from distributions:

```mermaid
---
config:
  xyChart:
    width: 550
    height: 230
---
xychart-beta
    title "Token Count Distribution (log-normal, median=500)"
    x-axis "Input Tokens" [0, 200, 400, 600, 800, 1000, 1500, 2000, 3000]
    y-axis "Frequency" 0 --> 50
    bar "Frequency" [5, 25, 45, 40, 28, 18, 10, 5, 2]
```

```mermaid
pie showData
    title "Model Selection (Categorical Distribution)"
    "gpt-4.1-mini (60%)" : 60
    "gpt-4.1 (25%)" : 25
    "claude-3.5-sonnet (10%)" : 10
    "claude-3.5-opus (5%)" : 5
```

```yaml
attributes:
  # Static attribute
  gen_ai.operation.name: chat
  
  # Distributed attribute (log-normal)
  gen_ai.usage.input_tokens:
    distribution: log_normal
    median: 500
    sigma: 0.8
    
  # Distributed attribute (categorical)
  gen_ai.request.model:
    distribution: categorical
    values:
      gpt-4.1-mini: 0.60
      gpt-4.1: 0.25
      claude-3.5-sonnet: 0.10
      claude-3.5-opus: 0.05
```

---

## Complete Example: Tool Retry Scenario

The `tool_retry.yaml` scenario demonstrates all features:

```mermaid
flowchart TB
    subgraph "Tool Retry Trace Structure"
        AT["{prefix}.a2a.orchestrate (root)"]
        
        AT --> AP["{prefix}.planner probability: 100%"]
        AP --> LLM1["{prefix}.llm.call (tool selection)"]
        
        AT --> TOOL["{prefix}.mcp.tool.execute with retry"]
        
        subgraph Retries["Retry Sequence"]
            R1["Attempt 1 FAIL (forced)"]
            R2["Attempt 2 75% success"]
            R3["Attempt 3 92% success"]
            R1 -->|"backoff 150ms"| R2
            R2 -->|"backoff 300ms"| R3
        end
        
        TOOL --> Retries
        
        AT --> LLM2["{prefix}.llm.call (final answer) probability: 95%"]
    end
    
    style R1 fill:#ffcdd2,stroke:#f44336
    style R2 fill:#fff3e0,stroke:#ff9800
    style R3 fill:#c8e6c9,stroke:#4caf50
    style TOOL fill:#e3f2fd,stroke:#1976d2
```

```yaml
name: tool_retry
description: >
  MCP tool call that fails on first attempt and succeeds after retry.

tags:
  - retry
  - failure
  - recovery

repeat_count: 50
interval_ms: 750

emit_metrics: true
emit_logs: true

root:
  type: vendor.a2a.orchestrate
  latency:
    distribution: log_normal
    median_ms: 2500
    sigma: 0.4
  error:
    rate: 0.05
    propagation:
      from_parent: 0.7
      cascade_to_siblings: 0.2

  children:
    - type: vendor.planner
      probability: 1.0
      latency:
        distribution: log_normal
        median_ms: 350
        sigma: 0.3
      children:
        - type: vendor.llm.call
          latency:
            distribution: log_normal
            median_ms: 450
            sigma: 0.5
          attributes:
            gen_ai.usage.input_tokens:
              distribution: log_normal
              median: 400
              sigma: 0.6

    - type: vendor.mcp.tool.execute
      latency:
        distribution: log_normal
        median_ms: 250
        sigma: 0.6
      error:
        rate: 0.8
        propagation:
          from_parent: 0.3
      retry:
        enabled: true
        max_attempts: 3
        force_initial_failure: true
        success_rate_per_attempt: [0.0, 0.75, 0.92]
      attributes:
        gen_ai.tool.name: database.query
        vendor.tool.system: mcp

    - type: vendor.llm.call
      probability: 0.95
      latency:
        distribution: log_normal
        median_ms: 650
        sigma: 0.4
```

### Running the Scenario

```bash
# Run 50 traces with retry behavior
telemetry-simulator scenario --name tool_retry --count 50

# Show full span output including retry attributes
telemetry-simulator scenario --name tool_retry --count 10 --show-full-spans
```

---

## Combining All Features

```mermaid
flowchart TB
    subgraph "Combined Scenario"
        ROOT["{prefix}.a2a.orchestrate"]
        
        ROOT --> PLAN["{prefix}.planner 100%"]
        
        ROOT --> TOOL["{prefix}.mcp.tool.execute 80%, count: poisson(1.5), latency: log_normal, retry: 2"]
        
        ROOT --> RAG["rag.retrieve probability: 30%"]
        
        ROOT --> LLM["{prefix}.llm.call 95%, tokens: log_normal, model: categorical"]
        
        TOOL --> T1["Tool 1"]
        TOOL --> T2["Tool 2?"]
        TOOL --> T3["Tool 3?"]
    end
    
    style TOOL fill:#e3f2fd,stroke:#1976d2
    style RAG fill:#fff3e0,stroke:#ff9800
    style LLM fill:#e8f5e9,stroke:#4caf50
```

```yaml
children:
  - type: vendor.mcp.tool.execute
    probability: 0.8              # 80% of traces
    count:
      distribution: poisson
      lambda: 1.5                 # Avg 1.5 calls
    latency:
      distribution: log_normal
      median_ms: 200
      sigma: 0.5
    error:
      rate: 0.1
      propagation:
        from_parent: 0.6
    retry:
      enabled: true
      max_attempts: 2
    attributes:
      gen_ai.tool.name:
        distribution: categorical
        values:
          database.query: 0.5
          http.request: 0.3
          file.read: 0.2
```

---

## Architecture

### Module Structure

```mermaid
flowchart LR
    subgraph statistics["src/simulator/statistics/"]
        dist["distributions.py"]
        corr["correlations.py"]
    end
    
    subgraph scenarios["src/simulator/scenarios/"]
        loader["scenario_loader.py"]
        runner["scenario_runner.py"]
        defs["sample definitions/*.yaml\n(or --scenarios-dir)"]
    end

    defs --> loader
    loader --> dist
    loader --> corr
    runner --> loader
    
    style statistics fill:#e3f2fd,stroke:#1976d2
    style scenarios fill:#e8f5e9,stroke:#4caf50
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `Distribution` | Base class for all distributions |
| `DistributionFactory` | Creates distributions from YAML config |
| `ErrorPropagation` | Models correlated errors |
| `RetryConfig` | Retry behavior settings |
| `RetrySequence` | Generates multi-attempt sequences |
| `ScenarioStep` | Span config with all generation features |
| `ScenarioLoader` | Parses YAML into scenarios |

---

## Best Practices

1. **Use log-normal for latencies** - More realistic than normal/Gaussian
2. **Set realistic sigma values** - 0.5-1.0 for moderate spread
3. **Model error propagation** - Real systems have correlated failures
4. **Start simple** - Add distributions and features incrementally
5. **Test with small counts first** - Verify behavior before large runs
6. **Use tags** - Organize scenarios by purpose (baseline, failure, load-test)

---

## See Also

- [Generating Telemetry](./generating-telemetry.md) - Quick start and examples
- [README](../README.md) - CLI reference and pipeline integration
- [successful_agent_turn.yaml](../src/simulator/scenarios/definitions/successful_agent_turn.yaml) - Sample baseline scenario
- [tool_retry.yaml](../src/simulator/scenarios/definitions/tool_retry.yaml) - Sample retry scenario
- Use `--scenarios-dir` to load scenarios from a custom folder
