"""Shared OpenTelemetry export timing (batch + flush boundaries)."""

# Upper bound for force_flush on MeterProvider, TracerProvider, LoggerProvider during
# normal shutdown. Slow collectors still complete or time out within this window.
FLUSH_TIMEOUT_MILLIS: int = 5_000

# After Ctrl+C: minimize blocking in shutdown while still attempting one flush cycle.
FLUSH_SHUTDOWN_FAST_MILLIS: int = 1_000

# BatchSpanProcessor: cap per-export work; aligns with OTEL BSP env guidance.
BSP_SCHEDULE_DELAY_MILLIS: int = 2_000
BSP_MAX_EXPORT_BATCH_SIZE: int = 512
BSP_EXPORT_TIMEOUT_MILLIS: int = 5_000
