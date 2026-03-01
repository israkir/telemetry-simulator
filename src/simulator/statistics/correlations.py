"""
Correlation models for error propagation and dependent sampling.

Models how errors cascade through trace hierarchies and how related
attributes (e.g., input/output tokens) correlate with each other.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .distributions import CategoricalDistribution


class ErrorType(Enum):
    """Error types aligned with conventions/semconv.yaml (error.type allowed_values)."""

    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_ERROR = "tool_error"
    PROTOCOL_ERROR = "protocol_error"


@dataclass
class RetryConfig:
    """Configuration for retry behavior modeling."""

    max_attempts: int = 3
    backoff_base_ms: float = 100.0
    backoff_multiplier: float = 2.0
    backoff_jitter: float = 0.2
    success_rate_per_attempt: list[float] = field(default_factory=lambda: [0.0, 0.7, 0.85, 0.95])
    retryable_errors: list[ErrorType] = field(
        default_factory=lambda: [ErrorType.TIMEOUT, ErrorType.UNAVAILABLE]
    )

    def get_backoff_ms(self, attempt: int) -> float:
        """Calculate backoff delay for given attempt number (1-indexed)."""
        base = self.backoff_base_ms * (self.backoff_multiplier ** (attempt - 1))
        jitter = random.uniform(-self.backoff_jitter, self.backoff_jitter)
        return base * (1 + jitter)

    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """Determine if a retry should be attempted."""
        if attempt >= self.max_attempts:
            return False
        return error_type in self.retryable_errors

    def retry_succeeds(self, attempt: int) -> bool:
        """Determine if retry attempt succeeds based on cumulative success rate."""
        if attempt < len(self.success_rate_per_attempt):
            rate = self.success_rate_per_attempt[attempt]
        else:
            rate = self.success_rate_per_attempt[-1]
        return random.random() < rate


@dataclass
class ErrorPropagation:
    """
    Models how errors propagate through span hierarchies.

    Error correlation is important for telemetry:
    - If a child span fails, the parent often fails too
    - Cascading failures affect sibling spans
    - Recovery patterns show decreasing error rates on retries
    """

    base_rate: float = 0.02
    propagation_from_parent: float = 0.8
    cascade_to_siblings: float = 0.3
    error_type_distribution: CategoricalDistribution | None = None

    def __post_init__(self):
        if self.error_type_distribution is None:
            self.error_type_distribution = CategoricalDistribution(
                categories=[e.value for e in ErrorType],
                weights=[0.25, 0.25, 0.2, 0.2, 0.1],
            )

    def should_error(self, parent_errored: bool = False, sibling_errored: bool = False) -> bool:
        """
        Determine if a span should have an error, considering parent/sibling state.

        Args:
            parent_errored: Whether the parent span has an error
            sibling_errored: Whether any sibling span has an error

        Returns:
            True if this span should be marked as errored
        """
        if parent_errored and random.random() < self.propagation_from_parent:
            return True

        if sibling_errored and random.random() < self.cascade_to_siblings:
            return True

        return random.random() < self.base_rate

    def sample_error_type(self) -> str:
        """Sample an error type based on configured distribution (SemConv-aligned)."""
        if self.error_type_distribution is None:
            return ErrorType.UNAVAILABLE.value
        return str(self.error_type_distribution.sample())

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ErrorPropagation":
        """Create ErrorPropagation from YAML config."""
        error_types = config.get("types", [e.value for e in ErrorType])
        type_weights = config.get("type_weights")

        if type_weights is None:
            type_weights = [1.0] * len(error_types)

        propagation = config.get("propagation", {})

        return cls(
            base_rate=config.get("rate", config.get("base_rate", 0.02)),
            propagation_from_parent=propagation.get("from_parent", 0.8),
            cascade_to_siblings=propagation.get("cascade_to_siblings", 0.3),
            error_type_distribution=CategoricalDistribution(
                categories=error_types,
                weights=type_weights,
            ),
        )


@dataclass
class CorrelatedSampler:
    """
    Sample correlated values for attributes that depend on each other.

    Examples:
    - output_tokens correlates with input_tokens (longer prompts → longer responses)
    - latency correlates with token count (more tokens → higher latency)
    - error rate may correlate with latency (timeouts more likely for slow requests)
    """

    correlation: float = 0.0
    base_sampler: Any = None
    dependent_sampler: Any = None

    def sample_pair(self) -> tuple[float, float]:
        """
        Sample two correlated values.

        Uses Cholesky decomposition for generating correlated normal samples,
        then transforms if needed.
        """
        if self.base_sampler is None or self.dependent_sampler is None:
            raise ValueError("Both samplers must be configured")

        base_val = self.base_sampler.sample()

        if abs(self.correlation) < 0.01:
            dep_val = self.dependent_sampler.sample()
        else:
            z1 = random.gauss(0, 1)
            z2 = random.gauss(0, 1)
            correlated_z = self.correlation * z1 + (1 - self.correlation**2) ** 0.5 * z2

            dep_mean = getattr(self.dependent_sampler, "mean", 0)
            dep_std = getattr(self.dependent_sampler, "stddev", 1)
            dep_val = dep_mean + dep_std * correlated_z

        return base_val, dep_val


@dataclass
class RetrySequence:
    """
    Generate a complete retry sequence with timing and outcomes.

    Models the full lifecycle of a retriable operation:
    1. Initial attempt (may fail)
    2. Backoff delay
    3. Retry attempt(s)
    4. Final success or failure
    """

    config: RetryConfig = field(default_factory=RetryConfig)
    error_propagation: ErrorPropagation = field(default_factory=ErrorPropagation)

    def generate(
        self,
        base_latency_ms: float,
        force_initial_failure: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Generate a sequence of attempts for a retriable operation.

        Args:
            base_latency_ms: Base latency for each attempt
            force_initial_failure: If True, first attempt always fails

        Returns:
            List of attempt records with:
            - attempt: Attempt number (1-indexed)
            - latency_ms: Duration of this attempt
            - success: Whether this attempt succeeded
            - error_type: Error type if failed (None if success)
            - backoff_ms: Backoff before next attempt (0 for last)
            - cumulative_ms: Total time including all backoffs
        """
        attempts = []
        cumulative_ms = 0.0

        for attempt_num in range(1, self.config.max_attempts + 1):
            latency_variation = random.uniform(0.8, 1.2)
            attempt_latency = base_latency_ms * latency_variation

            if attempt_num == 1:
                if force_initial_failure:
                    success = False
                else:
                    success = not self.error_propagation.should_error()
            else:
                success = self.config.retry_succeeds(attempt_num)

            error_type = None
            if not success:
                error_type = self.error_propagation.sample_error_type()

            cumulative_ms += attempt_latency

            backoff_ms = 0.0
            if not success and attempt_num < self.config.max_attempts:
                error_enum = ErrorType(error_type) if error_type else ErrorType.UNAVAILABLE
                if self.config.should_retry(error_enum, attempt_num):
                    backoff_ms = self.config.get_backoff_ms(attempt_num)
                    cumulative_ms += backoff_ms

            attempts.append(
                {
                    "attempt": attempt_num,
                    "latency_ms": attempt_latency,
                    "success": success,
                    "error_type": error_type,
                    "backoff_ms": backoff_ms,
                    "cumulative_ms": cumulative_ms,
                }
            )

            if success:
                break

            if error_type:
                error_enum = ErrorType(error_type)
                if not self.config.should_retry(error_enum, attempt_num):
                    break

        return attempts

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RetrySequence":
        """Create RetrySequence from YAML config."""
        retry_config = config.get("retry", {})
        error_config = config.get("error", {})

        return cls(
            config=RetryConfig(
                max_attempts=retry_config.get("max_attempts", 3),
                backoff_base_ms=retry_config.get("backoff_base_ms", 100.0),
                backoff_multiplier=retry_config.get("backoff_multiplier", 2.0),
                backoff_jitter=retry_config.get("backoff_jitter", 0.2),
                success_rate_per_attempt=retry_config.get(
                    "success_rate_per_attempt", [0.0, 0.7, 0.85, 0.95]
                ),
                retryable_errors=(
                    [
                        ErrorType(e)
                        for e in retry_config.get("retryable_errors", ["timeout", "unavailable"])
                        if e in [x.value for x in ErrorType]
                    ]
                    or [ErrorType.TIMEOUT, ErrorType.UNAVAILABLE]
                ),
            ),
            error_propagation=ErrorPropagation.from_config(error_config),
        )
