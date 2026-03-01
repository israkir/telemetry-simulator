"""
Statistical distributions for telemetry data generation.

Provides various probability distributions that model real-world latency,
token counts, retry attempts, and other telemetry characteristics.
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class Distribution(ABC):
    """Base class for statistical distributions."""

    @abstractmethod
    def sample(self) -> float:
        """Draw a single sample from the distribution."""
        pass

    def sample_int(self) -> int:
        """Draw a sample and round to integer."""
        return int(round(self.sample()))

    def sample_bounded(self, min_val: float | None = None, max_val: float | None = None) -> float:
        """Draw a sample with optional bounds."""
        value = self.sample()
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
        return value


@dataclass
class NormalDistribution(Distribution):
    """
    Normal (Gaussian) distribution.

    Good for: symmetric variations around a mean (temperature, calibrated sensors).
    """

    mean: float = 0.0
    stddev: float = 1.0

    def sample(self) -> float:
        return random.gauss(self.mean, self.stddev)


@dataclass
class LogNormalDistribution(Distribution):
    """
    Log-normal distribution - better fit for latencies than normal.

    Real-world latencies are typically right-skewed: most requests are fast,
    but some take much longer. Log-normal captures this naturally.

    Parameters:
        median: The median value (50th percentile)
        sigma: Shape parameter controlling the spread/skew
               - sigma=0.5: tight distribution, ~90% within 2x of median
               - sigma=1.0: moderate spread, long tail
               - sigma=1.5: heavy tail, occasional very large values
    """

    median: float = 100.0
    sigma: float = 0.8

    def sample(self) -> float:
        mu = math.log(self.median)
        return random.lognormvariate(mu, self.sigma)


@dataclass
class ExponentialDistribution(Distribution):
    """
    Exponential distribution - models time between events.

    Good for: inter-arrival times, time-to-failure, request gaps.
    Memoryless property: P(X > s+t | X > s) = P(X > t)
    """

    mean: float = 1.0

    def sample(self) -> float:
        return random.expovariate(1.0 / self.mean)


@dataclass
class UniformDistribution(Distribution):
    """Uniform distribution over [low, high]."""

    low: float = 0.0
    high: float = 1.0

    def sample(self) -> float:
        return random.uniform(self.low, self.high)


@dataclass
class PoissonDistribution(Distribution):
    """
    Poisson distribution - models count of events in fixed interval.

    Good for: number of tool calls per turn, number of retries, request counts.

    Note: Uses normal approximation for lambda > 30 for efficiency.
    """

    lambda_: float = 1.0

    def sample(self) -> float:
        if self.lambda_ > 30:
            return max(0, random.gauss(self.lambda_, math.sqrt(self.lambda_)))
        return self._poisson_small()

    def _poisson_small(self) -> int:
        L = math.exp(-self.lambda_)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1


@dataclass
class GeometricDistribution(Distribution):
    """
    Geometric distribution - models number of trials until first success.

    Good for: number of retries before success, session length in turns.
    """

    p: float = 0.5  # Success probability

    def sample(self) -> float:
        if self.p <= 0 or self.p > 1:
            raise ValueError("p must be in (0, 1]")
        return math.ceil(math.log(1 - random.random()) / math.log(1 - self.p))


@dataclass
class CategoricalDistribution(Distribution):
    """
    Categorical distribution - discrete choices with weights.

    Good for: model selection, error type selection, status codes.

    Example:
        dist = CategoricalDistribution(
            categories=["gpt-4", "gpt-3.5", "claude"],
            weights=[0.5, 0.3, 0.2]
        )
    """

    categories: list[Any] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.weights and self.categories:
            self.weights = [1.0] * len(self.categories)
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def sample(self) -> Any:
        if not self.categories:
            raise ValueError("CategoricalDistribution requires at least one category")
        return random.choices(self.categories, weights=self.weights)[0]

    def sample_index(self) -> int:
        """Return index of selected category."""
        if not self.categories:
            raise ValueError("CategoricalDistribution requires at least one category")
        return random.choices(range(len(self.categories)), weights=self.weights)[0]


@dataclass
class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution - single binary outcome.

    Good for: error occurrence, feature flags, A/B testing.
    """

    p: float = 0.5

    def sample(self) -> float:
        return 1.0 if random.random() < self.p else 0.0

    def sample_bool(self) -> bool:
        """Return boolean outcome."""
        return random.random() < self.p


@dataclass
class MixtureDistribution(Distribution):
    """
    Mixture of distributions - models multi-modal behavior.

    Good for: cache hit/miss latencies, different code paths,
    bimodal response times (fast path vs slow path).

    Example:
        # 70% cache hits (fast), 30% cache misses (slow)
        dist = MixtureDistribution(
            components=[
                NormalDistribution(mean=10, stddev=2),   # Cache hit
                LogNormalDistribution(median=200, sigma=0.5)  # Cache miss
            ],
            weights=[0.7, 0.3]
        )
    """

    components: list[Distribution] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.weights and self.components:
            self.weights = [1.0] * len(self.components)
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def sample(self) -> float:
        if not self.components:
            raise ValueError("MixtureDistribution requires at least one component")
        idx = random.choices(range(len(self.components)), weights=self.weights)[0]
        return self.components[idx].sample()


class DistributionFactory:
    """Factory for creating distributions from configuration dictionaries."""

    @classmethod
    def create(cls, config: dict[str, Any]) -> Distribution:
        """
        Create a distribution from a configuration dictionary.

        Examples:
            # Normal distribution
            {"distribution": "normal", "mean": 100, "stddev": 20}

            # Log-normal distribution
            {"distribution": "log_normal", "median": 200, "sigma": 0.8}

            # Categorical distribution
            {"distribution": "categorical", "values": {"gpt-4": 0.5, "gpt-3.5": 0.3}}

            # Mixture distribution
            {
                "distribution": "mixture",
                "components": [
                    {"weight": 0.7, "distribution": "normal", "mean": 50},
                    {"weight": 0.3, "distribution": "log_normal", "median": 500}
                ]
            }
        """
        dist_type = config.get("distribution", "normal").lower().replace("-", "_")

        if dist_type == "normal":
            return NormalDistribution(
                mean=config.get("mean", config.get("mean_ms", 100.0)),
                stddev=config.get(
                    "stddev", config.get("mean", 100.0) * config.get("variance", 0.3)
                ),
            )

        if dist_type == "log_normal":
            return LogNormalDistribution(
                median=config.get("median", config.get("median_ms", 100.0)),
                sigma=config.get("sigma", 0.8),
            )

        if dist_type == "exponential":
            return ExponentialDistribution(
                mean=config.get("mean", config.get("mean_ms", 100.0)),
            )

        if dist_type == "uniform":
            return UniformDistribution(
                low=config.get("low", config.get("min", 0.0)),
                high=config.get("high", config.get("max", 1.0)),
            )

        if dist_type == "poisson":
            return PoissonDistribution(
                lambda_=config.get("lambda", config.get("mean", 1.0)),
            )

        if dist_type == "geometric":
            return GeometricDistribution(
                p=config.get("p", 0.5),
            )

        if dist_type == "categorical":
            values = config.get("values", {})
            if isinstance(values, dict):
                return CategoricalDistribution(
                    categories=list(values.keys()),
                    weights=list(values.values()),
                )
            return CategoricalDistribution(
                categories=values,
                weights=config.get("weights", []),
            )

        if dist_type == "bernoulli":
            return BernoulliDistribution(
                p=config.get("p", config.get("probability", 0.5)),
            )

        if dist_type == "mixture":
            components = []
            weights = []
            for comp_config in config.get("components", []):
                weight = comp_config.pop("weight", 1.0)
                weights.append(weight)
                components.append(cls.create(comp_config))
            return MixtureDistribution(components=components, weights=weights)

        raise ValueError(f"Unknown distribution type: {dist_type}")

    @classmethod
    def create_latency(cls, config: dict[str, Any]) -> Distribution:
        """
        Create a latency distribution with sensible defaults.

        If config contains "distribution", delegates to create(); otherwise
        builds a log-normal from mean_ms and variance.
        """
        if "distribution" in config:
            return cls.create(config)

        mean_ms = config.get("mean_ms", config.get("mean", 100.0))
        variance = config.get("variance", 0.3)

        return LogNormalDistribution(
            median=mean_ms,
            sigma=variance,
        )
