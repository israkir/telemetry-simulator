"""Statistical distributions and sampling for realistic telemetry generation."""

from .correlations import (
    CorrelatedSampler,
    ErrorPropagation,
    RetryConfig,
    RetrySequence,
)
from .distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    Distribution,
    DistributionFactory,
    ExponentialDistribution,
    GeometricDistribution,
    LogNormalDistribution,
    MixtureDistribution,
    NormalDistribution,
    PoissonDistribution,
    UniformDistribution,
)

__all__ = [
    "Distribution",
    "NormalDistribution",
    "LogNormalDistribution",
    "ExponentialDistribution",
    "UniformDistribution",
    "PoissonDistribution",
    "GeometricDistribution",
    "CategoricalDistribution",
    "MixtureDistribution",
    "BernoulliDistribution",
    "DistributionFactory",
    "CorrelatedSampler",
    "ErrorPropagation",
    "RetryConfig",
    "RetrySequence",
]
