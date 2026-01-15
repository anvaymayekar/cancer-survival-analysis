# Abstract base class and concrete implementations of distributions

from abc import ABC, abstractmethod
from typing import Union, Dict
from functools import lru_cache
import numpy as np
from src.utils.types import FloatArray
from scipy.special import gamma


class SurvivalDistribution(ABC):
    """
    Abstract base class for survival time probability distributions.
    Defines the interface that all concrete distributions must implement.
    """

    def __init__(self, params: Dict[str, float]) -> None:
        """
        Initialize the distribution with parameters.

        Args:
            params: Dictionary of distribution parameters
        """
        self._params = params
        self._validate_parameters()

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate that parameters are appropriate for the distribution."""
        pass

    @abstractmethod
    def survival_function(
        self, t: Union[float, FloatArray]
    ) -> Union[float, FloatArray]:
        """
        Calculate S(t) = P(T > t).

        Args:
            t: Time point(s)

        Returns:
            Survival probability at time t
        """
        pass

    @abstractmethod
    def probability_density(
        self, t: Union[float, FloatArray]
    ) -> Union[float, FloatArray]:
        """
        Calculate f(t) - probability density function.

        Args:
            t: Time point(s)

        Returns:
            Probability density at time t
        """
        pass

    @abstractmethod
    def hazard_function(self, t: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """
        Calculate h(t) - hazard rate function.

        Args:
            t: Time point(s)

        Returns:
            Hazard rate at time t
        """
        pass

    @abstractmethod
    def mean(self) -> float:
        """Calculate the mean of the distribution."""
        pass

    @abstractmethod
    def median(self) -> float:
        """Calculate the median of the distribution."""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Calculate the variance of the distribution."""
        pass

    def cumulative_distribution(
        self, t: Union[float, FloatArray]
    ) -> Union[float, FloatArray]:
        """
        Calculate F(t) = P(T ≤ t) = 1 - S(t).

        Args:
            t: Time point(s)

        Returns:
            Cumulative probability at time t
        """
        return 1.0 - self.survival_function(t)

    def survival_probability_interval(self, t1: float, t2: float) -> float:
        """
        Calculate P(t1 < T ≤ t2) = S(t1) - S(t2).

        Args:
            t1: Start time
            t2: End time

        Returns:
            Probability of event in interval

        Raises:
            ValueError: If t1 >= t2
        """
        if t1 >= t2:
            raise ValueError(f"t1 ({t1}) must be less than t2 ({t2})")
        return self.survival_function(t1) - self.survival_function(t2)

    @property
    def parameters(self) -> Dict[str, float]:
        """Get distribution parameters."""
        return self._params.copy()


class ExponentialDistribution(SurvivalDistribution):
    """
    Exponential distribution for survival analysis.
    Characterized by constant hazard rate (memoryless property).

    Note: lru_cache is applied to scalar methods (mean, median, variance)
    but NOT to survival_function/probability_density because:
    1. They accept arrays (unhashable, can't be cached)
    2. numpy.exp() is already highly optimized
    3. Caching arrays would consume excessive memory
    """

    def __init__(self, lambda_rate: float) -> None:
        """
        Initialize exponential distribution.

        Args:
            lambda_rate: Rate parameter (λ > 0)
        """
        # Set _lambda before calling super().__init__
        self._lambda = lambda_rate
        super().__init__({"lambda": lambda_rate})

    def _validate_parameters(self) -> None:
        """Validate that lambda is positive."""
        if self._lambda <= 0:
            raise ValueError(f"Lambda must be positive, got {self._lambda}")

    def survival_function(
        self, t: Union[float, FloatArray]
    ) -> Union[float, FloatArray]:
        """
        S(t) = exp(-λt)

        Note: Not cached because:
        - Accepts arrays (unhashable)
        - np.exp() is already highly optimized
        - Would cache large arrays unnecessarily
        """
        return np.exp(-self._lambda * t)

    def probability_density(
        self, t: Union[float, FloatArray]
    ) -> Union[float, FloatArray]:
        """
        f(t) = λ * exp(-λt)

        Note: Not cached for same reasons as survival_function
        """
        return self._lambda * np.exp(-self._lambda * t)

    def hazard_function(self, t: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """
        h(t) = λ (constant hazard)

        Note: Not cached because return type depends on input type
        """
        if isinstance(t, (int, float)):
            return self._lambda
        return np.full_like(t, self._lambda, dtype=np.float64)

    @lru_cache(maxsize=1)
    def mean(self) -> float:
        """
        E[T] = 1/λ

        Cached because:
        - Pure function of λ (deterministic)
        - Called frequently in analysis
        - Returns single float (hashable)
        """
        return 1.0 / self._lambda

    @lru_cache(maxsize=1)
    def median(self) -> float:
        """
        Median = ln(2)/λ

        Cached for same reasons as mean()
        """
        return np.log(2.0) / self._lambda

    @lru_cache(maxsize=1)
    def variance(self) -> float:
        """
        Var(T) = 1/λ²

        Cached for same reasons as mean()
        """
        return 1.0 / (self._lambda**2)

    def conditional_survival(self, t: float, s: float) -> float:
        """
        P(T > t+s | T > s) = S(t) (memoryless property).

        Args:
            t: Additional time
            s: Already survived time

        Returns:
            Conditional survival probability
        """
        if s < 0 or t < 0:
            raise ValueError("Time values must be non-negative")
        return self.survival_function(t)

    @property
    def lambda_rate(self) -> float:
        """Get the rate parameter."""
        return self._lambda


# Example of how to add Weibull with caching
class WeibullDistribution(SurvivalDistribution):
    """
    Weibull distribution for survival analysis.
    Flexible hazard rate (increasing, decreasing, or constant).
    """

    def __init__(self, shape: float, scale: float) -> None:
        """
        Initialize Weibull distribution.

        Args:
            shape: Shape parameter k > 0 (affects hazard shape)
            scale: Scale parameter λ > 0 (characteristic life)
        """
        self._shape = shape
        self._scale = scale
        super().__init__({"shape": shape, "scale": scale})

    def _validate_parameters(self) -> None:
        """Validate that shape and scale are positive."""
        if self._shape <= 0:
            raise ValueError(f"Shape must be positive, got {self._shape}")
        if self._scale <= 0:
            raise ValueError(f"Scale must be positive, got {self._scale}")

    def survival_function(
        self, t: Union[float, FloatArray]
    ) -> Union[float, FloatArray]:
        """S(t) = exp(-(t/λ)^k)"""
        return np.exp(-((t / self._scale) ** self._shape))

    def probability_density(
        self, t: Union[float, FloatArray]
    ) -> Union[float, FloatArray]:
        """f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)"""
        return (
            (self._shape / self._scale)
            * ((t / self._scale) ** (self._shape - 1))
            * np.exp(-((t / self._scale) ** self._shape))
        )

    def hazard_function(self, t: Union[float, FloatArray]) -> Union[float, FloatArray]:
        """h(t) = (k/λ) * (t/λ)^(k-1)"""
        return (self._shape / self._scale) * ((t / self._scale) ** (self._shape - 1))

    @lru_cache(maxsize=1)
    def mean(self) -> float:
        """E[T] = λ * Γ(1 + 1/k)"""

        return self._scale * gamma(1.0 + 1.0 / self._shape)

    @lru_cache(maxsize=1)
    def median(self) -> float:
        """Median = λ * (ln(2))^(1/k)"""
        return self._scale * (np.log(2.0) ** (1.0 / self._shape))

    @lru_cache(maxsize=1)
    def variance(self) -> float:
        """Var(T) = λ² * [Γ(1 + 2/k) - Γ²(1 + 1/k)]"""

        return self._scale**2 * (
            gamma(1.0 + 2.0 / self._shape) - gamma(1.0 + 1.0 / self._shape) ** 2
        )

    @property
    def shape_parameter(self) -> float:
        """Get the shape parameter."""
        return self._shape

    @property
    def scale_parameter(self) -> float:
        """Get the scale parameter."""
        return self._scale
