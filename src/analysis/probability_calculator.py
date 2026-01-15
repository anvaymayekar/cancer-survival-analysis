# Calculates survival probabilities and related metrics

from typing import Dict, List, Optional
from src.models.survival_distribution import SurvivalDistribution
from functools import lru_cache


class ProbabilityCalculator:
    """
    Calculator for survival probabilities and related statistical measures.
    """

    def __init__(self, distribution: SurvivalDistribution) -> None:
        """
        Initialize calculator with a survival distribution.

        Args:
            distribution: Survival distribution model
        """
        self._distribution = distribution
        self._cached_survival = lru_cache(maxsize=128)(self._survival_scalar)

    def _survival_scalar(self, t: float) -> float:
        """Internal cached version for scalar time points"""
        return float(self._distribution.survival_function(t))

    def calculate_survival_at_time(self, t: float) -> float:
        """Use cached version"""
        return self._cached_survival(t)

    def calculate_survival_at_time(self, t: float) -> float:
        """
        Calculate probability of surviving beyond time t.

        Args:
            t: Time point in years

        Returns:
            Survival probability S(t)
        """
        return float(self._distribution.survival_function(t))

    def calculate_survival_at_times(self, times: List[float]) -> Dict[float, float]:
        """
        Calculate survival probabilities at multiple time points.

        Args:
            times: List of time points

        Returns:
            Dictionary mapping time to survival probability
        """
        return {t: self.calculate_survival_at_time(t) for t in times}

    def calculate_interval_probability(self, t1: float, t2: float) -> float:
        """
        Calculate probability of event occurring in interval (t1, t2].

        Args:
            t1: Start time
            t2: End time

        Returns:
            Interval probability P(t1 < T ≤ t2)
        """
        return float(self._distribution.survival_probability_interval(t1, t2))

    def calculate_key_probabilities(
        self, time_points: Optional[List[float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate survival, cumulative, and density at standard time points.

        Args:
            time_points: List of time points (default: [1, 2, 3, 5, 10, 15, 20])

        Returns:
            Nested dictionary with probabilities for each time point
        """
        if time_points is None:
            time_points = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]

        results: Dict[str, Dict[str, float]] = {}

        for t in time_points:
            results[f"{int(t)}_year"] = {
                "survival": float(self._distribution.survival_function(t)),
                "cumulative": float(self._distribution.cumulative_distribution(t)),
                "density": float(self._distribution.probability_density(t)),
                "hazard": float(self._distribution.hazard_function(t)),
            }

        return results

    def get_median_survival(self) -> float:
        """Get the median survival time."""
        return self._distribution.median()

    def get_mean_survival(self) -> float:
        """Get the mean survival time."""
        return self._distribution.mean()

    @property
    def distribution(self) -> SurvivalDistribution:
        """Get the underlying distribution."""
        return self._distribution
