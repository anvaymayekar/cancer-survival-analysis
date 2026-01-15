# Statistical validation and hypothesis testing

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from src.models.survival_distribution import SurvivalDistribution
from src.utils.types import ValidationResult, ConfidenceInterval, FloatArray


class StatisticalAnalyzer:
    """
    Performs comprehensive statistical analysis and validation of survival data.
    """

    def __init__(
        self, survival_data: FloatArray, theoretical_distribution: SurvivalDistribution
    ) -> None:
        """
        Initialize analyzer with data and theoretical distribution.

        Args:
            survival_data: Observed survival times
            theoretical_distribution: Expected distribution model
        """
        self._data = survival_data
        self._distribution = theoretical_distribution
        self._n_samples = len(survival_data)

    def goodness_of_fit_ks_test(self) -> ValidationResult:
        """
        Perform Kolmogorov-Smirnov goodness-of-fit test.

        Returns:
            ValidationResult with test statistics and conclusion
        """
        # Extract parameters for scipy
        mean_survival = self._distribution.mean()

        # Perform K-S test
        ks_statistic, p_value = stats.kstest(
            self._data, "expon", args=(0, mean_survival)
        )

        alpha = 0.05
        conclusion = (
            "Fail to reject H₀ (Good fit)"
            if p_value > alpha
            else "Reject H₀ (Poor fit)"
        )
        interpretation = (
            "Data follows exponential distribution"
            if p_value > alpha
            else "Data does not follow exponential distribution"
        )

        result: ValidationResult = {
            "test_name": "Kolmogorov-Smirnov Test",
            "statistic": float(ks_statistic),
            "p_value": float(p_value),
            "critical_value": None,
            "conclusion": conclusion,
            "interpretation": interpretation,
        }

        return result

    def goodness_of_fit_anderson_darling(self) -> ValidationResult:
        """
        Perform Anderson-Darling goodness-of-fit test.

        Returns:
            ValidationResult with test statistics and conclusion
        """
        result_obj = stats.anderson(self._data, dist="expon")

        # Find critical value for 5% significance level
        significance_levels = result_obj.significance_level
        critical_values = result_obj.critical_values

        idx_5_percent = np.where(significance_levels == 5.0)[0]
        critical_5 = (
            critical_values[idx_5_percent[0]]
            if len(idx_5_percent) > 0
            else critical_values[2]
        )

        conclusion = (
            "Fail to reject H₀ (Good fit)"
            if result_obj.statistic < critical_5
            else "Reject H₀ (Poor fit)"
        )

        result: ValidationResult = {
            "test_name": "Anderson-Darling Test",
            "statistic": float(result_obj.statistic),
            "p_value": None,
            "critical_value": float(critical_5),
            "conclusion": conclusion,
            "interpretation": f"Test statistic: {result_obj.statistic:.6f}, Critical value (5%): {critical_5:.6f}",
        }

        return result

    def bootstrap_confidence_interval(
        self, confidence_level: float = 0.95, n_bootstrap: int = 10000
    ) -> ConfidenceInterval:
        """
        Calculate bootstrap confidence interval for mean survival time.

        Args:
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples

        Returns:
            ConfidenceInterval with bounds and diagnostics
        """
        rng = np.random.default_rng(42)
        bootstrap_means: List[float] = []

        for _ in range(n_bootstrap):
            bootstrap_sample = rng.choice(
                self._data, size=self._n_samples, replace=True
            )
            bootstrap_means.append(float(np.mean(bootstrap_sample)))

        bootstrap_array = np.array(bootstrap_means)
        alpha = 1.0 - confidence_level

        lower_percentile = (alpha / 2.0) * 100.0
        upper_percentile = (1.0 - alpha / 2.0) * 100.0

        ci_lower = float(np.percentile(bootstrap_array, lower_percentile))
        ci_upper = float(np.percentile(bootstrap_array, upper_percentile))
        sample_mean = float(np.mean(self._data))
        theoretical_mean = self._distribution.mean()

        result: ConfidenceInterval = {
            "confidence_level": confidence_level,
            "sample_mean": sample_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "margin_of_error": (ci_upper - ci_lower) / 2.0,
            "theoretical_mean": theoretical_mean,
            "contains_theoretical": ci_lower <= theoretical_mean <= ci_upper,
        }

        return result

    def compare_theoretical_empirical(self) -> Dict[str, Dict[str, float]]:
        """
        Compare theoretical distribution parameters with empirical estimates.

        Returns:
            Dictionary with comparison metrics for mean, std_dev, and median
        """
        empirical_mean = float(np.mean(self._data))
        empirical_std = float(np.std(self._data, ddof=1))
        empirical_median = float(np.median(self._data))

        theoretical_mean = self._distribution.mean()
        theoretical_std = np.sqrt(self._distribution.variance())
        theoretical_median = self._distribution.median()

        def calculate_error(empirical: float, theoretical: float) -> float:
            return abs(empirical - theoretical) / theoretical * 100.0

        comparison: Dict[str, Dict[str, float]] = {
            "mean": {
                "theoretical": theoretical_mean,
                "empirical": empirical_mean,
                "error_percent": calculate_error(empirical_mean, theoretical_mean),
            },
            "std_dev": {
                "theoretical": theoretical_std,
                "empirical": empirical_std,
                "error_percent": calculate_error(empirical_std, theoretical_std),
            },
            "median": {
                "theoretical": theoretical_median,
                "empirical": empirical_median,
                "error_percent": calculate_error(empirical_median, theoretical_median),
            },
        }

        return comparison

    def validate_survival_probabilities(
        self, time_points: Optional[List[float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare theoretical and empirical survival probabilities.

        Args:
            time_points: Time points to validate (default: [1, 3, 5, 10, 15])

        Returns:
            Dictionary with comparison at each time point
        """
        if time_points is None:
            time_points = [1.0, 3.0, 5.0, 10.0, 15.0]

        results: Dict[str, Dict[str, float]] = {}

        for t in time_points:
            theoretical_prob = float(self._distribution.survival_function(t))
            empirical_prob = float(np.sum(self._data > t) / self._n_samples)
            difference = abs(theoretical_prob - empirical_prob)
            error_percent = (
                (difference / theoretical_prob) * 100.0 if theoretical_prob > 0 else 0.0
            )

            results[f"t_{int(t)}"] = {
                "time": t,
                "theoretical": theoretical_prob,
                "empirical": empirical_prob,
                "difference": difference,
                "error_percent": error_percent,
            }

        return results

    @property
    def data(self) -> FloatArray:
        """Get the survival data."""
        return self._data.copy()

    @property
    def sample_size(self) -> int:
        """Get the sample size."""
        return self._n_samples
