# Simulates survival data from probability distributions

from typing import List, Tuple
import numpy as np
from scipy import stats
from src.models.survival_distribution import (
    SurvivalDistribution,
    ExponentialDistribution,
)
from src.utils.types import (
    SimulationConfig,
    CohortData,
    StatisticsDict,
    FloatArray,
    DistributionType,
)


class SurvivalDataSimulator:
    """
    Simulator for generating synthetic cancer patient survival times.
    Uses probability distributions to create realistic survival data.
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize the simulator.

        Args:
            config: Simulation configuration parameters
        """
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._distribution = self._create_distribution()

    def _create_distribution(self) -> SurvivalDistribution:
        """Create the appropriate distribution based on config."""
        if self._config.distribution_type == DistributionType.EXPONENTIAL:
            lambda_rate = 1.0 / self._config.mean_survival_years
            return ExponentialDistribution(lambda_rate)
        else:
            raise NotImplementedError(
                f"Distribution {self._config.distribution_type} not implemented"
            )

    def generate_survival_times(self) -> FloatArray:
        """
        Generate survival times for the configured number of patients.

        Returns:
            Array of survival times in years
        """
        survival_times = self._rng.exponential(
            scale=self._config.mean_survival_years, size=self._config.n_patients
        )
        return survival_times.astype(np.float64)

    def generate_multiple_cohorts(
        self, n_cohorts: int, mean_range: Tuple[float, float] = (3.0, 7.0)
    ) -> List[CohortData]:
        """
        Generate multiple patient cohorts with different survival characteristics.

        Args:
            n_cohorts: Number of cohorts to generate
            mean_range: Range of mean survival times (min, max)

        Returns:
            List of CohortData objects
        """
        cohorts: List[CohortData] = []
        mean_values = np.linspace(mean_range[0], mean_range[1], n_cohorts)

        for i, mean_val in enumerate(mean_values):
            # Create temporary config for this cohort
            cohort_config = SimulationConfig(
                mean_survival_years=mean_val,
                n_patients=self._config.n_patients,
                random_seed=self._config.random_seed + i,
            )

            # Generate data
            temp_simulator = SurvivalDataSimulator(cohort_config)
            survival_times = temp_simulator.generate_survival_times()

            # Create cohort data object
            cohort = CohortData(
                name=f"Cohort_{i+1}",
                survival_times=survival_times,
                mean_survival=mean_val,
                lambda_rate=1.0 / mean_val,
                description=f"Mean Survival: {mean_val:.1f} years",
                sample_size=len(survival_times),
            )
            cohorts.append(cohort)

        return cohorts

    def calculate_statistics(self, survival_times: FloatArray) -> StatisticsDict:
        """
        Calculate comprehensive descriptive statistics.

        Args:
            survival_times: Array of survival times

        Returns:
            Dictionary containing statistical measures
        """
        statistics: StatisticsDict = {
            "count": len(survival_times),
            "mean": float(np.mean(survival_times)),
            "median": float(np.median(survival_times)),
            "std_dev": float(np.std(survival_times, ddof=1)),
            "variance": float(np.var(survival_times, ddof=1)),
            "min": float(np.min(survival_times)),
            "max": float(np.max(survival_times)),
            "q25": float(np.percentile(survival_times, 25)),
            "q50": float(np.percentile(survival_times, 50)),
            "q75": float(np.percentile(survival_times, 75)),
            "iqr": float(
                np.percentile(survival_times, 75) - np.percentile(survival_times, 25)
            ),
            "skewness": float(stats.skew(survival_times)),
            "kurtosis": float(stats.kurtosis(survival_times)),
        }
        return statistics

    @property
    def distribution(self) -> SurvivalDistribution:
        """Get the underlying distribution."""
        return self._distribution

    @property
    def config(self) -> SimulationConfig:
        """Get the simulation configuration."""
        return self._config
