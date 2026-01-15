# Custom type definitions and type aliases

from typing import TypedDict, List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float64]
TimeArray = FloatArray
ProbabilityArray = FloatArray


class DistributionType(Enum):
    """Enumeration of supported probability distributions"""

    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"
    LOG_NORMAL = "log_normal"


class StatisticsDict(TypedDict):
    """Type definition for statistics dictionary"""

    count: int
    mean: float
    median: float
    std_dev: float
    variance: float
    min: float
    max: float
    q25: float
    q50: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float


class ValidationResult(TypedDict):
    """Type definition for validation test results"""

    test_name: str
    statistic: float
    p_value: Optional[float]
    critical_value: Optional[float]
    conclusion: str
    interpretation: str


class ConfidenceInterval(TypedDict):
    """Type definition for confidence interval results"""

    confidence_level: float
    sample_mean: float
    ci_lower: float
    ci_upper: float
    margin_of_error: float
    theoretical_mean: float
    contains_theoretical: bool


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for survival data simulation"""

    mean_survival_years: float
    n_patients: int
    random_seed: int = 42
    distribution_type: DistributionType = DistributionType.EXPONENTIAL

    def __post_init__(self) -> None:
        if self.mean_survival_years <= 0:
            raise ValueError("Mean survival must be positive")
        if self.n_patients <= 0:
            raise ValueError("Number of patients must be positive")


@dataclass(frozen=True)
class CohortData:
    """Data structure for patient cohort information"""

    name: str
    survival_times: FloatArray
    mean_survival: float
    lambda_rate: float
    description: str
    sample_size: int


@dataclass
class AnalysisResults:
    """Container for complete analysis results"""

    simulation_config: SimulationConfig
    survival_data: FloatArray
    statistics: StatisticsDict
    validation_results: Dict[str, ValidationResult]
    confidence_interval: ConfidenceInterval
    key_probabilities: Dict[str, float]
