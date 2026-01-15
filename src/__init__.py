"""
Cancer Survival Analysis Package

This package provides tools for analyzing cancer patient survival times
using probability distributions, with emphasis on the exponential distribution.
"""

__version__ = "1.0.0"
__author__ = "Anvay Mayekar"
__email__ = "anvay.mayekar24@sakec.ac.in"

from src.models.survival_distribution import (
    SurvivalDistribution,
    ExponentialDistribution,
)
from src.simulation.data_simulator import SurvivalDataSimulator
from src.analysis.probability_calculator import ProbabilityCalculator
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.visualization.survival_visualizer import SurvivalVisualizer
from src.utils.types import (
    SimulationConfig,
    CohortData,
    AnalysisResults,
    StatisticsDict,
    ValidationResult,
    ConfidenceInterval,
)

__all__ = [
    # Distribution models
    "SurvivalDistribution",
    "ExponentialDistribution",
    # Simulation
    "SurvivalDataSimulator",
    # Analysis
    "ProbabilityCalculator",
    "StatisticalAnalyzer",
    # Visualization
    "SurvivalVisualizer",
    # Types
    "SimulationConfig",
    "CohortData",
    "AnalysisResults",
    "StatisticsDict",
    "ValidationResult",
    "ConfidenceInterval",
]
