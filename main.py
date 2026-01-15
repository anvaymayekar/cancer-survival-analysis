# entry point

from typing import Dict, Any, Optional
import sys
from pathlib import Path
from src.analysis.probability_calculator import ProbabilityCalculator
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.visualization.survival_visualizer import SurvivalVisualizer
from src.simulation.data_simulator import SurvivalDataSimulator
from src.models.survival_distribution import ExponentialDistribution
import numpy as np
import matplotlib.pyplot as plt
from src.utils.types import (
    SimulationConfig,
    AnalysisResults,
    StatisticsDict,
    FloatArray,
)
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import datetime


class CancerSurvivalAnalysis:
    """
    Main orchestrator class for complete cancer survival analysis.
    Coordinates all components and manages the analysis workflow.
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize analysis with configuration.

        Args:
            config: Simulation configuration
        """
        self._config = config
        self._simulator: Optional[SurvivalDataSimulator] = None
        self._survival_data: Optional[FloatArray] = None
        self._statistics: Optional[StatisticsDict] = None
        self._calculator: Optional[ProbabilityCalculator] = None
        self._analyzer: Optional[StatisticalAnalyzer] = None
        self._visualizer = SurvivalVisualizer()
        self._results: Optional[AnalysisResults] = None

    def run_complete_analysis(self) -> AnalysisResults:
        """
        Execute the complete analysis pipeline.

        Returns:
            AnalysisResults object with all computed metrics
        """
        print(self._format_header("CANCER SURVIVAL ANALYSIS - COMPLETE IMPLEMENTATION"))
        print("Author: Anvay Mayekar (SYECS1-26)")
        print("Course: Mathematical Techniques (ECCOR1PC201)\n")

        # Step 1: Simulation
        self._run_simulation()

        # Step 2: Probability Calculations
        self._calculate_probabilities()

        # Step 3: Statistical Analysis
        self._perform_statistical_analysis()

        # Step 4: Generate Visualizations
        self._generate_visualizations()

        # Step 5: Create Results Object
        self._compile_results()

        # Step 6: Display Summary
        self._display_summary()

        return self._results

    def _run_simulation(self) -> None:
        """Execute simulation step."""
        print(self._format_header("STEP 1: DATA SIMULATION"))

        print(f"Simulation Parameters:")
        print(f"  - Mean Survival Time: {self._config.mean_survival_years} years")
        print(f"  - Rate Parameter (λ): {1/self._config.mean_survival_years:.4f}")
        print(f"  - Number of Patients: {self._config.n_patients:,}")
        print(f"  - Random Seed: {self._config.random_seed}")

        self._simulator = SurvivalDataSimulator(self._config)
        self._survival_data = self._simulator.generate_survival_times()
        self._statistics = self._simulator.calculate_statistics(self._survival_data)

        print(f"\n✓ Successfully generated {len(self._survival_data):,} survival times")

        self._print_statistics()

    def _calculate_probabilities(self) -> None:
        """Calculate survival probabilities."""
        print(self._format_header("STEP 2: PROBABILITY CALCULATIONS"))

        distribution = self._simulator.distribution
        self._calculator = ProbabilityCalculator(distribution)

        # Key survival probabilities
        print(self._format_subsection("Key Survival Probabilities"))
        time_points = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]

        print(f"{'Time (years)':<15} {'S(t)':<12} {'F(t)':<12} {'Percentage'}")
        print("-" * 55)

        for t in time_points:
            s_t = self._calculator.calculate_survival_at_time(t)
            f_t = 1.0 - s_t
            print(f"{t:<15.0f} {s_t:<12.6f} {f_t:<12.6f} {s_t*100:>6.2f}%")

        # Interval probabilities
        print(self._format_subsection("Interval Probabilities"))
        intervals = [(0, 3), (3, 5), (5, 10), (10, 15), (15, 20)]

        print(f"{'Interval (years)':<20} {'Probability':<15} {'Percentage'}")
        print("-" * 50)

        for t1, t2 in intervals:
            prob = self._calculator.calculate_interval_probability(t1, t2)
            print(
                f"({t1:>2.0f}, {t2:>2.0f}]"
                + " " * 12
                + f"{prob:<15.6f} {prob*100:>6.2f}%"
            )

        # Median survival
        print(self._format_subsection("Median Survival Time"))
        median = self._calculator.get_median_survival()
        print(f"  Theoretical Median: {median:.4f} years")
        print(f"  Interpretation: 50% of patients survive beyond {median:.2f} years")

    def _perform_statistical_analysis(self) -> None:
        """Perform statistical validation."""
        print(self._format_header("STEP 3: STATISTICAL ANALYSIS & VALIDATION"))

        distribution = self._simulator.distribution
        self._analyzer = StatisticalAnalyzer(self._survival_data, distribution)

        # K-S Test
        print(self._format_subsection("Kolmogorov-Smirnov Test"))
        ks_result = self._analyzer.goodness_of_fit_ks_test()
        print(f"  Test Statistic: {ks_result['statistic']:.6f}")
        print(f"  P-value:        {ks_result['p_value']:.6f}")
        print(f"  Conclusion:     {ks_result['conclusion']}")
        print(f"  Interpretation: {ks_result['interpretation']}")

        # Anderson-Darling Test
        print(self._format_subsection("Anderson-Darling Test"))
        ad_result = self._analyzer.goodness_of_fit_anderson_darling()
        print(f"  Test Statistic:    {ad_result['statistic']:.6f}")
        print(f"  Critical Value:    {ad_result['critical_value']:.6f} (5% level)")
        print(f"  Conclusion:        {ad_result['conclusion']}")

        # Confidence Interval
        print(self._format_subsection("Bootstrap Confidence Interval"))
        ci_result = self._analyzer.bootstrap_confidence_interval(confidence_level=0.95)
        print(f"  Sample Mean:       {ci_result['sample_mean']:.4f} years")
        print(
            f"  95% CI:            [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]"
        )
        print(f"  Margin of Error:   ±{ci_result['margin_of_error']:.4f} years")
        print(f"  Theoretical Mean:  {ci_result['theoretical_mean']:.4f} years")
        print(f"  CI Contains θ:     {ci_result['contains_theoretical']}")

        # Theoretical vs Empirical
        print(self._format_subsection("Theoretical vs Empirical Comparison"))
        comparison = self._analyzer.compare_theoretical_empirical()

        print(f"\n  {'Metric':<15} {'Theoretical':<15} {'Empirical':<15} {'Error %'}")
        print("  " + "-" * 60)

        for metric_name, values in comparison.items():
            print(
                f"  {metric_name.replace('_', ' ').title():<15} "
                f"{values['theoretical']:<15.4f} "
                f"{values['empirical']:<15.4f} "
                f"{values['error_percent']:>7.2f}%"
            )

    def _generate_visualizations(self) -> None:
        """Generate all visualizations."""
        print(self._format_header("STEP 4: GENERATING VISUALIZATIONS"))

        distribution = self._simulator.distribution

        print("Creating visualizations...")

        # 1. Histogram
        print("  [1/5] Histogram with Theoretical PDF...")
        fig1 = self._visualizer.plot_histogram_with_pdf(
            self._survival_data, distribution
        )

        # 2. Survival Curve
        print("  [2/5] Survival Probability Curve...")
        fig2 = self._visualizer.plot_survival_curve(
            distribution, empirical_data=self._survival_data, max_time=25.0
        )

        # 3. Multiple Scenarios
        print("  [3/5] Multiple Scenario Comparison...")
        lambda_values = [0.1, 0.15, 0.2, 0.25, 0.33]
        distributions = [ExponentialDistribution(lam) for lam in lambda_values]
        labels = [f"λ={lam:.2f} (Mean={1/lam:.1f} years)" for lam in lambda_values]

        fig3 = self._visualizer.plot_multiple_scenarios(
            distributions, labels, max_time=20.0
        )

        # 4. Probability Heatmap
        print("  [4/5] Probability Heatmap...")
        time_range = np.linspace(0, 20, 100)
        lambda_range = np.linspace(0.05, 0.5, 100)
        fig4 = self._visualizer.plot_probability_heatmap(time_range, lambda_range)

        # 5. Comprehensive Dashboard
        print("  [5/5] Comprehensive Analysis Dashboard...")
        fig5 = self._visualizer.plot_comprehensive_dashboard(
            self._survival_data, distribution, self._statistics
        )

        print("\n✓ All visualizations created successfully!")

    def _compile_results(self) -> None:
        """Compile all results into AnalysisResults object."""
        validation_results = {
            "ks_test": self._analyzer.goodness_of_fit_ks_test(),
            "ad_test": self._analyzer.goodness_of_fit_anderson_darling(),
            "comparison": self._analyzer.compare_theoretical_empirical(),
            "survival_validation": self._analyzer.validate_survival_probabilities(),
        }

        key_probs = self._calculator.calculate_key_probabilities()
        key_prob_dict = {f"{k}_survival": v["survival"] for k, v in key_probs.items()}

        self._results = AnalysisResults(
            simulation_config=self._config,
            survival_data=self._survival_data,
            statistics=self._statistics,
            validation_results=validation_results,
            confidence_interval=self._analyzer.bootstrap_confidence_interval(),
            key_probabilities=key_prob_dict,
        )

    def _display_summary(self, display: bool = False) -> None:
        """Display final summary report."""
        print(self._format_header("FINAL SUMMARY REPORT"))

        print("SIMULATION SUMMARY:")
        print(f"  • Simulated {self._config.n_patients:,} patient survival times")
        print(f"  • Mean survival time: {self._config.mean_survival_years} years")
        print(f"  • Model: Exponential Distribution (OOP Implementation)")

        print("\nVALIDATION RESULTS:")
        ks = self._results.validation_results["ks_test"]
        print(f"  • K-S Test p-value: {ks['p_value']:.6f} → {ks['conclusion']}")

        comparison = self._results.validation_results["comparison"]
        print(f"  • Mean error: {comparison['mean']['error_percent']:.2f}%")
        print(f"  • Std Dev error: {comparison['std_dev']['error_percent']:.2f}%")
        print(
            f"  • 95% CI contains theoretical mean: {self._results.confidence_interval['contains_theoretical']}"
        )

        print("\nKEY FINDINGS:")
        for year in [1, 3, 5, 10]:
            key = f"{year}_year_survival"
            if key in self._results.key_probabilities:
                prob = self._results.key_probabilities[key]
                print(f"  • {year}-year survival probability: {prob*100:.2f}%")

        print(
            f"  • Median survival time: {self._calculator.get_median_survival():.2f} years"
        )

        print("\nCONCLUSION:")
        print("  The exponential distribution provides an excellent fit for modeling")
        print(
            "  cancer patient survival times. Statistical tests confirm that simulated"
        )
        print(
            "  data closely matches theoretical expectations. This implementation uses"
        )
        print("  modern Python best practices with OOP design and type hints.")

        print(self._format_header("ANALYSIS COMPLETE"))
        print("All figures have been saved. Closing the program.\n")

        if display:
            plt.show()

    def _print_statistics(self) -> None:
        """Print descriptive statistics."""
        print(self._format_subsection("Descriptive Statistics"))

        stats_items = [
            ("Mean", self._statistics["mean"], "years"),
            ("Median", self._statistics["median"], "years"),
            ("Standard Deviation", self._statistics["std_dev"], "years"),
            ("Variance", self._statistics["variance"], ""),
            ("Minimum", self._statistics["min"], "years"),
            ("Maximum", self._statistics["max"], "years"),
            ("25th Percentile", self._statistics["q25"], "years"),
            ("75th Percentile", self._statistics["q75"], "years"),
            ("IQR", self._statistics["iqr"], "years"),
            ("Skewness", self._statistics["skewness"], ""),
            ("Kurtosis", self._statistics["kurtosis"], ""),
        ]

        for name, value, unit in stats_items:
            unit_str = f" {unit}" if unit else ""
            print(f"  {name:<20}: {value:.4f}{unit_str}")

    @staticmethod
    def _format_header(text: str) -> str:
        """Format section header."""
        return f"\n{'='*5} {text} {'='*5}\n"

    @staticmethod
    def _format_subsection(text: str) -> str:
        """Format subsection header."""
        return f"\n--- {text} ---"

    def save_results(self, output_dir: Path) -> None:
        """Save complete analysis results to PDF."""

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H%M%S")
        pdf_path = output_dir / f"{timestamp}.pdf"

        with PdfPages(pdf_path) as pdf:
            # Generate all plots
            plots = [
                (
                    "Survival Time Distribution",
                    self._visualizer.plot_histogram_with_pdf(
                        self._survival_data, self._simulator.distribution
                    ),
                ),
                (
                    "Survival Curve Analysis",
                    self._visualizer.plot_survival_curve(
                        self._simulator.distribution, self._survival_data
                    ),
                ),
                (
                    "Multiple Scenario Comparison",
                    self._visualizer.plot_multiple_scenarios(
                        [ExponentialDistribution(1 / x) for x in [3, 5, 8, 10]],
                        [f"Mean = {x} years" for x in [3, 5, 8, 10]],
                    ),
                ),
                (
                    "Survival Probability Heatmap",
                    self._visualizer.plot_probability_heatmap(
                        np.linspace(0, 20, 100), np.linspace(0.05, 0.5, 100)
                    ),
                ),
                (
                    "Analysis Dashboard",
                    self._visualizer.plot_comprehensive_dashboard(
                        self._survival_data,
                        self._simulator.distribution,
                        self._statistics,
                    ),
                ),
            ]

            # Save each plot to PDF
            for _, fig in plots:
                pdf.savefig(fig)
                plt.close(fig)

            # Add PDF metadata
            d = pdf.infodict()
            d["Title"] = "Cancer Survival Analysis Report"
            d["Author"] = "Anvay Mayekar"
            d["Subject"] = "Statistical Analysis of Cancer Survival Data"
            d["Keywords"] = "survival analysis, cancer, statistics"
            d["CreationDate"] = datetime.datetime.today()
            d["ModDate"] = datetime.datetime.today()

        print(f"\n✓ Analysis results saved to: {pdf_path}")


def main() -> int:
    try:
        config = SimulationConfig(
            mean_survival_years=8.0, n_patients=10000, random_seed=42
        )

        analysis = CancerSurvivalAnalysis(config)
        analysis.run_complete_analysis()

        # Save results to PDF
        output_dir = Path("outputs")
        analysis.save_results(output_dir)

        return 0

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
