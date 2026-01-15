# Creates publication-quality visualizations

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.models.survival_distribution import SurvivalDistribution
from src.utils.types import FloatArray, StatisticsDict
from scipy import stats


class SurvivalVisualizer:
    """
    Creates comprehensive visualizations for survival analysis.
    All methods return Figure objects for flexibility in saving/displaying.
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        """
        Initialize visualizer with plotting style.

        Args:
            style: Matplotlib style name
        """
        self._style = style
        plt.style.use(style)

    def plot_histogram_with_pdf(
        self,
        survival_data: FloatArray,
        distribution: SurvivalDistribution,
        bins: int = 60,
        max_time: float = 20.0,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Figure:
        """
        Create histogram of survival times with theoretical PDF overlay.

        Args:
            survival_data: Observed survival times
            distribution: Theoretical distribution
            bins: Number of histogram bins
            figsize: Figure size (width, height)

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(
            survival_data,
            bins=bins,
            density=True,
            alpha=0.7,
            color="#3498db",
            edgecolor="black",
            linewidth=0.5,
            label="Simulated Data",
        )

        # Overlay theoretical PDF
        x_range = np.linspace(0, float(np.max(survival_data)), 1000)
        y_pdf = distribution.probability_density(x_range)

        ax.plot(
            x_range,
            y_pdf,
            "r-",
            linewidth=2.5,
            label=f"Theoretical PDF (λ={1/distribution.mean():.3f})",
        )

        # Add mean line
        mean_val = distribution.mean()
        ax.axvline(
            mean_val,
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {mean_val:.2f} years",
        )

        self._format_axes(
            ax,
            xlabel="Time (years)",
            ylabel="Survival Probability S(t)",
            title="Cancer Patient Survival Curve\nExponential Distribution Model",
        )

        ax.legend(fontsize=11, loc="upper right")
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, max_time])
        plt.tight_layout()

        return fig

    def plot_multiple_scenarios(
        self,
        distributions: List[SurvivalDistribution],
        labels: List[str],
        max_time: float = 20.0,
        figsize: Tuple[int, int] = (14, 7),
    ) -> Figure:
        """
        Compare survival curves for multiple scenarios.

        Args:
            distributions: List of survival distributions
            labels: Labels for each distribution
            max_time: Maximum time for x-axis
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        time_range = np.linspace(0, max_time, 1000)

        for i, (dist, label) in enumerate(zip(distributions, labels)):
            survival_prob = dist.survival_function(time_range)
            color = colors[i % len(colors)]

            ax.plot(time_range, survival_prob, linewidth=2.5, label=label, color=color)

        self._format_axes(
            ax,
            xlabel="Time (years)",
            ylabel="Survival Probability S(t)",
            title="Comparison of Survival Curves\nDifferent Mean Survival Times",
        )

        ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, max_time])
        plt.tight_layout()

        return fig

    def plot_probability_heatmap(
        self,
        time_range: FloatArray,
        lambda_range: FloatArray,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Figure:
        """
        Create heatmap of survival probabilities for different parameters.

        Args:
            time_range: Array of time values
            lambda_range: Array of lambda values
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create meshgrid
        T, L = np.meshgrid(time_range, lambda_range)
        S = np.exp(-L * T)

        # Create heatmap
        im = ax.contourf(T, L, S, levels=20, cmap="RdYlGn")

        # Add contour lines
        contours = ax.contour(
            T,
            L,
            S,
            levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            colors="black",
            linewidths=1.5,
            alpha=0.6,
        )
        ax.clabel(contours, inline=True, fontsize=10, fmt="%.2f")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Survival Probability S(t)", fontsize=12, fontweight="bold")

        self._format_axes(
            ax,
            xlabel="Time (years)",
            ylabel="Rate Parameter λ",
            title="Survival Probability Heatmap\nP(T > t) vs Time and Rate Parameter",
        )

        plt.tight_layout()
        return fig

    def plot_comprehensive_dashboard(
        self,
        survival_data: FloatArray,
        distribution: SurvivalDistribution,
        statistics: StatisticsDict,
        figsize: Tuple[int, int] = (16, 12),
    ) -> Figure:
        """Create comprehensive dashboard with improved layout."""
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])

        # Create panels
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_histogram_panel(ax1, survival_data, distribution)

        ax2 = fig.add_subplot(gs[0, 1])
        self._create_survival_panel(ax2, survival_data, distribution)

        ax3 = fig.add_subplot(gs[1, 0])
        self._create_qq_panel(ax3, survival_data)

        ax4 = fig.add_subplot(gs[1, 1])
        self._create_box_panel(ax4, survival_data)

        ax5 = fig.add_subplot(gs[2, :])
        self._create_stats_panel(ax5, statistics)

        fig.suptitle(
            "Comprehensive Cancer Survival Analysis Dashboard",
            fontsize=14,
            fontweight="bold",
            y=0.95,
        )

        return fig

    def _create_histogram_panel(
        self,
        ax: Axes,
        data: FloatArray,
        distribution: SurvivalDistribution,
    ) -> None:
        """Create histogram panel."""
        ax.hist(data, bins=50, density=True, alpha=0.7)
        x = np.linspace(0, max(data), 200)
        ax.plot(x, distribution.probability_density(x), color="red", lw=2)
        ax.set_title("A) Survival Time Distribution")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Density")

    def _create_survival_panel(
        self,
        ax: Axes,
        data: FloatArray,
        distribution: SurvivalDistribution,
    ) -> None:
        """Create survival curve panel."""
        time_range = np.linspace(0, max(data), 200)
        survival_prob = distribution.survival_function(time_range)
        ax.plot(time_range, survival_prob, color="#2c3e50", lw=2)

        # Add empirical curve
        sorted_times = np.sort(data)
        n = len(sorted_times)
        survival_probs = np.arange(n, 0, -1) / n
        ax.step(
            sorted_times,
            survival_probs,
            where="post",
            color="#e74c3c",
            alpha=0.7,
        )

        ax.set_title("B) Survival Function")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("S(t)")
        ax.set_ylim([0, 1.05])

    def _create_qq_panel(
        self,
        ax: Axes,
        data: FloatArray,
    ) -> None:
        """Create Q-Q plot panel."""
        stats.probplot(data, dist="expon", plot=ax)
        ax.set_title("C) Q-Q Plot")

    def _create_box_panel(
        self,
        ax: Axes,
        data: FloatArray,
    ) -> None:
        """Create box plot panel."""
        ax.boxplot(data)
        ax.set_title("D) Box Plot")
        ax.set_ylabel("Time (years)")

    def _create_stats_panel(
        self,
        ax: Axes,
        statistics: StatisticsDict,
    ) -> None:
        """Create statistics panel."""
        ax.axis("off")
        stats_text = [
            f"{k.replace('_', ' ').title()}: {v:.4f}" for k, v in statistics.items()
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(stats_text),
            transform=ax.transAxes,
            va="top",
            fontfamily="monospace",
        )

    def _add_survival_markers(
        self,
        ax: Axes,
        distribution: SurvivalDistribution,
        time_points: List[float],
        max_time: float,
    ) -> None:
        """Add marker annotations to survival curve."""
        for t in time_points:
            if t <= max_time:
                s_t = float(distribution.survival_function(t))
                ax.plot(t, s_t, "o", markersize=10, color="#27ae60", zorder=5)
                ax.annotate(
                    f"{s_t:.1%}",
                    xy=(t, s_t),
                    xytext=(10, -15),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="yellow",
                        alpha=0.8,
                        edgecolor="black",
                    ),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5
                    ),
                )

    def _format_axes(self, ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
        """Apply consistent formatting to axes."""
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3, linestyle="--")

    def plot_survival_curve(
        self,
        distribution: SurvivalDistribution,
        empirical_data: Optional[FloatArray] = None,
        max_time: float = 20.0,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Figure:
        """Create survival curve plot."""
        fig, ax = plt.subplots(figsize=figsize)

        # Plot theoretical survival curve
        time_range = np.linspace(0, max_time, 1000)
        survival_prob = distribution.survival_function(time_range)
        # Remove format string, use color keyword only
        ax.plot(
            time_range,
            survival_prob,
            linewidth=2.5,
            color="#2c3e50",
            label="Theoretical S(t)",
        )

        # Plot empirical if provided
        if empirical_data is not None:
            sorted_times = np.sort(empirical_data)
            n = len(sorted_times)
            survival_probs = np.arange(n, 0, -1) / n
            ax.step(
                sorted_times,
                survival_probs,
                where="post",
                color="#e74c3c",
                alpha=0.7,
                label="Empirical",
            )

        self._format_axes(
            ax,
            xlabel="Time (years)",
            ylabel="Survival Probability S(t)",
            title="Cancer Patient Survival Curve",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, max_time])

        return fig
