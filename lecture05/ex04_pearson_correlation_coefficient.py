#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

"""This script demonstrates how to calculate the Pearson correlation
coefficient."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import pearsonr, linregress


# Plot with Linear Regression Line
def plot_with_regression(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                         slope: np.float64, intercept: np.float64,
                         color: str, title: str, r_value: np.float64):
    """Plot the data with a linear regression line.

    Args:
        ax (plt.Axes): The matplotlib axes object.
        x (np.ndarray): The x values.
        y (np.ndarray): The y values.
        slope (np.float64): The slope of the regression line.
        intercept (np.float64): The intercept of the regression line.
        color (str): The color of the data points.
        title (str): The title of the plot.
        r_value (np.float64): The Pearson correlation coefficient.
    """
    ax.scatter(x, y, color=color)
    ax.plot(x, slope * x + intercept, color="#44546A",
            label=f"y = {slope:.2f}x + {intercept:.2f}")
    ax.set_title(f"{title} (r = {r_value:.2f})")
    ax.set_xlabel("X values")
    ax.set_ylabel("Y values")
    ax.legend()


# Set random seed for reproducibility.
np.random.seed(42)

# Generate 100 samples for different correlation scenarios.
x = np.random.randint(10, 100, 100)

# 1. Positive correlation.
y_pos = x + np.random.normal(0, 10, 100) # y is positively correlated with x.
r_pos, _ = pearsonr(x, y_pos)
slope_pos, intercept_pos, _, _, _ = linregress(x, y_pos)

# 2. Negative correlation.
y_neg = -x + np.random.normal(0, 10, 100) # y is negatively correlated with x.
r_neg, _ = pearsonr(x, y_neg)
slope_neg, intercept_neg, _, _, _ = linregress(x, y_neg)

# 3. Weak correlation (~0.25).
y_weak = 0.25 * x + np.random.normal(0, 30, 100) # Weak correlation around 0.25.
r_weak, _ = pearsonr(x, y_weak)
slope_weak, intercept_weak, _, _, _ = linregress(x, y_weak)

# 4. No correlation.
y_no_corr = np.random.normal(50, 10, 100) # y has no correlation with x.
r_no_corr, _ = pearsonr(x, y_no_corr)
slope_no_corr, intercept_no_corr, _, _, _ = linregress(x, y_no_corr)

# Define the plot style.
with plt.style.context("fast"):

    # Create a 2x2 plot.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Positive correlation plot.
    plot_with_regression(
        axes[0], x, y_pos, slope_pos, intercept_pos,
        "#179E86", "Positive Correlation", r_pos
    )

    # Negative correlation plot.
    plot_with_regression(
        axes[1], x, y_neg, slope_neg, intercept_neg,
        "#2580B7", "Negative Correlation", r_neg
    )

    # Weak correlation plot.
    plot_with_regression(
        axes[2], x, y_weak, slope_weak, intercept_weak,
        "#F59B11", "Weak Correlation", r_weak
    )

    # No correlation plot.
    plot_with_regression(
        axes[3], x, y_no_corr, slope_no_corr, intercept_no_corr,
        "#C03B26", "No Correlation", r_no_corr
    )

    # Adjust layout.
    plt.tight_layout()
    plt.show()
