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

"""This script demonstrates how to calculate the distance metrics between two
points."""

# Import the required libraries.
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_distance(axes: np.ndarray, points: Tuple, distances: Dict):
    """
    Plot the distances between two points on the provided axes.

    Args:
        axes (np.ndarray): The matplotlib axes object.
        points (Tuple): A tuple containing two points.
        distances (Dict): A dictionary containing the distance metrics
    """

    # Unpack the points.
    point1, point2 = points

    # Plot the distance metrics.
    for i, (key, value) in enumerate(distances.items()):
        if key == "Euclidean":
            # Euclidean Distance Plot.
            axes[i].plot(
                [point1[0], point2[0]], [point1[1], point2[1]], "#179E86", lw=2)

        elif key == "Manhattan":
            # Manhattan Distance Plot.
            x_steps = [point1[0]]
            y_steps = [point1[1], point1[1]]
            for j in range(1, 7):
                x_steps += [point1[0] + j, point1[0] + j]
                y_steps += [point1[1] + j, point1[1] + j]
            x_steps.append(point2[0])
            axes[i].step(x_steps, y_steps, "#F59B11", lw=2)

        elif key == "Chebyshev":
            # Chebyshev Distance Plot.
            axes[i].plot(
                [point1[0], point1[0]], [point1[1], point2[1]], "#44546A", linestyle="dashed", lw=1)
            axes[i].plot(
                [point1[0], point2[0]], [point2[1], point2[1]], "#C03B26", lw=2)

        # Common settings for all plots.
        axes[i].scatter(point1[0], point1[1], color="black")
        axes[i].text(
            point1[0] - 0.2, point1[1], f"u=({point1[0]}, {point1[1]})",
            fontsize=10, ha="right"
        )
        axes[i].scatter(point2[0], point2[1], color="black")
        axes[i].text(
            point2[0] + 0.2, point2[1], f"v=({point2[0]}, {point2[1]})",
            fontsize=10, ha="left"
        )
        axes[i].set_title(f"{key} = {value:.2f}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].set_xlim(0, 10)
        axes[i].set_ylim(0, 10)
        axes[i].xaxis.set_major_locator(plt.MultipleLocator(1))
        axes[i].yaxis.set_major_locator(plt.MultipleLocator(1))


# Define two points.
point1 = np.array([2, 2])
point2 = np.array([8, 8])

# Calculate the distances.
distances = {
    "Euclidean": np.linalg.norm(point1 - point2),
    "Manhattan": np.sum(np.abs(point1 - point2)),
    "Chebyshev": np.max(np.abs(point1 - point2))
}

# Define the plot style.
with plt.style.context("seaborn-v0_8"):

    # Create the plot.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Call the function with the points, distances, and types to plot.
    plot_distance(axes, (point1, point2), distances)

    # Adjust layout and show the plot.
    plt.tight_layout()
    plt.show()
