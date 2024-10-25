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

"""This script demonstrates how to use dendrograms to visualize hierarchical
clustering."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster


# Seed for reproducibility.
np.random.seed(42)

# Load the data.
df = pd.read_csv("./lecture09/data/social_media_exercise.csv")

# Define the methods to be used for linkage.
methods = ["single", "complete", "average", "weighted", "centroid", "ward"]

# Plotting the distributions.
with plt.style.context("./lecture05/data/themes/rose-pine.mplstyle"):

    # Create a figure with 2x3 subplots.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Loop through each method and plot the corresponding dendrogram.
    for ax, method in zip(axes.flatten(), methods):

        # Perform hierarchical/agglomerative clustering.
        Z = linkage(df[["social_media", "physical_exercise"]], method=method)
        
        # Plot the dendrogram.
        dendrogram(
            Z, labels=df["name"].values,
            distance_sort="ascending", show_leaf_counts=True,
            ax=ax
        )
        
        ax.set_title(f"{method.capitalize()} Linkage")
        ax.set_ylabel("Euclidean Distances")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="x", rotation=45, labelsize=10)

    plt.tight_layout()
    plt.show()


    # Define the distances and colors to be used for clustering.
    distances = [2.5, 6, 5, 5, 5, 10]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create a figure with 2x3 subplots for scatter plots.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Loop through each method and plot the corresponding scatter plot.
    for ax, method, distance in zip(axes.flatten(), methods, distances):

        # Perform hierarchical/agglomerative clustering.
        Z = linkage(df[["social_media", "physical_exercise"]], method=method)
        
        # Assign cluster labels.
        clusters = fcluster(Z, t=distance, criterion="distance")
        
        # Plot the scatter plot using the colors defined in rose-pine.mplstyle.
        ax.scatter(
            df["social_media"], df["physical_exercise"],
            c=[colors[i % len(colors)] for i in clusters], s=50
        )

        # Annotate the points with the names.
        for i, name in enumerate(df["name"]):
            ax.annotate(
                name,
                (df["social_media"][i], df["physical_exercise"][i] + 0.3), ha="center"
            )

        ax.set_title(f"{method.capitalize()} Linkage")
        ax.set_xlabel("Social Media")
        ax.set_ylabel("Physical Exercise")
        ax.grid(True, linestyle="--")
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)

    plt.tight_layout()
    plt.show()
