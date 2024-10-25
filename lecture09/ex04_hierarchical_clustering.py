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

"""This script demonstrates how to use the hierarchical clustering algorithm to
cluster data points."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage


# Define the colors to be used in the plot.
colors = [
    "#2580B7", # Blue
    "#F59B11", # Yellow
    "#C03B26", # Red
    "#44546A", # Gray
]

# Seed for reproducibility.
np.random.seed(42)

# Load the data.
df = pd.read_csv("./lecture09/data/social_media_exercise.csv")

# Agglomerative Clustering.
agg_cluster = AgglomerativeClustering(metric="euclidean", linkage="single")
df["agg_cluster"] = agg_cluster.fit_predict(
    df[["social_media", "physical_exercise"]]
)

# Agglomerative Clustering Plot.
plt.scatter(
    df["social_media"], df["physical_exercise"],
    c=[colors[color] for color in df["agg_cluster"]],
    s=50, edgecolor=colors[-1], alpha=0.6
)

# Annotate the points with the names.
for i, name in enumerate(df["name"]):
    plt.annotate(
        name,
        (df["social_media"][i], df["physical_exercise"][i] + 0.3), ha="center"
    )

# Plot information.
plt.title("Agglomerative Clustering")
plt.xlabel("Hours on Social Media")
plt.ylabel("Hours of Physical Exercise")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, 15)
plt.ylim(0, 15)

plt.tight_layout()
plt.show()

# Dendrogram for Agglomerative Clustering.
# Perform hierarchical/agglomerative clustering.
Z = linkage(df[["social_media", "physical_exercise"]], method="single")

# Plot the dendrogram
dendrogram(
    Z, labels=df["name"].values,
    distance_sort="ascending", show_leaf_counts=True
)

plt.title("Agglomerative Clustering Dendrogram")
plt.ylabel("Euclidean Distances")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tick_params(axis="x", rotation=45, labelsize=10)

plt.tight_layout()
plt.show()
