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

"""This script demonstrates how to use the k-means algorithm to cluster data
points."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

from kneed import KneeLocator
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs


# Set the seed for reproducibility.
np.random.seed(42)

# Generate sample data.
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Elbow method to find the optimal number of clusters.
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph.
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, "o-", markersize=8, linewidth=2, color="#2580B7")

# Add labels to the plot.
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Define the colors to be used in the plot.
colors = [
    "#2580B7", # Blue
    "#179E86", # Dark Green
    "#9EBE5B", # Light Green
    "#F59B11", # Yellow
    "#C03B26", # Red
    "#633248", # Brown
    "#44546A", # Gray
]

# Use the KneeLocator from the kneed library to find the elbow point.
kl = KneeLocator(K, inertia, curve="convex", direction="decreasing")
elbow_point = kl.elbow
print(f"The optimal number of clusters is: {elbow_point}.")

# Apply the k-means algorithm.
kmeans = KMeans(n_clusters=elbow_point, random_state=42)
kmeans.fit(X)

# Plot the data points and the centroids.
plt.scatter(
    X[:, 0], X[:, 1], c=[colors[int(label)] for label in kmeans.labels_],
    s=50, edgecolor=colors[-1], alpha=0.6
)
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    s=200, c=colors[4], marker="*", label="Centroids"
)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
