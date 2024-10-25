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

"""This script demonstrates how to use the silhouette score to evaluate the
clustering performance of KMeans and Agglomerative Clustering."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# Define the colors to be used in the plot.
colors = [
    "#2580B7", # Blue
    "#F59B11", # Yellow
    "#44546A", # Gray
]

# Set the seed for reproducibility.
np.random.seed(42)

# Generate sample data.
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(X)

# Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=2)
agglo_labels = agglo.fit_predict(X)

# Calculate Silhouette Scores
kmeans_silhouette = silhouette_score(X, kmeans_labels)
agglo_silhouette = silhouette_score(X, agglo_labels)

print(f"KMeans Silhouette Score: {kmeans_silhouette}")
print(f"Agglomerative Clustering Silhouette Score: {agglo_silhouette}")

# Plotting the clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

ax1.scatter(
    X[:, 0], X[:, 1], c=[colors[int(label)] for label in kmeans_labels],
    s=50, edgecolor=colors[-1], alpha=0.6
)
ax1.set_title(
    f"KMeans Clustering (Silhouette Score: {kmeans_silhouette:.2f})")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.grid(True)

ax2.scatter(
    X[:, 0], X[:, 1], c=[colors[int(label)] for label in agglo_labels],
    s=50, edgecolor=colors[-1], alpha=0.6
)
ax2.set_title(
    f"Agglomerative Clustering (Silhouette Score: {agglo_silhouette:.2f})")
ax2.set_xlabel("Feature 1")
ax2.grid(True)

plt.show()
