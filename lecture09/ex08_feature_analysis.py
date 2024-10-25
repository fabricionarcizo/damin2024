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

"""This script demonstrates how to use heatmaps to analyze the average feature
values across clusters."""

# Import the required libraries.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.datasets import load_iris


# Load the Iris dataset.
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Perform k-Means clustering.
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

# Calculate average feature values for each cluster.
cluster_averages = df.groupby("Cluster").mean()

# Plot the heatmap for average feature values across clusters.
plt.figure(figsize=(12, 8))
sns.heatmap(
    cluster_averages, annot=True, cmap="Reds", fmt=".2f", linewidths=.5,
    linecolor="black", cbar_kws={"shrink": 0.8}
)
plt.title("Heatmap of Average Feature Values Across Clusters")
plt.xlabel("Features")
plt.ylabel("Clusters")
plt.tight_layout()
plt.show()
