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

"""This script demonstrates how to identify anomalies using one-class SVM."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm


# Set random seed for reproducibility.
np.random.seed(42)

# Define the colors to be used in the plot.
colors = [
    "#2580B7", # Blue
    "#C03B26", # Red
    "#44546A", # Gray
]

# Generate 100 samples for the cluster with some outliers.
cluster_samples = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
data = np.vstack((cluster_samples, outliers))

# Fit the model.
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(data)

# Predict the labels (1 for inliers, -1 for outliers).
predictions = clf.predict(data)
outliers = predictions == -1

# Create a mesh grid for plotting decision boundary.
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the results.
plt.scatter(
    data[:, 0], data[:, 1],
    c=np.where(outliers, colors[1], colors[0]),
    s=50, edgecolor=colors[-1], alpha=0.6
)

# Plot decision boundary and margins.
plt.contour(
    xx, yy, Z, levels=[0], linewidths=2, colors=colors[-1]
)
plt.contourf(
    xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap="gray", alpha=0.5)

plt.title("Outlier Detection using One-Class SVM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
