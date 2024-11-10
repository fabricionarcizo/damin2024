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

"""This script demonstrates how to apply dimensionality reduction techniques to
a real-world dataset."""

# Import necessary libraries.
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE


# Define the colors to be used in the plot.
colors = [
    "#F59B11", # Yellow
    "#179E86", # Dark Green
    "#44546A", # Gray
]


def plot_2d_subplot(features_list: list, labels: np.ndarray, titles: list):
    """
    Create a 1x3 subplot to show the distribution of the classes in the dataset.

    Args:
        features_list (list): A list containing the features to be plotted.
        labels (np.ndarray): The labels associated with the features.
        titles (list): The titles for each subplot.
    """
    _, axes = plt.subplots(1, 3, figsize=(24, 6))
    for ax, features_2d, title in zip(axes, features_list, titles):
        ax.scatter(
            features_2d[labels == 0, 0], features_2d[labels == 0, 1],
            label="Cats", c=colors[0], s=50, edgecolor=colors[-1], alpha=0.6
        )
        ax.scatter(
            features_2d[labels == 1, 0], features_2d[labels == 1, 1],
            label="Dogs", c=colors[1], s=50, edgecolor=colors[-1], alpha=0.6
        )
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.legend()
        ax.grid(True)

    axes[0].set_ylabel("Component 2")
    plt.tight_layout()
    plt.show()


# Load features and labels from .npy files.
features = np.load("lecture11/data/features.npy")
labels = np.load("lecture11/data/labels.npy")

# Apply PCA.
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Apply LDA.
lda = LDA(n_components=1)
features_lda = lda.fit_transform(features, labels)
features_lda = np.hstack((features_lda, np.zeros_like(features_lda)))

# Apply t-SNE.
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

# Plot PCA, LDA, and t-SNE results in a 1x3 subplot.
plot_2d_subplot(
    [features_pca, features_lda, features_tsne],
    labels, ["PCA", "LDA", "t-SNE"]
)
