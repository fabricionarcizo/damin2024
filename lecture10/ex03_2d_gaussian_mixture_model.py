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

"""This script demonstrates how to use Gaussian Mixture Models to cluster 2D
data distribution."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.mixture import GaussianMixture


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

# Anisotropicly distributed data.
N_SAMPLES = 500
RANDOM_STATE = 170
X, _ = datasets.make_blobs(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)

# Creating a DataFrame for the dataset.
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])


# Applying Gaussian Mixture Model.
gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
df["GMM_Cluster"] = gmm.fit_predict(X)

# Plotting the Gaussian Mixture Model clusters.
plt.scatter(
    df["Feature 1"], df["Feature 2"],
    c=[colors[color] for color in df["GMM_Cluster"]],
    s=50, edgecolor=colors[-1], alpha=0.6
)
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(-6, 4)
plt.ylim(-4, 6)
plt.grid(True)
plt.tight_layout()
plt.show()


# Plotting the KDE distribution for the features.
sns.kdeplot(
    x=df["Feature 1"], y=df["Feature 2"],
    color=colors[-1], alpha=0.3
)
plt.scatter(
    df["Feature 1"], df["Feature 2"],
    c=[colors[color] for color in df["GMM_Cluster"]],
    s=50, edgecolor=colors[-1], alpha=0.6
)
plt.title("KDE Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(-6, 4)
plt.ylim(-4, 6)
plt.grid(True)
plt.tight_layout()
plt.show()
