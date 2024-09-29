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

"""This script demonstrates how to use the Logistic Regression algorithm to
classify data points using the scikit-learn library."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification


# Generate a 1D binary classification dataset.
X, y = make_classification(
    n_samples=100, n_features=1, n_informative=1, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)

# Initialize and train the Logistic Regression model.
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities for the logistic curve.
X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

# Plotting the dataset and logistic curve.
plt.figure(figsize=(10, 6))

# Scatter plot of the dataset.
plt.scatter(X[y == 0], y[y == 0], color="#2580B7", marker="o", label="Class 0")
plt.scatter(X[y == 1], y[y == 1], color="#179E86", marker="x", label="Class 1")

# Plot logistic sigmoid curve.
plt.plot(X_test, y_prob, color="#C03B26", label="Sigmoid Curve")

# Plot the decision boundary.
plt.axvline(
    x=model.intercept_ / -model.coef_, color="#44546A", linestyle="--",
    label="Decision Boundary"
)

# Labels and legend.
plt.title("Logistic Regression with Sigmoid Curve")
plt.xlabel("Feature 1")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()
