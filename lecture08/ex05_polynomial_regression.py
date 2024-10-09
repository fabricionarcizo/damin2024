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

"""This script demonstrates how to use polynomial regression to predict the
price of a house based on its lotsize."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, r2_score


# Load the Stock Market Predictions dataset.
df = pd.read_csv("lecture08/data/simple_windsor.csv")
colors = ["#179E86", "#2580B7", "#9EBE5B", "#C03B26", "#44546A"]

# Extract the features and target variable.
X = df[["lotsize"]]
y = df["price"]

# Define polynomial degrees to plot.
degrees = [ 2, 3, 4, 5 ]

# Create a 2x2 plot.
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i, degree in enumerate(degrees):

    # Create polynomial features.
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit the polynomial regression model.
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict using the polynomial model.
    y_pred = model.predict(X_poly)

    # Calculate and print metrics.
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\nDegree {degree} - Mean Absolute Error: {mae}.")
    print(f"Degree {degree} - R^2 Score: {r2}.")

    # Predict using the polynomial model.
    X_test = np.linspace(X.min() - 20, X.max() + 20, 1000).reshape(-1, 1)
    X_test_poly = poly.transform(pd.DataFrame(X_test, columns=X.columns))
    y_test_pred = model.predict(X_test_poly)

    # Plot the polynomial regression results
    ax = axs[i // 2, i % 2]
    ax.scatter(X, y, color=colors[i])
    ax.plot(X_test, y_test_pred, color=colors[-1])
    if i // 2 == 1:
        ax.set_xlabel("lotsize")
    if i % 2 == 0:
        ax.set_ylabel("price")
    ax.set_title(
        f"Polynomial Regression (degree={degree}, MAE={mae:.2f}, R²={r2:.2f})")

plt.tight_layout()
plt.show()
