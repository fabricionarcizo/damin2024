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

"""This script demonstrates how to apply Principal Component Analysis (PCA) to
a real-world dataset."""

# Import the required libraries.
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


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


def render_covariance_matrix(cov_matrix: pd.DataFrame):
    """
    Render the covariance matrix as a heatmap.

    Args:
        cov_matrix (pd.DataFrame): The covariance matrix to be rendered.
    """
    plt.figure(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list(
        "custom_green", ["#FFFFFF", "#179E86"])
    sns.heatmap(
        cov_matrix, annot=True, fmt="f", cmap=cmap,
        xticklabels=["Systolic Pressure", "Diastolic Pressure"],
        yticklabels=["Systolic Pressure", "Diastolic Pressure"]
    )
    plt.title("Covariance Matrix")


def render_transformation_matrix(matrix: pd.DataFrame):
    """
    Render the transformation matrix as a heatmap.

    Args:
        matrix (pd.DataFrame): The transformation matrix to be rendered.
    """
    plt.figure(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list(
        "custom_green", ["#FFFFFF", "#179E86"])
    sns.heatmap(
        matrix, annot=True, fmt="f", cmap=cmap,
        xticklabels=["First Eigenvector", "Second Eigenvector"],
        yticklabels=["X", "Y"]
    )
    plt.title("Transformation Matrix")


def plot_data(df: pd.DataFrame, x_label: str = "Systolic Pressure",
              y_label: str = "Diastolic Pressure"):
    """
    Create a scatter plot to show the relationship between Systolic Pressure and
    Diastolic Pressure, and modifying the circle size to show data overlapping.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be plotted.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
    """
    sns.scatterplot(
        data=df, x=x_label, y=y_label, color=colors[0],
        size=df.groupby(
            [x_label, y_label]
        ).transform("size"),
        sizes=(50, 200)
    )
    plt.title("Systolic and Diastolic Rate (Medical Data)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    x_extra = (df[x_label].max() - df[x_label].min()) * 0.15
    y_extra = (df[y_label].max() - df[y_label].min()) * 0.15
    plt.xlim(
        df[x_label].min() - x_extra,
        df[x_label].max() + x_extra
    )
    plt.ylim(
        df[y_label].min() - y_extra,
        df[y_label].max() + y_extra
    )

    plt.grid(True)
    plt.tight_layout()


def plot_second_order(coefficients: list):
    """
    Plot the roots of a second-order equation.

    Args:
        coefficients (list): The coefficients of the second-order equation.
    """
    roots = np.roots(coefficients)
    print("Roots of the equation:", roots)

    # Plot the roots on a graph.
    x = np.linspace(-6.225, 50, 400)
    y = x**2 - 43.77*x + 439.94
    plt.plot(
        x, y, color=colors[0], label=r"$\lambda^2 - 43.77\lambda + 439.94 = 0$")

    # Plot the roots and the y-intercept.
    plt.scatter(roots, [0, 0], color=colors[4], zorder=5)
    plt.text(roots[0]-15, 0.5, f"x1 = {roots[1]:.2f}", ha="right")
    plt.text(roots[0]+2.5, 0.5, f"x2 = {roots[0]:.2f}", ha="left")

    plt.scatter([0], [coefficients[-1]], color=colors[4], zorder=5)
    plt.text(0.75, coefficients[-1], f"c = {coefficients[-1]:.2f}", ha="left")

    plt.title("Second Order Equation")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$f(\lambda)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# Load the data.
df = pd.read_csv("./lecture11/data/bdp.csv")
df = df[["Systolic Pressure", "Diastolic Pressure"]]
plot_data(df)
plt.show()

# Step 1: Center the data by subtracting the mean of each column.
df_centered = df - df.mean()
plot_data(df_centered)
plt.show()

# Step 2: Calculate the covariance matrix.
cov_matrix = df_centered.cov()
render_covariance_matrix(cov_matrix)
plt.show()

# Step 3: Calculate the eigenvectors and eigenvalues of the covariance matrix.
_, S, Vt = np.linalg.svd(df_centered, full_matrices=False)
eigenvectors = Vt.T
eigenvalues = S ** 2 / (len(df_centered) - 1)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# Plot the result of the equation λ^2 - 43.77λ + 439.94 = 0
coefficients = [1, -43.77, 439.94]
plot_second_order(coefficients)
plt.show()

# Plot arrows to illustrate the eigenvectors.
plot_data(df_centered)
origin = np.array([[0, 0], [0, 0]])
plt.quiver(
    *origin, eigenvectors[0, :], eigenvectors[1, :], color=colors[4], scale=5)
plt.title("Eigenvectors")
plt.show()

# Step 4: Sort and select the principal components.
transformation_matrix = eigenvectors[::1]
render_transformation_matrix(transformation_matrix)
plt.show()

# Step 5: Transform the data using the transformation matrix.
df_transformed = df_centered.dot(transformation_matrix)
df_transformed.columns = ["Principal Component 1", "Principal Component 2"]
plot_data(df_transformed, "Principal Component 1", "Principal Component 2")

origin = np.array([[0, 0],[0, 0]])
eigenvectors_transformed = eigenvectors.T.dot(transformation_matrix)
plt.quiver(*origin,
           eigenvectors_transformed[0, :],
           eigenvectors_transformed[1, :],
           color=colors[4], scale=5)
plt.title("Principal Components")
plt.show()

# Step 6: Interpret the results.

# Calculate the variance of the transformed data.
variance = df_transformed.var()
print("Variance of the transformed data:", variance)

cov_matrix = df_transformed.cov()
render_covariance_matrix(cov_matrix)
plt.show()

# Step 7: Reduce the dimensionality of the data.
df_reduced = df_transformed.copy()
df_reduced["y"] = 0
plot_data(df_reduced, "Principal Component 1", "y")
plt.title("1D Systolic and Diastolic Rate (Medical Data)")
plt.show()
