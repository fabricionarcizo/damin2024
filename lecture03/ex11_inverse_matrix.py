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

"""This script plots one matrix and its inverse."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Define a square matrix A.
A = np.array([[4, 7], [2, 6]])

# Calculate the inverse of matrix A.
A_inverse = np.linalg.inv(A)

def create_index_based_colormap(matrix: np.ndarray,
                                base_color: str) -> np.ndarray:
    """Define a function to create a color grid based on matrix indices

    Args:
        matrix (np.ndarray): The matrix to be visualized.
        base_color (str): The base color for the color gradient.

    Returns:
        np.ndarray: The color grid based on the matrix indices.
    """
    rows, cols = matrix.shape
    color_grid = np.zeros((rows, cols, 3))  # Empty grid for RGB colors.

    for i in range(rows):
        for j in range(cols):
            # Map the indices to a color gradient based on a normalized
            # index-based intensity.
            color_intensity = (i + j) / (rows + cols - 2)
            color_grid[i, j] = np.array(
                plt.cm.colors.hex2color(base_color)
            ) * color_intensity + (1 - color_intensity)

    return color_grid

# Create color grids for matrices A, B, and the result.
color_grid_A = create_index_based_colormap(A, "#179E86")
color_grid_inverse = create_index_based_colormap(A_inverse, "#C03B26")

# Create a figure with subplots to display the matrices.
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display matrix A.
axs[0].imshow(color_grid_A, aspect="auto")
axs[0].set_title("Matrix A")
for (i, j), val in np.ndenumerate(A):
    axs[0].text(j, i, f"{val}", ha="center", va="center", fontsize=14)

# Display the inverse of matrix A.
axs[1].imshow(color_grid_inverse, aspect="auto")
axs[1].set_title("Inverse of A")
for (i, j), val in np.ndenumerate(A_inverse):
    axs[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=14)

# Adjust layout and show the plots.
plt.tight_layout()
plt.show()
