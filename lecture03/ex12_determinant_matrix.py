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

"""This script plots one matrix and its determinant."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Define a square matrix A.
A = np.array([[4, 7], [2, 6]])

# Calculate the determinant of matrix A
determinant_A = np.linalg.det(A)

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

# Create a figure to display the matrix and its determinant.
fig, ax = plt.subplots(figsize=(5, 5))

# Display matrix A.
ax.imshow(color_grid_A, aspect="auto")
ax.set_title(f"Matrix A\nDeterminant: {determinant_A:.2f}", pad=20)
for (i, j), val in np.ndenumerate(A):
    ax.text(j, i, f"{val}", ha="center", va="center", fontsize=14)

# Adjust layout and show the plot.
plt.tight_layout()
plt.show()
