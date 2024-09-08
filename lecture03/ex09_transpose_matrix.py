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

"""This script plots one matrix and its transpose."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

# Define a matrix A.
A = np.array([[1, 2, 3], [4, 5, 6]])

# Calculate the transpose of matrix A.
A_transpose = A.T

# Create color maps using the given base colors.
cmap_A = LinearSegmentedColormap.from_list(
    "custom_blue", ["#FFFFFF", "#179E86"])
cmap_T = LinearSegmentedColormap.from_list(
    "custom_red", ["#FFFFFF", "#C03B26"])

# Create a figure with subplots to display the matrices.
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display matrix A.
axs[0].matshow(A, cmap=cmap_A)
axs[0].set_title("Matrix A")
for (i, j), val in np.ndenumerate(A):
    axs[0].text(j, i, f"{val}", ha="center", va="center", fontsize=14)

# Display matrix A Transpose.
axs[1].matshow(A_transpose, cmap=cmap_T)
axs[1].set_title("Transpose of A")
for (i, j), val in np.ndenumerate(A_transpose):
    axs[1].text(j, i, f"{val}", ha="center", va="center", fontsize=14)

# Adjust layout and show the plots.
plt.tight_layout()
plt.show()
