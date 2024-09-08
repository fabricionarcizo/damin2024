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

"""This script plots two matrices and their multiplication."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

# Define two matrices A and B for multiplication.
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Calculate the multiplication of matrices A and B.
C_multiplication = np.dot(A, B)

# Create color maps using the given base colors.
cmap_A = LinearSegmentedColormap.from_list(
    "custom_blue", ["#FFFFFF", "#179E86"])
cmap_B = LinearSegmentedColormap.from_list(
    "custom_green", ["#FFFFFF", "#F59B11"])
cmap_C = LinearSegmentedColormap.from_list(
    "custom_red", ["#FFFFFF", "#C03B26"])

# Create a figure with subplots to display the matrices.
fig, axs = plt.subplots(1, 3, figsize=(15, 6))

# Display matrix A.
axs[0].matshow(A, cmap=cmap_A)
axs[0].set_title("Matrix A")
for (i, j), val in np.ndenumerate(A):
    axs[0].text(j, i, f"{val}", ha="center", va="center", fontsize=14)

# Display matrix B.
axs[1].matshow(B, cmap=cmap_B)
axs[1].set_title("Matrix B")
for (i, j), val in np.ndenumerate(B):
    axs[1].text(j, i, f"{val}", ha="center", va="center", fontsize=14)

# Display matrix A x B.
axs[2].matshow(C_multiplication, cmap=cmap_C)
axs[2].set_title("A x B")
for (i, j), val in np.ndenumerate(C_multiplication):
    axs[2].text(j, i, f"{val}", ha="center", va="center", fontsize=14)

# Adjust layout and show the plots.
plt.tight_layout()
plt.show()
