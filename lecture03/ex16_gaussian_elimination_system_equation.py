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

"""This script solves a system of linear equations using Gaussian
Elimination."""

# Import the required libraries.
import numpy as np

# Define the augmented matrix [A|b].
augmented_matrix = np.array([[2, 3, 5],
                             [4, -1, 6]], dtype=float)

# Perform Gaussian elimination.
rows, cols = augmented_matrix.shape

# Forward elimination process.
for i in range(rows):

    # Normalize the current row by dividing by the diagonal element.
    augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]

    # Eliminate the current variable from the subsequent rows.
    for j in range(i + 1, rows):
        augmented_matrix[j] = \
            augmented_matrix[j] - augmented_matrix[j][i] * augmented_matrix[i]

# Back substitution process.
solutions = np.zeros(rows)
for i in range(rows - 1, -1, -1):
    solutions[i] = \
        augmented_matrix[i][-1] - np.sum(
            augmented_matrix[i][i+1:-1] * solutions[i+1:]
        )

# Display the results.
print("Augmented Matrix after Gaussian Elimination:")
print(augmented_matrix)

print("\nSolution for the variables:")
for index, value in zip(["x", "y"], solutions):
    print(f"{index} = {value:.2f}")
