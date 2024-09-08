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

"""This script solves a system of linear equations using matrix inversion."""

# Import the required libraries.
import numpy as np

# Define the coefficient matrix A and the constant matrix b.
A = np.array([[2, 3], [4, 1]])
b = np.array([6, 5])

# Calculate the inverse of matrix A.
A_inv = np.linalg.inv(A)

# Solve for X using matrix multiplication.
X = np.dot(A_inv, b)

# Display the results
print("Coefficient Matrix A:")
print(A)
print("\nConstant Matrix b:")
print(b)
print("\nInverse of Matrix A:")
print(A_inv)
print("\nSolution Matrix X (values of x and y):")
print(X)

# Display the solution for x and y.
x, y = X
print(f"\nSolution: x = {x}, y = {y}.")
