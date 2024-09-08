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

"""This script calculates the dot product of two vectors and the angle between
them."""

# Import the required libraries.
import math

# Define the vectors.
u = (2, 3)
v = (4, 1)

# Step 1: Calculate the dot product of u and v.
DOT_PRODUCT = u[0] * v[0] + u[1] * v[1]

# Step 2: Calculate the magnitudes of u and v.
magnitude_u = math.sqrt(u[0]**2 + u[1]**2)
magnitude_v = math.sqrt(v[0]**2 + v[1]**2)

# Step 3: Use the equation to solve for cos(theta).
cos_theta = DOT_PRODUCT / (magnitude_u * magnitude_v)

# Step 4: Calculate the angle theta in radians and degrees.
theta_radians = math.acos(cos_theta)
theta_degrees = math.degrees(theta_radians)

# Output the results.
print(f"Dot product (u · v): {DOT_PRODUCT}.")
print(f"Magnitude of u: {magnitude_u}.")
print(f"Magnitude of v: {magnitude_v}.")
print(f"cos(θ): {cos_theta}.")
print(f"θ in radians: {theta_radians}.")
print(f"θ in degrees: {theta_degrees}.")
