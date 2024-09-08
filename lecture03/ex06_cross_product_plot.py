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

"""This script plots two vectors in 3D and their cross product."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

# Define colors based on provided RGB values.
COLOR_U = "#179E86"  # RGB color for vector u.
COLOR_V = "#F59B11"  # RGB color for vector v.
COLOR_CROSS = "#C03B26" # A distinct color for the cross product vector.

# Modify the vectors to ensure a positive cross product in the Z direction.
u = np.array([2, 1, 0])
v = np.array([1, 3, 0])

# Calculate the cross product.
cross_product = np.cross(u, v)

# Create a 3D plot.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the vectors u and v.
ax.quiver(0, 0, 0, u[0], u[1], u[2], color=COLOR_U)
ax.quiver(0, 0, 0, v[0], v[1], v[2], color=COLOR_V)

# Plot the cross product.
ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2],
          color=COLOR_CROSS)

# Mark the points (u1, u2), (v1, v2), and (u1-v1, u2-v2).
ax.scatter(
    [u[0], v[0], cross_product[0]],
    [u[1], v[1], cross_product[1]],
    [u[2], v[2], cross_product[2]],
    color="black"
)

ax.text(
    u[0], u[1] + 0.1, u[2],
    f"u=({u[0]}, {u[1]}, {u[2]})",
    fontsize=10, ha="left"
)
ax.text(
    v[0], v[1] + 0.1, v[2],
    f"v=({v[0]}, {v[1]}, {v[2]})",
    fontsize=10, ha="left"
)
ax.text(
    cross_product[0], cross_product[1] + 0.1, cross_product[2],
    f"u x v=({cross_product[0]}, {cross_product[1]}, {cross_product[2]})",
    fontsize=10, ha="left"
)

# Setting the limits for better visualization.
ax.set_xlim([-1, 5])
ax.set_ylim([-1, 5])
ax.set_zlim([-1, 5])

# Labeling the axes.
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Show the plot.
plt.show()
