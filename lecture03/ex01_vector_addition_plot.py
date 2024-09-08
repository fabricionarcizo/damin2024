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

"""This script plots two vectors and their sum vector, along with dashed lines
to show the vector projections."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

# Define colors based on provided RGB values.
COLOR_U = "#179E86"  # RGB color for vector u.
COLOR_V = "#F59B11"  # RGB color for vector v.
COLOR_SUM = "#C03B26" # A distinct color for the sum vector.
COLOR_GRAY = "#44546A" # A gray color for the dashed lines.

# Define vectors u and v.
u = np.array([2, 3])
v = np.array([3, 1])
uv_sum = u + v # Addition of vectors.

# Create a figure and axis.
fig, ax = plt.subplots()

# Plot vectors u and v.
ax.quiver(
    0, 0, u[0], u[1], angles="xy", scale_units="xy", scale=1,
    color=COLOR_U
)
ax.quiver(
    0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
    color=COLOR_V
)
ax.quiver(
    0, 0, uv_sum[0], uv_sum[1], angles="xy", scale_units="xy", scale=1,
    color=COLOR_SUM
)

# Plot dashed lines to show the vector projections.
ax.plot(
    [u[0], uv_sum[0]], [u[1], uv_sum[1]],
    color=COLOR_GRAY, linestyle="dashed", linewidth=0.75
)
ax.plot(
    [v[0], uv_sum[0]], [v[1], uv_sum[1]],
    color=COLOR_GRAY, linestyle="dashed", linewidth=0.75
)

# Mark the points (u1, u2), (v1, v2), and (u1+v1, u2+v2).
ax.scatter(
    [u[0], v[0], uv_sum[0]], [u[1], v[1], uv_sum[1]], color="black"
)
ax.text(
    u[0], u[1] + 0.1, f"u=({u[0]}, {u[1]})",
    fontsize=10, ha="right"
)
ax.text(
    v[0], v[1] + 0.1, f"v=({v[0]}, {v[1]})",
    fontsize=10, ha="right"
)
ax.text(
    uv_sum[0], uv_sum[1] + 0.1, f"u+v=({uv_sum[0]}, {uv_sum[1]})",
    fontsize=10, ha="left"
)

# Set labels and limits.
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Ensure the aspect ratio is equal.
ax.set_aspect("equal")

# Show the plot
plt.show()
