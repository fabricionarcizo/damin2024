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

"""This script plots two vectors and their dot product."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches

# Define colors based on provided RGB values.
COLOR_U = "#179E86"  # RGB color for vector u.
COLOR_V = "#F59B11"  # RGB color for vector v.
COLOR_GRAY = "#44546A" # A gray color for the dashed lines.

# Define vectors u and v.
u = np.array([2, 3])
v = np.array([4, 1])

# Calculate the dot product.
dot_product = np.dot(u, v)

# Calculate the angle between the vectors using the dot product formula.
angle = np.arccos(dot_product / (np.linalg.norm(u) * np.linalg.norm(v)))

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

# Plot the dashed line to represent the projection.
projection_length = dot_product / np.linalg.norm(v)
projection_vector = (projection_length / np.linalg.norm(v)) * v
ax.plot(
    [u[0], projection_vector[0]], [u[1], projection_vector[1]],
    color=COLOR_GRAY, linestyle="dashed", linewidth=0.75
)

# Calculate the position for the 90-degree angle marker.
angle_rect_x = projection_vector[0] - 1.25 * (v[1] / np.linalg.norm(v))
angle_rect_y = projection_vector[1] - 0.05 * (v[0] / np.linalg.norm(v))

# Add the 90-degree angle marker using a small rectangle on the inner side.
RECT_SIZE = 0.3  # Size of the right-angle square.
angle_rect = patches.Rectangle(
    (angle_rect_x, angle_rect_y),
    RECT_SIZE, RECT_SIZE,
    angle=np.degrees(np.arctan2(v[1], v[0])),
    color=COLOR_GRAY, fill=False
)
ax.add_patch(angle_rect)

# Draw the arc to represent the angle between the vectors.
ARC_RADIUS = 0.75  # Radius of the arc.
arc = patches.Arc(
    (-0.1, 0.175), 2 * ARC_RADIUS, 2 * ARC_RADIUS,
    angle=0, theta1=0, theta2=np.degrees(angle),
    color=COLOR_GRAY, linewidth=1.5
)
ax.add_patch(arc)

# Display the angle between the vectors.
ax.text(
    1.2, 0.6, f"{np.degrees(angle):.2f}°", fontsize=10,
    ha="center", va="center"
)

# Display the dot product value on the plot.
ax.text(
    0.05, 4.75, f"Dot Product: {dot_product}", fontsize=10, ha="center",
    va="center", bbox={"facecolor": 'white', "alpha": 0.5}
)

# Mark the points (u1, u2) and (v1, v2).
ax.scatter(
    [u[0], v[0]], [u[1], v[1]], color="black"
)
ax.text(
    u[0], u[1] + 0.1, f"u=({u[0]}, {u[1]})",
    fontsize=10, ha="right"
)
ax.text(
    v[0], v[1] + 0.1, f"v=({v[0]}, {v[1]})",
    fontsize=10, ha="right"
)

# Set labels and limits.
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Ensure the aspect ratio is equal.
ax.set_aspect("equal")

# Show the plot.
plt.show()
