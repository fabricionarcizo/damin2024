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

"""This script plots a vector and its scaled, rotated, and reflected versions
using linear transformations."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Define colors based on provided RGB values.
COLOR_V = "#179E86"  # RGB color for vector v.
COLOR_T = "#C03B26" # A distinct color for the transformed vector.


def rotation_matrix(theta: float) -> np.ndarray:
    """Define the rotation matrix.

    Args:
        theta (float): The angle of rotation (in radians).

    Returns:
        np.ndarray: The rotation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """Define the scaling matrix.

    Args:
        sx (float): The scaling factor in the X-direction.
        sy (float): The scaling factor in the Y-direction.

    Returns:
        np.ndarray: The scaling matrix.
    """
    return np.array([
        [sx, 0],
        [0, sy]
    ])


def reflection_matrix() -> np.ndarray:
    """Define the reflection matrix over the X-axis.

    Returns:
        np.ndarray: The reflection matrix.
    """
    return np.array([
        [1, 0],
        [0, -1]
    ])


# Define the vector to transform.
v = np.array([1, 2])

# Define the transformations.
scaling = scaling_matrix(2, 0.5)  # Scale by 2 on the X- and 0.5 on the Y-axis.
rotation = rotation_matrix(np.pi / 4)  # 45 degrees rotation.
reflection = reflection_matrix()  # Reflection over the X-axis.

# Apply the transformations.
scaled_v = np.dot(scaling, v)
rotated_v = np.dot(rotation, v)
reflected_v = np.dot(reflection, v)


# Scaling plot.
plt.figure()
plt.quiver(
    0, 0, v[0], v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_V, label="Original Vector"
)
plt.quiver(
    0, 0, scaled_v[0], scaled_v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_T, label="Scaled Vector"
)

# Mark the points (v1, v2), and (v'1, v2').
plt.scatter(
    [v[0], scaled_v[0]], [v[1], scaled_v[1]], color="black"
)
plt.text(
    v[0], v[1] + 0.2, f"v=({v[0]}, {v[1]})",
    fontsize=10, ha="left"
)
plt.text(
    scaled_v[0], scaled_v[1] + 0.2,
    f"v'=({scaled_v[0]:.0f}, {scaled_v[1]:.0f})",
    fontsize=10, ha="left"
)

# Set plot limits.
plt.xlim(-3.0, 3.0)
plt.ylim(-3.0, 3.0)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Ensure the aspect ratio is equal.
plt.gca().set_aspect("equal", adjustable="box")

# Define the legend.
plt.legend()


# Rotation plot.
plt.figure()
plt.quiver(
    0, 0, v[0], v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_V, label="Original Vector"
)
plt.quiver(
    0, 0, rotated_v[0], rotated_v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_T, label="Rotated Vector"
)

# Mark the points (v1, v2), and (v'1, v2').
plt.scatter(
    [v[0], rotated_v[0]], [v[1], rotated_v[1]], color="black"
)
plt.text(
    v[0], v[1] + 0.2, f"v=({v[0]}, {v[1]})",
    fontsize=10, ha="left"
)
plt.text(
    rotated_v[0], rotated_v[1] + 0.2,
    f"v'=({rotated_v[0]:.2f}, {rotated_v[1]:.2f})",
    fontsize=10, ha="left"
)

# Set plot limits.
plt.xlim(-3.0, 3.0)
plt.ylim(-3.0, 3.0)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Ensure the aspect ratio is equal.
plt.gca().set_aspect("equal", adjustable="box")

# Define the legend.
plt.legend()


# Reflection plot.
plt.figure()
plt.quiver(
    0, 0, v[0], v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_V, label="Original Vector"
)
plt.quiver(
    0, 0, reflected_v[0], reflected_v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_T, label="Reflected Vector"
)

# Mark the points (v1, v2), and (v'1, v2').
plt.scatter(
    [v[0], reflected_v[0]], [v[1], reflected_v[1]], color="black"
)
plt.text(
    v[0], v[1] + 0.2, f"v=({v[0]}, {v[1]})",
    fontsize=10, ha="left"
)
plt.text(
    reflected_v[0], reflected_v[1] + 0.2,
    f"v'=({reflected_v[0]:.0f}, {reflected_v[1]:.0f})",
    fontsize=10, ha="left"
)

# Set plot limits.
plt.xlim(-3.0, 3.0)
plt.ylim(-3.0, 3.0)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Ensure the aspect ratio is equal.
plt.gca().set_aspect("equal", adjustable="box")

# Define the legend.
plt.legend()

# Show all plots.
plt.show()
