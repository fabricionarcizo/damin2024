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

"""This script plots a vector and the result of applying two linear
transformations to it."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Define colors based on provided RGB values.
COLOR_V = "#179E86"  # RGB color for vector v.
COLOR_T1 = "#2580B7" # A distinct color for the first transformed vector.
COLOR_T2 = "#F59B11" # A distinct color for the second transformed vector.
COLOR_T = "#C03B26" # A distinct color for the composition transformed vector.


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


# Define the vector to transform.
v = np.array([1, 2])

# Define the matrices representing the transformations T1 and T2.
R = rotation_matrix(np.pi / 4)  # 45 degrees rotation.
S = scaling_matrix(2, 0.5)  # Scale by 2 on the X- and 0.5 on the Y-axis.

# Calculate the composition of the transformations T1 and T2.
T = np.dot(R, S)
print(T)

# Apply the individual transformations.
T1_v = np.dot(R, v)
T2_v = np.dot(S, v)

# Apply the composition of the transformations.
T_v = np.dot(T, v)

# Original vector.
plt.figure()
plt.quiver(
    0, 0, v[0], v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_V, label="Original Vector"
)

# T1 (R) transformation.
plt.quiver(
    0, 0, T1_v[0], T1_v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_T1, label="T1 (R) Applied"
)

# T2 (S) transformation.
plt.quiver(
    0, 0, T2_v[0], T2_v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_T2, label="T2 (S) Applied"
)

# Composition of T1 and T2.
plt.quiver(
    0, 0, T_v[0], T_v[1], angles="xy", scale_units="xy",
    scale=1, color=COLOR_T, label="T1 x T2 Applied"
)

# Mark the points.
plt.scatter(
    [v[0], T1_v[0], T2_v[0], T_v[0]], [v[1], T1_v[1], T2_v[1], T_v[1]],
    color="black"
)
plt.text(
    v[0] + 0.2, v[1], f"v=({v[0]}, {v[1]})",
    fontsize=10, ha="left"
)
plt.text(
    T1_v[0], T1_v[1] + 0.2,
    f"T1=({T1_v[0]:.2f}, {T1_v[1]:.2f})",
    fontsize=10, ha="left"
)
plt.text(
    T2_v[0], T2_v[1] + 0.2,
    f"T2=({T2_v[0]:.0f}, {T2_v[1]:.0f})",
    fontsize=10, ha="left"
)
plt.text(
    T_v[0], T_v[1] + 0.2,
    f"v'=({T_v[0]:.2f}, {T_v[1]:.2f})",
    fontsize=10, ha="left"
)

# Set plot limits.
plt.xlim(-1.0, 3.0)
plt.ylim(0.0, 3.5)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Ensure the aspect ratio is equal.
plt.gca().set_aspect("equal", adjustable="box")

# Define the legend.
plt.legend()

# Show all plots.
plt.show()
