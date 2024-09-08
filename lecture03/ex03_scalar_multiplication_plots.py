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

"""This script generates plots for scalar multiplication of a vector."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np


def plot_scalar_multiplication(scalar: float, ax: plt.Axes):
    """Function to generate plots for each scalar.

    Args:
        scalar (float): The scalar value to multiply the vector by.
        ax (plt.Axes): The axis to plot the vectors.
    """

    # Define the vector u.
    u = np.array([2, 3])

    # Scale the vector by the scalar.
    u_scaled = scalar * u

    # Plot the vector and the scaled vector.
    ax.quiver(
        0, 0, u_scaled[0], u_scaled[1], angles="xy", scale_units="xy", scale=1,
        color="#C03B26"
    )
    ax.text(
        u_scaled[0] + 0.2, u_scaled[1], f"scalar={scalar}", fontsize=10,
        ha="left"
    )

    # Plot the original vector.
    ax.quiver(
        0, 0, u[0], u[1], angles="xy", scale_units="xy", scale=1,
        color="#179E86", zorder=scalar >= 1
    )
    ax.text(
        u[0], u[1] + 0.1, f"u=({u[0]}, {u[1]})", fontsize=10, ha="right")

    # Set labels and limits.
    ax.set_xlim(-5, 6)
    ax.set_ylim(-6, 8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Ensure the aspect ratio is equal.
    ax.set_aspect("equal")


# Create subplots for each scalar multiplication.
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Positive scalar.
plot_scalar_multiplication(2.5, axs[0])

# Scalar between 0 and 1.
plot_scalar_multiplication(0.5, axs[1])

# Negative scalar.
plot_scalar_multiplication(-1.5, axs[2])

# Adjust the layout and show the plot.
plt.tight_layout()
plt.show()
