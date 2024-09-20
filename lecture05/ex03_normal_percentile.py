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

"""This script demonstrates the relationship between z-scores and percentiles.
"""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator
from scipy import stats


# Define the range of z-scores.
z_scores = np.linspace(-4, 4, 1000)

# Calculate the Gaussian distribution (mean=0, std=1 for normal distribution).
pdf = stats.norm.pdf(z_scores, 0, 1)

# Create the plot.
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.margins(y=0)

# Remove the y-axis lines.
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Plot and fill the Gaussian curve with color #C03B26 (Red).
ax1.fill_between(z_scores, pdf, color="#C03B26")

# Label the X-axis (z-scores).
ax1.set_xlabel("Z-scores")
ax1.set_xlim(-4, 4)

# Remove Y-axis.
ax1.get_yaxis().set_visible(False)

# Create a new X-axis at the top for the percentiles.
ax2 = ax1.twiny()

# Remove the y-axis lines.
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Define the range of percentiles (0 to 100 without the % sign).
percentiles_for_axis = np.linspace(0, 100, 101)

# Set the tick positions on the top X-axis to align with percentiles.
ax2.set_xticks(np.linspace(0, 100, 101))
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.tick_params(which="major", length=10, width=1)
ax2.tick_params(which="minor", length=5, color="#44546A")

# Set the labels for every fifth percentile and empty labels for others.
ax2.set_xticklabels([
    f"{int(p)}" if int(p) % 10 == 0 else "" for p in np.hstack(
        (0, percentiles_for_axis[::5])
    )
])

# Label the new X-axis (percentiles).
ax2.set_xlabel("Percentiles")
ax2.set_xlim(0, 100)

# Calculate the z-scores for each percentile (1st to 100th percentiles).
percentiles = np.linspace(0, 1, 101)
z_percentiles = stats.norm.ppf(percentiles)

# Draw lines for each percentile from the bottom to the top of the bell curve.
normalized_z = np.linspace(-4, 4, 101)
for index, z in enumerate(z_percentiles):

    # Define the line color and thickness.
    LINE_COLOR = "#000000" if index % 5 == 0 else "#44546A"
    LINE_THICKNESS = 1 if index % 5 == 0 else 0.5

    # Calculate the height of the bell curve at this z-score.
    pdf_value = stats.norm.pdf(z, 0, 1)

    # Draw a line from (z, 0) to (z, pdf_value).
    ax1.plot([z, z], [0, pdf_value],
             color=LINE_COLOR, linewidth=LINE_THICKNESS)

    # Draw a line from (z, pdf_value) to the corresponding z percentiles in the
    # percentile axis.
    ax1.plot([z, normalized_z[index]], [pdf_value, 0.5],
             color=LINE_COLOR, linewidth=LINE_THICKNESS)

# Show the plot.
plt.show()
