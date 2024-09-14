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

"""This script demonstrates how to deal with noisy data in a dataset using
various techniques."""

# Import the required libraries.
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from scipy.ndimage import uniform_filter1d
from scipy import stats


# Sample noisy data (x values and corresponding noisy y values).
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, size=x.shape)  # Linear trend + noise.

# Generate 5 random outliers.
random_outliers = np.random.uniform(low=x.min(), high=x.max(), size=5)
x = np.concatenate([x, random_outliers])

random_outliers = np.random.uniform(
    low=-y.max() * 4, high=y.max() * 4, size=5)
y = np.concatenate([y, random_outliers])

# Order the variables x and y by the values in x.
sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

# Create a DataFrame for ease of manipulation.
df = pd.DataFrame({"x": x, "y": y})

# Plot original noisy data
plt.figure(figsize=(10, 8))
plt.scatter(x, y, color="gray", label="Noisy Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# 1. Binning: Smooth data using mean of bins.
def binning(data: pd.Series, bin_size: int) -> Tuple[np.ndarray, int]:
    """Bin the data and calculate the mean of each bin.

    Args:
        data (pd.Series): The data to be binned.
        bin_size (int): The number of bins.

    Returns:
        Tuple[np.ndarray, int]: The bins and the mean of each bin.
    """
    bins = np.linspace(data.min(), data.max(), bin_size)
    bin_indices = np.digitize(data, bins)
    bin_means = [ data[bin_indices == i].mean() for i in range(1, len(bins)) ]
    return bins[1:], bin_means


bins, binned_data = binning(df["x"], 10)
plt.figure(figsize=(10, 8))
plt.scatter(df["x"], df["y"], color="gray", label="Noisy Data")
plt.plot(bins, binned_data, color="#C03B26", label="Binned Data", marker="o")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# 2. Regression: Fit a linear regression model to reduce noise.
model = LinearRegression()
model.fit(df[["x"]], df["y"])
y_pred = model.predict(df[["x"]])

plt.figure(figsize=(10, 8))
plt.scatter(df["x"], df["y"], color="gray", label="Noisy Data")
plt.plot(df["x"], y_pred, color="#2580B7", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# 3. Smoothing: Apply a moving average to smooth the noisy data.
smoothed_data = uniform_filter1d(df["y"], size=5)

plt.figure(figsize=(10, 8))
plt.scatter(df["x"], df["y"], color="gray", label="Noisy Data")
plt.plot(df["x"], smoothed_data, color="#179E86",
         label="Smoothed Data (Moving Average)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# 4. Outlier Detection: Using Z-score to detect and remove outliers.
z_scores = np.abs(stats.zscore(df["y"]))
outliers = z_scores > 2  # Z-score threshold for outliers.

# Remove outliers.
df_no_outliers = df[~outliers]

plt.figure(figsize=(10, 8))
plt.scatter(df["x"], df["y"], color="#C03B26", label="Noisy Data")
plt.scatter(df_no_outliers["x"], df_no_outliers["y"], color="#F59B11",
            label="Data Without Outliers")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
