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

"""This script demonstrates how to detect and deal with outliers in a dataset
using various techniques."""

# Import the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# Sample data with outliers.
np.random.seed(42)
data = np.random.normal(50, 10, 100)  # Generate normal data.
data_with_outliers = np.append(data, [150, 160, 170])  # Add extreme outliers.

df = pd.DataFrame({"Data": data_with_outliers})


# 1. Z-Score Method.
z_scores = np.abs(stats.zscore(df["Data"]))
THRESHOLD = 3  # Common threshold for z-scores.
outliers_z = df[z_scores > THRESHOLD]

print("Outliers detected using Z-Score method:\n", outliers_z)


# 2. Interquartile Range (IQR) Method.
Q1 = df["Data"].quantile(0.25)
Q3 = df["Data"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df["Data"] < lower_bound) | (df["Data"] > upper_bound)]

print("\nOutliers detected using IQR method:\n", outliers_iqr)


# 3. Visual Detection.
plt.figure(figsize=(12, 6))

# Scatter Plot.
plt.subplot(1, 2, 1)
plt.scatter(
    range(len(df)), df["Data"], color="#2580B7", label="Data Points")
plt.axhline(
    y=upper_bound, color="#C03B26", linestyle="--", label="Upper Bound (IQR)")
plt.axhline(
    y=lower_bound, color="#C03B26", linestyle="--", label="Lower Bound (IQR)")

plt.title("Scatter Plot for Outlier Detection")
plt.xlabel("Index")
plt.ylabel("Data Values")
plt.ylim(0, 200)
plt.legend()

# Box Plot.
plt.subplot(1, 2, 2)
plt.boxplot(
    df["Data"], vert=True, patch_artist=True,
    boxprops={
        "facecolor": "#2580B7"
    }
)

plt.title("Box Plot for Outlier Detection (Vertical)")
plt.ylabel("Data Values")
plt.ylim(0, 200)

plt.show()


# 4. Removing or Correcting Outliers.

# Removing outliers using the IQR method.
df_no_outliers = df[(df["Data"] >= lower_bound) & (df["Data"] <= upper_bound)]

print("\nData after removing outliers (using IQR):\n", df_no_outliers)

# You could also transform outliers instead of removing them (e.g., capping
# them to bounds).
df_transformed = df.copy()
df_transformed["Data"] = np.clip(df["Data"], lower_bound, upper_bound)

print("\nData after correcting outliers (capping to bounds):\n", df_transformed)
