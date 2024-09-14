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

"""This script demonstrates how to normalize and standardize data using the
Min-Max and Z-score scaling techniques."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Sample synthetic healthcare data.
np.random.seed(42)

data = {
    # Patient IDs.
    "Patient_ID": np.arange(1, 11),
    # Random blood pressure values.
    "Blood_Pressure": np.random.randint(110, 180, 10),
    # Random heart rate values.
    "Heart_Rate": np.random.randint(60, 100, 10)
}

df = pd.DataFrame(data)
print("Original Data:\n", df)


# 1. Normalization (Min-Max Scaling).
scaler_normalization = MinMaxScaler()
df_normalized = df.copy()
df_normalized[["Blood_Pressure", "Heart_Rate"]] = \
    scaler_normalization.fit_transform(df[["Blood_Pressure", "Heart_Rate"]])

print("\nData after Normalization (range between 0 and 1):\n", df_normalized)


# 2. Standardization (Z-score Scaling).
scaler_standardization = StandardScaler()
df_standardized = df.copy()
df_standardized[["Blood_Pressure", "Heart_Rate"]] = \
    scaler_standardization.fit_transform(df[["Blood_Pressure", "Heart_Rate"]])

print("\nData after Standardization (mean = 0, std = 1):\n", df_standardized)


# 3. Plotting the Original, Normalized, and Standardized Data.

# Original Data Plot.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(df["Blood_Pressure"], df["Heart_Rate"],
                color="#179E86")
axes[0].set_title("Original Data")
axes[0].set_xlabel("Blood Pressure")
axes[0].set_ylabel("Heart Rate")

# Normalized Data Plot.
axes[1].scatter(df_normalized["Blood_Pressure"], df_normalized["Heart_Rate"],
                color="#C03B26")
axes[1].set_title("Normalized Data (0-1)")
axes[1].set_xlabel("Blood Pressure")
axes[1].set_ylabel("Heart Rate")

plt.tight_layout()
plt.show()

# Original Data Plot.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(
    df["Blood_Pressure"], df["Heart_Rate"],
    color="#179E86"
)
axes[0].set_title("Original Data")
axes[0].set_xlabel("Blood Pressure")
axes[0].set_ylabel("Heart Rate")

# Standardized Data Plot.
axes[1].scatter(
    df_standardized["Blood_Pressure"], df_standardized["Heart_Rate"],
    color="#C03B26"
)
axes[1].set_title("Standardized Data (mean = 0, std = 1)")
axes[1].set_xlabel("Blood Pressure")
axes[1].set_ylabel("Heart Rate")

plt.tight_layout()
plt.show()
