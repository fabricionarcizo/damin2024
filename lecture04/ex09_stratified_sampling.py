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

"""This script demonstrates how to perform stratified sampling on a dataset
using pandas."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# Sample data: Customer segments, incomes, and spending scores.
np.random.seed(42)
data = {
    # 100 customers.
    "Customer_ID": np.arange(1, 101),
    # Randomly assigned customer segments.
    "Segment": np.random.choice(["Youth", "Adult", "Senior"],
                                size=100, p=[0.3, 0.5, 0.2]),
    # Random income between 20k and 120k.
    "Income": np.random.randint(20000, 120000, 100),
    # Random spending score between 1 and 100.
    "Spending_Score": np.random.randint(1, 100, 100)
}

df = pd.DataFrame(data)
print("Original Data:\n", df["Segment"].value_counts())

# 1. Stratified Sampling: Maintain the same distribution of "Segment" in the
# sample.
stratified_sample, _ = train_test_split(
    df, test_size=0.8, stratify=df["Segment"], random_state=42)

print("\nStratified Sample:\n", stratified_sample)
print("\nDistribution of Segments in Stratified Sample:\n",
      stratified_sample["Segment"].value_counts())


# Plotting the entire dataset.
plt.figure(figsize=(10, 8))
plt.scatter(df["Spending_Score"], df["Income"], label="Entire Dataset",
            color="#179E86")

# Plotting the selected sample.
plt.scatter(stratified_sample["Spending_Score"], stratified_sample["Income"],
            label="Selected Sample", color="#C03B26")

# Adding labels and title.
plt.xlabel("Spending Score")
plt.ylabel("Income")
plt.title("Customer Data")

# Adding legend.
plt.legend()

# Displaying the plot.
plt.show()
