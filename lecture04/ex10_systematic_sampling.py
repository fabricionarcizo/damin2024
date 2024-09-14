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

"""This script demonstrates how to perform systematic sampling on a dataset
using pandas."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Sample dataset: Customer data with 100 records.
np.random.seed(42)
data = {
    # 100 customers.
    "Customer_ID": np.arange(1, 101),
    # Random ages between 18 and 70.
    "Age": np.random.randint(18, 70, 100),
    # Random income between 20k and 120k.
    "Income": np.random.randint(20000, 120000, 100),
    # Random spending scores between 1 and 100.
    "Spending_Score": np.random.randint(1, 100, 100)
}

df = pd.DataFrame(data)
print("Original Dataset (first 10 rows):\n", df.head(10))

# 1. Systematic Sampling: Select every 10th record.
INTERVAL = 10  # Sampling interval.
df_systematic = df.iloc[::INTERVAL]

print("\nSystematically Selected Sample (every 10th observation):\n",
      df_systematic)


# Plotting the entire dataset.
plt.figure(figsize=(10, 8))
plt.scatter(df["Age"], df["Income"], label="Entire Dataset",
            color="#179E86")

# Plotting the selected sample.
plt.scatter(df_systematic["Age"], df_systematic["Income"],
            label="Selected Sample", color="#C03B26")

# Adding labels and title.
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Customer Data")

# Adding legend.
plt.legend()

# Displaying the plot.
plt.show()
