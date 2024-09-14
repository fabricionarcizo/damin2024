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

"""This script demonstrates how to aggregate, discretize, and visualize
customer data using Pandas."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Sample synthetic customer data.
np.random.seed(42)
data = {
    # 20 customers.
    "Customer_ID": np.arange(1, 21),
    # Random ages between 18 and 70.
    "Age": np.random.randint(18, 70, 20),
    # Random income between 20k and 120k.
    "Income": np.random.randint(20000, 120000, 20),
    # Random spending scores between 1 and 100.
    "Spending_Score": np.random.randint(1, 100, 20),
    # Random purchase amounts between 100 and 10000.
    "Purchase_Amount": np.random.randint(100, 10000, 20)
}

df = pd.DataFrame(data)
print("Original Customer Data:\n", df)


# 1. Aggregation: Summarize data by calculating mean and sum.
# Example: Aggregating total purchase amounts and average spending score by
#          income groups.

# Define income bins (low, medium, high) for aggregation.
income_bins = [0, 40000, 80000, 120000]
income_labels = [
    "Low Income",
    "Medium Income",
    "High Income"
]
df["Income_Group"] = pd.cut(
    df["Income"], bins=income_bins, labels=income_labels)

# Aggregating data by income groups.
df_aggregated = df.groupby("Income_Group").agg({
    "Purchase_Amount": ["sum", "mean"],  # Total and average purchase amounts.
    "Spending_Score": "mean"  # Average spending score.
})

print("\nAggregated Data by Income Group:\n", df_aggregated)


# 2. Discretization: Convert continuous variables into categories (e.g., Age
#                    groups).
# Example: Discretizing age into categories like "Youth", "Adult", "Senior".

# Define age bins (youth, adult, senior) for discretization.
age_bins = [0, 25, 50, 70]
age_labels = [
    "Youth",
    "Adult",
    "Senior"
]
df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)

print("\nCustomer Data with Discretized Age Groups:\n",
      df[["Customer_ID", "Age", "Age_Group"]])


# 3. Applications: Visualizing aggregated data

# Plotting total purchase amount by income group.
df_aggregated["Purchase_Amount"]["sum"].plot(
    kind="bar", color=["#179E86", "#F59B11", "#C03B26"], figsize=(8, 5))
plt.title("Total Purchase Amount by Income Group")
plt.xticks(rotation=0)
plt.xlabel("")
plt.ylabel("Total Purchase Amount")
plt.show()

# Plotting average spending score by income group.
df_aggregated["Spending_Score"]["mean"].plot(
    kind="bar", color=["#179E86", "#F59B11", "#C03B26"], figsize=(8, 5))
plt.title("Average Spending Score by Income Group")
plt.xticks(rotation=0)
plt.xlabel("")
plt.ylabel("Average Spending Score")
plt.show()

# Plotting average purchase amount by age group.
df.groupby("Age_Group")["Purchase_Amount"].mean().plot(
    kind="bar", color=["#179E86", "#F59B11", "#C03B26"], figsize=(8, 5))
plt.title("Average Purchase Amount by Age Group")
plt.xticks(rotation=0)
plt.xlabel("")
plt.ylabel("Average Purchase Amount")
plt.show()
