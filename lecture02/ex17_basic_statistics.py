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

"""This script demonstrates the use of basic statistics in Pandas."""

# Import the Python libraries.
import pandas as pd

# Assuming the large_data.csv file has been loaded into a DataFrame.
df = pd.read_csv("lecture02/data/data.csv")

# Calculating the Mean.
mean_age = df["Age"].mean()
print("Mean Age:", mean_age)

# Calculating the Median.
median_age = df["Age"].median()
print("Median Age:", median_age)

# Calculating the Sum.
sum_age = df["Age"].sum()
print("Total Sum of Ages:", sum_age)

# Applying statistics to multiple columns (if applicable).
# For example, calculate mean for all numeric columns in the DataFrame.
mean_values = df.mean(numeric_only=True)
print("\nMean values for all numeric columns:")
print(mean_values)

# Calculating the median for all numeric columns.
median_values = df.median(numeric_only=True)
print("\nMedian values for all numeric columns:")
print(median_values)

# Calculating the sum for all numeric columns.
sum_values = df.sum(numeric_only=True)
print("\nSum of values for all numeric columns:")
print(sum_values)

# Descriptive statistics.
summary = df.describe()
print("\nSummary of Descriptive Statistics:")
print(summary)
