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

"""This script demonstrates the use of Pandas to load CSV files."""

# Step 1: Import the pandas library.
import pandas as pd

# Step 2: Load the CSV file into a DataFrame.
# Replace 'lecture02/data/data.csv' with the path to your actual CSV file.
df = pd.read_csv("lecture02/data/data.csv")

# Step 3: Understanding the shape of your DataFrame.
print(f"DataFrame shape: {df.shape}.")

# Step 4: Preview the first and last few rows of the DataFrame.
print("\nPreview of the first 5 rows of the DataFrame:")
print(df.head())  # Displays the first 5 rows.
print("\nPreview of the last 10 rows of the DataFrame:")
print(df.tail(10)) # Displays the last 10 rows.

# Step 5: Inspect the structure of the DataFrame.
print("\nDataFrame structure and data types:")
print(df.info())  # Provides a summary of the DataFrame.

# Optional: Handling common issues.
# Handling missing values: Checking for NaN values.
print("\nChecking for missing values:")
print(df.isnull().sum())  # Displays the number of missing values per column.

# Loading only a portion of a large file (e.g., first 100 rows).
# df = pd.read_csv("data.csv", nrows=100)

# If the CSV file is in a different directory, ensure the correct path is used
# df = pd.read_csv("/path/to/your/data.csv")
