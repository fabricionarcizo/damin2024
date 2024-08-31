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

"""This script demonstrates the use of Pandas to apply simple data manipulation.
"""

# Import the Python libraries.
import pandas as pd

# Assuming the large_data.csv file has been loaded into a DataFrame.
df = pd.read_csv("lecture02/data/data.csv")

# Selecting Columns.
# Single column selection.
ages = df["Age"]
print("Selected \"Age\" column:")
print(ages.head())

# Multiple columns selection.
selected_columns = df[["Name", "City"]]
print("\nSelected \"Name\" and \"City\" columns:")
print(selected_columns.head())

# Filtering Rows.
# Basic filtering: Filtering rows where "Age" is less than 30.
young_people = df[df["Age"] < 30]
print("\nFiltered rows where \"Age\" < 30:")
print(young_people.head())

# Filtering with multiple conditions: "Age" < 30 and "City" is "Dallas".
specific_group = df[(df["Age"] < 30) & (df["City"] == "Dallas")]
print("\nFiltered rows where \"Age\" < 30 and \"City\" is \"Dallas\":")
print(specific_group.head())

# Filtering using isin(): Filtering rows where "City" is either "New York" or
# "Dallas".
cities = df[df["City"].isin(["New York", "Dallas"])]
print("\nFiltered rows where \"City\" is \"New York\" or \"Dallas\":")
print(cities.head())
