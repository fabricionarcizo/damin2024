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

"""This script demonstrates how to integrate data from different sources using
schema alignment and merging techniques."""

# Import the required libraries.
import xml.etree.ElementTree as ET

import pandas as pd


# 1. Reading the CSV file (Sales Database).
df_sales = pd.read_csv("lecture04/data/customer_sales.csv")

print("Original Customer Sales Data:")
print("=" * 92)
print(df_sales.head())
print("=" * 92)
print(df_sales.tail())
print("=" * 92)


def parse_xml(file: str) -> pd.DataFrame:
    """Parse an XML file and return a DataFrame with the data.
    
    Args:
        file (str): The path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the XML file.
    """

    # Parse the XML file.
    tree = ET.parse(file)
    root = tree.getroot()

    # Extract the data from the XML file.
    data = []
    for customer in root.findall("customer"):
        id_ = int(customer.find("id").text)
        name = customer.find("full_name").text
        email = customer.find("email").text
        purchase = customer.find("purchase").text
        signup_date = customer.find("signup_date").text
        data.append([id_, name, email, purchase, signup_date])

    # Creating a DataFrame.
    df = pd.DataFrame(data, columns=[
        "customer_id", "name", "email", "sales_amount", "signup_date"
    ])

    return df


# 2. Reading the XML file (Marketing Database).
df_marketing = parse_xml("lecture04/data/customer_marketing.xml")

print("\nOriginal Customer Marketing Data:")
print("=" * 92)
print(df_marketing.head())
print("=" * 92)
print(df_marketing.tail())
print("=" * 92)


# 3. Schema Alignment: Convert columns to appropriate types.
df_marketing["sales_amount"] = pd.to_numeric(
    df_marketing["sales_amount"], errors="coerce").fillna(0)
df_sales["sales_date"] = pd.to_datetime(df_sales["sales_date"])
df_marketing["signup_date"] = pd.to_datetime(df_marketing["signup_date"])


# 4. Merging the CSV and XML data intelligently.
# Merge on "customer_id", prioritize "df_sales" data when duplicate columns
# exist.
df_merged = pd.merge(df_sales, df_marketing, on="customer_id", how="outer",
                     suffixes=("_sales", "_marketing"))

# Now we will resolve the duplicated columns by:
#  - Prioritizing non-null values from the sales data.
#  - Filling missing values with data from the marketing dataset.
df_merged["name"] = df_merged["name_sales"].combine_first(
    df_merged["name_marketing"]
)
df_merged["sales_amount"] = df_merged["sales_amount_sales"].combine_first(
    df_merged["sales_amount_marketing"]
)
df_merged["sales_date"] = df_merged["sales_date"].combine_first(
    df_merged["signup_date"]
)

# Dropping the unnecessary columns (those with suffixes).
df_merged = df_merged.drop(columns=[
    "name_sales",
    "name_marketing",
    "sales_amount_sales",
    "sales_amount_marketing",
    "signup_date"
])

# Reorder the columns.
df_merged = df_merged.reindex(columns=[
    "customer_id",
    "name",
    "email",
    "sales_amount",
    "sales_date"
])


# 5. Display the unified schema.
print("\nUnified Data after Schema Integration:")
print("=" * 92)
print(df_merged.head())
print("=" * 92)
print(df_merged.tail())
print("=" * 92)
