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

"""This script demonstrates how to use the FP-Growth algorithm to find
association rules."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from mlxtend.frequent_patterns import fpgrowth, association_rules


def render_mpl_table(
        data: pd.DataFrame, col_width: float = 4.0, row_height: float = 0.625,
        font_size: int = 14, header_color: str = "#179E86",
        row_colors: list = ["#f1f1f2", "#ffffff"], edge_color: str = "black",
        bbox: list = [0, 0, 1, 1], header_font_color: str = "white",
        ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """Function to display the DataFrame as a table with colored cells.

    Args:
        data (pd.DataFrame): The DataFrame to be displayed as a table.
        col_width (float, optional): The width of the columns. Defaults to 4.0.
        row_height (float, optional): The height of the rows. Defaults to 0.625.
        font_size (int, optional): The font size. Defaults to 14.
        header_color (str, optional): The color of the header. Defaults to
            "#179E86".
        row_colors (list, optional): The colors of the rows. Defaults to
            ["#f1f1f2", "#ffffff"].
        edge_color (str, optional): The color of the edges. Defaults to "black".
        bbox (list, optional): The bounding box. Defaults to [0, 0, 1, 1].
        header_font_color (str, optional): The color of the header font.
            Defaults to "white".
        ax (plt.Axes, optional): The axes to be used. Defaults to None.

    Returns:
        plt.Axes: The axes with the table.
    """

    # Create the figure and axes if not provided.
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * \
            np.array([col_width, row_height])
        _, ax = plt.subplots(figsize=size)
        ax.axis("off")

    # Create the table.
    mpl_table = ax.table(
        cellText=[
            [
                str(cell)
                    .replace("frozenset({", "")
                    .replace("})", "") for cell in row
            ] for row in data.values
        ],
        bbox=bbox,
        colLabels=data.columns,
        colWidths=[0.30, 0.70],
        **kwargs
    )

    # Set the font size.
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    # Styling the table.
    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)

        # Header row.
        if k[0] == 0:
            cell.set_text_props(weight="bold", color=header_font_color)
            cell.set_facecolor(header_color)

        # Alternate row coloring
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    return ax


def encode_units(x: int) -> int:
    """
    Function to encode units.

    Args:
        x (int): The quantity of the product.

    Returns:
        int: 1 if the quantity is greater than 0, 0 otherwise.
    """
    return 1 if x > 0 else 0


# Read the Excel file.
df = pd.read_excel("lecture13/data/online_retail.xlsx")

# Perform data cleaning to remove the rules that do not have invoice number.
df["Description"] = df["Description"].str.strip()
df["InvoiceNo"] = df["InvoiceNo"].astype("str")
df = df[~df["InvoiceNo"].str.contains("C")]

# Create a shopping cart.
shopping_cart = df[df["Country"] == "Portugal"].groupby(
    ["InvoiceNo", "Description"]
)["Quantity"].sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")

# Apply the encoding function to the shopping cart.
shopping_cart_sets = shopping_cart.map(encode_units)

# Drop the "POSTAGE" column if it exists.
if "POSTAGE" in shopping_cart_sets.columns:
    shopping_cart_sets.drop("POSTAGE", inplace=True, axis=1)

# Apply the FP-Growth algorithm with a minimum support of 0.07.
frequent_itemsets = fpgrowth(
    shopping_cart_sets, min_support=0.07, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

# Render the table.
render_mpl_table(frequent_itemsets.head(10))
plt.show()

# Generate the association rules.
rules = association_rules(
    frequent_itemsets, num_itemsets=len(frequent_itemsets),
    metric="lift", min_threshold=1
)

# Select the rules with a lift greater than 6 and a confidence greater than 0.8.
selected_rules = rules[(rules["lift"] >= 6) & (rules["confidence"] >= 0.8)]
print(
    selected_rules.head(30)[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ]
)

# Plot the top 50 items using TreeMap.
df["all"] = "Top 50 items"
fig = px.treemap(
    df.head(50), path=["all", "Description"], values="Quantity",
    color=df["Quantity"].head(50), hover_data=["Description"],
    color_continuous_scale="Blues",
)
fig.show()
