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

"""This script demonstrates how to perform One-Hot and Label Encoding on
categorical data using Pandas and Scikit-learn."""

# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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
        cellText=data.values, bbox=bbox,
        colLabels=data.columns, **kwargs
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


# Sample data: Customer segments and payment methods
data = {
    "Customer_ID": [1, 2, 3, 4, 5],
    "Segment": ["Youth", "Adult", "Senior", "Youth", "Senior"],
    "Payment_Method": ["Credit Card", "Debit Card", "Paypal", "Paypal", "Credit Card"]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)
render_mpl_table(df)
plt.show()

# 1. One-Hot Encoding.
# Create dummy variables for "Segment" and "Payment_Method".
one_hot_encoder_segment = OneHotEncoder()
one_hot_encoder_payment = OneHotEncoder()

df_one_hot = df.copy()
df_one_hot["Segment"] = \
    one_hot_encoder_segment.fit_transform(
        df[["Segment"]]).toarray().tolist()
df_one_hot["Payment_Method"] = \
    one_hot_encoder_payment.fit_transform(
        df[["Payment_Method"]]).toarray().tolist()

print("\nData after One-Hot Encoding:\n", df_one_hot)
render_mpl_table(df_one_hot, header_color="#F59B11")
plt.show()


# 2. Label Encoding.
# Apply Label Encoding for "Segment" and "Payment_Method".
label_encoder_segment = LabelEncoder()
label_encoder_payment = LabelEncoder()

df_label_encoded = df.copy()
df_label_encoded["Segment"] = label_encoder_segment.fit_transform(
    df["Segment"])
df_label_encoded["Payment_Method"] = label_encoder_payment.fit_transform(
    df["Payment_Method"])

print("\nData after Label Encoding:\n", df_label_encoded)
render_mpl_table(df_label_encoded, header_color="#C03B26")
plt.show()
