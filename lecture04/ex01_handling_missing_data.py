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

"""This script demonstrates how to handle missing data in a dataset using
various techniques."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer


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
        cellText=np.round(data.values, decimals=1), bbox=bbox,
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


# Create a sample dataset with missing values.
data = {
    "Feature1": [2.5, np.nan, 3.1, 4.0, np.nan],
    "Feature2": [np.nan, 2.7, np.nan, 3.3, 5.1],
    "Feature3": [1.2, 2.3, np.nan, 4.1, 5.5],
    "Feature4": [5.0, 6.2, 7.3, 8.5, 9.1],
    "Target": [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
print("\nOriginal data:\n", df)

# Render the table.
ax = render_mpl_table(df)
plt.show()

# 1. Deletion Method: Dropping rows with missing values
df_deletion = df.dropna()
print("Data after deletion of rows with missing values:\n", df_deletion)

# 2. Imputation Techniques

# Mean Imputation.
mean_imputer = SimpleImputer(strategy="mean")
df_mean_imputed = df.copy()
df_mean_imputed.iloc[:, :4] = mean_imputer.fit_transform(
    df_mean_imputed[["Feature1", "Feature2", "Feature3", "Feature4"]]
)
print("\nData after mean imputation:\n", df_mean_imputed)

# Render the table.
ax = render_mpl_table(df_mean_imputed, header_color="#F59B11")
plt.show()

# KNN Imputation.
knn_imputer = KNNImputer(n_neighbors=2)
df_knn_imputed = df.copy()
df_knn_imputed.iloc[:, :4] = knn_imputer.fit_transform(
    df_knn_imputed.iloc[:, :4]
)
print("\nData after KNN imputation:\n", df_knn_imputed)

# Render the table.
ax = render_mpl_table(df_knn_imputed, header_color="#C03B26")
plt.show()
