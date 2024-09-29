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

"""This script demonstrates how to use the Decision Tree algorithm to classify
data points using the scikit-learn library."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import plot_tree


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


# Step 1: Define the dataset.
dataset = [
    ["Green (0)", 3, "Apple"],
    ["Yellow (2)", 3, "Apple"],
    ["Red (1)", 1, "Grape"],
    ["Red (1)", 1, "Grape"],
    ["Yellow (2)", 3, "Lemon"],
]

# Step 2: Convert the dataset into a DataFrame.
df = pd.DataFrame(dataset, columns=["Color", "Diameter", "Fruit"])
render_mpl_table(df)
plt.show()

# Step 3: Split the dataset into features (X) and labels (y).
X = np.array([[row[0], row[1]] for row in dataset]) # Features (Color, Diameter)
y = np.array([row[2] for row in dataset]) # Labels (Fruit)

# Step 4: Convert categorical data (Color) into numerical data using
# LabelEncoder.
label_encoder_color = LabelEncoder()
X[:, 0] = label_encoder_color.fit_transform(X[:, 0])  # Encode "Color"

# Step 5: Train a Decision Tree Classifier.
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Step 6: Test the model with new data.
new_fruits = [
    ["Green (0)", 3],
    ["Yellow (2)", 4],
    ["Red (1)", 2],
    ["Green (0)", 2]
]

# Encode the color of new fruits.
new_fruits_encoded = np.array(new_fruits)
new_fruits_encoded[:, 0] = label_encoder_color.transform(
    new_fruits_encoded[:, 0])

# Step 6: Make predictions.
predictions = clf.predict(new_fruits_encoded)

# Display the results.
for i, fruit in enumerate(new_fruits):
    print(f"Fruit with color {fruit[0]} and diameter {fruit[1]} "
          f"is predicted as: {predictions[i]}.")

# Step 7: Plot the Decision Tree.
plt.figure(figsize=(16, 10))
plot_tree(
    clf, feature_names=["Color", "Diameter"],
    class_names=clf.classes_, filled=True,
    rounded=True, proportion=True,
    precision=2, fontsize=12
)
plt.title("Decision Tree Visualization")
plt.show()
