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

"""This script demonstrates how to apply dimensionality reduction techniques to
a real-world dataset."""

# Import necessary libraries.
from typing import Tuple

import base64
import glob

import dash

import numpy as np
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

from dash import dcc, html, Input, Output

from plotly.subplots import make_subplots


# Define the colors to be used in the plot.
colors = [
    "#F59B11", # Yellow
    "#179E86", # Dark Green
    "#44546A", # Gray
]

def encode_image(image_path: str) -> str:
    """
    Encode an image to base64 format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    return f"data:image/png;base64,{encoded_string}"


# Load features and labels from .npy files.
features = np.load("lecture11/data/features.npy")
labels = np.load("lecture11/data/labels.npy")

# Load image paths.
image_paths = glob.glob("lecture11/data/**/*.jpg", recursive=True)
image_paths.sort()
encoded_images = [ encode_image(path) for path in image_paths ]

# Apply PCA.
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Apply LDA.
lda = LDA(n_components=1)
features_lda = lda.fit_transform(features, labels)
features_lda = np.hstack((features_lda, np.zeros_like(features_lda)))

# Apply t-SNE.
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)


# Initialize the Dash app.
app = dash.Dash(__name__)

# Create the 1x3 subplot figure.
fig = make_subplots(rows=1, cols=3, subplot_titles=["PCA", "LDA", "t-SNE"])


def add_scatter(features_2d:np.ndarray, col: int):
    """
    Add a scatter plot to the specified subplot.

    Args:
        features_2d (np.ndarray): The 2D features to be plotted.
        col (int): The column of the subplot.
    """
    fig.add_trace(
        go.Scatter(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            mode="markers",
            marker={ "color": [colors[label] for label in labels] },
            customdata=encoded_images,  # Attach images to points.
            hoverinfo="none"  # Disable default hover for custom tooltip.
        ),
        row=1, col=col
    )

# Add the PCA, LDA, and t-SNE scatter plots.
add_scatter(features_pca, 1)
add_scatter(features_lda, 2)
add_scatter(features_tsne, 3)

# Set figure layout.
fig.update_layout(
    height=600, width=1800, title_text="Dimensionality Reduction Techniques")

# Dash layout with dcc.Graph and dcc.Tooltip.
app.layout = html.Div([
    dcc.Graph(id="scatter-plot", figure=fig),
    dcc.Tooltip(id="graph-tooltip")
])

# Callback to display an image tooltip on hover.
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("scatter-plot", "hoverData")
)


def display_tooltip(hover_data: dict) -> Tuple[bool, dict, html.Div]:
    """
    Display a tooltip with an image when hovering over a point.

    Args:
        hover_data (dict): The data of the point being hovered over.

    Returns:
        Tuple[bool, dict, html.Div]: A tuple containing the tooltip display
            status, the bounding box of the hovered point, and the tooltip
            content.
    """
    if hover_data is None:
        return False, None, None

    # Get the index of the point being hovered over.
    point_index = hover_data["points"][0]["pointIndex"]
    bbox = hover_data["points"][0]["bbox"]

    # Set the tooltip content with the image.
    tooltip_content = html.Div([
        html.Img(src=encoded_images[point_index],
                 style={"width": "150px", "height": "150px"})
    ])

    return True, bbox, tooltip_content


# Run the app.
if __name__ == "__main__":
    app.run_server(debug=True)
