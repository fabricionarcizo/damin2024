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

"""This script demonstrates how to identify anomalies using autoencoders."""

# Import the required libraries.
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


# Set random seed for reproducibility.
np.random.seed(42)

# Define the colors to be used in the plot.
colors = [
    "#2580B7", # Blue
    "#C03B26", # Red
    "#44546A", # Gray
]

# Generate 100 samples for the cluster with some outliers.
cluster_samples = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
data = np.vstack((cluster_samples, outliers))

# Define the autoencoder model.
input_dim = data.shape[1]
ENCODING_DIM = 2

input_layer = Input(shape=(input_dim,))
encoder = Dense(ENCODING_DIM, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

# Train the autoencoder.
autoencoder.fit(
    data, data, epochs=100, batch_size=10, shuffle=True,
    validation_split=0.2, verbose=0
)

# Get the reconstruction error for each data point.
reconstructions = autoencoder.predict(data)
reconstruction_errors = np.mean(np.square(data - reconstructions), axis=1)

# Define a threshold for anomaly detection.
threshold = np.percentile(reconstruction_errors, 95)

# Identify anomalies.
anomalies = reconstruction_errors > threshold

# Plot the results.
plt.scatter(
    data[:, 0], data[:, 1],
    c=np.where(anomalies, colors[1], colors[0]),
    s=50, edgecolor=colors[-1], alpha=0.6
)

# Custom legend for clarity.
initial_patch = mpatches.Patch(color=colors[0], alpha=0.6, label="Normal Data")
final_patch = mpatches.Patch(color=colors[1], alpha=0.6, label="Anomaly")
plt.legend(
    handles=[initial_patch, final_patch]
)

plt.title("Outlier Detection using Autoencoders")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
