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

"""This script demonstrates how to use the k-means algorithm to cluster data
points."""

# Import the required libraries.
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans


# Define the colors to be used in the plot.
colors = [
    "#2580B7", # Blue
    "#F59B11", # Yellow
    "#C03B26", # Red
    "#44546A", # Gray
]

# Load the dataset.
data = pd.read_csv("./lecture09/data/starwars.csv")

# Extract the relevant columns.
height_mass_data = data[["height", "mass"]].dropna()

# Remove outliers using standard deviation
mean_mass = height_mass_data["mass"].mean()
std_mass = height_mass_data["mass"].std()

height_mass_clen_data = height_mass_data[
    (height_mass_data["mass"] > mean_mass - 3 * std_mass) &
    (height_mass_data["mass"] < mean_mass + 3 * std_mass)
]

# Perform K-Means clustering for height_mass_data
kmeans_original = KMeans(n_clusters=3, random_state=42)
kmeans_original.fit(height_mass_data)
clusters_original = kmeans_original.predict(height_mass_data)

# Add the cluster information to the original dataset
height_mass_data["Cluster"] = clusters_original

# Perform K-Means clustering for height_mass_clen_data
kmeans_cleaned = KMeans(n_clusters=3, random_state=42)
kmeans_cleaned.fit(height_mass_clen_data)
clusters_cleaned = kmeans_cleaned.predict(height_mass_clen_data)

# Add the cluster information to the cleaned dataset
height_mass_clen_data["Cluster"] = clusters_cleaned

# Plot for original data.
labels = ["Small", "Medium", "Large"]
for i, uid in enumerate([2, 0, 1]):
    plt.scatter(
        height_mass_data[height_mass_data["Cluster"] == uid]["height"],
        height_mass_data[height_mass_data["Cluster"] == uid]["mass"],
        c=colors[i], s=50, edgecolor=colors[-1], alpha=0.6, label=labels[i]
    )

plt.legend(loc="upper left")
plt.grid(True)
plt.xlabel("Height")
plt.ylabel("Mass")
plt.title("Height vs Mass Clustering (Original Data)")
plt.tight_layout()
plt.show()

# Plot for cleaned data.
for i, uid in enumerate([1, 0, 2]):
    plt.scatter(
        height_mass_clen_data[
            height_mass_clen_data["Cluster"] == uid]["height"],
        height_mass_clen_data[
            height_mass_clen_data["Cluster"] == uid]["mass"],
        c=colors[i], s=50, edgecolor=colors[-1], alpha=0.6, label=labels[i]
    )

plt.legend(["Small", "Medium", "Large"], loc="upper left")
plt.grid(True)
plt.xlabel("Height")
plt.ylabel("Mass")
plt.title("Height vs Mass Clustering (Cleaned Data)")
plt.tight_layout()
plt.show()
