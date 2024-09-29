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

"""This script demonstrates how to use the k-NN algorithm to classify data
points using the scikit-learn library."""

# Import the required libraries.
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


# Load the Iris dataset.
# pylint: disable=E1101
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

# Filter the dataset to show only two classes (0 = setosa and 1 = versicolor).
filtered_iris_df = iris_df[iris_df["target"].isin([0, 1])].iloc[:, [0, 1, -1]]

# Generate three random data.
random.seed(2)
random_data = []
for _ in range(2):
    random_data.append(
        np.array([random.uniform(4.5, 7.0), random.uniform(2.0, 4.5)])
    )

# Find the nearest points from setosa and versicolor.
setosa_points = filtered_iris_df[
    filtered_iris_df["target"] == 0].iloc[:, :-1].values
versicolor_points = filtered_iris_df[
    filtered_iris_df["target"] == 1].iloc[:, :-1].values

nearest_setosa = setosa_points[
    np.argmin(
        np.linalg.norm(
            setosa_points - versicolor_points.mean(axis=0), axis=1
        )
    )
]
nearest_versicolor = versicolor_points[
    np.argmin(
        np.linalg.norm(
            versicolor_points - setosa_points.mean(axis=0), axis=1
        )
    )
]

# Calculate the midpoint between the nearest setosa and versicolor points.
midpoint = (nearest_setosa + nearest_versicolor) / 2
random_data.append(midpoint)
random_data = np.asarray(random_data)

# Plot the dataset before classification.
plt.figure(figsize=(8, 6))
plt.scatter(
    filtered_iris_df[filtered_iris_df["target"] == 0]["sepal length (cm)"],
    filtered_iris_df[filtered_iris_df["target"] == 0]["sepal width (cm)"],
    color="#179E86", label="Setosa", marker="o"
)
plt.scatter(
    filtered_iris_df[filtered_iris_df["target"] == 1]["sepal length (cm)"],
    filtered_iris_df[filtered_iris_df["target"] == 1]["sepal width (cm)"],
    color="#C03B26", label="Versicolor", marker="x"
)

# Plot the new random data.
plt.scatter(
    random_data[:, 0], random_data[:, 1],
    color="#44546A", label="Unknown", marker="*", s=200
)

# Plot information.
plt.title("Iris Dataset (Setosa and Versicolor) Before Classification")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(loc="upper left")
plt.show()

# Split the dataset into training and testing sets.
X = filtered_iris_df.drop("target", axis=1)
y = filtered_iris_df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the k-NN classifier.
k = 9  # Number of neighbors.
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Classify the random data using k-NN.
predicted_class = knn.predict(random_data)

# Plot the dataset after classification, including the random data.
plt.figure(figsize=(8, 6))
plt.scatter(
    filtered_iris_df[filtered_iris_df["target"] == 0]["sepal length (cm)"],
    filtered_iris_df[filtered_iris_df["target"] == 0]["sepal width (cm)"],
    color="#179E86", label="Setosa", marker="o"
)
plt.scatter(
    filtered_iris_df[filtered_iris_df["target"] == 1]["sepal length (cm)"],
    filtered_iris_df[filtered_iris_df["target"] == 1]["sepal width (cm)"],
    color="#C03B26", label="Versicolor", marker="x"
)

# Plot the classified random data.
for i in range(2):
    COLOR = "#C03B26" if i else "#179E86"
    CLASS_NAME = "versicolor" if i else "setosa"
    plt.scatter(
        random_data[predicted_class == i, 0],
        random_data[predicted_class == i, 1], color=COLOR,
        label=f"Random Point (Classified as {CLASS_NAME})",
        marker="*", s=200
    )

# Plot information.
plt.title(f"Iris Dataset (Setosa and Versicolor) After Classification (k={k})")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(loc="upper left")
plt.show()
